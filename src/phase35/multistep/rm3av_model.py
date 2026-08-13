"""Executable architecture interventions for the RM3 independent-audit screen."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import Phase35ProtocolError
from .gatec_contracts import GateCModelConfig
from .gatec_model import (
    CoordinateRestrictedResponse,
    ZeroLocalResponse,
    build_gatec_model,
)
from .rm3_prediction import RM3FairPredictionAdapter, RM3PredictionConfig
from .rm3av_contracts import RM3AV_CANDIDATE_IDS


_BASE_CANDIDATE = {
    **{key: "P3_gatec_paired_free" for key in ("C00", "C03", "C25", "C28")},
    **{
        key: "P4_gatec_a1_scheduled"
        for key in (
            "C01", "C07", "C08", "C09", "C10", "C11", "C17", "C18",
            "C19", "C20", "C21", "C22", "C23", "C24", "C26", "C29",
        )
    },
    **{
        key: "P5_hybrid_joint_latent"
        for key in (
            "C02", "C04", "C05", "C06", "C12", "C13", "C14", "C15",
            "C16", "C27", "C30", "C31",
        )
    },
}


@dataclass(frozen=True)
class RM3AVModelConfig:
    candidate_id: str
    window: int
    horizon: int
    n_features: int
    d_model: int = 64
    latent_dim: int = 32
    dropout: float = 0.1

    def validate(self) -> None:
        if self.candidate_id not in RM3AV_CANDIDATE_IDS:
            raise Phase35ProtocolError("RM3-AV candidate is not frozen in the matrix")
        if min(self.window, self.horizon, self.n_features, self.d_model, self.latent_dim) < 1:
            raise Phase35ProtocolError("RM3-AV model dimensions must be positive")
        if self.horizon != 60:
            raise Phase35ProtocolError("RM3-AV model horizon must remain H60")
        if not 0.0 <= self.dropout < 1.0:
            raise Phase35ProtocolError("RM3-AV dropout must be in [0,1)")


class HistoryOnlyTerminalBypass(nn.Module):
    """Capacity-limited terminal correction with no future-action argument."""

    def __init__(self, n_features: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, horizon * 2),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.network(history[:, -1]).reshape(-1, self.horizon, 2)


class StructuredPIValveDecoder(nn.Module):
    """Causal PI core using the measured terminal-temperature feedback proxy."""

    def __init__(self, context_dim: int, *, residual: bool) -> None:
        super().__init__()
        self.residual = residual
        self.kp_raw = nn.Parameter(torch.full((2,), -2.0))
        self.ki_raw = nn.Parameter(torch.full((2,), -5.0))
        self.context_gain = nn.Linear(context_dim, 4)
        nn.init.zeros_(self.context_gain.weight)
        nn.init.zeros_(self.context_gain.bias)
        if residual:
            self.cell = nn.GRUCell(6, context_dim)
            self.output = nn.Linear(context_dim, 2)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        context: torch.Tensor,
        future_sp: torch.Tensor,
        baseline_valve: torch.Tensor,
        baseline_temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if baseline_temperature is None:
            raise Phase35ProtocolError("structured PI valve decoder requires temperature feedback")
        scheduled = self.context_gain(context).reshape(-1, 2, 2)
        kp = F.softplus(self.kp_raw)[None] * torch.exp(0.1 * torch.tanh(scheduled[..., 0]))
        ki = F.softplus(self.ki_raw)[None] * torch.exp(0.1 * torch.tanh(scheduled[..., 1]))
        integral = torch.zeros_like(baseline_valve)
        valve = baseline_valve
        hidden = context
        values: list[torch.Tensor] = []
        for step in range(future_sp.shape[1]):
            error = (future_sp[:, step] - baseline_temperature) / 20.0
            integral = torch.clamp(integral + error, -20.0, 20.0)
            increment = -(kp * error + ki * integral)
            if self.residual:
                hidden = self.cell(
                    torch.cat((error, integral / 20.0, valve / 100.0), dim=1), hidden
                )
                increment = increment + 0.5 * torch.tanh(self.output(hidden))
            valve = torch.clamp(valve + 2.0 * torch.tanh(increment), 0.0, 100.0)
            values.append(valve)
        return torch.stack(values, dim=1)


class DiagonalOnlyResponse(nn.Module):
    """Remove cross-side effects while retaining each fitted diagonal channel."""

    def __init__(self, operator: nn.Module) -> None:
        super().__init__()
        self.operator = operator
        self.route = f"{getattr(operator, 'route', 'response')}_diagonal_only"

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        side_effects: list[torch.Tensor] = []
        full = self.operator(context, future_valve, baseline_valve)
        for side in range(2):
            restricted = baseline_valve[:, None].expand_as(future_valve).clone()
            restricted[..., side] = future_valve[..., side]
            response = self.operator(context, restricted, baseline_valve)
            side_effects.append(response["effect"][..., side])
        return {
            **full,
            "effect": torch.stack(side_effects, dim=-1),
            "operator_family": self.route,
        }


class AuditShapeResponse(nn.Module):
    """Small response-shape family used only to test the audit's timing hypotheses."""

    def __init__(self, shape: str, horizon: int, dt_seconds: float = 10.0) -> None:
        super().__init__()
        allowed = {
            "one_pole", "two_pole", "power_basis", "linear_ramp",
            "three_pole_bounded_dead_time", "unconstrained_diagnostic",
        }
        if shape not in allowed:
            raise Phase35ProtocolError("unknown RM3-AV response shape")
        self.shape = shape
        self.route = f"rm3av_{shape}"
        self.horizon = horizon
        self.dt_seconds = dt_seconds
        bases = {"one_pole": 1, "two_pole": 2}.get(shape, 3)
        self.bases = bases
        tau = torch.tensor((60.0, 240.0, 900.0))[:bases]
        self.decay_logits = nn.Parameter(torch.logit(torch.exp(-dt_seconds / tau)))
        if shape == "unconstrained_diagnostic":
            self.mixing = nn.Parameter(torch.eye(2))
        else:
            self.diagonal_raw = nn.Parameter(torch.zeros(2))
            self.cross_raw = nn.Parameter(torch.full((2,), -3.0))
        self.basis_weights = nn.Parameter(torch.zeros(2, bases))
        self.dead_time_logits = nn.Parameter(torch.zeros(4)) if "dead_time" in shape else None
        self.power_raw = nn.Parameter(torch.zeros(bases)) if shape == "power_basis" else None

    def _mixing(self) -> torch.Tensor:
        if self.shape == "unconstrained_diagnostic":
            return self.mixing
        diagonal = F.softplus(self.diagonal_raw) + 1e-3
        cross = 0.2 * torch.sigmoid(self.cross_raw)
        return torch.stack(
            (torch.stack((diagonal[0], cross[0])), torch.stack((cross[1], diagonal[1])))
        )

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        del context
        dose = torch.clamp(future_valve / 100.0, 0.0, 1.0) - torch.clamp(
            baseline_valve[:, None] / 100.0, 0.0, 1.0
        )
        drive = torch.einsum("bhi,oi->bho", dose, self._mixing())
        if self.dead_time_logits is not None:
            delayed = []
            for delay in range(4):
                if delay == 0:
                    delayed.append(drive)
                else:
                    delayed.append(
                        torch.cat((torch.zeros_like(drive[:, :delay]), drive[:, :-delay]), dim=1)
                    )
            drive = torch.sum(
                torch.stack(delayed, dim=-1)
                * F.softmax(self.dead_time_logits, dim=0).view(1, 1, 1, -1),
                dim=-1,
            )
        decay = 0.05 + 0.945 * torch.sigmoid(self.decay_logits)
        state = future_valve.new_zeros((len(future_valve), 2, self.bases))
        states: list[torch.Tensor] = []
        effects: list[torch.Tensor] = []
        weights = F.softmax(self.basis_weights, dim=1)
        for step in range(future_valve.shape[1]):
            target = drive[:, step, :, None]
            if self.shape == "power_basis":
                assert self.power_raw is not None
                lags = torch.arange(
                    step + 1, device=drive.device, dtype=drive.dtype
                ) + 1.0
                exponents = 0.25 + 1.75 * torch.sigmoid(self.power_raw)
                kernels = lags[:, None].pow(-exponents[None])
                kernels = kernels / kernels.sum(dim=0, keepdim=True)
                history = drive[:, : step + 1].flip(1)
                state = torch.einsum("bti,tk->bik", history, kernels)
            elif self.shape == "linear_ramp":
                state = state + (1.0 - decay)[None, None] * torch.clamp(
                    target - state, -0.05, 0.05
                )
            else:
                state = decay[None, None] * state + (1.0 - decay)[None, None] * target
            effects.append(torch.sum(weights[None] * state, dim=2))
            padded = F.pad(state, (0, max(0, 3 - self.bases)))[:, :, :3]
            states.append(padded.reshape(len(future_valve), 6))
        return {
            "effect": torch.stack(effects, dim=1),
            "state": torch.stack(states, dim=1),
            "stable_poles": decay,
            "operator_family": self.route,
        }


def _module_scoped_initialization(model: nn.Module) -> None:
    """Name-seed random neural tensors while preserving deliberate physical/zero initializers."""

    physical_tokens = (
        "tau_logits",
        "pole_weights",
        "diagonal_gain",
        "cross_gain_logits",
        "power_logits",
        "decay_logits",
        "rate_raw",
        "input_scale",
        "output_weights",
        "memory_decay_logits",
        "dead_time_logits",
        "kp_raw",
        "ki_raw",
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(token in name for token in physical_tokens):
                continue
            if bool(torch.count_nonzero(parameter).item()) is False:
                # Zero-initialized response/readout heads are part of the architecture contract.
                continue
            if name.endswith(("normalization.weight", "norm.weight")):
                parameter.fill_(1.0)
                continue
            digest = hashlib.sha256(name.encode("utf-8")).digest()
            generator = torch.Generator(device=parameter.device).manual_seed(
                int.from_bytes(digest[:8], "little") % (2**63 - 1)
            )
            parameter.uniform_(-0.05, 0.05, generator=generator)


def module_state_hashes(model: nn.Module) -> dict[str, str]:
    """Content hashes by semantic module, before training mutates parameters."""

    groups = {
        "encoder": ("base.model.encoder.", "base.model.backbone.", "base.model.context_projection."),
        "valve_policy": ("base.model.valve_policy.",),
        "tin": ("base.model.tin_forecaster.", "base.model.tin_head."),
        "free_residual": ("base.model.residual_head.",),
        "response": ("base.model.local_response.",),
        "downstream": ("base.model.downstream.", "base.model.joint."),
        "bypass": ("extra_bypass.", "base.model.joint.terminal_bypass."),
    }
    state = model.state_dict()
    result = {}
    for group, prefixes in groups.items():
        digest = hashlib.sha256()
        found = False
        for name in sorted(state):
            if not name.startswith(prefixes):
                continue
            found = True
            value = state[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        result[group] = digest.hexdigest() if found else "not_applicable"
    return result


class RM3AVModel(nn.Module):
    def __init__(self, config: RM3AVModelConfig, feature_names: Sequence[str]) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.feature_names = tuple(feature_names)
        base_id = _BASE_CANDIDATE[config.candidate_id]
        base_config = RM3PredictionConfig(
            candidate_id=base_id,
            window=config.window,
            horizon=config.horizon,
            n_features=config.n_features,
            d_model=config.d_model,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.base = RM3FairPredictionAdapter(base_config, feature_names)
        self.base_candidate_id = base_id
        self.valve_decoder_family = "gru"
        self.extra_bypass: HistoryOnlyTerminalBypass | None = None
        self.register_buffer("action_shield", torch.eye(config.n_features), persistent=True)
        self.register_buffer(
            "_action_shield_fitted",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        candidate = config.candidate_id

        if candidate in {"C07", "C08"}:
            residual_capacity = "small" if candidate == "C07" else "large"
            self.base.model = build_gatec_model(
                GateCModelConfig(
                    window=config.window,
                    horizon=config.horizon,
                    n_features=config.n_features,
                    d_model=config.d_model,
                    latent_dim=config.latent_dim,
                    local_state_dim=6,
                    response_route="a1phys_three_pole",
                    residual_capacity=residual_capacity,
                    response_scheduling="scheduled",
                    dropout=config.dropout,
                ),
                feature_names,
            )
        if candidate == "C03":
            self.extra_bypass = HistoryOnlyTerminalBypass(
                config.n_features, min(16, config.d_model), config.horizon
            )
        if candidate == "C06":
            self.base.model.local_response = ZeroLocalResponse(6)
        if candidate == "C17":
            response = self.base.model.local_response.operator
            self.base.model.local_response = CoordinateRestrictedResponse(response, "common_only")
        if candidate == "C18":
            response = self.base.model.local_response.operator
            self.base.model.local_response = DiagonalOnlyResponse(response)
        shapes = {
            "C19": "one_pole",
            "C20": "two_pole",
            "C21": "power_basis",
            "C22": "linear_ramp",
            "C23": "three_pole_bounded_dead_time",
            "C24": "unconstrained_diagnostic",
        }
        if candidate in shapes:
            self.base.model.local_response = AuditShapeResponse(
                shapes[candidate], config.horizon
            )
        if candidate in {"C15", "C16"}:
            residual = candidate == "C16"
            self.base.model.valve_policy = StructuredPIValveDecoder(
                config.d_model, residual=residual
            )
            self.valve_decoder_family = "structured_pi_plus_gru" if residual else "structured_pi"
        if candidate not in {"C00", "C01", "C02"}:
            _module_scoped_initialization(self)

        self.valve_indices = self.base.valve_indices
        self.tin_indices = self.base.tin_indices
        self.tout_indices = self.base.tout_indices
        self.terminal_indices = self.base.terminal_indices

    def set_history_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        self.base.set_history_normalization(center, scale)

    def explicit_response(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        if self.base_candidate_id == "P3_gatec_paired_free":
            return ZeroLocalResponse(6)(context, future_valve, baseline_valve)
        return self.base.model.local_response(context, future_valve, baseline_valve)

    def set_action_shield(self, projector: torch.Tensor) -> None:
        if self.config.candidate_id != "C09":
            raise Phase35ProtocolError("action shield may be fitted only for C09")
        if projector.shape != self.action_shield.shape or not torch.isfinite(projector).all():
            raise Phase35ProtocolError("RM3-AV action shield shape/value is invalid")
        if not torch.allclose(projector, projector.T, atol=1e-5, rtol=0.0):
            raise Phase35ProtocolError("RM3-AV action shield must be symmetric")
        self.action_shield.copy_(projector.detach().to(self.action_shield))
        self._action_shield_fitted.fill_(True)

    @property
    def action_shield_fitted(self) -> bool:
        return bool(self._action_shield_fitted.item())

    def _p5_context(self, history: torch.Tensor) -> torch.Tensor:
        model = self.base.model
        normalized = (history - model.history_center) / model.history_scale
        _, flat = model.backbone(normalized)
        return torch.tanh(model.context_projection(flat))

    def _p5_forward_with_state(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        initial_local_state: torch.Tensor | None = None,
        initial_latent_state: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        baseline_valve = history[:, -1, self.base.valve_indices]
        baseline_tin = history[:, -1, self.base.tin_indices]
        baseline_tout = history[:, -1, self.base.tout_indices]
        return self.base.model(
            history,
            future_sp,
            baseline_valve=baseline_valve,
            baseline_tin=baseline_tin,
            baseline_local=baseline_tin - baseline_tout,
            baseline_terminal=history[:, -1, self.base.terminal_indices],
            initial_local_state=initial_local_state,
            initial_latent_state=initial_latent_state,
        )

    def _apply_action_shield(
        self, history: torch.Tensor, output: dict[str, Any]
    ) -> dict[str, Any]:
        if self.config.candidate_id != "C09" or not self.action_shield_fitted:
            return output
        model = self.base.model
        shielded_history = torch.einsum("bwf,fg->bwg", history, self.action_shield)
        normalized = (shielded_history - model.history_center) / model.history_scale
        residual_context = model.encoder(normalized)
        residual_delta = model.residual_head(residual_context).reshape(
            -1, self.config.horizon, 2
        )
        baseline_tin = history[:, -1, self.base.tin_indices]
        baseline_tout = history[:, -1, self.base.tout_indices]
        residual_local = baseline_tin[:, None] - baseline_tout[:, None] + residual_delta
        local_drop = residual_local + output["local_effect"]
        tout = output["boundary_used"] - local_drop
        original_normalized = (history - model.history_center) / model.history_scale
        original_context = model.encoder(original_normalized)
        baseline_terminal = history[:, -1, self.base.terminal_indices]
        terminal, latent = model.downstream(original_context, tout, baseline_terminal)
        return {
            **output,
            "residual_local_delta_prediction": residual_delta,
            "residual_local_prediction": residual_local,
            "local_drop_prediction": local_drop,
            "tout_prediction": tout,
            "terminal_prediction": terminal,
            "latent_state": latent,
        }

    def _diagnostic_action_forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        action: torch.Tensor,
        *,
        boundary_tin: torch.Tensor | None = None,
        local_override: torch.Tensor | None = None,
        response_off: bool = False,
    ) -> dict[str, Any]:
        baseline_valve = history[:, -1, self.valve_indices]
        baseline_tin = history[:, -1, self.tin_indices]
        baseline_tout = history[:, -1, self.tout_indices]
        baseline_local = baseline_tin - baseline_tout
        baseline_terminal = history[:, -1, self.terminal_indices]
        if self.base_candidate_id in {"P3_gatec_paired_free", "P4_gatec_a1_scheduled"}:
            model = self.base.model
            normalized = (history - model.history_center) / model.history_scale
            context = model.encoder(normalized)
            tin = model.tin_forecaster(context, baseline_tin, self.config.horizon)
            boundary = tin if boundary_tin is None else boundary_tin
            residual_delta = model.residual_head(context).reshape(-1, self.config.horizon, 2)
            residual = baseline_local[:, None] + residual_delta
            if response_off:
                effect = torch.zeros_like(residual)
                state = residual.new_zeros((*residual.shape[:2], 6))
            else:
                response = self.explicit_response(context, action, baseline_valve)
                effect, state = response["effect"], response["state"]
            local = residual + effect if local_override is None else local_override
            tout = boundary - local
            terminal, latent = model.downstream(context, tout, baseline_terminal)
            result = {
                "valve_prediction": action,
                "tin_prediction": tin,
                "boundary_used": boundary,
                "residual_local_prediction": residual,
                "local_drop_prediction": local,
                "tout_prediction": tout,
                "terminal_prediction": terminal,
                "terminal_physical_prediction": terminal,
                "terminal_bypass": torch.zeros_like(terminal),
                "explicit_local_effect": effect,
                "local_effect": effect,
                "local_state": state,
                "latent_state": latent,
            }
        else:
            model = self.base.model
            normalized = (history - model.history_center) / model.history_scale
            _, flat = model.backbone(normalized)
            context = torch.tanh(model.context_projection(flat))
            tin = baseline_tin[:, None] + model.tin_head(context).reshape(
                -1, self.config.horizon, 2
            )
            boundary = tin if boundary_tin is None else boundary_tin
            if response_off:
                effect = torch.zeros_like(boundary)
                state = boundary.new_zeros((*boundary.shape[:2], 6))
            else:
                response = self.explicit_response(context, action, baseline_valve)
                effect, state = response["effect"], response["state"]
            if local_override is None:
                joint = model.joint(
                    context,
                    boundary,
                    effect,
                    baseline_valve=baseline_valve,
                    baseline_tin=baseline_tin,
                    baseline_local=baseline_local,
                    baseline_terminal=baseline_terminal,
                )
            else:
                # Local truth replaces the explicit interface only for oracle attribution.
                joint = model.joint(
                    context,
                    boundary,
                    local_override - baseline_local[:, None],
                    baseline_valve=baseline_valve,
                    baseline_tin=baseline_tin,
                    baseline_local=baseline_local,
                    baseline_terminal=baseline_terminal,
                )
            result = {
                **joint,
                "valve_prediction": action,
                "tin_prediction": tin,
                "local_state": state,
                "local_effect": effect,
            }
        terminal = result["terminal_prediction"]
        physical = result.get("terminal_physical_prediction", terminal)
        bypass = result.get("terminal_bypass", torch.zeros_like(terminal))
        if self.config.candidate_id == "C03":
            assert self.extra_bypass is not None
            bypass = self.extra_bypass(history)
            physical = terminal
            terminal = physical + bypass
        elif self.config.candidate_id == "C04":
            bypass = torch.zeros_like(bypass)
            terminal = physical
        elif self.config.candidate_id == "C05":
            physical = baseline_terminal[:, None].expand_as(terminal)
            terminal = physical + bypass
        return {
            **result,
            "terminal_prediction": terminal,
            "terminal_physical_prediction": physical,
            "terminal_bypass": bypass,
            "explicit_local_effect": effect,
        }

    def diagnostic_forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        logged_future_valve: torch.Tensor,
        logged_future_tin: torch.Tensor,
        local_target: torch.Tensor,
    ) -> dict[str, dict[str, Any]]:
        if not (
            logged_future_valve.shape == logged_future_tin.shape == local_target.shape == future_sp.shape
        ):
            raise Phase35ProtocolError("RM3-AV diagnostic future tensor shape mismatch")
        normal = self(history, future_sp)
        predicted = normal["valve_prediction"]
        shuffled = logged_future_valve.flip(0)
        wrong_side = logged_future_valve.flip(-1)
        lead = torch.cat((logged_future_valve[:, 1:], logged_future_valve[:, -1:]), dim=1)
        bypass_off = {**normal, "terminal_prediction": normal["terminal_physical_prediction"]}
        baseline_terminal = history[:, -1, self.terminal_indices]
        bypass_only = {
            **normal,
            "terminal_prediction": baseline_terminal[:, None] + normal["terminal_bypass"],
        }
        return {
            "normal": normal,
            "bypass_off": bypass_off,
            "bypass_only": bypass_only,
            "response_off": self._diagnostic_action_forward(
                history, future_sp, predicted, response_off=True
            ),
            "predicted_valve": self._diagnostic_action_forward(history, future_sp, predicted),
            "logged_valve": self._diagnostic_action_forward(
                history, future_sp, logged_future_valve
            ),
            "logged_valve_oracle_tin": self._diagnostic_action_forward(
                history, future_sp, logged_future_valve, boundary_tin=logged_future_tin
            ),
            "oracle_local": self._diagnostic_action_forward(
                history,
                future_sp,
                logged_future_valve,
                boundary_tin=logged_future_tin,
                local_override=local_target,
            ),
            "shuffled": self._diagnostic_action_forward(history, future_sp, shuffled),
            "wrong_side": self._diagnostic_action_forward(history, future_sp, wrong_side),
            "lead": self._diagnostic_action_forward(history, future_sp, lead),
        }

    def action_alignment_sensitivity(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        logged_future_valve: torch.Tensor,
        logged_future_tin: torch.Tensor,
        local_target: torch.Tensor,
    ) -> dict[str, dict[str, Any]]:
        if not (
            logged_future_valve.shape == logged_future_tin.shape == local_target.shape == future_sp.shape
        ):
            raise Phase35ProtocolError("RM3-AV alignment future tensor shape mismatch")
        result = {}
        for shift_steps in range(-3, 4):
            shifted = torch.roll(logged_future_valve, shifts=shift_steps, dims=1)
            if shift_steps < 0:
                shifted[:, shift_steps:] = logged_future_valve[:, -1:]
            elif shift_steps > 0:
                shifted[:, :shift_steps] = logged_future_valve[:, :1]
            result[str(shift_steps * 10)] = self._diagnostic_action_forward(
                history,
                future_sp,
                shifted,
                boundary_tin=logged_future_tin,
            )
        return result

    def forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        logged_future_valve: torch.Tensor | None = None,
        oof_action_residual: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        auxiliary_candidates = {"C10", "C11", "C12", "C13"}
        if logged_future_valve is not None and self.config.candidate_id not in auxiliary_candidates:
            raise Phase35ProtocolError("logged future valve is restricted to declared RM3-AV auxiliaries")
        if oof_action_residual is not None and self.config.candidate_id not in {"C11", "C12"}:
            raise Phase35ProtocolError("OOF action residual is restricted to R-loss candidates")
        if self.base_candidate_id == "P4_gatec_a1_scheduled" and logged_future_valve is not None:
            output = self.base.model(
                history,
                future_sp,
                boundary_mode="forecast_boundary",
                logged_future_valve_for_aux=logged_future_valve,
            )
            output = {
                **output,
                "action_used": output["valve_prediction"],
                "action_access": "future_sp_to_predicted_valve",
                "deployable": True,
                "prefix_causal_action_path": True,
            }
        else:
            output = self.base(history, future_sp)
            if logged_future_valve is not None:
                baseline_valve = history[:, -1, self.valve_indices]
                logged = self.explicit_response(
                    self._p5_context(history), logged_future_valve, baseline_valve
                )
                output = {
                    **output,
                    "logged_local_effect": logged["effect"],
                    "logged_local_drop_prediction": (
                        output["local_drop_prediction"]
                        - output["explicit_local_effect"]
                        + logged["effect"]
                    ),
                }

        output = self._apply_action_shield(history, output)
        if oof_action_residual is not None:
            if oof_action_residual.shape != future_sp.shape:
                raise Phase35ProtocolError("RM3-AV OOF action residual shape mismatch")
            baseline_valve = history[:, -1, self.valve_indices]
            residual_response = self.explicit_response(
                self._p5_context(history)
                if self.base_candidate_id == "P5_hybrid_joint_latent"
                else self.base.model.encoder(
                    (history - self.base.model.history_center) / self.base.model.history_scale
                ),
                baseline_valve[:, None] + oof_action_residual,
                baseline_valve,
            )
            output = {**output, "oof_response_prediction": residual_response["effect"]}

        terminal = output["terminal_prediction"]
        explicit = output.get("explicit_local_effect", output.get("local_effect"))
        if explicit is None:
            explicit = terminal.new_zeros(terminal.shape)
        physical = output.get("terminal_physical_prediction", terminal)
        bypass = output.get("terminal_bypass", torch.zeros_like(terminal))
        if self.config.candidate_id == "C03":
            assert self.extra_bypass is not None
            bypass = self.extra_bypass(history)
            physical = terminal
            terminal = physical + bypass
        elif self.config.candidate_id == "C04":
            bypass = torch.zeros_like(bypass)
            terminal = physical
        elif self.config.candidate_id == "C05":
            baseline = history[:, -1, self.terminal_indices]
            physical = baseline[:, None].expand_as(terminal)
            terminal = physical + bypass
        return {
            **output,
            "terminal_prediction": terminal,
            "terminal_physical_prediction": physical,
            "terminal_bypass": bypass,
            "explicit_local_effect": explicit,
            "rm3av_candidate_id": self.config.candidate_id,
            "base_candidate_id": self.base_candidate_id,
            "test_accessed": False,
            "action_shield_fitted": self.action_shield_fitted,
        }

    def forward_two_window(
        self,
        first_history: torch.Tensor,
        first_sp: torch.Tensor,
        second_history: torch.Tensor,
        second_sp: torch.Tensor,
    ) -> dict[str, dict[str, Any]]:
        if self.config.candidate_id != "C31":
            raise Phase35ProtocolError("two-window continuation is restricted to C31")
        first = self._p5_forward_with_state(first_history, first_sp)
        local_state = first["local_state"][:, -1]
        latent_state = first["latent_state"][:, -1]
        second = self._p5_forward_with_state(
            second_history,
            second_sp,
            initial_local_state=local_state,
            initial_latent_state=latent_state,
        )
        second = {
            **second,
            "continuation_initial_local_state": local_state,
            "continuation_initial_latent_state": latent_state,
        }
        return {"first": first, "second": second}


def build_rm3av_model(
    config: RM3AVModelConfig, feature_names: Sequence[str]
) -> RM3AVModel:
    return RM3AVModel(config, feature_names)
