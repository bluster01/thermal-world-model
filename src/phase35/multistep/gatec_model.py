"""Dual-interface measured-boundary latent MIMO modules for MS3-R Gate C."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import Phase35ProtocolError
from .gatec_contracts import BOUNDARY_MODES, GateCModelConfig, RESPONSE_ROUTES


class PairedHistoryEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.recurrent = nn.GRU(d_model, d_model, batch_first=True)
        self.normalization = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = F.gelu(self.input_projection(history))
        _, final = self.recurrent(self.dropout(encoded))
        return self.normalization(final[-1])


class CausalValvePolicyDecoder(nn.Module):
    def __init__(self, context_dim: int, dropout: float) -> None:
        super().__init__()
        self.initial = nn.Linear(context_dim, context_dim)
        self.cell = nn.GRUCell(4, context_dim)
        self.output = nn.Linear(context_dim, 2)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, context: torch.Tensor, future_sp: torch.Tensor, baseline_valve: torch.Tensor
    ) -> torch.Tensor:
        hidden = torch.tanh(self.initial(context))
        previous = baseline_valve
        values: list[torch.Tensor] = []
        for step in range(future_sp.shape[1]):
            inputs = torch.cat((future_sp[:, step] / 600.0, previous / 100.0), dim=1)
            hidden = self.cell(inputs, self.dropout(hidden))
            delta = 2.0 * torch.tanh(self.output(hidden))
            previous = previous + delta
            values.append(previous)
        return torch.stack(values, dim=1)


class TinBoundaryForecaster(nn.Module):
    def __init__(self, context_dim: int, dropout: float) -> None:
        super().__init__()
        self.initial = nn.Linear(context_dim, context_dim)
        self.cell = nn.GRUCell(2, context_dim)
        self.output = nn.Linear(context_dim, 2)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, context: torch.Tensor, baseline_tin: torch.Tensor, horizon: int
    ) -> torch.Tensor:
        hidden = torch.tanh(self.initial(context))
        previous = baseline_tin
        values: list[torch.Tensor] = []
        for _ in range(horizon):
            hidden = self.cell(previous / 600.0, self.dropout(hidden))
            previous = previous + 1.5 * torch.tanh(self.output(hidden))
            values.append(previous)
        return torch.stack(values, dim=1)


class ZeroLocalResponse(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.route = "none"

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        shape = (*future_valve.shape[:2], 2)
        state_shape = (*future_valve.shape[:2], self.state_dim)
        return {
            "effect": future_valve.new_zeros(shape),
            "state": future_valve.new_zeros(state_shape),
            "stable_poles": future_valve.new_zeros((0,)),
            "operator_family": self.route,
        }


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-5, 1.0 - 1e-5))


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value).clamp_min(1e-12))


class MonotoneValveOpening(nn.Module):
    """Normalized monotone proxy for the unmeasured valve-to-flow curve."""

    def __init__(self) -> None:
        super().__init__()
        self.power_logits = nn.Parameter(torch.zeros(3))

    def forward(self, valve: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(valve / 100.0, 0.0, 1.0)
        powers = torch.stack((normalized, normalized.square(), normalized.pow(3)), dim=-1)
        return torch.sum(F.softmax(self.power_logits, dim=0) * powers, dim=-1)


class StableMIMOResponseBase(nn.Module):
    """Shared monotone opening and positive equilibrium mixing, not shared dynamics."""

    def __init__(
        self,
        *,
        route: str,
        context_dim: int,
        state_dim: int,
        dt_seconds: float,
        scheduled: bool,
    ) -> None:
        super().__init__()
        if route not in RESPONSE_ROUTES - {"none"}:
            raise Phase35ProtocolError(f"unsupported Gate C response route={route!r}")
        if state_dim != 6:
            raise Phase35ProtocolError("Gate C response requires three bases per mode")
        self.route = route
        self.state_dim = state_dim
        self.bases = state_dim // 2
        self.dt_seconds = float(dt_seconds)
        self.scheduled = bool(scheduled)
        self.opening = MonotoneValveOpening()
        self.diagonal_gain = nn.Parameter(torch.full((2,), 2.0))
        self.cross_gain_logits = nn.Parameter(torch.full((2,), -2.0))
        self.context_equilibrium = nn.Linear(context_dim, 2) if scheduled else None
        if self.context_equilibrium is not None:
            nn.init.zeros_(self.context_equilibrium.weight)
            nn.init.zeros_(self.context_equilibrium.bias)

    def _equilibrium_modes(
        self,
        context: torch.Tensor,
        future_valve: torch.Tensor,
        baseline_valve: torch.Tensor,
    ) -> torch.Tensor:
        dose = self.opening(future_valve) - self.opening(baseline_valve)[:, None, :]
        diagonal = F.softplus(self.diagonal_gain) + 1e-3
        cross = 0.35 * torch.sigmoid(self.cross_gain_logits) * diagonal.flip(0)
        mixing = torch.stack(
            (
                torch.stack((diagonal[0], cross[0])),
                torch.stack((cross[1], diagonal[1])),
            )
        )
        equilibrium_side = torch.einsum("bhi,oi->bho", dose, mixing)
        if self.context_equilibrium is not None:
            schedule = torch.exp(0.25 * torch.tanh(self.context_equilibrium(context)))
            equilibrium_side = equilibrium_side * schedule[:, None, :]
        common = 0.5 * (equilibrium_side[..., 0] + equilibrium_side[..., 1])
        differential = 0.5 * (equilibrium_side[..., 0] - equilibrium_side[..., 1])
        return torch.stack((common, differential), dim=-1)

    @staticmethod
    def _modes_to_sides(modes: torch.Tensor) -> torch.Tensor:
        return torch.stack((modes[..., 0] + modes[..., 1], modes[..., 0] - modes[..., 1]), dim=-1)


class A1PhysThreePoleResponse(StableMIMOResponseBase):
    """Explicit three-time-constant response in common/differential modes."""

    def __init__(
        self,
        *,
        context_dim: int,
        state_dim: int,
        dt_seconds: float,
        tau_min_seconds: float,
        tau_max_seconds: float,
        scheduled: bool,
    ) -> None:
        super().__init__(
            route="a1phys_three_pole",
            context_dim=context_dim,
            state_dim=state_dim,
            dt_seconds=dt_seconds,
            scheduled=scheduled,
        )
        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = float(tau_max_seconds)
        initial_tau = torch.tensor((30.0, 180.0, 900.0)).clamp(
            tau_min_seconds * 1.01, tau_max_seconds * 0.99
        )
        fraction = (initial_tau - tau_min_seconds) / (tau_max_seconds - tau_min_seconds)
        self.tau_logits = nn.Parameter(_inverse_sigmoid(fraction).repeat(2, 1))
        self.pole_weights = nn.Parameter(torch.zeros(2, self.bases))

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        batch, horizon, _ = future_valve.shape
        equilibrium = self._equilibrium_modes(context, future_valve, baseline_valve)
        tau = self.tau_min_seconds + (
            self.tau_max_seconds - self.tau_min_seconds
        ) * torch.sigmoid(self.tau_logits)
        decay = torch.exp(-self.dt_seconds / tau)
        state = future_valve.new_zeros((batch, 2, self.bases))
        states: list[torch.Tensor] = []
        effects: list[torch.Tensor] = []
        weights = F.softmax(self.pole_weights, dim=1)
        for step in range(horizon):
            state = decay[None] * state + (1.0 - decay[None]) * equilibrium[:, step, :, None]
            modes = torch.sum(weights[None] * state, dim=2)
            states.append(state.reshape(batch, self.state_dim))
            effects.append(self._modes_to_sides(modes))
        return {
            "effect": torch.stack(effects, dim=1),
            "state": torch.stack(states, dim=1),
            "stable_poles": decay,
            "operator_family": self.route,
        }


class StableLPVKoopmanResponse(StableMIMOResponseBase):
    """Stable operating-condition-dependent lifted representation."""

    def __init__(
        self, *, context_dim: int, state_dim: int, dt_seconds: float, scheduled: bool
    ) -> None:
        super().__init__(
            route="stable_koopman_lpv",
            context_dim=context_dim,
            state_dim=state_dim,
            dt_seconds=dt_seconds,
            scheduled=scheduled,
        )
        initial_tau = torch.tensor((40.0, 220.0, 850.0))
        initial_decay = torch.exp(-dt_seconds / initial_tau)
        fraction = (initial_decay - 0.05) / 0.945
        self.decay_logits = nn.Parameter(_inverse_sigmoid(fraction).repeat(2, 1))
        self.input_scale = nn.Parameter(torch.zeros(2, self.bases))
        self.output_weights = nn.Parameter(torch.zeros(2, self.bases))
        self.context_decay = nn.Linear(context_dim, state_dim) if scheduled else None
        self.context_lift = nn.Linear(context_dim, state_dim) if scheduled else None
        for layer in (self.context_decay, self.context_lift):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        batch, horizon, _ = future_valve.shape
        modes_input = self._equilibrium_modes(context, future_valve, baseline_valve)
        decay_logits = self.decay_logits[None].expand(batch, -1, -1)
        if self.context_decay is not None:
            decay_logits = decay_logits + 0.25 * torch.tanh(
                self.context_decay(context).reshape(batch, 2, self.bases)
            )
        decay = 0.05 + 0.945 * torch.sigmoid(decay_logits)
        lift_scale = F.softplus(self.input_scale)[None] + 0.1
        if self.context_lift is not None:
            lift_scale = lift_scale * torch.exp(
                0.25 * torch.tanh(self.context_lift(context).reshape(batch, 2, self.bases))
            )
        state = future_valve.new_zeros((batch, 2, self.bases))
        weights = F.softmax(self.output_weights, dim=1)
        states: list[torch.Tensor] = []
        effects: list[torch.Tensor] = []
        for step in range(horizon):
            value = modes_input[:, step]
            lifted = torch.stack(
                (value, value * value.abs() / (1.0 + value.abs()), torch.tanh(value)),
                dim=-1,
            )
            equilibrium = lift_scale * lifted
            state = decay * state + (1.0 - decay) * equilibrium
            modes = torch.sum(weights[None] * state, dim=2)
            states.append(state.reshape(batch, self.state_dim))
            effects.append(self._modes_to_sides(modes))
        return {
            "effect": torch.stack(effects, dim=1),
            "state": torch.stack(states, dim=1),
            "stable_poles": decay,
            "operator_family": self.route,
        }


class PINeuralODEResponse(StableMIMOResponseBase):
    """Dissipative neural ODE closure with exact exponential integration."""

    def __init__(
        self, *, context_dim: int, state_dim: int, dt_seconds: float, scheduled: bool
    ) -> None:
        super().__init__(
            route="pi_neural_ode",
            context_dim=context_dim,
            state_dim=state_dim,
            dt_seconds=dt_seconds,
            scheduled=scheduled,
        )
        initial_rate = 1.0 / torch.tensor((35.0, 240.0, 950.0))
        self.rate_raw = nn.Parameter(_inverse_softplus(initial_rate).repeat(2, 1))
        self.output_weights = nn.Parameter(torch.zeros(2, self.bases))
        self.context_rate = nn.Linear(context_dim, state_dim) if scheduled else None
        gate_features = context_dim + 2 if scheduled else 2
        self.closure_gate = nn.Linear(gate_features, state_dim)
        nn.init.zeros_(self.closure_gate.weight)
        nn.init.zeros_(self.closure_gate.bias)
        if self.context_rate is not None:
            nn.init.zeros_(self.context_rate.weight)
            nn.init.zeros_(self.context_rate.bias)

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        batch, horizon, _ = future_valve.shape
        modes_input = self._equilibrium_modes(context, future_valve, baseline_valve)
        rate_raw = self.rate_raw[None].expand(batch, -1, -1)
        if self.context_rate is not None:
            rate_raw = rate_raw + 0.25 * torch.tanh(
                self.context_rate(context).reshape(batch, 2, self.bases)
            )
        rate = F.softplus(rate_raw) + 1e-5
        decay = torch.exp(-self.dt_seconds * rate)
        state = future_valve.new_zeros((batch, 2, self.bases))
        weights = F.softmax(self.output_weights, dim=1)
        states: list[torch.Tensor] = []
        effects: list[torch.Tensor] = []
        for step in range(horizon):
            value = modes_input[:, step]
            gate_input = torch.cat((context, value.abs()), dim=1) if self.scheduled else value.abs()
            gate = torch.exp(
                0.35 * torch.tanh(self.closure_gate(gate_input).reshape(batch, 2, self.bases))
            )
            equilibrium = value[:, :, None] * gate
            state = decay * state + (1.0 - decay) * equilibrium
            modes = torch.sum(weights[None] * state, dim=2)
            states.append(state.reshape(batch, self.state_dim))
            effects.append(self._modes_to_sides(modes))
        return {
            "effect": torch.stack(effects, dim=1),
            "state": torch.stack(states, dim=1),
            "stable_poles": decay,
            "operator_family": self.route,
        }


class CausalDeepONetResponse(StableMIMOResponseBase):
    """Prefix-causal branch/memory/trunk response operator."""

    def __init__(
        self, *, context_dim: int, state_dim: int, dt_seconds: float, scheduled: bool
    ) -> None:
        super().__init__(
            route="deeponet_response",
            context_dim=context_dim,
            state_dim=state_dim,
            dt_seconds=dt_seconds,
            scheduled=scheduled,
        )
        initial_tau = torch.tensor((45.0, 260.0, 1000.0))
        initial_decay = torch.exp(-dt_seconds / initial_tau)
        fraction = (initial_decay - 0.05) / 0.945
        self.memory_decay_logits = nn.Parameter(_inverse_sigmoid(fraction).repeat(2, 1))
        branch_features = context_dim + 2 if scheduled else 2
        self.branch_gate = nn.Linear(branch_features, state_dim)
        self.trunk = nn.Sequential(nn.Linear(1, 12), nn.Tanh(), nn.Linear(12, state_dim))
        nn.init.zeros_(self.branch_gate.weight)
        nn.init.zeros_(self.branch_gate.bias)

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        batch, horizon, _ = future_valve.shape
        modes_input = self._equilibrium_modes(context, future_valve, baseline_valve)
        decay = 0.05 + 0.945 * torch.sigmoid(self.memory_decay_logits)
        state = future_valve.new_zeros((batch, 2, self.bases))
        states: list[torch.Tensor] = []
        effects: list[torch.Tensor] = []
        for step in range(horizon):
            value = modes_input[:, step]
            branch_input = torch.cat((context, value.abs()), dim=1) if self.scheduled else value.abs()
            branch_gain = torch.exp(
                0.5 * torch.tanh(self.branch_gate(branch_input).reshape(batch, 2, self.bases))
            )
            branch = value[:, :, None] * branch_gain
            state = decay[None] * state + (1.0 - decay[None]) * branch
            normalized_time = future_valve.new_full((batch, 1), (step + 1) / horizon)
            trunk = F.softmax(self.trunk(normalized_time).reshape(batch, 2, self.bases), dim=2)
            modes = torch.sum(trunk * state, dim=2)
            states.append(state.reshape(batch, self.state_dim))
            effects.append(self._modes_to_sides(modes))
        return {
            "effect": torch.stack(effects, dim=1),
            "state": torch.stack(states, dim=1),
            "stable_poles": decay,
            "operator_family": self.route,
        }

def build_local_response_operator(
    *,
    route: str,
    context_dim: int,
    state_dim: int,
    horizon: int,
    dt_seconds: float,
    scheduled: bool,
    tau_min_seconds: float = 20.0,
    tau_max_seconds: float = 1200.0,
) -> nn.Module:
    del horizon  # The recurrent operator accepts any finite rollout horizon.
    if route == "none":
        return ZeroLocalResponse(state_dim)
    common = {
        "context_dim": context_dim,
        "state_dim": state_dim,
        "dt_seconds": dt_seconds,
        "scheduled": scheduled,
    }
    if route == "a1phys_three_pole":
        return A1PhysThreePoleResponse(
            **common,
            tau_min_seconds=tau_min_seconds,
            tau_max_seconds=tau_max_seconds,
        )
    if route == "stable_koopman_lpv":
        return StableLPVKoopmanResponse(**common)
    if route == "pi_neural_ode":
        return PINeuralODEResponse(**common)
    if route == "deeponet_response":
        return CausalDeepONetResponse(**common)
    raise Phase35ProtocolError(f"unsupported Gate C response route={route!r}")


class StableDownstreamLatentMixer(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.initial = nn.Linear(context_dim, latent_dim)
        self.input_projection = nn.Linear(2, latent_dim, bias=False)
        self.output_projection = nn.Linear(latent_dim, 2, bias=False)
        self.decay_logits = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        context: torch.Tensor,
        future_tout: torch.Tensor,
        baseline_terminal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.tanh(self.initial(context))
        decay = 0.5 + 0.49 * torch.sigmoid(self.decay_logits)
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for step in range(future_tout.shape[1]):
            centered = (future_tout[:, step] - baseline_terminal) / 20.0
            state = decay[None] * state + (1.0 - decay[None]) * torch.tanh(
                self.input_projection(centered)
            )
            outputs.append(baseline_terminal + self.output_projection(state))
            states.append(state)
        return torch.stack(outputs, dim=1), torch.stack(states, dim=1)


class DirectDownstreamMixer(nn.Module):
    """Causal no-state ablation for testing whether downstream memory is needed."""

    def __init__(self, context_dim: int) -> None:
        super().__init__()
        self.context_projection = nn.Linear(context_dim, 2)
        self.input_projection = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        context: torch.Tensor,
        future_tout: torch.Tensor,
        baseline_terminal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centered = (future_tout - baseline_terminal[:, None, :]) / 20.0
        delta = self.input_projection(centered) + self.context_projection(context)[:, None, :]
        terminal = baseline_terminal[:, None, :] + delta
        no_latent = future_tout.new_zeros((*future_tout.shape[:2], 0))
        return terminal, no_latent


class CoordinateRestrictedResponse(nn.Module):
    """Restrict only the explicit response effect to the supported coordinate subspace."""

    def __init__(self, operator: nn.Module, mode: str) -> None:
        super().__init__()
        self.operator = operator
        self.mode = mode
        self.route = getattr(operator, "route", "unknown")

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, Any]:
        output = self.operator(context, future_valve, baseline_valve)
        if self.mode == "common_only":
            common = output["effect"].mean(dim=-1, keepdim=True)
            output = {**output, "effect": common.expand_as(output["effect"])}
        return output


class MeasuredBoundaryMIMOWorldModel(nn.Module):
    def __init__(
        self, config: GateCModelConfig, feature_names: Sequence[str]
    ) -> None:
        super().__init__()
        config.validate()
        if len(feature_names) != config.n_features or len(set(feature_names)) != len(feature_names):
            raise Phase35ProtocolError("Gate C feature contract does not match model config")
        self.config = config
        self.feature_names = tuple(feature_names)
        self.register_buffer("history_center", torch.zeros(config.n_features))
        self.register_buffer("history_scale", torch.ones(config.n_features))
        required = [
            *(f"{side}::二级减温调节门阀位" for side in ("A", "B")),
            *(f"{side}::二级减温器入口温度" for side in ("A", "B")),
            *(f"{side}::二级减温器出口温度" for side in ("A", "B")),
            *(f"{side}::末级过热器出口汽温" for side in ("A", "B")),
        ]
        missing = set(required) - set(self.feature_names)
        if missing:
            raise Phase35ProtocolError(f"Gate C model is missing features: {sorted(missing)}")
        self.valve_indices = [self.feature_names.index(f"{side}::二级减温调节门阀位") for side in ("A", "B")]
        self.tin_indices = [self.feature_names.index(f"{side}::二级减温器入口温度") for side in ("A", "B")]
        self.tout_indices = [self.feature_names.index(f"{side}::二级减温器出口温度") for side in ("A", "B")]
        self.terminal_indices = [self.feature_names.index(f"{side}::末级过热器出口汽温") for side in ("A", "B")]
        self.encoder = PairedHistoryEncoder(config.n_features, config.d_model, config.dropout)
        self.valve_policy = CausalValvePolicyDecoder(config.d_model, config.dropout)
        self.tin_forecaster = TinBoundaryForecaster(config.d_model, config.dropout)
        hidden_multiplier = {"small": 1, "base": 2, "large": 4}[config.residual_capacity]
        hidden = config.d_model * hidden_multiplier
        self.residual_head = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.horizon * 2),
        )
        nn.init.normal_(self.residual_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)
        response = build_local_response_operator(
            route=config.response_route,
            context_dim=config.d_model,
            state_dim=config.local_state_dim,
            horizon=config.horizon,
            dt_seconds=config.dt_seconds,
            scheduled=config.response_scheduling == "scheduled",
            tau_min_seconds=config.tau_min_seconds,
            tau_max_seconds=config.tau_max_seconds,
        )
        self.local_response = CoordinateRestrictedResponse(
            response, config.response_coordinate_mode
        )
        self.downstream = (
            StableDownstreamLatentMixer(config.d_model, config.latent_dim)
            if config.downstream_mode == "latent_mimo"
            else DirectDownstreamMixer(config.d_model)
        )

    def set_history_normalization(
        self, center: torch.Tensor, scale: torch.Tensor
    ) -> None:
        if center.shape != self.history_center.shape or scale.shape != self.history_scale.shape:
            raise Phase35ProtocolError("Gate C history normalization shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise Phase35ProtocolError("Gate C history normalization values are invalid")
        self.history_center.copy_(center.detach().to(self.history_center))
        self.history_scale.copy_(scale.detach().to(self.history_scale))

    def forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        boundary_mode: str,
        boundary_future: torch.Tensor | None = None,
        logged_future_valve_for_aux: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if boundary_mode not in BOUNDARY_MODES:
            raise Phase35ProtocolError(f"unknown Gate C boundary mode={boundary_mode!r}")
        if history.shape[1:] != (self.config.window, self.config.n_features):
            raise Phase35ProtocolError("Gate C history shape mismatch")
        if future_sp.shape[1:] != (self.config.horizon, 2):
            raise Phase35ProtocolError("Gate C future SP shape mismatch")
        if boundary_mode == "forecast_boundary" and boundary_future is not None:
            raise Phase35ProtocolError("Gate C forecast mode must not receive future Tin truth")
        if boundary_mode != "forecast_boundary":
            if boundary_future is None:
                raise Phase35ProtocolError("Gate C oracle/scenario boundary requires future Tin")
            if boundary_future.shape != future_sp.shape or not torch.isfinite(boundary_future).all():
                raise Phase35ProtocolError("Gate C future Tin boundary is invalid")
        normalized_history = (history - self.history_center[None, None, :]) / self.history_scale[
            None, None, :
        ]
        context = self.encoder(normalized_history)
        baseline_valve = history[:, -1, self.valve_indices]
        baseline_tin = history[:, -1, self.tin_indices]
        baseline_tout = history[:, -1, self.tout_indices]
        baseline_local_drop = baseline_tin - baseline_tout
        baseline_terminal = history[:, -1, self.terminal_indices]
        valve = self.valve_policy(context, future_sp, baseline_valve)
        tin_prediction = self.tin_forecaster(context, baseline_tin, self.config.horizon)
        boundary_used = tin_prediction if boundary_mode == "forecast_boundary" else boundary_future
        assert boundary_used is not None
        residual_local_delta = self.residual_head(context).reshape(-1, self.config.horizon, 2)
        residual_local = baseline_local_drop[:, None, :] + residual_local_delta
        response = self.local_response(context, valve, baseline_valve)
        local_drop = residual_local + response["effect"]
        tout = boundary_used - local_drop
        terminal, latent = self.downstream(context, tout, baseline_terminal)
        reference_tout = boundary_used - residual_local
        terminal_reference, _ = self.downstream(context, reference_tout, baseline_terminal)
        result = {
            "valve_prediction": valve,
            "tin_prediction": tin_prediction,
            "boundary_used": boundary_used,
            "residual_local_prediction": residual_local,
            "residual_local_delta_prediction": residual_local_delta,
            "local_drop_prediction": local_drop,
            "tout_prediction": tout,
            "terminal_prediction": terminal,
            "local_effect": response["effect"],
            "terminal_effect": terminal - terminal_reference,
            "local_state": response["state"],
            "local_stable_poles": response["stable_poles"],
            "local_operator_family": response["operator_family"],
            "latent_state": latent,
            "boundary_mode": boundary_mode,
            "response_route": self.config.response_route,
        }
        if logged_future_valve_for_aux is not None:
            if (
                logged_future_valve_for_aux.shape != future_sp.shape
                or not torch.isfinite(logged_future_valve_for_aux).all()
            ):
                raise Phase35ProtocolError("Gate C logged future valve auxiliary is invalid")
            logged_response = self.local_response(
                context, logged_future_valve_for_aux, baseline_valve
            )
            result["logged_local_effect"] = logged_response["effect"]
            result["logged_local_drop_prediction"] = residual_local + logged_response["effect"]
        return result


def build_gatec_model(
    config: GateCModelConfig, feature_names: Sequence[str]
) -> MeasuredBoundaryMIMOWorldModel:
    return MeasuredBoundaryMIMOWorldModel(config, feature_names)
