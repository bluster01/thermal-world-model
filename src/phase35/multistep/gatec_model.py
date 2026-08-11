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
        }


class StableLocalMIMOResponse(nn.Module):
    """Stable two-output response core shared by every Gate-C operator adapter."""

    def __init__(
        self,
        *,
        route: str,
        context_dim: int,
        state_dim: int,
        dt_seconds: float,
        tau_min_seconds: float,
        tau_max_seconds: float,
        scheduled: bool,
    ) -> None:
        super().__init__()
        if route not in RESPONSE_ROUTES - {"none"}:
            raise Phase35ProtocolError(f"unsupported Gate C response route={route!r}")
        if state_dim % 2:
            raise Phase35ProtocolError("Gate C response state must split into two sides")
        self.route = route
        self.state_dim = state_dim
        self.poles = state_dim // 2
        self.dt_seconds = float(dt_seconds)
        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = float(tau_max_seconds)
        self.scheduled = bool(scheduled)
        initial_tau = torch.linspace(tau_min_seconds * 1.5, tau_max_seconds * 0.5, self.poles)
        fraction = (initial_tau - tau_min_seconds) / (tau_max_seconds - tau_min_seconds)
        logits = torch.logit(fraction.clamp(1e-4, 1 - 1e-4))
        self.tau_logits = nn.Parameter(logits.repeat(2, 1))
        self.diagonal_gain = nn.Parameter(torch.full((2, 2), -1.0))
        self.off_diagonal_gain = nn.Parameter(torch.zeros(2, 2))
        self.pole_weights = nn.Parameter(torch.zeros(2, self.poles))
        self.context_schedule = nn.Linear(context_dim, 2) if scheduled else None
        if route == "stable_koopman_lpv":
            self.route_scale = nn.Parameter(torch.tensor(0.0))
        elif route == "pi_neural_ode":
            self.route_scale = nn.Parameter(torch.tensor(-0.2))
        elif route == "deeponet_response":
            self.route_scale = nn.Parameter(torch.tensor(0.2))
        else:
            self.route_scale = nn.Parameter(torch.tensor(-0.5))

    def _mixing(self) -> torch.Tensor:
        diagonal = F.softplus(torch.diagonal(self.diagonal_gain)) + 1e-3
        mixing = 0.15 * torch.tanh(self.off_diagonal_gain)
        mixing = mixing - torch.diag_embed(torch.diagonal(mixing))
        return mixing + torch.diag(diagonal)

    def forward(
        self, context: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, horizon, _ = future_valve.shape
        tau = self.tau_min_seconds + (
            self.tau_max_seconds - self.tau_min_seconds
        ) * torch.sigmoid(self.tau_logits)
        decay = torch.exp(-self.dt_seconds / tau)
        dose = future_valve - baseline_valve[:, None, :]
        mixed = torch.einsum("bhi,oi->bho", dose, self._mixing())
        if self.context_schedule is not None:
            schedule = torch.exp(0.25 * torch.tanh(self.context_schedule(context)))
            mixed = mixed * schedule[:, None, :]
        mixed = mixed * torch.exp(0.1 * torch.tanh(self.route_scale))
        state = future_valve.new_zeros((batch, 2, self.poles))
        states: list[torch.Tensor] = []
        effects: list[torch.Tensor] = []
        weights = F.softmax(self.pole_weights, dim=1)
        for step in range(horizon):
            state = decay[None] * state + (1.0 - decay[None]) * mixed[:, step, :, None]
            effect = torch.sum(weights[None] * state, dim=2)
            states.append(state.reshape(batch, self.state_dim))
            effects.append(effect)
        return {"effect": torch.stack(effects, dim=1), "state": torch.stack(states, dim=1)}


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
    return StableLocalMIMOResponse(
        route=route,
        context_dim=context_dim,
        state_dim=state_dim,
        dt_seconds=dt_seconds,
        tau_min_seconds=tau_min_seconds,
        tau_max_seconds=tau_max_seconds,
        scheduled=scheduled,
    )


class StableDownstreamLatentMixer(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.initial = nn.Linear(context_dim, latent_dim)
        self.input_projection = nn.Linear(2, latent_dim, bias=False)
        self.output_projection = nn.Linear(latent_dim, 2, bias=False)
        self.decay_logits = nn.Parameter(torch.zeros(latent_dim))

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
        required = [
            *(f"{side}::二级减温调节门阀位" for side in ("A", "B")),
            *(f"{side}::二级减温器入口温度" for side in ("A", "B")),
            *(f"{side}::末级过热器出口汽温" for side in ("A", "B")),
        ]
        missing = set(required) - set(self.feature_names)
        if missing:
            raise Phase35ProtocolError(f"Gate C model is missing features: {sorted(missing)}")
        self.valve_indices = [self.feature_names.index(f"{side}::二级减温调节门阀位") for side in ("A", "B")]
        self.tin_indices = [self.feature_names.index(f"{side}::二级减温器入口温度") for side in ("A", "B")]
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
        self.local_response = build_local_response_operator(
            route=config.response_route,
            context_dim=config.d_model,
            state_dim=config.local_state_dim,
            horizon=config.horizon,
            dt_seconds=config.dt_seconds,
            scheduled=config.response_scheduling == "scheduled",
            tau_min_seconds=config.tau_min_seconds,
            tau_max_seconds=config.tau_max_seconds,
        )
        self.downstream = StableDownstreamLatentMixer(config.d_model, config.latent_dim)

    def forward(
        self,
        history: torch.Tensor,
        future_sp: torch.Tensor,
        *,
        boundary_mode: str,
        boundary_future: torch.Tensor | None = None,
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
        context = self.encoder(history)
        baseline_valve = history[:, -1, self.valve_indices]
        baseline_tin = history[:, -1, self.tin_indices]
        baseline_terminal = history[:, -1, self.terminal_indices]
        valve = self.valve_policy(context, future_sp, baseline_valve)
        tin_prediction = self.tin_forecaster(context, baseline_tin, self.config.horizon)
        boundary_used = tin_prediction if boundary_mode == "forecast_boundary" else boundary_future
        assert boundary_used is not None
        residual_local = self.residual_head(context).reshape(-1, self.config.horizon, 2)
        response = self.local_response(context, valve, baseline_valve)
        local_drop = residual_local + response["effect"]
        tout = boundary_used - local_drop
        terminal, latent = self.downstream(context, tout, baseline_terminal)
        reference_tout = boundary_used - residual_local
        terminal_reference, _ = self.downstream(context, reference_tout, baseline_terminal)
        return {
            "valve_prediction": valve,
            "tin_prediction": tin_prediction,
            "boundary_used": boundary_used,
            "residual_local_prediction": residual_local,
            "local_drop_prediction": local_drop,
            "tout_prediction": tout,
            "terminal_prediction": terminal,
            "local_effect": response["effect"],
            "terminal_effect": terminal - terminal_reference,
            "local_state": response["state"],
            "latent_state": latent,
            "boundary_mode": boundary_mode,
            "response_route": self.config.response_route,
        }


def build_gatec_model(
    config: GateCModelConfig, feature_names: Sequence[str]
) -> MeasuredBoundaryMIMOWorldModel:
    return MeasuredBoundaryMIMOWorldModel(config, feature_names)
