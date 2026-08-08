"""Valve-level A1phys model with causal, physically scaled intervention response."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import ExperimentConfig, Phase35ProtocolError


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse softplus requires a positive value")
    return math.log(math.expm1(value))


class MonotoneValveMap(nn.Module):
    """Differentiable piecewise-linear opening map with fixed endpoints 0 and 100.

    Positive interval slopes enforce monotonicity. Endpoint normalization removes
    the otherwise unidentifiable scale trade-off between the map and physical gain K.
    """

    def __init__(self, mode: str = "identity", knots: tuple[float, ...] | None = None):
        super().__init__()
        if mode not in {"identity", "equal_percentage_r50", "monotone"}:
            raise Phase35ProtocolError(f"unknown valve map mode={mode!r}")
        self.mode = mode
        knots = knots or (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 100.0)
        if knots[0] != 0.0 or knots[-1] != 100.0 or any(b <= a for a, b in zip(knots, knots[1:])):
            raise Phase35ProtocolError("valve knots must be strictly increasing from 0 to 100")
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32))
        if mode == "monotone":
            self.raw_slopes = nn.Parameter(torch.full((len(knots) - 1,), _inverse_softplus(1.0)))
        else:
            self.register_parameter("raw_slopes", None)

    def knot_values(self) -> torch.Tensor:
        if self.mode == "identity":
            return self.knots
        if self.mode == "equal_percentage_r50":
            return self._equal_percentage(self.knots)
        widths = self.knots[1:] - self.knots[:-1]
        increments = F.softplus(self.raw_slopes) * widths
        values = torch.cat([torch.zeros(1, device=increments.device), torch.cumsum(increments, dim=0)])
        return values / values[-1].clamp_min(1e-8) * 100.0

    @staticmethod
    def _equal_percentage(valve: torch.Tensor, ratio: float = 50.0) -> torch.Tensor:
        # Fixed engineering prior retained only as an ablation.  The output is
        # normalized to effective-opening percent; it is not claimed as kg/s.
        base = torch.as_tensor(ratio, dtype=valve.dtype, device=valve.device)
        raw = torch.pow(base, valve / 100.0 - 1.0)
        return (raw - 1.0 / base) / (1.0 - 1.0 / base) * 100.0

    def forward(self, valve: torch.Tensor) -> torch.Tensor:
        valve = valve.clamp(0.0, 100.0)
        if self.mode == "identity":
            return valve
        if self.mode == "equal_percentage_r50":
            return self._equal_percentage(valve)
        values = self.knot_values()
        idx = torch.bucketize(valve.contiguous(), self.knots[1:-1])
        x0, x1 = self.knots[idx], self.knots[idx + 1]
        y0, y1 = values[idx], values[idx + 1]
        weight = (valve - x0) / (x1 - x0).clamp_min(1e-8)
        return y0 + weight * (y1 - y0)


class HistoryEncoder(nn.Module):
    """Past-only temporal/variable encoder with per-window normalization.

    Logged valve history is a legitimate pre-treatment state.  The encoder is
    isolated only from the *future* valve path used by the intervention branch.
    """

    def __init__(self, n_features: int, target_index: int, window: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.target_index = target_index
        self.window = window
        self.temporal = nn.Sequential(
            nn.Linear(window, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.variable_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.context = nn.Sequential(
            nn.Linear(n_features * d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
        )
        self.context_dim = d_model * 2

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[1:] != (self.window, self.n_features):
            raise Phase35ProtocolError(
                f"history must have shape [B,{self.window},{self.n_features}], got {tuple(history.shape)}"
            )
        mean = history.mean(dim=1, keepdim=True).detach()
        std = (history.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        normalized = (history - mean) / std
        tokens = self.temporal(normalized.transpose(1, 2))
        attended, _ = self.variable_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        context = self.context(tokens.flatten(1))
        target_mean = mean[:, 0, self.target_index]
        target_std = std[:, 0, self.target_index]
        target_last = history[:, -1, self.target_index]
        return context, target_mean, target_std, target_last


class ActionAdapter(nn.Module):
    """Turn alternate logged action representations into a causal physical drive."""

    def __init__(self, mode: str, opening_map: MonotoneValveMap):
        super().__init__()
        self.mode = mode
        self.opening_map = opening_map

    def forward(self, future_valve: torch.Tensor, baseline_valve: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if future_valve.ndim != 2 or baseline_valve.ndim != 1 or future_valve.shape[0] != baseline_valve.shape[0]:
            raise Phase35ProtocolError("future_valve must be [B,H] and baseline_valve must be [B]")
        previous = torch.cat([baseline_valve[:, None], future_valve[:, :-1]], dim=1)
        delta = future_valve - previous
        if self.mode == "none":
            return torch.zeros_like(future_valve), torch.zeros_like(delta)
        if self.mode == "delta_no_baseline":
            # Historical naive representation: a step is an impulse and loses its absolute operating point.
            return delta, delta
        if self.mode == "delta_with_baseline":
            reconstructed = baseline_valve[:, None] + torch.cumsum(delta, dim=1)
            dose = self.opening_map(reconstructed) - self.opening_map(baseline_valve)[:, None]
            return dose, delta
        if self.mode in {"absolute", "absolute_plus_delta"}:
            dose = self.opening_map(future_valve) - self.opening_map(baseline_valve)[:, None]
            return dose, delta
        raise Phase35ProtocolError(f"unsupported action mode={self.mode!r}")


@dataclass
class PhysicalParameters:
    gain: torch.Tensor
    tau: torch.Tensor
    rate_gain: torch.Tensor | None


class TwoStageValveResponse(nn.Module):
    """Causal two-stage inertia with non-positive steady valve-to-temperature gain."""

    def __init__(self, context_dim: int, rate_branch: bool, tau_min: float = 1.5, tau_max: float = 120.0):
        super().__init__()
        self.rate_branch = rate_branch
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.trunk = nn.Sequential(nn.Linear(context_dim, context_dim), nn.GELU())
        self.gain_head = nn.Linear(context_dim, 1)
        self.tau_head = nn.Linear(context_dim, 2)
        nn.init.zeros_(self.gain_head.weight)
        nn.init.constant_(self.gain_head.bias, _inverse_softplus(0.05))
        nn.init.zeros_(self.tau_head.weight)
        tau_init = 18.0
        frac = (tau_init - tau_min) / (tau_max - tau_min)
        nn.init.constant_(self.tau_head.bias, math.log(frac / (1.0 - frac)))
        if rate_branch:
            self.rate_head = nn.Linear(context_dim, 1)
            nn.init.zeros_(self.rate_head.weight)
            nn.init.constant_(self.rate_head.bias, _inverse_softplus(0.005))
        else:
            self.rate_head = None

    def parameters_for(self, context: torch.Tensor) -> PhysicalParameters:
        hidden = self.trunk(context)
        gain = -F.softplus(self.gain_head(hidden))
        tau_frac = torch.sigmoid(self.tau_head(hidden))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_frac
        rate_gain = -F.softplus(self.rate_head(hidden)) if self.rate_head is not None else None
        return PhysicalParameters(gain=gain, tau=tau, rate_gain=rate_gain)

    @staticmethod
    def _first_order(signal: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        alpha = (1.0 / tau).clamp(1e-4, 1.0)
        state = torch.zeros_like(signal[:, 0])
        outputs = []
        for k in range(signal.shape[1]):
            state = state + alpha[:, 0] * (signal[:, k] - state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, context: torch.Tensor, dose: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, PhysicalParameters]:
        params = self.parameters_for(context)
        signal = params.gain * dose
        response = self._first_order(signal, params.tau[:, 0:1])
        response = self._first_order(response, params.tau[:, 1:2])
        if params.rate_gain is not None:
            rate_signal = params.rate_gain * delta
            response = response + self._first_order(rate_signal, params.tau[:, 0:1])
        return response, params


class A1PhysValveWM(nn.Module):
    """Phase 3.5 residual gray-box model in physical output units."""

    def __init__(self, config: ExperimentConfig, n_features: int, target_index: int):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = HistoryEncoder(
            n_features=n_features,
            target_index=target_index,
            window=config.window,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        context_dim = self.encoder.context_dim
        self.free_head_enabled = config.free_head
        if config.free_head:
            self.free_head = nn.Sequential(
                nn.Linear(context_dim, context_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(context_dim * 2, config.horizon * 2),
            )
        else:
            self.free_head = None
        self.opening_map = MonotoneValveMap(config.opening_map)
        self.action_adapter = ActionAdapter(config.action_mode, self.opening_map)
        self.physics = TwoStageValveResponse(context_dim, rate_branch=config.rate_branch)

    def _free_prediction(
        self,
        context: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        target_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        if self.free_head is None:
            mu = target_last[:, None].expand(batch, self.config.horizon)
            sigma = target_std[:, None].expand_as(mu).clamp_min(0.05)
            return mu, sigma
        raw = self.free_head(context).reshape(batch, self.config.horizon, 2)
        mu = target_mean[:, None] + target_std[:, None] * raw[..., 0]
        sigma = (F.softplus(raw[..., 1]) + 1e-3) * target_std[:, None]
        return mu, sigma.clamp_min(0.02)

    def forward(self, history: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor) -> dict[str, torch.Tensor]:
        context, mean, std, last = self.encoder(history)
        free_mu, sigma = self._free_prediction(context, mean, std, last)
        dose, delta = self.action_adapter(future_valve, baseline_valve)
        effect, params = self.physics(context, dose, delta)
        if self.config.action_mode == "none":
            effect = torch.zeros_like(effect)
        return {
            "mu": free_mu + effect,
            "sigma": sigma,
            "free_mu": free_mu,
            "effect": effect,
            "dose": dose,
            "gain": params.gain,
            "tau": params.tau,
        }

    def intervention_effect(self, history: torch.Tensor, future_valve: torch.Tensor, baseline_valve: torch.Tensor) -> torch.Tensor:
        return self.forward(history, future_valve, baseline_valve)["effect"]


@torch.no_grad()
def assert_constant_valve_identity(model: A1PhysValveWM, history: torch.Tensor, baseline_valve: torch.Tensor, atol: float = 1e-6) -> None:
    model.eval()
    future = baseline_valve[:, None].expand(len(baseline_valve), model.config.horizon)
    effect = model.intervention_effect(history, future, baseline_valve)
    if not torch.allclose(effect, torch.zeros_like(effect), atol=atol):
        raise AssertionError(f"constant-valve intervention is nonzero: max={effect.abs().max().item():.3e}")
