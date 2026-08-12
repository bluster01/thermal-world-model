"""Joint latent physical interfaces for RM3 architecture comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import Phase35ProtocolError


@dataclass(frozen=True)
class RM3JointConfig:
    context_dim: int
    latent_dim: int = 32
    horizon: int = 60
    terminal_bypass_hidden: int = 16

    def validate(self) -> None:
        if min(self.context_dim, self.latent_dim, self.horizon) < 1:
            raise Phase35ProtocolError("RM3 joint-latent dimensions must be positive")
        if self.terminal_bypass_hidden < 1 or self.terminal_bypass_hidden > self.latent_dim:
            raise Phase35ProtocolError("RM3 terminal bypass capacity is invalid")


class ActionInvariantTerminalBypass(nn.Module):
    """History-only correction; future action is deliberately absent from the API."""

    def __init__(self, context_dim: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, horizon * 2),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history_context: torch.Tensor) -> torch.Tensor:
        return self.network(history_context).reshape(-1, self.horizon, 2)


class JointLatentPhysicalInterfaces(nn.Module):
    """Shared stable dynamics with measured boundary and explicit local-action interfaces."""

    def __init__(self, config: RM3JointConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.initial = nn.Linear(config.context_dim, config.latent_dim)
        self.boundary_projection = nn.Linear(2, config.latent_dim, bias=False)
        self.local_response_projection = nn.Linear(2, config.latent_dim, bias=False)
        self.state_readout = nn.Linear(config.latent_dim, 8)
        self.decay_logits = nn.Parameter(torch.zeros(config.latent_dim))
        self.terminal_bypass = ActionInvariantTerminalBypass(
            config.context_dim, config.terminal_bypass_hidden, config.horizon
        )

    def forward(
        self,
        history_context: torch.Tensor,
        boundary_tin: torch.Tensor,
        explicit_local_effect: torch.Tensor,
        *,
        baseline_valve: torch.Tensor,
        baseline_tin: torch.Tensor,
        baseline_local: torch.Tensor,
        baseline_terminal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = (self.config.horizon, 2)
        if boundary_tin.shape[1:] != expected or explicit_local_effect.shape[1:] != expected:
            raise Phase35ProtocolError("RM3 joint-latent interface shape mismatch")
        baselines = (baseline_valve, baseline_tin, baseline_local, baseline_terminal)
        if any(value.shape[1:] != (2,) for value in baselines):
            raise Phase35ProtocolError("RM3 joint-latent baseline shape mismatch")
        if not all(
            torch.isfinite(value).all()
            for value in (history_context, boundary_tin, explicit_local_effect, *baselines)
        ):
            raise Phase35ProtocolError("RM3 joint-latent inputs must be finite")
        state = torch.tanh(self.initial(history_context))
        decay = 0.5 + 0.49 * torch.sigmoid(self.decay_logits)
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for step in range(self.config.horizon):
            drive = self.boundary_projection(
                (boundary_tin[:, step] - baseline_tin) / 20.0
            ) + self.local_response_projection(explicit_local_effect[:, step] / 10.0)
            state = decay[None] * state + (1.0 - decay[None]) * torch.tanh(drive)
            outputs.append(self.state_readout(state))
            states.append(state)
        readout = torch.stack(outputs, dim=1)
        bypass = self.terminal_bypass(history_context)
        valve = baseline_valve[:, None] + readout[..., 0:2]
        tin = boundary_tin + readout[..., 2:4]
        local = baseline_local[:, None] + readout[..., 4:6] + explicit_local_effect
        terminal_physical = baseline_terminal[:, None] + readout[..., 6:8]
        terminal = terminal_physical + bypass
        return {
            "valve_prediction": valve,
            "tin_prediction": tin,
            "local_drop_prediction": local,
            "terminal_physical_prediction": terminal_physical,
            "terminal_bypass": bypass,
            "terminal_prediction": terminal,
            "latent_state": torch.stack(states, dim=1),
            "stable_poles": decay,
            "explicit_local_effect": explicit_local_effect,
        }


def oracle_forecast_consistency_loss(
    forecast: dict[str, torch.Tensor], oracle: dict[str, torch.Tensor]
) -> torch.Tensor:
    keys = ("local_drop_prediction", "terminal_physical_prediction")
    if any(key not in forecast or key not in oracle for key in keys):
        raise Phase35ProtocolError("RM3 consistency outputs are incomplete")
    return sum(F.smooth_l1_loss(forecast[key], oracle[key].detach()) for key in keys)


def inject_oof_residuals(
    prediction: torch.Tensor,
    residual_bank: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Deployment-shaped robustness noise; residuals must come from frozen OOF errors."""

    if residual_bank.ndim != prediction.ndim or residual_bank.shape[1:] != prediction.shape[1:]:
        raise Phase35ProtocolError("RM3 OOF residual bank shape mismatch")
    if indices.ndim != 1 or len(indices) != len(prediction):
        raise Phase35ProtocolError("RM3 OOF residual indices are invalid")
    if indices.min() < 0 or indices.max() >= len(residual_bank):
        raise Phase35ProtocolError("RM3 OOF residual index is out of bounds")
    return prediction + residual_bank[indices]
