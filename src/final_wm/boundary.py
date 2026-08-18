"""Dual-mode future boundary model.

The world model must never silently consume true future Tin/load/pressure.
Two explicitly labeled modes are provided:

- `forecast`: a GRU encoder-decoder predicts a diagonal Gaussian over future
  boundary channels from history and an optional declared scenario; the
  method signature has no slot for true future boundary data;
- `oracle`: wraps supplied true future boundary values for diagnostics;
  the result is labeled `mode="oracle"` and downstream components reject it
  unless the run is explicitly in oracle mode.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    ACTION_NORM,
    BOUNDARY_ELEMENTS,
    BOUNDARY_NORM,
    BoundaryModelConfig,
    FinalWMProtocolError,
    validate_boundary_config,
)


class BoundarySequence(NamedTuple):
    mu: torch.Tensor        # (B, H, 7), physical units
    logvar: torch.Tensor    # (B, H, 7), log variance in physical units
    mode: str               # "forecast" | "oracle"

    def sample(self) -> torch.Tensor:
        return self.mu + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mu)


class BoundaryModel(nn.Module):
    def __init__(self, config: BoundaryModelConfig) -> None:
        super().__init__()
        validate_boundary_config(config)
        self.config = config
        self.register_buffer(
            "loc", torch.tensor([loc for loc, _s in BOUNDARY_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "scale", torch.tensor([s for _l, s in BOUNDARY_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "action_loc", torch.tensor([loc for loc, _s in ACTION_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "action_scale", torch.tensor([s for _l, s in ACTION_NORM], dtype=torch.float32)
        )
        n_boundary = len(BOUNDARY_ELEMENTS)
        in_dim = n_boundary + len(ACTION_ELEMENTS)
        self.encoder = nn.GRU(input_size=in_dim, hidden_size=config.d_hidden, batch_first=True)
        dec_in = n_boundary + config.scenario_dim
        self.decoder = nn.GRUCell(input_size=dec_in, hidden_size=config.d_hidden)
        self.mu_head = nn.Linear(config.d_hidden, n_boundary)
        self.logvar_head = nn.Linear(config.d_hidden, n_boundary)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -3.0)

    def forecast(
        self,
        history_boundary: torch.Tensor,
        history_actions: torch.Tensor,
        *,
        scenario: torch.Tensor | None = None,
        horizon: int | None = None,
    ) -> BoundarySequence:
        """Predict p(b_{1:H} | H, scenario).  No true-future input exists here."""
        steps = self.config.history_steps
        if history_boundary.shape[1] != steps or history_boundary.shape[2] != len(BOUNDARY_ELEMENTS):
            raise FinalWMProtocolError(f"history_boundary must be (B, {steps}, 7)")
        if history_actions.shape[:2] != history_boundary.shape[:2]:
            raise FinalWMProtocolError("history_actions shape mismatch")
        horizon = int(horizon or self.config.horizon)
        if horizon < 1:
            raise FinalWMProtocolError("horizon must be >= 1")
        if self.config.scenario_dim > 0:
            if scenario is None or scenario.shape != (history_boundary.shape[0], self.config.scenario_dim):
                raise FinalWMProtocolError("declared scenario vector is required by this config")
        elif scenario is not None:
            raise FinalWMProtocolError("scenario supplied but scenario_dim is 0")

        bnd_n = (history_boundary - self.loc) / self.scale
        act_n = (history_actions - self.action_loc) / self.action_scale
        _out, hidden = self.encoder(torch.cat([bnd_n, act_n], dim=-1))
        state = hidden[-1]
        prev = torch.zeros(history_boundary.shape[0], len(BOUNDARY_ELEMENTS), device=history_boundary.device)
        mus, logvars = [], []
        for _t in range(horizon):
            dec_in = prev if scenario is None else torch.cat([prev, scenario], dim=-1)
            state = self.decoder(dec_in, state)
            mus.append(self.mu_head(state))
            logvars.append(self.logvar_head(state))
            prev = mus[-1]
        mu = torch.stack(mus, dim=1) * self.scale + self.loc
        logvar = torch.stack(logvars, dim=1) + 2.0 * torch.log(self.scale)
        return BoundarySequence(mu=mu, logvar=logvar, mode="forecast")

    def oracle(self, true_future_boundary: torch.Tensor, *, sigma: float = 1e-3) -> BoundarySequence:
        """Wrap true future boundary values for oracle diagnostics only."""
        if true_future_boundary.ndim != 3 or true_future_boundary.shape[2] != len(BOUNDARY_ELEMENTS):
            raise FinalWMProtocolError("oracle boundary must be (B, H, 7)")
        if sigma <= 0:
            raise FinalWMProtocolError("oracle sigma must be positive")
        logvar = torch.full_like(true_future_boundary, 2.0 * float(torch.log(torch.tensor(sigma))))
        return BoundarySequence(mu=true_future_boundary, logvar=logvar, mode="oracle")


def require_mode(sequence: BoundarySequence, mode: str) -> None:
    """Fail-closed check used by the assembled world model."""
    if sequence.mode != mode:
        raise FinalWMProtocolError(
            f"boundary sequence mode is '{sequence.mode}' but the run requires '{mode}'"
        )
