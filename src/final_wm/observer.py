"""Probabilistic initial-state observer q(x0 | H).

The observer maps a history window H = {y, u, b} (past measurements, past
actions, past boundary) to a diagonal Gaussian posterior over the packed
physical+latent state.  Contractual properties:

- it never reads future information: the signature accepts history tensors
  only, and the window length is fixed by `ObserverConfig.history_steps`;
- posterior support is bounded by construction (tanh-squashed normalized
  mean mapped through fixed physical loc/scale constants), so an untrained
  observer cannot produce physically absurd states;
- adjacent-window state continuity is evaluated with
  `state_continuity_error`, in normalized units.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    ACTION_NORM,
    BOUNDARY_ELEMENTS,
    BOUNDARY_NORM,
    FinalWMProtocolError,
    OBSERVATION_ELEMENTS,
    OBSERVATION_NORM,
    ObserverConfig,
    PHYSICAL_STATE_NORM,
    StateLayout,
    validate_observer_config,
)


class ProbabilisticObserver(nn.Module):
    def __init__(self, config: ObserverConfig, layout: StateLayout) -> None:
        super().__init__()
        validate_observer_config(config)
        if config.latent_dim != layout.latent_dim:
            raise FinalWMProtocolError("observer latent_dim must match the state layout")
        self.config = config
        self.layout = layout

        loc = [loc for loc, _s in PHYSICAL_STATE_NORM] + [0.0] * layout.latent_dim
        scale = [s for _l, s in PHYSICAL_STATE_NORM] + [1.0] * layout.latent_dim
        self.register_buffer("state_loc", torch.tensor(loc, dtype=torch.float32))
        self.register_buffer("state_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer(
            "obs_loc", torch.tensor([loc for loc, _s in OBSERVATION_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "obs_scale", torch.tensor([s for _l, s in OBSERVATION_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "boundary_loc", torch.tensor([loc for loc, _s in BOUNDARY_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "boundary_scale", torch.tensor([s for _l, s in BOUNDARY_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "action_loc", torch.tensor([loc for loc, _s in ACTION_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "action_scale", torch.tensor([s for _l, s in ACTION_NORM], dtype=torch.float32)
        )

        in_dim = len(OBSERVATION_ELEMENTS) + len(ACTION_ELEMENTS) + len(BOUNDARY_ELEMENTS)
        self.encoder = nn.GRU(input_size=in_dim, hidden_size=config.d_hidden, batch_first=True)
        self.mu_head = nn.Linear(config.d_hidden, layout.dim)
        self.logvar_head = nn.Linear(config.d_hidden, layout.dim)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # sigma_norm ~= 0.12 at init

    def _check_history(
        self,
        history_obs: torch.Tensor,
        history_actions: torch.Tensor,
        history_boundary: torch.Tensor,
    ) -> None:
        steps = self.config.history_steps
        for name, tensor, width in (
            ("history_obs", history_obs, len(OBSERVATION_ELEMENTS)),
            ("history_actions", history_actions, len(ACTION_ELEMENTS)),
            ("history_boundary", history_boundary, len(BOUNDARY_ELEMENTS)),
        ):
            if tensor.ndim != 3 or tensor.shape[1] != steps or tensor.shape[2] != width:
                raise FinalWMProtocolError(
                    f"{name} must have shape (B, {steps}, {width}); the observer reads history only"
                )

    def encode(
        self,
        history_obs: torch.Tensor,
        history_actions: torch.Tensor,
        history_boundary: torch.Tensor,
    ) -> torch.Tensor:
        self._check_history(history_obs, history_actions, history_boundary)
        obs_n = (history_obs - self.obs_loc) / self.obs_scale
        act_n = (history_actions - self.action_loc) / self.action_scale
        bnd_n = (history_boundary - self.boundary_loc) / self.boundary_scale
        features = torch.cat([obs_n, act_n, bnd_n], dim=-1)
        _output, hidden = self.encoder(features)
        return hidden[-1]

    def posterior(
        self,
        history_obs: torch.Tensor,
        history_actions: torch.Tensor,
        history_boundary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, sigma) in physical units, shapes (B, dim)."""
        hidden = self.encode(history_obs, history_actions, history_boundary)
        mu_norm = torch.tanh(self.mu_head(hidden))
        sigma_norm = F.softplus(self.logvar_head(hidden)) + 1e-3
        mu = self.state_loc + self.state_scale * mu_norm
        sigma = self.state_scale * sigma_norm
        return mu, sigma

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu + sigma * torch.randn_like(mu)

    def state_continuity_error(
        self,
        state_end_previous: torch.Tensor,
        mu_next: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized L2 distance between a rolled-forward state and the next
        window's posterior mean (per-sample vector, shape (B,))."""
        if state_end_previous.shape != mu_next.shape:
            raise FinalWMProtocolError("continuity inputs must have identical shapes")
        diff = (state_end_previous - mu_next) / self.state_scale
        return torch.linalg.vector_norm(diff, dim=-1)
