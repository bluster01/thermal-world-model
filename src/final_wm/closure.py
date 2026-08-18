"""Action-blind residual closure for the final world model.

The closure corrects the transition for unmeasured disturbances and
structural mismatch.  Two contractual properties are enforced here and by
`contracts.validate_closure_config`:

1. **Action-blind**: features are built only from the current physical/latent
   state and the whitelisted current boundary channels
   (`CLOSURE_BOUNDARY_CHANNELS`, which excludes the measured total spray
   flow).  No action, setpoint, valve prediction, or any future descendant
   of them can influence the residual.
2. **Fixed injection positions**: residuals enter only as per-stage power
   corrections with signs resolved by the declared injection mode
   (`none` / `steam_only` / `conservative`).  The rejected "double"
   pattern that injected the same residual into both metal and steam energy
   is not representable.

Amplitudes saturate via `tanh` at `residual_scale_kw` (kW), matching the
legacy probe scale (30% of a typical per-stage heat flow).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.final_wm.contracts import (
    BOUNDARY_ELEMENTS,
    CLOSURE_BOUNDARY_CHANNELS,
    BOUNDARY_NORM,
    ClosureConfig,
    FinalWMProtocolError,
    PHYSICAL_STATE_NORM,
    StateLayout,
    validate_closure_config,
)
from src.final_wm.transition import ResidualInjection


class ActionBlindClosure(nn.Module):
    """Residual vector field r_theta(x_t, b_t, epsilon_t)."""

    def __init__(self, config: ClosureConfig, layout: StateLayout) -> None:
        super().__init__()
        validate_closure_config(config)
        self.config = config
        self.layout = layout

        state_loc = torch.tensor([loc for loc, _s in PHYSICAL_STATE_NORM], dtype=torch.float32)
        state_scale = torch.tensor([s for _l, s in PHYSICAL_STATE_NORM], dtype=torch.float32)
        self.register_buffer("state_loc", state_loc)
        self.register_buffer("state_scale", state_scale)
        boundary_index = {name: i for i, name in enumerate(BOUNDARY_ELEMENTS)}
        channel_index = [boundary_index[name] for name in CLOSURE_BOUNDARY_CHANNELS]
        self.register_buffer("boundary_index", torch.tensor(channel_index, dtype=torch.long))
        bnd_loc = torch.tensor([BOUNDARY_NORM[i][0] for i in channel_index], dtype=torch.float32)
        bnd_scale = torch.tensor([BOUNDARY_NORM[i][1] for i in channel_index], dtype=torch.float32)
        self.register_buffer("boundary_loc", bnd_loc)
        self.register_buffer("boundary_scale", bnd_scale)

        feature_dim = layout.physical_dim + layout.latent_dim + len(CLOSURE_BOUNDARY_CHANNELS)
        if config.stochastic:
            feature_dim += 4  # epsilon channel width
        self.feature_dim = feature_dim
        out_dim = 3 + layout.latent_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim), nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.Tanh(),
            nn.Linear(config.hidden_dim, out_dim),
        )
        # Zero-init the output head: the closure starts as exactly zero, so
        # enabling it never discontinuously changes a validated skeleton.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, state: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """Whitelisted feature vector; action channels are not representable."""
        if state.shape[-1] != self.layout.dim:
            raise FinalWMProtocolError("closure state input does not match the layout")
        physical = state[..., : self.layout.physical_dim]
        physical_n = (physical - self.state_loc) / self.state_scale
        parts = [physical_n]
        if self.layout.latent_dim > 0:
            parts.append(state[..., self.layout.latent_slice])
        bnd = boundary.index_select(-1, self.boundary_index)
        parts.append((bnd - self.boundary_loc) / self.boundary_scale)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
        *,
        epsilon: torch.Tensor | None = None,
    ) -> ResidualInjection:
        if self.config.stochastic:
            if epsilon is None or epsilon.shape != (state.shape[0], 4):
                raise FinalWMProtocolError("stochastic closure requires epsilon of shape (B, 4)")
        elif epsilon is not None:
            raise FinalWMProtocolError("deterministic closure must not receive noise")
        feats = self.features(state, boundary)
        if epsilon is not None:
            feats = torch.cat([feats, epsilon], dim=-1)
        raw = self.net(feats)
        z = self.config.residual_scale_kw * torch.tanh(raw[..., :3])
        mode = self.config.injection_mode
        if mode == "none":
            steam = torch.zeros_like(z)
            metal = torch.zeros_like(z)
        elif mode == "steam_only":
            steam = z
            metal = torch.zeros_like(z)
        else:  # conservative: heat is moved into the steam side, not created twice
            steam = z
            metal = -z
        latent_step = None
        if self.layout.latent_dim > 0:
            latent_step = self.config.latent_scale * torch.tanh(raw[..., 3:])
        return ResidualInjection(steam_power=steam, metal_power=metal, latent_step=latent_step)
