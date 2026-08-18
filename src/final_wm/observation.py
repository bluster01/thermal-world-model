"""Probabilistic multi-measurement observation model.

The measurement mean is the transition's physical output equation
g(x, b, u); this module only adds the measurement-noise distribution.  All
five anchor measurements (Tin side, attemperator outlets, terminal) are
produced simultaneously with positive, bounded sigmas.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.final_wm.contracts import (
    FinalWMProtocolError,
    OBSERVATION_ELEMENTS,
    ObservationConfig,
    PHYSICAL_STATE_NORM,
    StateLayout,
    validate_observation_config,
)


class ObservationModel(nn.Module):
    """Heteroscedastic Gaussian measurement noise over the five anchors."""

    def __init__(self, layout: StateLayout, config: ObservationConfig) -> None:
        super().__init__()
        validate_observation_config(config)
        self.layout = layout
        self.config = config
        n_out = len(OBSERVATION_ELEMENTS)
        init_raw = math.log(math.expm1(config.init_sigma_c))
        self.logsigma_bias = nn.Parameter(torch.full((n_out,), init_raw, dtype=torch.float32))
        if config.heteroscedastic:
            loc = torch.tensor([l for l, _s in PHYSICAL_STATE_NORM], dtype=torch.float32)
            scale = torch.tensor([s for _l, s in PHYSICAL_STATE_NORM], dtype=torch.float32)
            self.register_buffer("state_loc", loc)
            self.register_buffer("state_scale", scale)
            self.head = nn.Linear(layout.physical_dim, n_out)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = None

    def sigma(self, state: torch.Tensor) -> torch.Tensor:
        """Per-measurement sigma (degC), shape (..., 5), bounded positive."""
        if state.shape[-1] != self.layout.dim:
            raise FinalWMProtocolError("observation state input does not match the layout")
        raw = self.logsigma_bias
        if self.head is not None:
            physical = state[..., : self.layout.physical_dim]
            feats = (physical - self.state_loc) / self.state_scale
            raw = raw + 0.1 * self.head(feats)
        sigma = F.softplus(raw)
        return sigma.clamp(self.config.min_sigma_c, self.config.max_sigma_c)

    def distribution(
        self,
        mean: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mean.shape[-1] != len(OBSERVATION_ELEMENTS):
            raise FinalWMProtocolError("observation mean must have 5 channels")
        return mean, self.sigma(state)
