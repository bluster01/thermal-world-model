"""Probabilistic initial-state observer q(x0 | H) — anchor-relative.

Repair batch 1-B (design 2026-08-21, amendment v0.3 item 1): the observer no
longer learns an absolute posterior.  It outputs a bounded CORRECTION delta to
the five-point steady anchor, conditioned on pressure-regime features
(sub/supercritical soft indicators at 22.064 MPa).  Zero-initialised heads
make an untrained observer return the anchor exactly, removing the O1
degradation channel (learned posterior dragging exact enthalpy anchors off
by +2.7/+1.5 degC at H1).  Contractual properties:

- it never reads future information: the signature accepts history tensors
  only, and the window length is fixed by `ObserverConfig.history_steps`;
- corrections are bounded by construction (0.1 x state scale, tanh-squashed);
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
            "boundary_loc", torch.tensor([loc for loc, _s in BOUNDARY_NORM] + [0.0] * config.history_extension_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "boundary_scale", torch.tensor([s for _l, s in BOUNDARY_NORM] + [1.0] * config.history_extension_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_loc", torch.tensor([loc for loc, _s in ACTION_NORM], dtype=torch.float32)
        )
        self.register_buffer(
            "action_scale", torch.tensor([s for _l, s in ACTION_NORM], dtype=torch.float32)
        )

        in_dim = len(OBSERVATION_ELEMENTS) + len(ACTION_ELEMENTS) + len(BOUNDARY_ELEMENTS) + config.history_extension_dim
        if config.encoder_type == "gru":
            # Keep the historical module name and shapes so existing GRU
            # checkpoints continue to load strictly.
            self.encoder = nn.GRU(input_size=in_dim, hidden_size=config.d_hidden, batch_first=True)
        else:
            n_patches = 1 + (config.history_steps - config.patch_length) // config.patch_stride
            self.patch_projection = nn.Linear(config.patch_length, config.d_hidden)
            self.variable_embedding = nn.Parameter(torch.empty(in_dim, config.d_hidden))
            self.patch_embedding = nn.Parameter(torch.empty(n_patches, config.d_hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_hidden,
                nhead=config.attention_heads,
                dim_feedforward=4 * config.d_hidden,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=config.attention_layers)
            self.state_queries = nn.Parameter(torch.empty(layout.dim, config.d_hidden))
            self.state_cross_attention = nn.MultiheadAttention(
                config.d_hidden,
                config.attention_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.state_query_norm = nn.LayerNorm(config.d_hidden)
            nn.init.normal_(self.variable_embedding, std=0.02)
            nn.init.normal_(self.patch_embedding, std=0.02)
            nn.init.normal_(self.state_queries, std=0.02)
        # Repair 1-B: pressure-segmented inversion.  The T<->h sensitivity and
        # the spray dynamics differ across the critical point, so the
        # correction heads condition on soft sub/supercritical indicators of
        # the last separator pressure plus the normalized pressure itself.
        head_width = layout.dim if config.encoder_type == "gru" else 1
        self.mu_head = nn.Linear(config.d_hidden + 3, head_width)
        self.logvar_head = nn.Linear(config.d_hidden + 3, head_width)
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
            ("history_boundary", history_boundary, len(BOUNDARY_ELEMENTS) + self.config.history_extension_dim),
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
        if self.config.encoder_type == "gru":
            _output, hidden = self.encoder(features)
            return hidden[-1]
        # Preserve the public (B, d_hidden) encoding contract used by JEPA.
        # Posterior construction below retains the state-specific queries.
        return self._token_state_features(features).mean(dim=1)

    def _token_state_features(self, normalized_history: torch.Tensor) -> torch.Tensor:
        """Return one past-conditioned representation per packed state.

        Each normalized scalar channel is patched independently, so the
        observer can distinguish variables without mixing their raw units.
        Learned state queries then cross-attend to the past-only token memory.
        """
        patches = normalized_history.transpose(1, 2).unfold(
            dimension=-1,
            size=self.config.patch_length,
            step=self.config.patch_stride,
        )
        tokens = self.patch_projection(patches)
        tokens = tokens + self.variable_embedding[None, :, None, :]
        tokens = tokens + self.patch_embedding[None, None, :, :]
        memory = self.token_encoder(tokens.flatten(1, 2))
        queries = self.state_queries[None, :, :].expand(normalized_history.shape[0], -1, -1)
        attended, _weights = self.state_cross_attention(
            queries, memory, memory, need_weights=False
        )
        return self.state_query_norm(queries + attended)

    # Critical pressure of water/steam in the boundary's MPa units.
    _PC_MPA = 22.064

    def _pressure_features(self, history_boundary: torch.Tensor) -> torch.Tensor:
        pm_idx = BOUNDARY_ELEMENTS.index("separator_pressure")
        pm = history_boundary[:, -1, pm_idx]
        pm_loc = self.boundary_loc[pm_idx]
        pm_scale = self.boundary_scale[pm_idx]
        return torch.stack([
            F.softplus(pm - self._PC_MPA),
            F.softplus(self._PC_MPA - pm),
            (pm - pm_loc) / pm_scale,
        ], dim=-1)

    def posterior(
        self,
        history_obs: torch.Tensor,
        history_actions: torch.Tensor,
        history_boundary: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, sigma) in physical units, shapes (B, dim).

        mu = anchor + delta, delta bounded by 0.1 x state scale (tanh-squared
        head, zero-initialised: an untrained observer returns the anchor).
        The mode-specific mask (which dims may move) is applied by the caller
        (`FinalWorldModel._initial_state`).
        """
        if anchor.shape[-1] != self.layout.dim:
            raise FinalWMProtocolError("anchor last dim must match the state layout")
        if self.config.encoder_type == "gru":
            hidden = self.encode(history_obs, history_actions, history_boundary)
            feats = torch.cat([hidden, self._pressure_features(history_boundary)], dim=-1)
            delta_raw = self.mu_head(feats)
            logvar_raw = self.logvar_head(feats)
        else:
            self._check_history(history_obs, history_actions, history_boundary)
            obs_n = (history_obs - self.obs_loc) / self.obs_scale
            act_n = (history_actions - self.action_loc) / self.action_scale
            bnd_n = (history_boundary - self.boundary_loc) / self.boundary_scale
            state_features = self._token_state_features(torch.cat([obs_n, act_n, bnd_n], dim=-1))
            pressure = self._pressure_features(history_boundary)
            pressure = pressure[:, None, :].expand(-1, self.layout.dim, -1)
            feats = torch.cat([state_features, pressure], dim=-1)
            delta_raw = self.mu_head(feats).squeeze(-1)
            logvar_raw = self.logvar_head(feats).squeeze(-1)
        delta = 0.1 * self.state_scale * torch.tanh(delta_raw)
        sigma_norm = F.softplus(logvar_raw) + 1e-3
        sigma = self.state_scale * sigma_norm
        return anchor + delta, sigma

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
