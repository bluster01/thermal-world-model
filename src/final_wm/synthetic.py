"""Synthetic operating-point data for local micro-smoke and unit tests.

Generates smooth, physically plausible boundary/action/history trajectories
around a nominal 660 MW supercritical operating point.  These data are for
interface verification only: they carry no scientific content and must not
be used for model selection.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from src.final_wm.contracts import BOUNDARY_NORM, FinalWMProtocolError
from src.final_wm.model import HistoryWindow


class SyntheticBatch(NamedTuple):
    history: HistoryWindow
    future_boundary: torch.Tensor   # (B, H, 7)
    future_actions: torch.Tensor    # (B, H, 2)
    future_obs: torch.Tensor | None # (B, H, 5), present for teacher rollouts


def _ar1(generator: torch.Generator, shape: tuple[int, ...], rho: float, scale: float) -> torch.Tensor:
    noise = torch.randn(shape, generator=generator)
    out = torch.zeros_like(noise)
    acc = torch.zeros(shape[0], shape[-1])
    for t in range(shape[1]):
        acc = rho * acc + (1.0 - rho) ** 0.5 * noise[:, t]
        out[:, t] = acc
    return out * scale


def synthetic_history(
    batch: int = 4,
    history_steps: int = 96,
    horizon: int = 60,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> SyntheticBatch:
    """Independent synthetic history/future around the nominal operating point."""
    if batch < 1 or history_steps < 1 or horizon < 1:
        raise FinalWMProtocolError("synthetic batch dimensions must be positive")
    gen = torch.Generator().manual_seed(seed)
    total = history_steps + horizon

    pm = 17.0 + 2.0 * _ar1(gen, (batch, total, 1), 0.995, 1.0)
    d_flow = 350.0 + 30.0 * _ar1(gen, (batch, total, 1), 0.995, 1.0)
    u_b = 250.0 + 20.0 * _ar1(gen, (batch, total, 1), 0.995, 1.0)
    tm_sep = 360.0 + 12.0 * _ar1(gen, (batch, total, 1), 0.99, 1.0) + 2.0 * (pm - 17.0)
    tfw = 280.0 + 8.0 * _ar1(gen, (batch, total, 1), 0.995, 1.0)
    p_out = pm - 1.2 - 0.3 * _ar1(gen, (batch, total, 1), 0.99, 1.0)
    v1 = (0.35 + 0.12 * _ar1(gen, (batch, total, 1), 0.98, 1.0)).clamp(0.02, 0.95)
    v2 = (0.40 + 0.15 * _ar1(gen, (batch, total, 1), 0.98, 1.0)).clamp(0.02, 0.95)
    spray = (3.6 * (2.0 * v1 + 4.0 * v2) + 0.5 * _ar1(gen, (batch, total, 1), 0.9, 1.0)).clamp(min=0.0)

    boundary = torch.cat([d_flow, u_b, pm, tm_sep, tfw, p_out, spray], dim=-1)
    actions = torch.cat([v1, v2], dim=-1)

    # Plausible observation anchors (qualitative; not physics-derived).
    spray_cooling = 4.0 * v1 + 9.0 * v2
    obs1 = tm_sep + 25.0 + 3.0 * _ar1(gen, (batch, total, 1), 0.99, 1.0)
    obs2 = obs1 - spray_cooling - 2.0
    obs3 = obs1 + 45.0 + 3.0 * _ar1(gen, (batch, total, 1), 0.99, 1.0)
    obs4 = obs3 - spray_cooling - 3.0
    obs5 = 565.0 + 2.0 * _ar1(gen, (batch, total, 1), 0.995, 1.0) - 0.5 * spray_cooling
    obs = torch.cat([obs1, obs2, obs3, obs4, obs5], dim=-1)

    history = HistoryWindow(
        obs=obs[:, :history_steps].to(device),
        actions=actions[:, :history_steps].to(device),
        boundary=boundary[:, :history_steps].to(device),
    )
    return SyntheticBatch(
        history=history,
        future_boundary=boundary[:, history_steps:].to(device),
        future_actions=actions[:, history_steps:].to(device),
        future_obs=obs[:, history_steps:].to(device),
    )


def teacher_rollout_obs(
    transition,
    batch: SyntheticBatch,
    *,
    seed: int = 0,
    noise_c: float = 0.3,
) -> torch.Tensor:
    """Same-type teacher: roll the transition from its steady init and add
    small measurement noise.  Used by the micro-smoke to check that the
    assembled model is self-consistent and trainable (known-truth solvability
    on synthetic data only)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        state_0 = transition.initial_steady_state(
            batch.future_boundary[:, 0],
            batch.future_actions[:, 0],
            batch.history.obs[:, -1],
        )
        _states, temps = transition.integrate(
            state_0, batch.future_boundary, batch.future_actions
        )
    return temps + noise_c * torch.randn(temps.shape, generator=gen)
