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


def synthetic_canonical_arrays(
    total_steps: int = 20_000,
    *,
    seed: int = 0,
    dt_seconds: float = 10.0,
    teacher=None,
    chunk: int = 2_000,
) -> dict:
    """Long synthetic timeline in canonical-record array form.

    With `teacher=None` the observation channels use the plausible
    independent anchors of `synthetic_history`; with a teacher transition
    they come from a full-timeline teacher rollout (D-SYN known-truth
    gate).  Split: 75% train / 15% val / 10% test(locked).
    """
    import numpy as np

    if total_steps < 100:
        raise FinalWMProtocolError("total_steps too small for a canonical record")
    gen = torch.Generator().manual_seed(seed)
    pm = 17.0 + 2.0 * _ar1(gen, (1, total_steps, 1), 0.999, 1.0)
    d_flow = 350.0 + 30.0 * _ar1(gen, (1, total_steps, 1), 0.999, 1.0)
    u_b = 250.0 + 20.0 * _ar1(gen, (1, total_steps, 1), 0.999, 1.0)
    tm_sep = 360.0 + 12.0 * _ar1(gen, (1, total_steps, 1), 0.998, 1.0) + 2.0 * (pm - 17.0)
    tfw = 280.0 + 8.0 * _ar1(gen, (1, total_steps, 1), 0.999, 1.0)
    p_out = pm - 1.2 - 0.3 * _ar1(gen, (1, total_steps, 1), 0.998, 1.0)
    v1 = (0.35 + 0.15 * _ar1(gen, (1, total_steps, 1), 0.995, 1.0)).clamp(0.02, 0.95)
    v2 = (0.40 + 0.18 * _ar1(gen, (1, total_steps, 1), 0.995, 1.0)).clamp(0.02, 0.95)
    spray = (3.6 * (2.0 * v1 + 4.0 * v2) + 0.5 * _ar1(gen, (1, total_steps, 1), 0.9, 1.0)).clamp(min=0.0)
    boundary = torch.cat([d_flow, u_b, pm, tm_sep, tfw, p_out, spray], dim=-1)[0]
    actions = torch.cat([v1, v2], dim=-1)[0]

    if teacher is None:
        spray_cooling = 4.0 * v1 + 9.0 * v2
        obs1 = tm_sep + 25.0 + 3.0 * _ar1(gen, (1, total_steps, 1), 0.99, 1.0)
        obs2 = obs1 - spray_cooling - 2.0
        obs3 = obs1 + 45.0 + 3.0 * _ar1(gen, (1, total_steps, 1), 0.99, 1.0)
        obs4 = obs3 - spray_cooling - 3.0
        obs5 = 565.0 + 2.0 * _ar1(gen, (1, total_steps, 1), 0.995, 1.0) - 0.5 * spray_cooling
        obs = torch.cat([obs1, obs2, obs3, obs4, obs5], dim=-1)[0]
    else:
        with torch.no_grad():
            anchor = torch.tensor([[385.0, 383.0, 405.0, 402.0, 565.0]])
            state = teacher.initial_steady_state(boundary[:1], actions[:1], anchor)
            temps_all = []
            for start in range(0, total_steps, chunk):
                end = min(start + chunk, total_steps)
                states, temps = teacher.integrate(state, boundary[start:end].unsqueeze(0), actions[start:end].unsqueeze(0))
                state = states[:, -1]
                temps_all.append(temps[0])
            obs = torch.cat(temps_all, dim=0)
        obs = obs + 0.3 * torch.randn(obs.shape, generator=gen)

    n = total_steps
    n_test = int(n * 0.10)
    n_val = int(n * 0.15)
    split = torch.zeros(n, dtype=torch.int64)
    split[n - n_val - n_test : n - n_test] = 1
    split[n - n_test :] = 2
    timestamps = torch.arange(n, dtype=torch.int64) * int(dt_seconds) + 1_700_000_000
    return {
        "boundary": boundary.numpy().astype("float32"),
        "actions": actions.numpy().astype("float32"),
        "obs": obs.numpy().astype("float32"),
        "valid": np.ones(n, dtype=bool),
        "timestamps": timestamps.numpy(),
        "split": split.numpy().astype("int8"),
    }


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
