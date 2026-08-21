"""Evaluation utilities for the discrimination matrix.

Everything here operates on validation windows only.  Estimators are
paired by UTC day so the block bootstrap respects the matrix's
dependence structure; single-seed results never produce a champion.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import CanonicalRecord, WindowBatch, sample_windows
from src.final_wm.model import FinalWorldModel


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closed-form CRPS of N(mu, sigma^2) at y (same shape as inputs)."""
    sigma = sigma.clamp_min(1e-6)
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    pdf = torch.exp(normal.log_prob(z))
    cdf = normal.cdf(z)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


class WindowMetrics(NamedTuple):
    nll: torch.Tensor        # (N, H) mean NLL over 5 channels per step
    mae: torch.Tensor        # (N, H)
    crps: torch.Tensor       # (N, H)
    day_ids: torch.Tensor    # (N,)


@torch.no_grad()
def evaluate_windows(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int,
    batch_size: int,
    history_steps: int,
    horizon: int,
    boundary_mode: str,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> WindowMetrics:
    """Rollout metrics on `n_windows` validation windows (batched)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    nlls, maes, crpss, days = [], [], [], []
    done = 0
    while done < n_windows:
        bsz = min(batch_size, n_windows - done)
        batch = sample_windows(record, split_id, bsz, history_steps, horizon, gen)
        history = batch.history.__class__(
            obs=batch.history.obs.to(device),
            actions=batch.history.actions.to(device),
            boundary=batch.history.boundary.to(device),
        )
        result = model.forecast(
            history,
            batch.future_actions.to(device),
            boundary_mode=boundary_mode,
            true_future_boundary=batch.future_boundary.to(device) if boundary_mode == "oracle" else None,
        )
        target = batch.future_obs.to(device)
        var = result.temps_sigma ** 2
        nll = (0.5 * (target - result.temps_mu) ** 2 / var + torch.log(result.temps_sigma)).mean(dim=-1)
        mae = (target - result.temps_mu).abs().mean(dim=-1)
        crps = gaussian_crps(result.temps_mu, result.temps_sigma, target).mean(dim=-1)
        nlls.append(nll.cpu())
        maes.append(mae.cpu())
        crpss.append(crps.cpu())
        days.append(batch.day_ids)
        done += bsz
    return WindowMetrics(
        nll=torch.cat(nlls), mae=torch.cat(maes), crps=torch.cat(crpss), day_ids=torch.cat(days)
    )


def horizon_summary(metrics: WindowMetrics, horizons: tuple[int, ...] = (1, 6, 18)) -> dict:
    out = {}
    for h in horizons:
        if h > metrics.nll.shape[1]:
            continue
        sl = slice(0, h)
        out[f"H{h}"] = {
            "nll": float(metrics.nll[:, sl].mean()),
            "mae": float(metrics.mae[:, sl].mean()),
            "crps": float(metrics.crps[:, sl].mean()),
        }
    return out


class ImprovementCI(NamedTuple):
    point: float
    ci_lo: float
    ci_hi: float
    n_days: int


def relative_improvement_ci(
    baseline: WindowMetrics,
    arm: WindowMetrics,
    *,
    horizon: int,
    metric: str = "nll",
    n_boot: int = 1000,
    seed: int = 0,
) -> ImprovementCI:
    """Day-paired relative improvement (baseline - arm) / baseline, with a
    UTC-day block bootstrap CI (resample days with replacement)."""
    base_values = getattr(baseline, metric)[:, :horizon].mean(dim=1)
    arm_values = getattr(arm, metric)[:, :horizon].mean(dim=1)
    base_days = baseline.day_ids
    arm_days = arm.day_ids
    common_days = torch.unique(base_days)
    common_days = common_days[torch.isin(common_days, arm_days)]
    if len(common_days) < 2:
        raise FinalWMProtocolError("fewer than two common UTC days; CI is not identifiable")
    base_by_day = torch.stack([base_values[base_days == d].mean() for d in common_days])
    arm_by_day = torch.stack([arm_values[arm_days == d].mean() for d in common_days])
    rel = (base_by_day - arm_by_day) / base_by_day.clamp_min(1e-9)
    point = float(rel.mean())
    gen = torch.Generator().manual_seed(seed)
    boots = []
    for _ in range(n_boot):
        idx = torch.randint(len(rel), (len(rel),), generator=gen)
        boots.append(float(rel[idx].mean()))
    boots_t = torch.tensor(boots)
    return ImprovementCI(
        point=point,
        ci_lo=float(torch.quantile(boots_t, 0.025)),
        ci_hi=float(torch.quantile(boots_t, 0.975)),
        n_days=len(common_days),
    )


@torch.no_grad()
def boundary_forecast_metrics(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int,
    batch_size: int,
    history_steps: int,
    horizon: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> WindowMetrics:
    """Boundary-model forecast NLL/CRPS against true future boundary."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    nlls, crpss, days = [], [], []
    done = 0
    while done < n_windows:
        bsz = min(batch_size, n_windows - done)
        batch = sample_windows(record, split_id, bsz, history_steps, horizon, gen)
        seq = model.boundary_model.forecast(
            batch.history.boundary.to(device), batch.history.actions.to(device), horizon=horizon
        )
        target = batch.future_boundary.to(device)
        var = torch.exp(seq.logvar)
        nll = (0.5 * (target - seq.mu) ** 2 / var + 0.5 * seq.logvar).mean(dim=-1)
        crps = gaussian_crps(seq.mu, torch.exp(0.5 * seq.logvar), target).mean(dim=-1)
        nlls.append(nll.cpu())
        crpss.append(crps.cpu())
        days.append(batch.day_ids)
        done += bsz
    nll = torch.cat(nlls)
    return WindowMetrics(nll=nll, mae=nll.clone(), crps=torch.cat(crpss), day_ids=torch.cat(days))


@torch.no_grad()
def persistence_boundary_metrics(
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int,
    batch_size: int,
    history_steps: int,
    horizon: int,
    seed: int = 0,
) -> WindowMetrics:
    """Persistence baseline: future boundary = last history value, with a
    per-channel random-walk sigma estimated from the training split."""
    train = record.split == 0
    diffs = record.boundary[train][1:] - record.boundary[train][:-1]
    step_std = diffs.std(dim=0).clamp_min(1e-6)
    gen = torch.Generator().manual_seed(seed)
    nlls, crpss, days = [], [], []
    done = 0
    steps = torch.arange(1, horizon + 1, dtype=torch.float32)
    while done < n_windows:
        bsz = min(batch_size, n_windows - done)
        batch = sample_windows(record, split_id, bsz, history_steps, horizon, gen)
        mu = batch.history.boundary[:, -1:].expand(-1, horizon, -1)
        sigma = step_std * steps.sqrt()[:, None]
        target = batch.future_boundary
        nll = (0.5 * (target - mu) ** 2 / sigma**2 + torch.log(sigma)).mean(dim=-1)
        crps = gaussian_crps(mu, sigma, target).mean(dim=-1)
        nlls.append(nll)
        crpss.append(crps)
        days.append(batch.day_ids)
        done += bsz
    nll = torch.cat(nlls)
    return WindowMetrics(nll=nll, mae=nll.clone(), crps=torch.cat(crpss), day_ids=torch.cat(days))


@torch.no_grad()
def step_response_direction(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int = 32,
    history_steps: int = 96,
    rollout_steps: int = 60,
    valve_index: int = 1,
    delta_v: float = 0.05,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """v2 (or v1) step +delta_v from window end; expected terminal response
    is negative (spray cools).  Returns mean delta and the fraction of
    windows with a negative long-run response."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, split_id, n_windows, history_steps, 1, gen)
    boundary_0 = batch.future_boundary[:, 0].to(device)
    action_0 = batch.future_actions[:, 0].to(device)
    obs_0 = batch.history.obs[:, -1].to(device)
    state_0 = model.transition.initial_steady_state(boundary_0, action_0, obs_0)
    boundary_seq = boundary_0.unsqueeze(1).repeat(1, rollout_steps, 1)
    base_actions = action_0.unsqueeze(1).repeat(1, rollout_steps, 1)
    step_actions = base_actions.clone()
    step_actions[:, :, valve_index] = (step_actions[:, :, valve_index] + delta_v).clamp(max=1.0)
    _s0, temps_base = model.transition.integrate(state_0, boundary_seq, base_actions)
    _s1, temps_step = model.transition.integrate(state_0, boundary_seq, step_actions)
    delta = (temps_step[:, -10:, 4] - temps_base[:, -10:, 4]).mean(dim=1)
    return {
        "mean_delta_c": float(delta.mean()),
        "frac_negative": float((delta < 0).float().mean()),
        "n_windows": int(n_windows),
    }


@torch.no_grad()
def residual_quantiles(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int = 64,
    history_steps: int = 96,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """Closure residual power quantiles over validation states (kW)."""
    if model.config.closure.injection_mode == "none":
        raise FinalWMProtocolError("closure is disabled in this configuration")
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, split_id, n_windows, history_steps, 1, gen)
    state_0 = model.transition.initial_steady_state(
        batch.future_boundary[:, 0].to(device),
        batch.future_actions[:, 0].to(device),
        batch.history.obs[:, -1].to(device),
    )
    residual = model.closure(state_0, batch.future_boundary[:, 0].to(device))
    power = residual.steam_power.flatten().abs()
    return {
        "p50_kw": float(torch.quantile(power, 0.5)),
        "p90_kw": float(torch.quantile(power, 0.9)),
        "max_kw": float(power.max()),
    }
