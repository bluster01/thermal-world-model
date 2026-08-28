"""Evaluation utilities for the discrimination matrix.

Everything here operates on validation windows only.  Estimators are
paired by UTC day so the block bootstrap respects the matrix's
dependence structure; single-seed results never produce a champion.
"""

from __future__ import annotations

import math
from statistics import NormalDist
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


class ScalarMetrics(NamedTuple):
    values: torch.Tensor     # (N,)
    day_ids: torch.Tensor    # (N,)


def day_block_mean_ci(
    values: torch.Tensor,
    day_ids: torch.Tensor,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Equal-day mean and UTC-day block bootstrap CI.

    A one-day smoke sample is reported but explicitly non-identifiable; it
    can never satisfy a formal evidence gate.
    """
    values = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
    day_ids = torch.as_tensor(day_ids).flatten().cpu()
    if values.numel() == 0 or values.shape != day_ids.shape:
        raise FinalWMProtocolError("values/day_ids must be non-empty aligned vectors")
    if not bool(torch.isfinite(values).all()):
        return {
            "point": None,
            "ci_lo": None,
            "ci_hi": None,
            "n_days": int(torch.unique(day_ids).numel()),
            "identifiable": False,
        }
    days = torch.unique(day_ids)
    day_means = torch.stack([values[day_ids == day].mean() for day in days])
    point = float(day_means.mean())
    if len(days) < 2:
        return {
            "point": point,
            "ci_lo": None,
            "ci_hi": None,
            "n_days": int(len(days)),
            "identifiable": False,
        }
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(days), (n_boot, len(days)), generator=gen)
    boot = day_means[idx].mean(dim=1)
    return {
        "point": point,
        "ci_lo": float(torch.quantile(boot, 0.025)),
        "ci_hi": float(torch.quantile(boot, 0.975)),
        "n_days": int(len(days)),
        "identifiable": True,
    }


@torch.no_grad()
def state_continuity_metrics(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int,
    history_steps: int,
    gap_steps: int = 18,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> ScalarMetrics:
    """Arm-specific adjacent-window normalized state-continuity errors."""
    if gap_steps < 1 or gap_steps > history_steps:
        raise FinalWMProtocolError("gap_steps must be in [1, history_steps]")
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, split_id, n_windows, history_steps, gap_steps, gen)
    history = batch.history.__class__(
        obs=batch.history.obs.to(device),
        actions=batch.history.actions.to(device),
        boundary=batch.history.boundary.to(device),
    )
    next_history = batch.history.__class__(
        obs=torch.cat([history.obs[:, gap_steps:], batch.future_obs.to(device)], dim=1),
        actions=torch.cat([history.actions[:, gap_steps:], batch.future_actions.to(device)], dim=1),
        boundary=torch.cat([history.boundary[:, gap_steps:], batch.future_boundary.to(device)], dim=1),
    )
    state = model._initial_state(history)
    boundary = model.boundary_model.oracle(batch.future_boundary.to(device))
    rolled = model._rollout(
        state, boundary, batch.future_actions.to(device), mode="oracle"
    ).states[:, -1]
    next_state = model._initial_state(next_history)
    values = model.observer.state_continuity_error(rolled, next_state)
    return ScalarMetrics(values=values.cpu(), day_ids=batch.day_ids)


def _stability_summary(states: torch.Tensor, temps: torch.Tensor, horizon: int) -> dict:
    finite = torch.isfinite(states).all() and torch.isfinite(temps).all()
    if not bool(finite):
        return {
            "horizon": int(horizon),
            "all_finite": False,
            "max_abs_drift_c": None,
            "p95_abs_drift_c": None,
            "max_settle_c": None,
            "bounded": False,
        }
    drift = (temps[:, -1, -1] - temps[:, 0, -1]).abs()
    settle_lag = min(6, horizon)
    settle = (temps[:, -1, -1] - temps[:, -settle_lag, -1]).abs()
    max_drift = float(drift.max())
    max_settle = float(settle.max())
    return {
        "horizon": int(horizon),
        "all_finite": True,
        "max_abs_drift_c": max_drift,
        "p95_abs_drift_c": float(torch.quantile(drift, 0.95)),
        "max_settle_c": max_settle,
        "bounded": bool(max_drift <= 60.0 and max_settle <= 5.0),
    }


@torch.no_grad()
def constant_condition_stability(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int,
    history_steps: int,
    rollout_steps: int = 60,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """Frozen-boundary/action rollout using the local drift/settle contract."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, split_id, n_windows, history_steps, 1, gen)
    boundary_values = batch.future_boundary[:, :1].to(device).repeat(1, rollout_steps, 1)
    actions = batch.future_actions[:, :1].to(device).repeat(1, rollout_steps, 1)
    state = model.transition.initial_steady_state(
        boundary_values[:, 0], actions[:, 0], batch.history.obs[:, -1].to(device)
    )
    result = model._rollout(
        state, model.boundary_model.oracle(boundary_values), actions, mode="oracle"
    )
    report = _stability_summary(result.states, result.temps_mu, rollout_steps)
    report["rollout_steps"] = int(rollout_steps)
    report["n_windows"] = int(n_windows)
    return report


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
    n_boot: int = 1000,
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
    ci = day_block_mean_ci(delta.cpu(), batch.day_ids, n_boot=n_boot, seed=seed + 1)
    return {
        "mean_delta_c": ci["point"],
        "ci_lo_c": ci["ci_lo"],
        "ci_hi_c": ci["ci_hi"],
        "n_days": ci["n_days"],
        "ci_identifiable": ci["identifiable"],
        "frac_negative": float((delta < 0).float().mean()),
        "n_windows": int(n_windows),
        "rollout_steps": int(rollout_steps),
        "valve_index": int(valve_index),
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


# ---------------------------------------------------------------------------
# CF credential probes (2026-08-21; checklist results/final_wm/
# world_model_credential_checklist_20260821.md, credentials B3/B4/B6/D1).
# Evaluation-only: no model/spec change, no fingerprint impact.
# ---------------------------------------------------------------------------

# Observation channel indices used by the probes (see contracts.OBSERVATION_ELEMENTS).
_SH1_OUT, _SH2_OUT, _FINAL = 1, 3, 4


@torch.no_grad()
def counterfactual_fidelity_synthetic(
    model: FinalWorldModel,
    teacher,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int = 64,
    history_steps: int = 96,
    horizon: int = 18,
    valve_index: int = 1,
    delta_v: float = 0.05,
    abduction: str = "replay",
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """CF-1: counterfactual delta-trajectory fidelity against the known teacher.

    Teacher chain: steady-state init at window start, then replay of the true
    history (near-perfect abduction).  Student chain: the same replay protocol
    on its own transition (abduction="replay"), or the observer
    (abduction="observer").  Both roll out baseline (true future actions) and
    counterfactual (valve += delta_v) trajectories; the metric compares the
    DELTA trajectories (cf - base), isolating interventional response fidelity
    from level error.
    """
    if abduction not in ("replay", "observer"):
        raise FinalWMProtocolError("abduction must be 'replay' or 'observer'")
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, split_id, n_windows, history_steps, horizon, gen)
    history = batch.history.__class__(
        obs=batch.history.obs.to(device),
        actions=batch.history.actions.to(device),
        boundary=batch.history.boundary.to(device),
    )
    fut_b = batch.future_boundary.to(device)
    fut_a = batch.future_actions.to(device)

    def _replay_state(transition) -> torch.Tensor:
        s = transition.initial_steady_state(
            history.boundary[:, 0], history.actions[:, 0], history.obs[:, -1]
        )
        states, _temps = transition.integrate(s, history.boundary, history.actions)
        return states[:, -1]  # integrate returns the full (B, H, dim) trajectory

    t_state = _replay_state(teacher)
    if abduction == "observer":
        s_state = model._initial_state(history)
    else:
        s_state = _replay_state(model.transition)

    cf_actions = fut_a.clone()
    cf_actions[:, :, valve_index] = (cf_actions[:, :, valve_index] + delta_v).clamp(max=1.0)

    _s, t_base = teacher.integrate(t_state, fut_b, fut_a)
    _s, t_cf = teacher.integrate(t_state, fut_b, cf_actions)
    _s, s_base = model.transition.integrate(s_state, fut_b, fut_a)
    _s, s_cf = model.transition.integrate(s_state, fut_b, cf_actions)

    d_teacher = t_cf - t_base          # (B, H, 5)
    d_student = s_cf - s_base
    diff = (d_student - d_teacher).abs()          # delta-trajectory error
    per_channel_mae = diff.mean(dim=(0, 1))       # (5,)
    mag_ratio = (
        d_student.abs().mean(dim=(0, 1)) / d_teacher.abs().mean(dim=(0, 1)).clamp_min(1e-9)
    )
    eps = 1e-6
    sign_mask = d_teacher[:, -1, :].abs() > eps
    sign_agree = (
        (torch.sign(d_student[:, -1, :]) == torch.sign(d_teacher[:, -1, :])) & sign_mask
    ).float().sum() / sign_mask.float().sum().clamp_min(1.0)
    return {
        "abduction": abduction,
        "valve_index": int(valve_index),
        "delta_v": float(delta_v),
        "horizon": int(horizon),
        "n_windows": int(n_windows),
        "delta_mae_per_channel": per_channel_mae.tolist(),
        "delta_mae": float(per_channel_mae.mean()),
        "delta_magnitude_ratio_per_channel": mag_ratio.tolist(),
        "terminal_sign_agreement": float(sign_agree),
        "teacher_delta_terminal": d_teacher[:, -1, :].mean(dim=0).tolist(),
        "student_delta_terminal": d_student[:, -1, :].mean(dim=0).tolist(),
    }


@torch.no_grad()
def constraint_checks(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int = 32,
    history_steps: int = 96,
    rollout_steps: int = 120,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """CF-4: physics constraint consistency (no ground truth needed).

    (a) monotonicity: larger v2 opening -> strictly more cooling at sh2_outlet;
    (b) zero-spray drift: removing all spray must warm the spray-outlet
    channels (dry blend), never cool them.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, split_id, n_windows, history_steps, 1, gen)
    b0 = batch.future_boundary[:, 0].to(device)
    a0 = batch.future_actions[:, 0].to(device)
    obs0 = batch.history.obs[:, -1].to(device)
    state0 = model.transition.initial_steady_state(b0, a0, obs0)
    boundary_seq = b0.unsqueeze(1).repeat(1, rollout_steps, 1)
    base_actions = a0.unsqueeze(1).repeat(1, rollout_steps, 1)
    _s, temps_base = model.transition.integrate(state0, boundary_seq, base_actions)

    terminal = temps_base[:, -10:, :].mean(dim=1)  # (B, 5)

    grid = (0.01, 0.02, 0.05, 0.10)
    mono: list[float] = []
    for dv in grid:
        step = base_actions.clone()
        step[:, :, 1] = (step[:, :, 1] + dv).clamp(max=1.0)
        _s, temps_step = model.transition.integrate(state0, boundary_seq, step)
        mono.append(float((temps_step[:, -10:, _SH2_OUT].mean(dim=1) - terminal[:, _SH2_OUT]).mean()))
    monotone = all(mono[i + 1] < mono[i] for i in range(len(mono) - 1)) and mono[-1] < 0.0

    zero = torch.zeros_like(base_actions)
    _s, temps_zero = model.transition.integrate(state0, boundary_seq, zero)
    drift = temps_zero[:, -10:, :].mean(dim=1) - terminal  # (B, 5)
    return {
        "monotonicity": {
            "delta_v_grid": list(grid),
            "mean_terminal_delta_c": mono,
            "monotone_cooling": bool(monotone),
            "rollout_steps": int(rollout_steps),
        },
        "zero_spray_drift": {
            "mean_drift_c": drift.mean(dim=0).tolist(),
            "frac_positive_sh1_out": float((drift[:, _SH1_OUT] > 0).float().mean()),
            "frac_positive_sh2_out": float((drift[:, _SH2_OUT] > 0).float().mean()),
        },
        "n_windows": int(n_windows),
    }


@torch.no_grad()
def calibration_coverage(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int = 256,
    batch_size: int = 32,
    history_steps: int = 96,
    horizon: int = 18,
    boundary_mode: str = "oracle",
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """D1: empirical coverage of the Gaussian predictive intervals.

    For each level, the central interval is mu +- sigma * Phi^{-1}((1+level)/2).
    Reported per channel at horizons {1, 6, horizon} and overall; a calibrated
    model's empirical coverage matches the nominal level.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    mus, sigmas, targets = [], [], []
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
        mus.append(result.temps_mu.cpu())
        sigmas.append(result.temps_sigma.cpu())
        targets.append(batch.future_obs)
        done += bsz
    mu = torch.cat(mus)          # (N, H, 5)
    sigma = torch.cat(sigmas).clamp_min(1e-6)
    target = torch.cat(targets)
    abs_err = (target - mu).abs()

    horizon_marks = sorted({1, min(6, horizon), horizon})
    coverage: dict = {"levels": list(levels), "horizons": horizon_marks, "per_channel": {}, "overall": {}}
    for h in horizon_marks:
        err_h = abs_err[:, :h, :]  # (N, h, 5)
        sig_h = sigma[:, :h, :]
        for level in levels:
            z = NormalDist().inv_cdf((1.0 + level) / 2.0)
            covered = (err_h <= z * sig_h).float()
            key = f"H{h}_L{level:.2f}"
            coverage["per_channel"][key] = covered.mean(dim=(0, 1)).tolist()
            coverage["overall"][key] = float(covered.mean())
    # Summary diagnostic: mean |coverage - nominal| overall.
    gaps = []
    for h in horizon_marks:
        for level in levels:
            gaps.append(abs(coverage["overall"][f"H{h}_L{level:.2f}"] - level))
    coverage["mean_abs_coverage_gap"] = float(sum(gaps) / len(gaps))
    coverage["n_windows"] = int(n_windows)
    return coverage
