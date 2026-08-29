"""Protocolized evidence-chain analyses (the audit pack).

Re-implements, as tested in-repo functions, the exploratory analyses behind
`results/final_wm/evidence_chain.md` (originally Linux-local /tmp scripts):
real-plant valve-step event study, persistence increment baseline, spray
sensitivity regression + mixing cooling reference, residual binning,
error-floor anchors, and the rewetting ablation probe.

Everything here reads the validation split only; the test split stays locked.
Model-based probes take a trained model explicitly and never train anything.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BOUNDARY_ELEMENTS,
    KAPPA_TPH_TO_KGS,
    OBSERVATION_ELEMENTS,
    FinalWMProtocolError,
)
from src.final_wm.data import CanonicalRecord, sample_windows
from src.final_wm.evaluation import step_response_direction
from src.final_wm.model import FinalWorldModel

EVENT_HORIZONS = (1, 6, 18, 60)

# Anchors for the steady-state mixing cooling reference (evidence chain §2):
# h_steam - h_spray ≈ 1674 kJ/kg, effective cp ≈ 2.2 kJ/kg/K reproduce the
# 1.36 °C (D=560 kg/s) to 3.8 °C (D=200 kg/s) band.
SPRAY_DELTA_H_KJ_KG = 1674.0
MIX_CP_KJ_KG_K = 2.2

FINAL_OBS_INDEX = OBSERVATION_ELEMENTS.index("final_outlet_temp")
STEAM_FLOW_INDEX = BOUNDARY_ELEMENTS.index("steam_flow")
SPRAY_FLOW_INDEX = BOUNDARY_ELEMENTS.index("spray_flow_total")


# ---------------------------------------------------------------------------
# Real-plant event study
# ---------------------------------------------------------------------------

def valve_step_events(
    record: CanonicalRecord,
    split_id: int,
    valve_index: int,
    *,
    min_step: float = 0.04,
    horizon: int = 60,
    obs_index: int = FINAL_OBS_INDEX,
) -> dict:
    """Collect isolated valve-step events from contiguous runs of a split.

    An event at t requires |v[t] - v[t-1]| >= min_step; events are excluded
    when any other step (either valve) occurs within `horizon` steps before
    or after t (contaminated baseline / contaminated response), or when the
    window [t-1, t+horizon] leaves the run.  ΔT at horizon h is
    obs[t-1+h] - obs[t-1] on `obs_index`.
    """
    if valve_index >= len(ACTION_ELEMENTS):
        raise FinalWMProtocolError("valve index out of range")
    actions = record.actions.numpy()
    obs = record.obs.numpy()
    out = {"up": [], "down": [], "n_excluded_followup": 0, "n_excluded_preceded": 0}
    horizons = tuple(h for h in EVENT_HORIZONS if h <= horizon)
    for start, end in record.split_runs(split_id):
        v = actions[start:end, valve_index]
        step = np.abs(np.diff(actions[start:end, :], axis=0)).max(axis=1)  # (L-1,) any valve
        for t_rel in range(1, end - start - horizon):
            dv = v[t_rel] - v[t_rel - 1]
            if abs(dv) < min_step:
                continue
            if step[t_rel : t_rel + horizon].max(initial=0.0) >= min_step:
                out["n_excluded_followup"] += 1
                continue
            if step[max(0, t_rel - 1 - horizon) : t_rel - 1].max(initial=0.0) >= min_step:
                out["n_excluded_preceded"] += 1
                continue
            t = start + t_rel
            base = obs[t - 1, obs_index]
            deltas = np.array([obs[t - 1 + h, obs_index] - base for h in horizons], dtype=np.float64)
            out["up" if dv > 0 else "down"].append(deltas)
    for key in ("up", "down"):
        rows = out[key]
        out[key] = {
            "n": len(rows),
            "horizons": horizons,
            "deltas": np.stack(rows) if rows else np.zeros((0, len(horizons))),
        }
    return out


def event_study_summary(events: dict) -> dict:
    """Per direction/horizon: mean ΔT and correct-direction fraction
    (up: ΔT < 0 = cooling; down: ΔT > 0 = warming)."""
    summary = {}
    for direction in ("up", "down"):
        block = events[direction]
        deltas = block["deltas"]
        entry = {"n": block["n"], "horizons": list(block["horizons"])}
        if block["n"] > 0:
            entry["mean_delta"] = deltas.mean(axis=0).tolist()
            correct = deltas < 0 if direction == "up" else deltas > 0
            entry["frac_correct"] = correct.astype(np.float64).mean(axis=0).tolist()
        else:
            entry["mean_delta"] = [float("nan")] * len(block["horizons"])
            entry["frac_correct"] = [float("nan")] * len(block["horizons"])
        summary[direction] = entry
    summary["n_excluded_followup"] = events["n_excluded_followup"]
    summary["n_excluded_preceded"] = events["n_excluded_preceded"]
    return summary


# ---------------------------------------------------------------------------
# Baselines and references (record-only)
# ---------------------------------------------------------------------------

def persistence_increment_mae(record: CanonicalRecord, split_id: int) -> dict:
    """Per-channel MAE of one-step observation increments within a split —
    the H1 persistence reference (AR-consistency baseline)."""
    obs = record.obs.numpy()
    diffs = []
    for start, end in record.split_runs(split_id):
        if end - start < 2:
            continue
        diffs.append(np.abs(np.diff(obs[start:end], axis=0)))
    if not diffs:
        raise FinalWMProtocolError("split has no usable increments")
    arr = np.concatenate(diffs, axis=0)
    return {name: float(arr[:, i].mean()) for i, name in enumerate(OBSERVATION_ELEMENTS)}


def spray_sensitivity(record: CanonicalRecord, split_id: int) -> dict:
    """Least-squares W ~ 1 + v1 + v2 on a split.  Slopes in t/h per full
    travel and kg/s per +2% valve.  Closed-loop data: descriptive only."""
    if not record.split_runs(split_id):
        raise FinalWMProtocolError("split is empty")
    mask = record.split == split_id
    v = record.actions[mask].to(torch.float64)
    w = record.boundary[mask, SPRAY_FLOW_INDEX].to(torch.float64)
    design = torch.column_stack([torch.ones(len(v), dtype=torch.float64), v])
    # Normal equations via torch's bundled LAPACK: numpy.linalg.lstsq aborts
    # the interpreter on this Windows MKL build when combined with torch.
    gram = design.T @ design
    coef = torch.linalg.solve(gram, design.T @ w)
    resid = w - design @ coef
    r2 = 1.0 - float(resid.var() / w.var().clamp_min(1e-12))
    return {
        "intercept_tph": float(coef[0]),
        "dW_dv1_tph_per_full": float(coef[1]),
        "dW_dv2_tph_per_full": float(coef[2]),
        "dW_dv1_kgs_per_2pct": float(coef[1] * 0.02 * KAPPA_TPH_TO_KGS),
        "dW_dv2_kgs_per_2pct": float(coef[2] * 0.02 * KAPPA_TPH_TO_KGS),
        "r2": r2,
        "closed_loop_warning": True,
    }


def mixing_cooling_reference(
    dWdv_kgs_per_2pct: float,
    *,
    d_lo: float = 200.0,
    d_hi: float = 560.0,
) -> dict:
    """Steady-state mixed-temperature drop per +2% valve: dW * Δh / (D * cp),
    evaluated at the low/high steam-flow ends."""
    return {
        "delta_t_at_d_lo": float(dWdv_kgs_per_2pct * SPRAY_DELTA_H_KJ_KG / (d_lo * MIX_CP_KJ_KG_K)),
        "delta_t_at_d_hi": float(dWdv_kgs_per_2pct * SPRAY_DELTA_H_KJ_KG / (d_hi * MIX_CP_KJ_KG_K)),
        "assumptions": {"delta_h_kj_kg": SPRAY_DELTA_H_KJ_KG, "cp_kj_kg_k": MIX_CP_KJ_KG_K},
    }


def error_floor_anchors(
    record: CanonicalRecord,
    split_id: int,
    *,
    n_bins: int = 5,
    median_window: int = 61,
) -> dict:
    """Error-floor anchors per observation channel.

    fast_sigma: σ of obs minus a rolling median (default 61 steps ≈ 10 min) —
    the fast, regime-free component.  within_bin_sigma: σ after removing
    steam-flow-quintile bin means — the regime-conditional remainder.
    """
    obs = record.obs.numpy()
    flow = record.boundary.numpy()[:, STEAM_FLOW_INDEX]
    fast, flat_obs, flat_flow = [], [], []
    for start, end in record.split_runs(split_id):
        seg = obs[start:end]
        if end - start < median_window:
            continue
        seg_t = torch.from_numpy(seg)
        med = seg_t.unfold(0, median_window, 1).median(dim=-1).values
        offset = median_window // 2
        core = seg_t[offset : offset + med.shape[0]]
        fast.append((core - med).numpy())
        flat_obs.append(seg)
        flat_flow.append(flow[start:end])
    if not fast:
        raise FinalWMProtocolError("split runs too short for the median window")
    fast_arr = np.concatenate(fast, axis=0)
    flat_obs_arr = np.concatenate(flat_obs, axis=0)
    flat_flow_arr = np.concatenate(flat_flow, axis=0)
    edges = np.quantile(flat_flow_arr, np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_id = np.clip(np.digitize(flat_flow_arr, edges), 0, n_bins - 1)
    centered = flat_obs_arr.copy()
    for b in range(n_bins):
        sel = bin_id == b
        if sel.any():
            centered[sel] -= flat_obs_arr[sel].mean(axis=0)
    return {
        "fast_sigma": {n: float(fast_arr[:, i].std()) for i, n in enumerate(OBSERVATION_ELEMENTS)},
        "within_bin_sigma": {n: float(centered[:, i].std()) for i, n in enumerate(OBSERVATION_ELEMENTS)},
    }


# ---------------------------------------------------------------------------
# Model-based probes
# ---------------------------------------------------------------------------

class WindowErrors(NamedTuple):
    abs_err: torch.Tensor   # (N, H, 5)
    load: torch.Tensor      # (N,) steam flow at first future step
    day_ids: torch.Tensor   # (N,)


@torch.no_grad()
def window_abs_errors(
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
) -> WindowErrors:
    """Per-window per-channel absolute rollout errors (mirrors
    `evaluate_windows` sampling so numbers stay comparable)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    errs, loads, days = [], [], []
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
        errs.append((batch.future_obs.to(device) - result.temps_mu).abs().cpu())
        loads.append(batch.future_boundary[:, 0, STEAM_FLOW_INDEX])
        days.append(batch.day_ids)
        done += bsz
    return WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads), day_ids=torch.cat(days))


def binning_stats(
    errors: WindowErrors,
    *,
    n_bins: int = 5,
    horizons: tuple[int, ...] = (1, 18),
) -> dict:
    """Per channel/horizon: load-quintile bin mean errors and the
    between-bin variance fraction (regime-bias signature)."""
    load = errors.load.numpy()
    edges = np.quantile(load, np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_id = np.clip(np.digitize(load, edges), 0, n_bins - 1)
    out = {}
    for h in horizons:
        if h > errors.abs_err.shape[1]:
            continue
        per_h = {}
        vals = errors.abs_err[:, : h, :].mean(dim=1).numpy()  # (N, 5)
        for c, name in enumerate(OBSERVATION_ELEMENTS):
            col = vals[:, c]
            means = np.array([col[bin_id == b].mean() if (bin_id == b).any() else np.nan
                              for b in range(n_bins)])
            total = float(col.var())
            between = float(np.nanvar(means))
            per_h[name] = {
                "bin_means": means.tolist(),
                "between_ratio": between / total if total > 1e-12 else 0.0,
            }
        out[f"H{h}"] = per_h
    return out


@torch.no_grad()
def rewetting_ablation(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int = 32,
    history_steps: int = 96,
    seed: int = 0,
    device: str | torch.device = "cpu",
    valve_index: int = 1,
    delta_v: float = 0.05,
    allow_extrapolation: bool = False,
) -> dict:
    """Step-response direction with the wall-rewetting gains intact vs zeroed
    (aW1/aW2 raw -> -30, softplus ≈ 0).  Parameters are restored afterwards."""
    probe = dict(
        record=record, split_id=split_id, n_windows=n_windows,
        history_steps=history_steps, seed=seed, device=device,
        valve_index=valve_index, delta_v=delta_v,
        allow_extrapolation=allow_extrapolation,
    )
    intact = step_response_direction(model, **probe)
    raws = [model.transition.raw[name] for name in ("aW1", "aW2")]
    saved = [r.data.clone() for r in raws]
    try:
        for r in raws:
            r.data.fill_(-30.0)
        zeroed = step_response_direction(model, **probe)
    finally:
        for r, backup in zip(raws, saved):
            r.data.copy_(backup)
    return {"intact": intact, "rewet_zeroed": zeroed}


# ---------------------------------------------------------------------------
# CF-3 position-binned local gain (credential B3; checklist 2026-08-21)
# ---------------------------------------------------------------------------

def position_binned_gain(
    record: CanonicalRecord,
    split_id: int,
    valve_index: int,
    *,
    n_bins: int = 4,
    min_step: float = 0.04,
    horizon: int = 60,
    obs_index: int = FINAL_OBS_INDEX,
    model: FinalWorldModel | None = None,
    history_steps: int = 96,
    rollout_steps: int = 60,
    delta_v: float = 0.05,
    n_windows: int = 256,
    seed: int = 0,
    device: str | torch.device = "cpu",
    allow_extrapolation: bool = False,
) -> dict:
    """Local spray gain vs absolute valve opening, data side + model side.

    Data: isolated valve-step events (same exclusion rules as
    `valve_step_events`), gain = ΔT(horizon)/dv, binned by pre-event opening.
    Model (optional): per-window step response at the same horizon, gain =
    terminal delta/delta_v, binned by window-end opening on the SAME edges.
    Tests the valve-nonlinearity hypothesis: if the installed characteristic
    is nonlinear, the data-side local gain varies with opening, and a faithful
    model's gain curve should track it.  Evidence-only (event counts are
    small); no CIs, no gate.
    """
    if valve_index >= len(ACTION_ELEMENTS):
        raise FinalWMProtocolError("valve index out of range")
    actions = record.actions.numpy()
    obs = record.obs.numpy()
    events: list[tuple[float, float]] = []  # (pre_opening, gain degC per full opening)
    for start, end in record.split_runs(split_id):
        v = actions[start:end, valve_index]
        step = np.abs(np.diff(actions[start:end, :], axis=0)).max(axis=1)
        for t_rel in range(1, end - start - horizon):
            dv = v[t_rel] - v[t_rel - 1]
            if abs(dv) < min_step:
                continue
            if step[t_rel: t_rel + horizon].max(initial=0.0) >= min_step:
                continue
            if step[max(0, t_rel - 1 - horizon): t_rel - 1].max(initial=0.0) >= min_step:
                continue
            t = start + t_rel
            delta_t = float(obs[t - 1 + horizon, obs_index] - obs[t - 1, obs_index])
            events.append((float(v[t_rel - 1]), delta_t / dv))
    if len(events) < n_bins:
        edges = np.linspace(0.0, 1.0, n_bins + 1).tolist()
        return {"valve_index": valve_index, "bin_edges": edges, "bins": [],
                "note": f"too few isolated events ({len(events)}) to bin"}
    openings = np.array([e[0] for e in events])
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(openings, quantiles)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.maximum.accumulate(edges).tolist()

    bins: list[dict] = [{"lo": float(edges[i]), "hi": float(edges[i + 1]),
                         "data": {"n": 0, "mean_gain": None}, "model": None}
                        for i in range(n_bins)]
    for opening, gain in events:
        bi = min(int(np.searchsorted(edges, opening, side="right")) - 1, n_bins - 1)
        bi = max(bi, 0)
        cell = bins[bi]["data"]
        cell["n"] += 1
        cell["mean_gain"] = gain if cell["mean_gain"] is None else (
            cell["mean_gain"] * (cell["n"] - 1) + gain) / cell["n"]

    if model is not None:
        model.eval()
        gen = torch.Generator().manual_seed(seed)
        batch = sample_windows(record, split_id, n_windows, history_steps, 1, gen)
        history = batch.history.__class__(
            obs=batch.history.obs.to(device),
            actions=batch.history.actions.to(device),
            boundary=batch.history.boundary.to(device),
        )
        b0 = batch.future_boundary[:, 0].to(device)
        a0 = batch.future_actions[:, 0].to(device)
        v_abs = batch.history.actions[:, -1, valve_index].numpy()
        boundary_seq = b0.unsqueeze(1).repeat(1, rollout_steps, 1)
        base = a0.unsqueeze(1).repeat(1, rollout_steps, 1)
        step = base.clone()
        step[:, :, valve_index] = (step[:, :, valve_index] + delta_v).clamp(max=1.0)
        base_result = model.counterfactual(
            history, base, boundary_mode="oracle", true_future_boundary=boundary_seq,
            allow_extrapolation=allow_extrapolation,
        )
        step_result = model.counterfactual(
            history, step, boundary_mode="oracle", true_future_boundary=boundary_seq,
            allow_extrapolation=allow_extrapolation,
        )
        if base_result.in_support is None or step_result.in_support is None:
            raise FinalWMProtocolError("counterfactual path did not return support masks")
        temps_base = base_result.temps_mu
        temps_step = step_result.temps_mu
        gains = ((temps_step[:, -10:, obs_index].mean(dim=1)
                  - temps_base[:, -10:, obs_index].mean(dim=1)) / delta_v).detach().cpu().numpy()
        for i, (opening, gain) in enumerate(zip(v_abs, gains)):
            bi = min(max(int(np.searchsorted(edges, float(opening), side="right")) - 1, 0), n_bins - 1)
            cell = bins[bi]
            if cell["model"] is None:
                cell["model"] = {"n": 0, "mean_gain": 0.0}
            cell["model"]["n"] += 1
            cell["model"]["mean_gain"] += (float(gain) - cell["model"]["mean_gain"]) / cell["model"]["n"]

    report = {"valve_index": valve_index, "obs_index": obs_index, "horizon": horizon,
              "bin_edges": [float(e) for e in edges], "bins": bins,
              "n_events": len(events)}
    if model is not None:
        report.update({
            "support_rate": float(step_result.in_support.float().mean()),
            "n_unsupported": int((~step_result.in_support).sum()),
            "in_support_mask": step_result.in_support.cpu().tolist(),
            "baseline_support_rate": float(base_result.in_support.float().mean()),
            "baseline_n_unsupported": int((~base_result.in_support).sum()),
            "baseline_in_support_mask": base_result.in_support.cpu().tolist(),
            "allow_extrapolation": bool(allow_extrapolation),
        })
    return report
