"""Pre-treatment-only event construction and quiet-control matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import Phase35Cache
from .schema import (
    LOAD_COLUMN,
    Phase35ProtocolError,
    SP_COLUMN,
    TARGET_COLUMN,
    TOUT2_COLUMN,
    VALVE_COLUMN,
)


MATCHING_COVARIATES = (
    "load",
    "main_temperature_error",
    "main_temperature_pretrend",
    "baseline_valve",
    "valve_pretrend",
    "second_stage_outlet_temperature",
)


@dataclass(frozen=True)
class ValveEvent:
    event_id: str
    anchor: int
    onset: int
    dose: float
    direction: str
    baseline_valve: float
    split: str


@dataclass(frozen=True)
class SPEvent:
    event_id: str
    anchor: int
    onset: int
    delta_sp: float
    valve_max_60s: float
    valve_max_300s: float
    valve_max_600s: float
    execution: str
    split: str


def _split_range(cache: Phase35Cache, split: str) -> tuple[int, int]:
    bounds = cache.split_bounds()
    if split not in bounds:
        raise Phase35ProtocolError(f"unknown split={split!r}")
    return bounds[split]


def detect_valve_events(
    cache: Phase35Cache,
    split: str,
    window: int,
    horizon: int,
    step_threshold: float = 0.8,
    dose_threshold: float = 1.0,
    dose_seconds: int = 60,
    isolate_seconds: int = 60,
    isolate_step_threshold: float = 0.5,
    stable_load_delta: float = 10.0,
    min_gap_seconds: int = 600,
    max_age_s: float = 180.0,
) -> list[ValveEvent]:
    step_s = int(cache.metadata.get("step_seconds", 10))
    dose_steps = max(1, dose_seconds // step_s)
    isolate_steps = max(1, isolate_seconds // step_s)
    gap_steps = max(1, min_gap_seconds // step_s)
    vi, li = cache.index(VALVE_COLUMN), cache.index(LOAD_COLUMN)
    valve, load = cache.values[:, vi], cache.values[:, li]
    valve_age, load_age = cache.ages_s[:, vi], cache.ages_s[:, li]
    lo, hi = _split_range(cache, split)
    start = max(lo + window, isolate_steps + 1)
    end = min(hi - horizon, len(valve) - max(horizon, dose_steps) - 1)
    events: list[ValveEvent] = []
    last_onset = -gap_steps
    for onset in range(start, end):
        if not np.isfinite(valve[onset - isolate_steps - 1:onset + dose_steps + 1]).all():
            continue
        if np.any(valve_age[onset - isolate_steps - 1:onset + dose_steps + 1] > max_age_s):
            continue
        step = float(valve[onset] - valve[onset - 1])
        if abs(step) < step_threshold:
            continue
        pre_steps = np.abs(np.diff(valve[onset - isolate_steps - 1:onset]))
        if len(pre_steps) and float(np.max(pre_steps)) >= isolate_step_threshold:
            continue
        dose = float(valve[onset + dose_steps - 1] - valve[onset - 1])
        if abs(dose) < dose_threshold:
            continue
        if not np.isfinite(load[onset - 1]) or not np.isfinite(load[onset + dose_steps - 1]):
            continue
        if load_age[onset - 1] > max_age_s or load_age[onset + dose_steps - 1] > max_age_s:
            continue
        if abs(float(load[onset + dose_steps - 1] - load[onset - 1])) > stable_load_delta:
            continue
        if onset - last_onset < gap_steps:
            continue
        direction = "open" if dose > 0 else "close"
        events.append(ValveEvent(
            event_id=f"{split}-valve-{onset}",
            anchor=onset - 1,
            onset=onset,
            dose=dose,
            direction=direction,
            baseline_valve=float(valve[onset - 1]),
            split=split,
        ))
        last_onset = onset
    return events


def detect_sp_execution_events(
    cache: Phase35Cache,
    split: str,
    window: int,
    horizon: int,
    sp_threshold: float = 1.0,
    hold_tolerance: float = 0.5,
    no_execution_threshold: float = 0.1,
    execution_threshold: float = 0.5,
    min_gap_seconds: int = 600,
    max_age_s: float = 180.0,
) -> list[SPEvent]:
    step_s = int(cache.metadata.get("step_seconds", 10))
    k60 = max(1, 60 // step_s)
    k300 = max(1, 300 // step_s)
    k600 = max(1, 600 // step_s)
    gap_steps = max(1, min_gap_seconds // step_s)
    si, vi = cache.index(SP_COLUMN), cache.index(VALVE_COLUMN)
    sp, valve = cache.values[:, si], cache.values[:, vi]
    sp_age, valve_age = cache.ages_s[:, si], cache.ages_s[:, vi]
    lo, hi = _split_range(cache, split)
    start = max(lo + window, 1)
    end = min(hi - max(horizon, k600), len(sp) - k600 - 1)
    events: list[SPEvent] = []
    last_onset = -gap_steps
    for onset in range(start, end):
        if not np.isfinite(sp[onset - 1:onset + k60 + 1]).all():
            continue
        if np.any(sp_age[onset - 1:onset + k60 + 1] > max_age_s):
            continue
        delta_sp = float(sp[onset] - sp[onset - 1])
        if abs(delta_sp) < sp_threshold:
            continue
        if float(np.max(np.abs(sp[onset:onset + k60 + 1] - sp[onset]))) > hold_tolerance:
            continue
        if onset - last_onset < gap_steps:
            continue
        baseline = valve[onset - 1]
        if not np.isfinite(baseline) or not np.isfinite(valve[onset:onset + k600 + 1]).all():
            continue
        if valve_age[onset - 1] > max_age_s or np.any(valve_age[onset:onset + k600 + 1] > max_age_s):
            continue
        max60 = float(np.max(np.abs(valve[onset:onset + k60 + 1] - baseline)))
        max300 = float(np.max(np.abs(valve[onset:onset + k300 + 1] - baseline)))
        max600 = float(np.max(np.abs(valve[onset:onset + k600 + 1] - baseline)))
        if max600 <= no_execution_threshold:
            execution = "no_execution"
        elif max60 >= execution_threshold:
            execution = "executed"
        else:
            execution = "ambiguous"
        events.append(SPEvent(
            event_id=f"{split}-sp-{onset}",
            anchor=onset - 1,
            onset=onset,
            delta_sp=delta_sp,
            valve_max_60s=max60,
            valve_max_300s=max300,
            valve_max_600s=max600,
            execution=execution,
            split=split,
        ))
        last_onset = onset
    return events


def quiet_control_candidates(
    cache: Phase35Cache,
    split: str,
    window: int,
    horizon: int,
    stride_seconds: int = 60,
    valve_tolerance: float = 0.2,
    stable_load_delta: float = 10.0,
    max_age_s: float = 180.0,
) -> np.ndarray:
    step_s = int(cache.metadata.get("step_seconds", 10))
    stride = max(1, stride_seconds // step_s)
    vi, li = cache.index(VALVE_COLUMN), cache.index(LOAD_COLUMN)
    valve, load = cache.values[:, vi], cache.values[:, li]
    valve_age, load_age = cache.ages_s[:, vi], cache.ages_s[:, li]
    lo, hi = _split_range(cache, split)
    anchors = np.arange(max(lo + window, window), hi - horizon, stride, dtype=np.int64)
    keep = []
    for anchor in anchors:
        vf = valve[anchor + 1:anchor + horizon + 1]
        if not np.isfinite(vf).all() or not np.isfinite(valve[anchor]):
            continue
        if valve_age[anchor] > max_age_s or np.any(valve_age[anchor + 1:anchor + horizon + 1] > max_age_s):
            continue
        if float(np.max(np.abs(vf - valve[anchor]))) > valve_tolerance:
            continue
        if not np.isfinite(load[anchor]) or not np.isfinite(load[anchor + horizon]):
            continue
        if load_age[anchor] > max_age_s or load_age[anchor + horizon] > max_age_s:
            continue
        if abs(float(load[anchor + horizon] - load[anchor])) > stable_load_delta:
            continue
        keep.append(anchor)
    return np.asarray(keep, dtype=np.int64)


def _pretreatment_covariates(cache: Phase35Cache, anchors: np.ndarray, pretrend_steps: int) -> np.ndarray:
    li = cache.index(LOAD_COLUMN)
    ti = cache.index(TARGET_COLUMN)
    si = cache.index(SP_COLUMN)
    vi = cache.index(VALVE_COLUMN)
    oi = cache.index(TOUT2_COLUMN)
    prev = anchors - pretrend_steps
    values = cache.values
    out = np.column_stack([
        values[anchors, li],
        values[anchors, ti] - values[anchors, si],
        values[anchors, ti] - values[prev, ti],
        values[anchors, vi],
        values[anchors, vi] - values[prev, vi],
        values[anchors, oi],
    ]).astype(np.float64)
    required_at_anchor = (li, ti, si, vi, oi)
    invalid_anchor = (cache.ages_s[anchors][:, required_at_anchor] > 180.0).any(axis=1)
    invalid_prev = (cache.ages_s[prev][:, (ti, vi)] > 180.0).any(axis=1)
    out[invalid_anchor | invalid_prev] = np.nan
    return out


def matching_diagnostics(
    cache: Phase35Cache,
    events: Sequence[ValveEvent],
    matches: Mapping[str, Sequence[int]],
    pretrend_seconds: int = 60,
) -> dict:
    """Report event-level matched balance using only pre-treatment values."""
    step_s = int(cache.metadata.get("step_seconds", 10))
    pre_steps = max(1, pretrend_seconds // step_s)
    kept_events, matched_anchors = [], []
    for event in events:
        anchors = matches.get(event.event_id, ())
        if anchors:
            kept_events.append(event.anchor)
            matched_anchors.append(np.asarray(anchors, dtype=np.int64))
    if not kept_events:
        return {"n_matched_events": 0, "status": "insufficient_events"}
    event_cov = _pretreatment_covariates(cache, np.asarray(kept_events), pre_steps)
    control_cov = np.stack([
        _pretreatment_covariates(cache, anchors, pre_steps).mean(axis=0)
        for anchors in matched_anchors
    ])
    finite = np.isfinite(event_cov).all(axis=1) & np.isfinite(control_cov).all(axis=1)
    event_cov, control_cov = event_cov[finite], control_cov[finite]
    if len(event_cov) < 3:
        return {"n_matched_events": int(len(event_cov)), "status": "insufficient_events"}
    pooled = np.sqrt((event_cov.var(axis=0, ddof=1) + control_cov.var(axis=0, ddof=1)) / 2.0)
    raw_difference = event_cov.mean(axis=0) - control_cov.mean(axis=0)
    smd = np.divide(raw_difference, pooled, out=np.zeros_like(raw_difference), where=pooled > 1e-8)
    # The target pretrend is already one of the pre-treatment matching variables;
    # retain its raw Celsius difference so the gate is interpretable.
    pretrend_index = MATCHING_COVARIATES.index("main_temperature_pretrend")
    return {
        "status": "ok",
        "n_matched_events": int(len(event_cov)),
        "covariates": {
            name: {
                "smd": float(smd[i]),
                "raw_mean_difference": float(raw_difference[i]),
            }
            for i, name in enumerate(MATCHING_COVARIATES)
        },
        "max_abs_smd": float(np.max(np.abs(smd))),
        "main_temperature_pretrend_difference_c": float(raw_difference[pretrend_index]),
    }


def match_quiet_controls(
    cache: Phase35Cache,
    events: Sequence[ValveEvent],
    controls: np.ndarray,
    controls_per_event: int = 5,
    pretrend_seconds: int = 60,
    min_time_separation_seconds: int = 1800,
) -> dict[str, list[int]]:
    if not events:
        return {}
    if len(controls) < controls_per_event:
        raise Phase35ProtocolError("not enough quiet controls for requested matching ratio")
    step_s = int(cache.metadata.get("step_seconds", 10))
    pre_steps = max(1, pretrend_seconds // step_s)
    min_sep = max(1, min_time_separation_seconds // step_s)
    event_anchors = np.asarray([e.anchor for e in events], dtype=np.int64)
    Xc = _pretreatment_covariates(cache, controls, pre_steps)
    Xe = _pretreatment_covariates(cache, event_anchors, pre_steps)
    finite_c = np.isfinite(Xc).all(axis=1)
    controls, Xc = controls[finite_c], Xc[finite_c]
    if len(controls) < controls_per_event:
        raise Phase35ProtocolError("not enough finite quiet controls")
    center = np.median(Xc, axis=0)
    scale = np.quantile(Xc, 0.75, axis=0) - np.quantile(Xc, 0.25, axis=0)
    scale[scale < 1e-6] = 1.0
    Xc = (Xc - center) / scale
    Xe = (Xe - center) / scale
    matches: dict[str, list[int]] = {}
    for event, x in zip(events, Xe):
        if not np.isfinite(x).all():
            continue
        allowed = np.abs(controls - event.anchor) >= min_sep
        idx = np.flatnonzero(allowed)
        if len(idx) < controls_per_event:
            continue
        distance = np.sum((Xc[idx] - x) ** 2, axis=1)
        chosen_local = np.argpartition(distance, controls_per_event - 1)[:controls_per_event]
        chosen = idx[chosen_local[np.argsort(distance[chosen_local])]]
        matches[event.event_id] = [int(v) for v in controls[chosen]]
    return matches


def matched_empirical_irf(
    cache: Phase35Cache,
    events: Sequence[ValveEvent],
    matches: Mapping[str, Sequence[int]],
    outcome_column: str,
    horizon: int,
    max_age_s: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    oi = cache.index(outcome_column)
    curves, doses, event_ids = [], [], []
    y = cache.values[:, oi]
    age = cache.ages_s[:, oi]
    for event in events:
        control_anchors = matches.get(event.event_id, ())
        if not control_anchors:
            continue
        event_curve = y[event.anchor + 1:event.anchor + horizon + 1] - y[event.anchor]
        control_curves = [y[a + 1:a + horizon + 1] - y[a] for a in control_anchors]
        if len(event_curve) != horizon or not np.isfinite(event_curve).all():
            continue
        if age[event.anchor] > max_age_s or np.any(age[event.anchor + 1:event.anchor + horizon + 1] > max_age_s):
            continue
        if any(age[a] > max_age_s or np.any(age[a + 1:a + horizon + 1] > max_age_s) for a in control_anchors):
            continue
        controls_arr = np.asarray(control_curves, dtype=np.float64)
        if controls_arr.shape != (len(control_anchors), horizon) or not np.isfinite(controls_arr).all():
            continue
        curves.append(event_curve - controls_arr.mean(axis=0))
        doses.append(event.dose)
        event_ids.append(event.event_id)
    return np.asarray(curves, dtype=np.float64), np.asarray(doses, dtype=np.float64), event_ids


def events_to_jsonable(events: Iterable[ValveEvent | SPEvent]) -> list[dict]:
    return [asdict(event) for event in events]
