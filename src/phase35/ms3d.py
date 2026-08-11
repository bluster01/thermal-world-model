"""Pure Phase 3.5-MS3-D event, response, and day-level diagnostic logic."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .data import Phase35Cache
from .schema import (
    LOAD_COLUMN,
    PRESSURE_COLUMN,
    SP_COLUMN,
    TARGET_COLUMN,
    TIN2_COLUMN,
    TOUT2_COLUMN,
    VALVE_COLUMN,
    Phase35ProtocolError,
)


REQUIRED_COLUMNS = (
    LOAD_COLUMN,
    PRESSURE_COLUMN,
    TIN2_COLUMN,
    TOUT2_COLUMN,
    TARGET_COLUMN,
    SP_COLUMN,
    VALVE_COLUMN,
)


def validate_ms3d_config(config: Mapping[str, Any]) -> None:
    """Fail closed if the local diagnostic could leak test or change estimands."""
    if config.get("protocol_version") != "phase3.5-ms3d-v1":
        raise Phase35ProtocolError("unsupported MS3-D protocol version")
    data = config.get("data_contract", {})
    if data.get("split") != "validation" or data.get("test_allowed") is not False:
        raise Phase35ProtocolError("MS3-D is validation-only and test must stay locked")
    if data.get("sides") != ["A", "B"] or data.get("step_seconds") != 10:
        raise Phase35ProtocolError("MS3-D requires the frozen A/B 10 s contract")
    event = config.get("event_contract", {})
    minimum = float(event.get("sp_step_min_abs_c", -1))
    maximum = float(event.get("sp_step_max_abs_c", -1))
    if not 0 < minimum <= maximum <= 3.0:
        raise Phase35ProtocolError("MS3-D SP-step bounds changed or are invalid")
    if event.get("sp_hold_seconds") != 600:
        raise Phase35ProtocolError("MS3-D must retain the full 600 s held-SP horizon")
    if event.get("pre_sp_stability_seconds") != 60 or not 0 < float(
        event.get("pre_sp_range_max_c", -1)
    ) <= 0.1:
        raise Phase35ProtocolError("MS3-D requires a stable 60 s pre-onset SP")
    response = config.get("response_contract", {})
    if response.get("horizons_seconds") != [60, 180, 300, 600]:
        raise Phase35ProtocolError("MS3-D response horizons changed")
    if response.get("baseline_seconds") != 60:
        raise Phase35ProtocolError("MS3-D baseline must remain 60 s")
    boundary = config.get("claim_boundary", {})
    forbidden = ("training_allowed", "linux_authorized", "causal_claim_allowed", "ms4_release_allowed")
    if any(boundary.get(key) is not False for key in forbidden):
        raise Phase35ProtocolError("MS3-D claim boundary was broadened")
    if boundary.get("ms3_decision_immutable") is not True:
        raise Phase35ProtocolError("MS3-D cannot revise the frozen MS3 decision")


def _range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


def _endpoint_median(values: np.ndarray, onset: int, step: int, width: int) -> float:
    start = onset + step - width + 1
    return float(np.median(values[start : onset + step + 1]))


def _timestamp_iso(timestamp_ns: int) -> str:
    return str(np.datetime64(int(timestamp_ns), "ns"))


def _stability_pass(
    values: np.ndarray,
    onset: int,
    indices: Mapping[str, int],
    layer: Mapping[str, Any],
    step_seconds: int,
) -> tuple[bool, dict[str, float]]:
    steps = int(layer["window_seconds"] // step_seconds)
    pre = values[onset - steps : onset]
    ranges = {
        "load_range_mw": _range(pre[:, indices[LOAD_COLUMN]]),
        "pressure_range_mpa": _range(pre[:, indices[PRESSURE_COLUMN]]),
        "terminal_temperature_range_c": _range(pre[:, indices[TARGET_COLUMN]]),
        "valve_range_pct": _range(pre[:, indices[VALVE_COLUMN]]),
    }
    passed = bool(
        ranges["load_range_mw"] <= float(layer["load_range_max_mw"])
        and ranges["pressure_range_mpa"] <= float(layer["pressure_range_max_mpa"])
        and ranges["terminal_temperature_range_c"]
        <= float(layer["terminal_temperature_range_max_c"])
    )
    return passed, ranges


def detect_ms3d_events(
    cache: Phase35Cache,
    other_cache: Phase35Cache,
    side: str,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Detect held-SP events without using any post-event outcome for selection."""
    validate_ms3d_config(config)
    if side not in {"A", "B"}:
        raise Phase35ProtocolError("MS3-D side must be A or B")
    for column in REQUIRED_COLUMNS:
        cache.index(column)
    other_cache.index(SP_COLUMN)
    other_cache.index(VALVE_COLUMN)
    if not np.array_equal(cache.timestamps_ns, other_cache.timestamps_ns):
        raise Phase35ProtocolError("MS3-D A/B caches do not share one timeline")
    if len(cache.timestamps_ns) != len(other_cache.timestamps_ns):
        raise Phase35ProtocolError("MS3-D A/B caches have different lengths")

    data = config["data_contract"]
    event = config["event_contract"]
    response = config["response_contract"]
    step_seconds = int(data["step_seconds"])
    lo, hi = cache.split_bounds()[data["split"]]
    indices = {column: cache.index(column) for column in REQUIRED_COLUMNS}
    other_sp = other_cache.values[:, other_cache.index(SP_COLUMN)].astype(float)
    other_valve = other_cache.values[:, other_cache.index(VALVE_COLUMN)].astype(float)
    values = cache.values.astype(float, copy=False)
    timestamps = cache.timestamps_ns
    sp = values[:, indices[SP_COLUMN]]
    valve = values[:, indices[VALVE_COLUMN]]
    target = values[:, indices[TARGET_COLUMN]]
    local_drop = values[:, indices[TIN2_COLUMN]] - values[:, indices[TOUT2_COLUMN]]

    horizon_steps = int(event["sp_hold_seconds"] // step_seconds)
    pre180_steps = int(event["steady_180s"]["window_seconds"] // step_seconds)
    strict_steps = int(event["strict_600s_support"]["window_seconds"] // step_seconds)
    baseline_steps = int(response["baseline_seconds"] // step_seconds)
    pre_sp_steps = int(event["pre_sp_stability_seconds"] // step_seconds)
    endpoint_width = int(response["endpoint_median_seconds"] // step_seconds)
    gap_steps = int(event["min_same_side_gap_seconds"] // step_seconds)
    expected_step_ns = int(step_seconds * 1_000_000_000)
    min_step = float(event["sp_step_min_abs_c"])
    max_step = float(event["sp_step_max_abs_c"])
    raw_candidates = np.flatnonzero(np.abs(np.diff(sp)) >= min_step) + 1
    raw_candidates = raw_candidates[(raw_candidates >= lo) & (raw_candidates < hi)]

    funnel: Counter[str] = Counter()
    funnel["raw_sp_step_candidates_ge_min"] = int(len(raw_candidates))
    accepted: list[dict[str, Any]] = []
    last_accepted = -10**12
    for onset in raw_candidates.tolist():
        delta_sp = float(sp[onset] - sp[onset - 1])
        if not math.isfinite(delta_sp) or abs(delta_sp) > max_step:
            funnel["rejected_above_max_sp_step"] += 1
            continue
        funnel["within_sp_step_bounds"] += 1
        if onset < lo + pre180_steps or onset + horizon_steps >= hi:
            funnel["rejected_split_boundary"] += 1
            continue
        if onset - last_accepted < gap_steps:
            funnel["rejected_same_side_gap"] += 1
            continue
        timeline = timestamps[onset - pre180_steps : onset + horizon_steps + 1]
        if np.any(np.diff(timeline) != expected_step_ns):
            funnel["rejected_irregular_timeline"] += 1
            continue
        pre_sp = sp[onset - pre_sp_steps : onset]
        if not np.isfinite(pre_sp).all() or _range(pre_sp) > float(
            event["pre_sp_range_max_c"]
        ):
            funnel["rejected_pre_sp_not_stable"] += 1
            continue
        hold_tolerance = max(
            float(event["sp_hold_abs_tolerance_c"]),
            float(event["sp_hold_relative_tolerance"]) * abs(delta_sp),
        )
        post_sp = sp[onset : onset + horizon_steps + 1]
        if not np.isfinite(post_sp).all() or _range(post_sp) > hold_tolerance:
            funnel["rejected_not_held_600s"] += 1
            continue
        required = values[
            onset - pre180_steps : onset + horizon_steps + 1,
            [indices[column] for column in REQUIRED_COLUMNS],
        ]
        if not np.isfinite(required).all():
            funnel["rejected_nonfinite_required_signal"] += 1
            continue
        pre180 = values[onset - pre180_steps : onset]
        operating = event["operating"]
        pre_load = float(np.mean(pre180[:, indices[LOAD_COLUMN]]))
        pre_pressure = float(np.mean(pre180[:, indices[PRESSURE_COLUMN]]))
        pre_target = float(np.mean(pre180[:, indices[TARGET_COLUMN]]))
        if not (
            pre_load >= float(operating["min_load_mw"])
            and pre_pressure >= float(operating["min_pressure_mpa"])
            and float(operating["terminal_temperature_min_c"])
            <= pre_target
            <= float(operating["terminal_temperature_max_c"])
        ):
            funnel["rejected_not_operating"] += 1
            continue

        stable60, ranges60 = _stability_pass(
            values, onset, indices, event["steady_60s"], step_seconds
        )
        stable180, ranges180 = _stability_pass(
            values, onset, indices, event["steady_180s"], step_seconds
        )
        strict600: bool | None = None
        strict600_clean: bool | None = None
        ranges600: dict[str, float] | None = None
        if onset >= lo + strict_steps:
            strict_timeline = timestamps[onset - strict_steps : onset + 1]
            if np.all(np.diff(strict_timeline) == expected_step_ns):
                strict600, ranges600 = _stability_pass(
                    values,
                    onset,
                    indices,
                    event["strict_600s_support"],
                    step_seconds,
                )
                strict600_clean = bool(
                    strict600
                    and ranges600["valve_range_pct"]
                    <= float(
                        event["strict_600s_support"][
                            "valve_range_max_pct_for_clean_chain"
                        ]
                    )
                )
        clean_chain = bool(
            stable60
            and stable180
            and ranges60["valve_range_pct"]
            <= float(event["steady_60s"]["valve_range_max_pct_for_clean_chain"])
            and ranges180["valve_range_pct"]
            <= float(event["steady_180s"]["valve_range_max_pct_for_clean_chain"])
        )
        primary = bool(stable60 and stable180)
        sign = float(np.sign(delta_sp))
        valve0 = float(np.median(valve[onset - baseline_steps : onset]))
        target0 = float(np.median(target[onset - baseline_steps : onset]))
        drop0 = float(np.median(local_drop[onset - baseline_steps : onset]))
        other_sp0 = float(np.median(other_sp[onset - baseline_steps : onset]))
        other_valve0 = float(np.median(other_valve[onset - baseline_steps : onset]))
        other_sp_change = float(
            np.max(np.abs(other_sp[onset : onset + horizon_steps + 1] - other_sp0))
        )
        other_valve_change = float(
            np.max(
                np.abs(other_valve[onset : onset + horizon_steps + 1] - other_valve0)
            )
        )
        quiet = event["other_loop_quiet_diagnostic"]
        row: dict[str, Any] = {
            "event_id": f"{side}_{int(timestamps[onset])}",
            "side": side,
            "split": "validation",
            "onset_index": int(onset),
            "timestamp_ns": int(timestamps[onset]),
            "timestamp_utc": _timestamp_iso(int(timestamps[onset])),
            "utc_day": _timestamp_iso(int(timestamps[onset]))[:10],
            "delta_sp_c": delta_sp,
            "abs_delta_sp_c": abs(delta_sp),
            "direction": "sp_up" if delta_sp > 0 else "sp_down",
            "hold_tolerance_c": hold_tolerance,
            "stable_60s": stable60,
            "stable_180s": stable180,
            "primary_dual_steady": primary,
            "strict_600s_support": strict600,
            "strict_600s_clean_chain": strict600_clean,
            "clean_chain": clean_chain,
            "analysis_layer": "primary_dual_steady" if primary else "dynamic_secondary",
            "pre_load_mean_mw": pre_load,
            "pre_pressure_mean_mpa": pre_pressure,
            "pre_terminal_temperature_mean_c": pre_target,
            "pre_60s": ranges60,
            "pre_180s": ranges180,
            "pre_600s": ranges600,
            "other_loop_sp_max_abs_change_c": other_sp_change,
            "other_loop_valve_max_abs_change_pct": other_valve_change,
            "other_loop_quiet": bool(
                other_sp_change <= float(quiet["sp_max_abs_change_c"])
                and other_valve_change <= float(quiet["valve_max_abs_change_pct"])
            ),
        }
        for horizon_seconds in response["horizons_seconds"]:
            step = int(horizon_seconds // step_seconds)
            valve_h = _endpoint_median(valve, onset, step, endpoint_width)
            drop_h = _endpoint_median(local_drop, onset, step, endpoint_width)
            target_h = _endpoint_median(target, onset, step, endpoint_width)
            raw_valve = valve_h - valve0
            raw_drop = drop_h - drop0
            raw_target = target_h - target0
            valve_expected = -sign * raw_valve
            drop_expected = -sign * raw_drop
            target_expected = sign * raw_target
            prefix = f"h{horizon_seconds}"
            row[f"{prefix}_raw_valve_delta_pct"] = raw_valve
            row[f"{prefix}_raw_local_drop_delta_c"] = raw_drop
            row[f"{prefix}_raw_terminal_delta_c"] = raw_target
            row[f"{prefix}_expected_valve_motion_pct"] = valve_expected
            row[f"{prefix}_expected_local_drop_motion_c"] = drop_expected
            row[f"{prefix}_expected_terminal_motion_c"] = target_expected
            row[f"{prefix}_valve_per_sp_pct_per_c"] = valve_expected / abs(delta_sp)
            row[f"{prefix}_local_drop_per_sp"] = drop_expected / abs(delta_sp)
            row[f"{prefix}_terminal_per_sp"] = target_expected / abs(delta_sp)
            if abs(raw_valve) >= float(response["gain_min_abs_valve_dose_pct"]):
                row[f"{prefix}_local_drop_gain_c_per_pct"] = raw_drop / raw_valve
                row[f"{prefix}_terminal_gain_c_per_pct"] = -raw_target / raw_valve
            else:
                row[f"{prefix}_local_drop_gain_c_per_pct"] = None
                row[f"{prefix}_terminal_gain_c_per_pct"] = None
        accepted.append(row)
        last_accepted = onset
        funnel["accepted_held_operating"] += 1
        funnel["accepted_primary_dual_steady" if primary else "accepted_dynamic_secondary"] += 1
        if clean_chain:
            funnel["accepted_clean_chain"] += 1
        if strict600 is True:
            funnel["accepted_strict_600s_support"] += 1
        if strict600_clean is True:
            funnel["accepted_strict_600s_clean_chain"] += 1
    return accepted, dict(sorted(funnel.items()))


def response_metric_keys(horizons_seconds: Iterable[int]) -> list[str]:
    keys: list[str] = []
    for horizon in horizons_seconds:
        prefix = f"h{int(horizon)}"
        keys.extend(
            [
                f"{prefix}_valve_per_sp_pct_per_c",
                f"{prefix}_local_drop_per_sp",
                f"{prefix}_terminal_per_sp",
                f"{prefix}_local_drop_gain_c_per_pct",
                f"{prefix}_terminal_gain_c_per_pct",
            ]
        )
    return keys


def _finite_values(events: Iterable[Mapping[str, Any]], key: str) -> np.ndarray:
    values = [event.get(key) for event in events]
    return np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )


def summarize_side_layer(
    events: list[dict[str, Any]], metric_keys: Iterable[str]
) -> dict[str, Any]:
    days = sorted({event["utc_day"] for event in events})
    result: dict[str, Any] = {
        "event_count": len(events),
        "utc_day_count": len(days),
        "utc_days": days,
        "sp_up_count": sum(event["direction"] == "sp_up" for event in events),
        "sp_down_count": sum(event["direction"] == "sp_down" for event in events),
        "clean_chain_count": sum(bool(event["clean_chain"]) for event in events),
        "other_loop_quiet_count": sum(bool(event["other_loop_quiet"]) for event in events),
        "metrics": {},
    }
    for key in metric_keys:
        values = _finite_values(events, key)
        day_values = []
        for day in days:
            selected = _finite_values(
                [event for event in events if event["utc_day"] == day], key
            )
            if len(selected):
                day_values.append(float(np.median(selected)))
        result["metrics"][key] = {
            "event_n": int(len(values)),
            "event_median": float(np.median(values)) if len(values) else None,
            "event_q25": float(np.quantile(values, 0.25)) if len(values) else None,
            "event_q75": float(np.quantile(values, 0.75)) if len(values) else None,
            "positive_event_fraction": float(np.mean(values > 0)) if len(values) else None,
            "day_n": len(day_values),
            "day_median": float(np.median(day_values)) if day_values else None,
        }
    return result


def _paired_day_values(
    events_by_side: Mapping[str, list[dict[str, Any]]], key: str
) -> tuple[list[str], np.ndarray]:
    daily: dict[str, dict[str, float]] = defaultdict(dict)
    for side in ("A", "B"):
        by_day: dict[str, list[float]] = defaultdict(list)
        for event in events_by_side[side]:
            value = event.get(key)
            if value is not None and math.isfinite(float(value)):
                by_day[event["utc_day"]].append(float(value))
        for day, values in by_day.items():
            daily[day][side] = float(np.median(values))
    common = sorted(day for day, values in daily.items() if set(values) == {"A", "B"})
    differences = np.asarray([daily[day]["B"] - daily[day]["A"] for day in common])
    return common, differences


def _bootstrap_median_ci(
    values: np.ndarray, *, samples: int, seed: int, block_length: int = 1
) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    if block_length == 1:
        indices = rng.integers(0, len(values), size=(samples, len(values)))
    else:
        block_count = math.ceil(len(values) / block_length)
        starts = rng.integers(0, len(values), size=(samples, block_count))
        offsets = np.arange(block_length, dtype=np.int64)
        indices = (starts[..., None] + offsets) % len(values)
        indices = indices.reshape(samples, -1)[:, : len(values)]
    boot = np.median(values[indices], axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def paired_day_contrasts(
    events_by_side: Mapping[str, list[dict[str, Any]]],
    metric_keys: Iterable[str],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    samples = int(statistics["bootstrap_samples"])
    seed = int(statistics["bootstrap_seed"])
    block_lengths = [int(value) for value in statistics["diagnostic_block_lengths_days"]]
    output: dict[str, Any] = {}
    for index, key in enumerate(metric_keys):
        days, differences = _paired_day_values(events_by_side, key)
        output[key] = {
            "contrast": "B_minus_A",
            "paired_utc_day_count": len(days),
            "paired_utc_days": days,
            "median_difference": float(np.median(differences)) if len(differences) else None,
            "ci95": _bootstrap_median_ci(
                differences, samples=samples, seed=seed + index * 100
            ),
            "diagnostic_circular_block_ci95": {
                str(block): _bootstrap_median_ci(
                    differences,
                    samples=samples,
                    seed=seed + index * 100 + block,
                    block_length=block,
                )
                for block in block_lengths
                if block <= len(differences)
            },
        }
    return output


def diagnosis_label(
    paired: Mapping[str, Any], checkpoint: Mapping[str, Any], minimum_ratio: float
) -> dict[str, Any]:
    """Apply a conservative descriptive decision without treating no-difference as equivalence."""
    model_ratio = float(checkpoint["B_to_A_abs_h600_effect_ratio_median"])
    physical_keys = (
        "h180_local_drop_per_sp",
        "h300_local_drop_per_sp",
        "h180_local_drop_gain_c_per_pct",
        "h300_local_drop_gain_c_per_pct",
        "h600_terminal_per_sp",
    )
    intervals = {key: paired[key]["ci95"] for key in physical_keys}
    consistently_b_larger = all(
        interval is not None and interval[0] > 0 for interval in intervals.values()
    )
    empirical_not_demonstrably_b_larger = all(
        interval is None or interval[0] <= 0 for interval in intervals.values()
    )
    if consistently_b_larger:
        label = "FIELD_A_WEAK_SUPPORTED"
    elif model_ratio >= minimum_ratio and empirical_not_demonstrably_b_larger:
        label = "MODEL_A_RESPONSE_ATTENUATION_EXCEEDS_FIELD_EVIDENCE"
    else:
        label = "INCONCLUSIVE_ASYMMETRY_DIAGNOSIS"
    return {
        "label": label,
        "model_large_asymmetry": model_ratio >= minimum_ratio,
        "model_B_to_A_abs_h600_effect_ratio_median": model_ratio,
        "field_chain_all_B_minus_A_ci_lower_positive": consistently_b_larger,
        "field_chain_no_prespecified_B_minus_A_ci_lower_positive": (
            empirical_not_demonstrably_b_larger
        ),
        "interpretation_boundary": (
            "Attenuation diagnosis only; intervals containing zero do not establish "
            "equivalence or identify free-head absorption or another mechanism."
        ),
    }
