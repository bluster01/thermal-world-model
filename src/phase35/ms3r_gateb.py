"""MS3-R Gate B paired path-closure and SP-IV feasibility diagnostics."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .data import Phase35Cache
from .ms3r import (
    SIDES,
    _point_series,
    _utc_days,
    _valve_delta,
    analysis_anchors,
    build_predictors,
    crossfit_residual_matrix,
    rolling_origin_splits,
    validate_aligned_caches,
)
from .schema import SP_COLUMN, VALVE_COLUMN, Phase35ProtocolError


def _all_horizons(config: Mapping[str, Any]) -> tuple[int, ...]:
    point = config["point_contract"]
    return tuple(sorted({*(int(v) for v in point["primary_horizons_steps"]), *(int(v) for v in point["diagnostic_horizons_steps"])}))


def validate_ms3r_gateb_config(config: Mapping[str, Any]) -> None:
    if config.get("data_contract", {}).get("split") != "validation":
        raise Phase35ProtocolError("MS3-R Gate B is validation-only")
    if config["data_contract"].get("test_allowed") is not False:
        raise Phase35ProtocolError("MS3-R Gate B must prohibit test access")
    if config["execution_contract"].get("automatic_scientific_pass") is not False:
        raise Phase35ProtocolError("MS3-R Gate B cannot auto-promote scientific claims")
    if config["iv_contract"].get("status") != "feasibility_diagnostic_only":
        raise Phase35ProtocolError("MS3-R Gate B IV must remain feasibility-only")
    if config["statistics"].get("independent_unit") != "UTC_day":
        raise Phase35ProtocolError("MS3-R Gate B independent unit must be UTC_day")
    if config["statistics"].get("no_separate_ci_comparison") is not True:
        raise Phase35ProtocolError("MS3-R Gate B requires direct paired contrasts")
    if tuple(int(v) for v in config["point_contract"]["primary_horizons_steps"]) != (6, 18):
        raise Phase35ProtocolError("MS3-R Gate B primary horizons must stay frozen at 60/180 s")
    if len(_all_horizons(config)) < 2 or min(_all_horizons(config)) < 1:
        raise Phase35ProtocolError("MS3-R Gate B horizons must be positive")
    confidence = float(config["statistics"]["simultaneous_interval_confidence"])
    if not (0.95 < confidence < 1.0):
        raise Phase35ProtocolError("MS3-R Gate B simultaneous confidence must exceed 95%")
    if config["execution_contract"].get("resource_capture_command") != "/usr/bin/time -v":
        raise Phase35ProtocolError("MS3-R Gate B must capture peak RSS with /usr/bin/time -v")


def _analysis_config(config: Mapping[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(dict(config))
    adapted["analysis"]["horizons_steps"] = list(_all_horizons(config))
    return adapted


def _history_predictors(
    caches: Mapping[str, Phase35Cache], anchors: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    lags = [int(v) for v in config["analysis"]["history_lags_steps"]]
    columns = config["point_contract"]["conditioning_columns"]
    features: list[np.ndarray] = []
    for side in SIDES:
        cache = caches[side]
        for column in columns:
            values = cache.values[:, cache.index(column)].astype(float)
            for lag in lags:
                features.append(values[anchors - lag])
    return np.stack(features, axis=1)


def _delta(cache: Phase35Cache, column: str, anchors: np.ndarray) -> np.ndarray:
    values = cache.values[:, cache.index(column)].astype(float)
    return values[anchors] - values[anchors - 1]


def _solve_two_by_two(gram: np.ndarray, cross: np.ndarray, epsilon: float) -> np.ndarray:
    a, b = float(gram[0, 0]), float(gram[0, 1])
    c, d = float(gram[1, 0]), float(gram[1, 1])
    determinant = a * d - b * c
    if not math.isfinite(determinant) or abs(determinant) <= epsilon:
        return np.full((2, cross.shape[1]), np.nan, dtype=float)
    output = np.empty((2, cross.shape[1]), dtype=float)
    output[0] = (d * cross[0] - b * cross[1]) / determinant
    output[1] = (-c * cross[0] + a * cross[1]) / determinant
    return output


def daily_mimo_matrices(
    action: np.ndarray,
    outcome: np.ndarray,
    days: np.ndarray,
    *,
    minimum_rows: int,
    ridge_alpha: float,
    epsilon: float,
    day_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned UTC-day 2x2 coefficients and row counts."""

    action = np.asarray(action, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if action.shape[1:] != (2,) or outcome.shape[1:] != (2,) or len(action) != len(outcome):
        raise Phase35ProtocolError("MS3-R Gate B MIMO arrays must be n×2")
    ordered = np.unique(days) if day_order is None else np.asarray(day_order)
    matrices = np.full((len(ordered), 2, 2), np.nan, dtype=float)
    counts = np.zeros(len(ordered), dtype=np.int64)
    for index, day in enumerate(ordered):
        selected = (days == day) & np.isfinite(action).all(axis=1) & np.isfinite(outcome).all(axis=1)
        counts[index] = int(selected.sum())
        if counts[index] < minimum_rows:
            continue
        u, y = action[selected], outcome[selected]
        gram = np.einsum("ni,nj->ij", u, u, optimize=False) + float(ridge_alpha) * np.eye(2)
        cross = np.einsum("ni,nj->ij", u, y, optimize=False)
        matrices[index] = _solve_two_by_two(gram, cross, epsilon)
    return ordered, matrices, counts


def _bootstrap_median_interval(
    values: np.ndarray, *, samples: int, seed: int, confidence: float
) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for start in range(0, samples, 512):
        stop = min(samples, start + 512)
        chosen = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = np.median(values[chosen], axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return [float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))]


def _nanmean_stack(values: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values, axis=0)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=0)
    total = np.where(finite, stacked, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _contrast_summary(
    values: np.ndarray,
    config: Mapping[str, Any],
    seed_offset: int,
    *,
    gate_eligible: bool = True,
) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    statistics = config["statistics"]
    interval = _bootstrap_median_interval(
        finite,
        samples=int(statistics["bootstrap_samples"]),
        seed=int(statistics["bootstrap_seed"]) + seed_offset,
        confidence=float(statistics["simultaneous_interval_confidence"]),
    )
    return {
        "utc_day_count": int(len(finite)),
        "day_median": float(np.median(finite)) if len(finite) else None,
        "simultaneous_interval": interval,
        "interval_confidence": float(statistics["simultaneous_interval_confidence"]),
        "supervisor_gate_component": (
            None
            if interval is None or not gate_eligible
            else bool(interval[0] > 0.0 and len(finite) >= int(statistics["minimum_utc_days"]))
        ),
    }


def paired_path_contrasts(
    future: Mapping[int, np.ndarray],
    lead: Mapping[int, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    primary = [int(v) for v in config["point_contract"]["primary_horizons_steps"]]
    future_short = _nanmean_stack([future[h] for h in primary])
    lead_short = _nanmean_stack([lead[h] for h in primary])
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "primary_horizons_steps": primary,
        "independent_unit": "UTC_day",
        "specificity_family": {},
        "timing_family": {},
        "automatic_scientific_pass": None,
        "interpretation_boundary": "Direct paired UTC-day contrasts; Supervisor, not code, decides closure.",
    }
    for index, side in enumerate(SIDES):
        other = 1 - index
        specificity = future_short[:, index, index] - np.abs(future_short[:, index, other])
        timing = future_short[:, index, index] - np.abs(lead_short[:, index, index])
        arrays[f"specificity_{side}"] = specificity
        arrays[f"timing_{side}"] = timing
        summary["specificity_family"][side] = _contrast_summary(specificity, config, 10 + index)
        summary["timing_family"][side] = _contrast_summary(timing, config, 20 + index)
    return summary, arrays


def _common_differential(matrix: np.ndarray) -> np.ndarray:
    transform = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    return np.einsum("ij,njk,kl->nil", transform, matrix, 0.5 * transform, optimize=False)


def _matrix_summary(matrix: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(matrix).all(axis=(1, 2))
    if not valid.any():
        return {
            "utc_day_count": 0,
            "day_median_matrix": None,
            "common_differential_day_median_matrix": None,
        }
    kept = matrix[valid]
    return {
        "utc_day_count": int(valid.sum()),
        "day_median_matrix": np.median(kept, axis=0).astype(float).tolist(),
        "common_differential_day_median_matrix": np.median(_common_differential(kept), axis=0).astype(float).tolist(),
    }


def _stratum_summary(values: np.ndarray, matrices: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values) & np.isfinite(matrices).all(axis=(1, 2))
    if int(finite.sum()) < 3:
        return {"available": False, "reason": "fewer_than_three_utc_days"}
    cuts = np.quantile(values[finite], [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(values <= cuts[0], 0, np.where(values <= cuts[1], 1, 2))
    groups: dict[str, Any] = {}
    for index, name in enumerate(("low", "middle", "high")):
        selected = finite & (labels == index)
        groups[name] = {
            "utc_day_count": int(selected.sum()),
            "day_median_matrix": np.nanmedian(matrices[selected], axis=0).astype(float).tolist() if selected.any() else None,
        }
    return {"available": True, "tertile_cuts": cuts.astype(float).tolist(), "groups": groups}


def _safe_correlation(x: np.ndarray, y: np.ndarray, epsilon: float) -> float | None:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or x.std() <= epsilon or y.std() <= epsilon:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))


def _iv_summary(
    z: np.ndarray,
    action: np.ndarray,
    future: Mapping[int, np.ndarray],
    lead: Mapping[int, np.ndarray],
    days: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    epsilon = float(config["analysis"]["variance_epsilon"])
    minimum = float(config["analysis"]["iv_minimum_abs_day_denominator"])
    output: dict[str, Any] = {"status": "feasibility_diagnostic_only", "sides": {}, "automatic_iv_validity": None}
    arrays: dict[str, np.ndarray] = {"iv_z": z, "iv_action": action}
    primary = [int(v) for v in config["point_contract"]["primary_horizons_steps"]]
    y_future = np.mean(np.stack([future[h] for h in primary]), axis=0)
    y_lead = np.mean(np.stack([lead[h] for h in primary]), axis=0)
    evaluated = np.isfinite(z).all(axis=1) & np.isfinite(action).all(axis=1)
    day_order = np.unique(days[evaluated])
    arrays["iv_day_order"] = day_order.astype("datetime64[D]").astype(np.int64)
    for index, side in enumerate(SIDES):
        zi, xi = z[:, index], action[:, index]
        correlation = _safe_correlation(zi, xi, epsilon)
        day_first = np.full(len(day_order), np.nan)
        day_correct = np.full(len(day_order), np.nan)
        day_wrong = np.full(len(day_order), np.nan)
        day_lead = np.full(len(day_order), np.nan)
        for day_index, day in enumerate(day_order):
            selected = (days == day) & np.isfinite(zi) & np.isfinite(xi)
            if int(selected.sum()) < int(config["analysis"]["minimum_day_rows"]):
                continue
            denominator_z = float(np.einsum("i,i->", zi[selected], zi[selected], optimize=False))
            denominator_iv = float(np.einsum("i,i->", zi[selected], xi[selected], optimize=False))
            if denominator_z > epsilon:
                day_first[day_index] = float(denominator_iv / denominator_z)
            if abs(denominator_iv) <= minimum:
                continue
            day_correct[day_index] = float(np.einsum("i,i->", zi[selected], y_future[selected, index], optimize=False) / denominator_iv)
            day_wrong[day_index] = float(np.einsum("i,i->", zi[selected], y_future[selected, 1 - index], optimize=False) / denominator_iv)
            day_lead[day_index] = float(np.einsum("i,i->", zi[selected], y_lead[selected, index], optimize=False) / denominator_iv)
        arrays[f"iv_first_stage_{side}"] = day_first
        arrays[f"iv_correct_{side}"] = day_correct
        arrays[f"iv_wrong_{side}"] = day_wrong
        arrays[f"iv_lead_{side}"] = day_lead
        output["sides"][side] = {
            "sample_count": int((np.isfinite(zi) & np.isfinite(xi)).sum()),
            "first_stage_correlation": correlation,
            "first_stage_partial_r2": None if correlation is None else float(correlation * correlation),
            "other_valve_coaction_correlation": _safe_correlation(zi, action[:, 1 - index], epsilon),
            "daily_first_stage": _contrast_summary(day_first, config, 100 + index, gate_eligible=False),
            "daily_2sls_correct": _contrast_summary(day_correct, config, 110 + index, gate_eligible=False),
            "daily_2sls_wrong_side": _contrast_summary(day_wrong, config, 120 + index, gate_eligible=False),
            "daily_2sls_lead": _contrast_summary(day_lead, config, 130 + index, gate_eligible=False),
            "boundary": "First-stage and Wald-ratio diagnostics do not verify exclusion or SP exogeneity.",
        }
    return output, arrays


def run_gateb_analysis(
    caches: Mapping[str, Phase35Cache], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_ms3r_gateb_config(config)
    adapted = _analysis_config(config)
    validate_aligned_caches(caches, adapted)
    anchors = analysis_anchors(caches, adapted)
    times = caches["A"].timestamps_ns[anchors]
    days = _utc_days(times)
    splits = rolling_origin_splits(
        times,
        folds=int(config["analysis"]["rolling_folds"]),
        minimum_training_days=int(config["analysis"]["minimum_training_days"]),
        minimum_evaluation_days=int(config["analysis"]["minimum_evaluation_days"]),
        minimum_rows=int(config["analysis"]["minimum_rows_per_fold"]),
    )
    labels: list[tuple[str, str, int, str]] = []
    targets: list[np.ndarray] = []
    for side in SIDES:
        labels.append(("action", side, 0, "now"))
        targets.append(_valve_delta(caches[side], anchors))
    point_kinds = ("local_drop", "tin", "target")
    for outcome_side in SIDES:
        for kind in point_kinds:
            series = _point_series(caches[outcome_side], kind)
            for horizon in _all_horizons(config):
                labels.append((kind, outcome_side, horizon, "future"))
                targets.append(series[anchors + horizon] - series[anchors - 1])
                labels.append((kind, outcome_side, horizon, "lead"))
                targets.append(series[anchors - 1] - series[anchors - horizon - 1])
    main_residual, fold_id = crossfit_residual_matrix(
        build_predictors(caches, anchors, adapted),
        np.stack(targets, axis=1),
        splits,
        alpha=float(config["analysis"]["ridge_alpha"]),
        epsilon=float(config["analysis"]["variance_epsilon"]),
    )
    lookup = {label: i for i, label in enumerate(labels)}
    u = np.stack([main_residual[:, lookup[("action", side, 0, "now")]] for side in SIDES], axis=1)
    point_residuals = {
        kind: {
            timing: {
                h: np.stack([
                    main_residual[:, lookup[(kind, side, h, timing)]] for side in SIDES
                ], axis=1)
                for h in _all_horizons(config)
            }
            for timing in ("future", "lead")
        }
        for kind in point_kinds
    }
    outcome_future = point_residuals["local_drop"]["future"]
    outcome_lead = point_residuals["local_drop"]["lead"]
    evaluated_rows = np.isfinite(u).all(axis=1)
    day_order = np.unique(days[evaluated_rows])
    future_matrices: dict[int, np.ndarray] = {}
    lead_matrices: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for horizon in _all_horizons(config):
        _, future_matrices[horizon], counts[horizon] = daily_mimo_matrices(
            u, outcome_future[horizon], days,
            minimum_rows=int(config["analysis"]["minimum_day_rows"]),
            ridge_alpha=float(config["analysis"]["daily_mimo_ridge_alpha"]),
            epsilon=float(config["analysis"]["variance_epsilon"]),
            day_order=day_order,
        )
        _, lead_matrices[horizon], _ = daily_mimo_matrices(
            u, outcome_lead[horizon], days,
            minimum_rows=int(config["analysis"]["minimum_day_rows"]),
            ridge_alpha=float(config["analysis"]["daily_mimo_ridge_alpha"]),
            epsilon=float(config["analysis"]["variance_epsilon"]),
            day_order=day_order,
        )
    paired, contrast_arrays = paired_path_contrasts(future_matrices, lead_matrices, config)

    # The IV residualization deliberately excludes the current SP delta.
    iv_labels: list[tuple[str, str, int, str]] = []
    iv_targets: list[np.ndarray] = []
    for side in SIDES:
        iv_labels.append(("sp", side, 0, "now"))
        iv_targets.append(_delta(caches[side], SP_COLUMN, anchors))
        iv_labels.append(("action", side, 0, "now"))
        iv_targets.append(_delta(caches[side], VALVE_COLUMN, anchors))
    for outcome_side in SIDES:
        local = _point_series(caches[outcome_side], "local_drop")
        for horizon in config["point_contract"]["primary_horizons_steps"]:
            h = int(horizon)
            iv_labels.append(("local", outcome_side, h, "future"))
            iv_targets.append(local[anchors + h] - local[anchors - 1])
            iv_labels.append(("local", outcome_side, h, "lead"))
            iv_targets.append(local[anchors - 1] - local[anchors - h - 1])
    iv_residual, _ = crossfit_residual_matrix(
        _history_predictors(caches, anchors, config),
        np.stack(iv_targets, axis=1),
        splits,
        alpha=float(config["analysis"]["ridge_alpha"]),
        epsilon=float(config["analysis"]["variance_epsilon"]),
    )
    iv_lookup = {label: i for i, label in enumerate(iv_labels)}
    z = np.stack([iv_residual[:, iv_lookup[("sp", side, 0, "now")]] for side in SIDES], axis=1)
    iv_action = np.stack([iv_residual[:, iv_lookup[("action", side, 0, "now")]] for side in SIDES], axis=1)
    iv_future = {
        int(h): np.stack([iv_residual[:, iv_lookup[("local", side, int(h), "future")]] for side in SIDES], axis=1)
        for h in config["point_contract"]["primary_horizons_steps"]
    }
    iv_lead = {
        int(h): np.stack([iv_residual[:, iv_lookup[("local", side, int(h), "lead")]] for side in SIDES], axis=1)
        for h in config["point_contract"]["primary_horizons_steps"]
    }
    iv_summary, iv_arrays = _iv_summary(z, iv_action, iv_future, iv_lead, days, config)

    day_load = np.full(len(day_order), np.nan)
    day_valve = np.full(len(day_order), np.nan)
    day_coal_load = np.full(len(day_order), np.nan)
    load = np.mean(np.stack([
        caches[side].values[:, caches[side].index("机组负荷")].astype(float)[anchors - 1]
        for side in SIDES
    ]), axis=0)
    coal = np.mean(np.stack([
        caches[side].values[:, caches[side].index("未校正总煤量")].astype(float)[anchors - 1]
        for side in SIDES
    ]), axis=0)
    valve_baseline = np.mean(np.stack([
        caches[side].values[:, caches[side].index(VALVE_COLUMN)].astype(float)[anchors - 1] for side in SIDES
    ]), axis=0)
    for index, day in enumerate(day_order):
        selected = days == day
        day_load[index] = float(np.median(load[selected]))
        day_valve[index] = float(np.median(valve_baseline[selected]))
        day_coal_load[index] = float(np.median(coal[selected] / np.maximum(load[selected], 1.0)))
    short_matrix = _nanmean_stack([
        future_matrices[int(h)] for h in config["point_contract"]["primary_horizons_steps"]
    ])
    direction_summary: dict[str, Any] = {}
    direction_arrays: dict[str, np.ndarray] = {}
    short_outcome = _nanmean_stack([
        outcome_future[int(h)] for h in config["point_contract"]["primary_horizons_steps"]
    ])
    for side_index, side in enumerate(SIDES):
        direction_summary[side] = {}
        for label, selected in (("opening", u[:, side_index] > 0.0), ("closing", u[:, side_index] < 0.0)):
            masked_action = np.where(selected[:, None], u, np.nan)
            masked_outcome = np.where(selected[:, None], short_outcome, np.nan)
            _, matrix, direction_count = daily_mimo_matrices(
                masked_action, masked_outcome, days,
                minimum_rows=int(config["analysis"]["minimum_direction_day_rows"]),
                ridge_alpha=float(config["analysis"]["daily_mimo_ridge_alpha"]),
                epsilon=float(config["analysis"]["variance_epsilon"]),
                day_order=day_order,
            )
            direction_arrays[f"direction_{side}_{label}_matrix"] = matrix
            direction_arrays[f"direction_{side}_{label}_count"] = direction_count
            direction_summary[side][label] = _matrix_summary(matrix)
    invariance = {
        "status": "diagnostic_only",
        "load_tertiles": _stratum_summary(day_load, short_matrix),
        "baseline_valve_tertiles": _stratum_summary(day_valve, short_matrix),
        "coal_per_load_tertiles": _stratum_summary(day_coal_load, short_matrix),
        "rolling_fold_day_medians": {
            str(i): _matrix_summary(short_matrix[np.asarray([np.nanmedian(fold_id[days == d]) == i for d in day_order])])
            for i in range(int(config["analysis"]["rolling_folds"]))
        },
        "action_innovation_direction": direction_summary,
        "boundary": "Strata diagnose heterogeneity; they do not select a favorable support region.",
    }

    arrays: dict[str, np.ndarray] = {
        "anchors": anchors,
        "timestamps_ns": times,
        "utc_day_order": day_order.astype("datetime64[D]").astype(np.int64),
        "fold_id": fold_id,
        "innovation_A": u[:, 0],
        "innovation_B": u[:, 1],
        "day_load": day_load,
        "day_baseline_valve": day_valve,
        "day_coal_per_load": day_coal_load,
        **contrast_arrays,
        **iv_arrays,
        **direction_arrays,
    }
    mimo_summary: dict[str, Any] = {
        "orientation": "positive_means_opening_increases_Tin_minus_Tout",
        "horizons": {},
        "diagnostic_point_maps": {
            "upstream_tin_placebo": {"orientation": "raw", "horizons": {}},
            "terminal_temperature": {"orientation": "sign_flipped_so_cooling_is_positive", "horizons": {}},
        },
    }
    for horizon in _all_horizons(config):
        arrays[f"outcome_future_H{horizon}"] = outcome_future[horizon]
        arrays[f"outcome_lead_H{horizon}"] = outcome_lead[horizon]
        arrays[f"mimo_future_H{horizon}"] = future_matrices[horizon]
        arrays[f"mimo_lead_H{horizon}"] = lead_matrices[horizon]
        arrays[f"mimo_day_counts_H{horizon}"] = counts[horizon]
        mimo_summary["horizons"][f"H{horizon}"] = {
            "horizon_seconds": int(horizon * config["data_contract"]["step_seconds"]),
            "future": _matrix_summary(future_matrices[horizon]),
            "lead": _matrix_summary(lead_matrices[horizon]),
            "primary": horizon in config["point_contract"]["primary_horizons_steps"],
        }
        for kind, summary_name, orientation in (
            ("tin", "upstream_tin_placebo", 1.0),
            ("target", "terminal_temperature", -1.0),
        ):
            _, diagnostic_future, diagnostic_counts = daily_mimo_matrices(
                u, point_residuals[kind]["future"][horizon], days,
                minimum_rows=int(config["analysis"]["minimum_day_rows"]),
                ridge_alpha=float(config["analysis"]["daily_mimo_ridge_alpha"]),
                epsilon=float(config["analysis"]["variance_epsilon"]),
                day_order=day_order,
            )
            _, diagnostic_lead, _ = daily_mimo_matrices(
                u, point_residuals[kind]["lead"][horizon], days,
                minimum_rows=int(config["analysis"]["minimum_day_rows"]),
                ridge_alpha=float(config["analysis"]["daily_mimo_ridge_alpha"]),
                epsilon=float(config["analysis"]["variance_epsilon"]),
                day_order=day_order,
            )
            oriented_future = diagnostic_future * orientation
            oriented_lead = diagnostic_lead * orientation
            arrays[f"{kind}_outcome_future_H{horizon}"] = point_residuals[kind]["future"][horizon]
            arrays[f"{kind}_outcome_lead_H{horizon}"] = point_residuals[kind]["lead"][horizon]
            arrays[f"{kind}_mimo_future_H{horizon}"] = diagnostic_future
            arrays[f"{kind}_mimo_lead_H{horizon}"] = diagnostic_lead
            arrays[f"{kind}_mimo_day_counts_H{horizon}"] = diagnostic_counts
            mimo_summary["diagnostic_point_maps"][summary_name]["horizons"][f"H{horizon}"] = {
                "horizon_seconds": int(horizon * config["data_contract"]["step_seconds"]),
                "future": _matrix_summary(oriented_future),
                "lead": _matrix_summary(oriented_lead),
                "automatic_gate_component": None,
            }
    summary = {
        "protocol_version": config["protocol_version"],
        "evidence_scope": config["evidence_scope"],
        "analysis_support": {
            "candidate_anchor_count": int(len(anchors)),
            "crossfit_evaluated_count": int(np.isfinite(u).all(axis=1).sum()),
            "rolling_fold_count": len(splits),
            "independent_utc_day_count": int(len(day_order)),
        },
        "paired_contrasts": paired,
        "mimo_response": mimo_summary,
        "invariance": invariance,
        "iv_feasibility": iv_summary,
        "automatic_scientific_pass": None,
        "test_accessed": False,
        "training_executed": False,
        "claim_boundary": "Validation-only closed-loop conditional response diagnostics; no open-loop plant, valid-IV, do(valve), independent-test, model-selection, Gate-C, or MS4 claim.",
    }
    return summary, arrays
