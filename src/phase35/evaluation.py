"""Preregistered forecast and event-response metrics for Phase 3.5."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def forecast_metrics(target: np.ndarray, prediction: np.ndarray, horizons: Iterable[int] = (1, 6, 18, 30, 60)) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must be equal [N,H] matrices")
    err = np.abs(target - prediction)
    out = {
        "n_windows": int(len(target)),
        "integrated_mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
    }
    for horizon in horizons:
        if 1 <= horizon <= target.shape[1]:
            out[f"mae_h{horizon}"] = float(err[:, horizon - 1].mean())
    return out


def response_direction_rate(dose: np.ndarray, curves: np.ndarray, horizon: int | None = None) -> float:
    dose = np.asarray(dose, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or len(dose) != len(curves) or len(dose) == 0:
        return float("nan")
    k = curves.shape[1] - 1 if horizon is None else horizon - 1
    return float(np.mean(np.sign(dose) * curves[:, k] < 0))


def onset_lag_seconds(curve: np.ndarray, step_seconds: int = 10, threshold_c: float = 0.1) -> float:
    curve = np.asarray(curve, dtype=np.float64)
    finite = np.isfinite(curve)
    if not finite.any():
        return float("nan")
    final_sign = np.sign(curve[np.flatnonzero(finite)[-1]])
    if final_sign == 0:
        return float("nan")
    eligible = finite & (np.sign(curve) == final_sign) & (np.abs(curve) >= threshold_c)
    idx = np.flatnonzero(eligible)
    return float((idx[0] + 1) * step_seconds) if len(idx) else float("nan")


def irf_wmae(empirical: np.ndarray, model: np.ndarray) -> float:
    empirical = np.asarray(empirical, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    if empirical.shape != model.shape or empirical.ndim != 1:
        raise ValueError("IRF curves must be equal one-dimensional arrays")
    # Later horizons receive modestly larger weight because the process has a known delay.
    weights = np.sqrt(np.arange(1, len(empirical) + 1, dtype=np.float64))
    weights /= weights.sum()
    return float(np.sum(weights * np.abs(empirical - model)))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including exact tie handling."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def dose_monotonicity(dose: np.ndarray, curves: np.ndarray, horizon: int | None = None) -> float:
    dose = np.asarray(dose, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    if len(dose) < 3 or curves.ndim != 2 or len(curves) != len(dose):
        return float("nan")
    k = curves.shape[1] - 1 if horizon is None else horizon - 1
    x, y = _ranks(np.abs(dose)), _ranks(np.abs(curves[:, k]))
    x, y = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def bootstrap_mean_curve(
    curves: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    cluster_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or len(curves) == 0:
        raise ValueError("curves must be a non-empty [events,H] matrix")
    if cluster_ids is None:
        units = curves
    else:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.ndim != 1 or len(cluster_ids) != len(curves):
            raise ValueError("cluster_ids must contain one id per curve")
        # Equal-weight calendar/episode blocks are the top-level sampling unit;
        # repeated events within one block are not treated as independent plants.
        units = np.stack([curves[cluster_ids == key].mean(axis=0) for key in np.unique(cluster_ids)])
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, curves.shape[1]), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(units), size=len(units))
        boot[i] = units[idx].mean(axis=0)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": units.mean(axis=0),
        "low": np.quantile(boot, alpha, axis=0),
        "high": np.quantile(boot, 1.0 - alpha, axis=0),
        "n_clusters": int(len(units)),
    }


def event_response_metrics(
    empirical_curves: np.ndarray,
    model_curves: np.ndarray,
    dose: np.ndarray,
    step_seconds: int = 10,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
    cluster_ids: np.ndarray | None = None,
) -> dict:
    empirical_curves = np.asarray(empirical_curves, dtype=np.float64)
    model_curves = np.asarray(model_curves, dtype=np.float64)
    dose = np.asarray(dose, dtype=np.float64)
    if empirical_curves.shape != model_curves.shape or empirical_curves.ndim != 2:
        raise ValueError("empirical/model event curves must be equal [events,H] matrices")
    if len(dose) != len(empirical_curves):
        raise ValueError("one dose is required per event")
    emp_boot = bootstrap_mean_curve(
        empirical_curves, bootstrap_replicates, seed=seed, cluster_ids=cluster_ids
    )
    model_boot = bootstrap_mean_curve(
        model_curves, bootstrap_replicates, seed=seed + 1, cluster_ids=cluster_ids
    )
    emp_mean, model_mean = emp_boot["mean"], model_boot["mean"]
    emp_lag = onset_lag_seconds(emp_mean, step_seconds)
    model_lag = onset_lag_seconds(model_mean, step_seconds)
    result = {
        "n_events": int(len(dose)),
        "n_clusters": emp_boot["n_clusters"],
        "empirical_direction_rate": response_direction_rate(dose, empirical_curves),
        "model_direction_rate": response_direction_rate(dose, model_curves),
        "empirical_onset_lag_s": emp_lag,
        "model_onset_lag_s": model_lag,
        "onset_lag_abs_error_s": float(abs(model_lag - emp_lag)) if np.isfinite(emp_lag + model_lag) else float("nan"),
        "irf_wmae": irf_wmae(emp_mean, model_mean),
        "empirical_dose_monotonicity": dose_monotonicity(dose, empirical_curves),
        "model_dose_monotonicity": dose_monotonicity(dose, model_curves),
        "empirical_mean_curve": emp_mean.tolist(),
        "empirical_ci_low": emp_boot["low"].tolist(),
        "empirical_ci_high": emp_boot["high"].tolist(),
        "model_mean_curve": model_mean.tolist(),
        "model_ci_low": model_boot["low"].tolist(),
        "model_ci_high": model_boot["high"].tolist(),
    }
    for horizon in (1, 6, 12, 18, 30, 60):
        if horizon <= empirical_curves.shape[1]:
            k = horizon - 1
            result[f"empirical_effect_h{horizon}"] = float(emp_mean[k])
            result[f"empirical_ci_low_h{horizon}"] = float(emp_boot["low"][k])
            result[f"empirical_ci_high_h{horizon}"] = float(emp_boot["high"][k])
            result[f"model_effect_h{horizon}"] = float(model_mean[k])
            result[f"model_ci_low_h{horizon}"] = float(model_boot["low"][k])
            result[f"model_ci_high_h{horizon}"] = float(model_boot["high"][k])
    return result


def empirical_response_summary(
    curves: np.ndarray,
    dose: np.ndarray,
    step_seconds: int = 10,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
    cluster_ids: np.ndarray | None = None,
) -> dict:
    curves = np.asarray(curves, dtype=np.float64)
    dose = np.asarray(dose, dtype=np.float64)
    if curves.ndim != 2 or len(curves) != len(dose) or len(dose) == 0:
        raise ValueError("empirical curves must be non-empty [events,H] with one dose per event")
    boot = bootstrap_mean_curve(
        curves, bootstrap_replicates, seed=seed, cluster_ids=cluster_ids
    )
    result = {
        "n_events": int(len(dose)),
        "n_clusters": boot["n_clusters"],
        "direction_rate": response_direction_rate(dose, curves),
        "onset_lag_s": onset_lag_seconds(boot["mean"], step_seconds),
        "dose_monotonicity": dose_monotonicity(dose, curves),
        "mean_curve": boot["mean"].tolist(),
        "ci_low": boot["low"].tolist(),
        "ci_high": boot["high"].tolist(),
    }
    for horizon in (1, 6, 12, 18, 30, 60):
        if horizon <= curves.shape[1]:
            k = horizon - 1
            result[f"effect_h{horizon}"] = float(boot["mean"][k])
            result[f"ci_low_h{horizon}"] = float(boot["low"][k])
            result[f"ci_high_h{horizon}"] = float(boot["high"][k])
    return result
