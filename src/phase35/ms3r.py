"""MS3-R Gate A point, timing, excitation, and dual-input rank diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


SIDES = ("A", "B")


def validate_ms3r_gate1_config(config: Mapping[str, Any]) -> None:
    if config.get("data_contract", {}).get("split") != "validation":
        raise Phase35ProtocolError("MS3-R Gate 1 is validation-only")
    if config["data_contract"].get("test_allowed") is not False:
        raise Phase35ProtocolError("MS3-R Gate 1 must prohibit test access")
    if config["execution_contract"].get("automatic_scientific_pass") is not False:
        raise Phase35ProtocolError("MS3-R Gate 1 cannot auto-promote scientific claims")
    semantics = config["branch_semantics"]
    if semantics.get("free_future_action_access") is not False:
        raise Phase35ProtocolError("the residual branch must not read future action")
    if semantics.get("response_requires_constant_action_identity") is not True:
        raise Phase35ProtocolError("the response branch must preserve constant-action identity")
    analysis = config["analysis"]
    lags = tuple(int(v) for v in analysis["history_lags_steps"])
    horizons = tuple(int(v) for v in analysis["horizons_steps"])
    if not lags or not horizons or min(lags + horizons) < 1:
        raise Phase35ProtocolError("MS3-R lags and horizons must be positive")
    if tuple(sorted(set(lags))) != lags or tuple(sorted(set(horizons))) != horizons:
        raise Phase35ProtocolError("MS3-R lags/horizons must be unique and sorted")
    if int(analysis["rolling_folds"]) < 2:
        raise Phase35ProtocolError("MS3-R requires at least two rolling folds")
    required = {LOAD_COLUMN, PRESSURE_COLUMN, SP_COLUMN, VALVE_COLUMN, TIN2_COLUMN, TOUT2_COLUMN, TARGET_COLUMN}
    missing = required - set(config["point_contract"]["conditioning_columns"])
    if missing:
        raise Phase35ProtocolError(f"MS3-R point contract is missing columns: {sorted(missing)}")


def validate_aligned_caches(
    caches: Mapping[str, Phase35Cache], config: Mapping[str, Any]
) -> None:
    if set(caches) != set(SIDES):
        raise Phase35ProtocolError("MS3-R requires exactly A and B caches")
    reference = caches["A"]
    for side in SIDES:
        cache = caches[side]
        if cache.metadata.get("side") != side:
            raise Phase35ProtocolError(f"MS3-R cache side mismatch for {side}")
        if cache.metadata.get("source", {}).get("sha256") != config["data_contract"]["source_sha256"]:
            raise Phase35ProtocolError(f"MS3-R source SHA mismatch for side {side}")
        if float(cache.metadata.get("step_seconds", -1)) != float(config["data_contract"]["step_seconds"]):
            raise Phase35ProtocolError(f"MS3-R step size mismatch for side {side}")
        missing = set(config["point_contract"]["conditioning_columns"]) - set(cache.columns)
        if missing:
            raise Phase35ProtocolError(f"MS3-R {side} cache missing columns: {sorted(missing)}")
    if not np.array_equal(reference.timestamps_ns, caches["B"].timestamps_ns):
        raise Phase35ProtocolError("MS3-R A/B cache timestamps must align exactly")
    if reference.split_bounds()["validation"] != caches["B"].split_bounds()["validation"]:
        raise Phase35ProtocolError("MS3-R A/B validation bounds differ")


def branch_semantics_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    semantics = config["branch_semantics"]
    return {
        "residual_branch_label": semantics["free_head_label"],
        "response_branch_label": semantics["response_head_label"],
        "residual_branch_future_action_access": False,
        "response_constant_action_identity_required": True,
        "semantic_promotion_allowed": False,
        "prohibited_claims": list(semantics["prohibited_claims"]),
    }


def structural_branch_information_probe() -> dict[str, Any]:
    """Verify the current model's action information flow without training."""

    import torch

    from .multistep.real_training import RealModelConfig, build_real_model
    from .schema import MS3_HISTORY_FEATURES

    torch.manual_seed(20260811)
    model = build_real_model(
        RealModelConfig(
            window=8,
            horizon=6,
            d_model=8,
            n_heads=2,
            dropout=0.0,
            opening_map="identity",
            poles=2,
            tau_min_seconds=20.0,
            tau_max_seconds=120.0,
            context_scheduled=True,
            schedule_log_scale=0.25,
        ),
        MS3_HISTORY_FEATURES,
        "joint_total",
    ).eval()
    history = torch.randn(4, 8, len(MS3_HISTORY_FEATURES))
    baseline = torch.full((4,), 50.0)
    future = baseline[:, None] + torch.linspace(0.0, 4.0, 6)[None, :]
    changed = future.clone()
    changed[:, 3:] += 7.0
    constant = baseline[:, None].expand_as(future)
    with torch.no_grad():
        normal = model(history, future, baseline)
        altered = model(history, changed, baseline)
        identity = model(history, constant, baseline)
    return {
        "probe_kind": "untrained_structural_information_flow_not_field_evidence",
        "residual_branch_future_action_permutation_max_error": float(
            (normal["free_prediction"] - altered["free_prediction"]).abs().max()
        ),
        "response_constant_action_identity_max_error": float(identity["effect"].abs().max()),
        "future_action_prefix_leakage_max_error": float(
            (normal["prediction"][:, :3] - altered["prediction"][:, :3]).abs().max()
        ),
        "semantic_promotion_allowed": False,
    }


def point_quality(cache: Phase35Cache, columns: Sequence[str]) -> dict[str, Any]:
    quality: dict[str, Any] = {}
    for column in columns:
        values = cache.values[:, cache.index(column)].astype(float)
        finite = np.isfinite(values)
        finite_values = values[finite]
        quality[column] = {
            "finite_fraction": float(finite.mean()),
            "finite_count": int(finite.sum()),
            "min": float(finite_values.min()) if len(finite_values) else None,
            "max": float(finite_values.max()) if len(finite_values) else None,
            "mean": float(finite_values.mean()) if len(finite_values) else None,
            "std": float(finite_values.std()) if len(finite_values) else None,
            "unique_count": int(len(np.unique(finite_values))) if len(finite_values) else 0,
        }
    return quality


def _utc_days(timestamps_ns: np.ndarray) -> np.ndarray:
    return timestamps_ns.astype("datetime64[ns]").astype("datetime64[D]")


def rolling_origin_splits(
    timestamps_ns: np.ndarray,
    *,
    folds: int,
    minimum_training_days: int,
    minimum_evaluation_days: int,
    minimum_rows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    days = _utc_days(timestamps_ns)
    unique_days = np.unique(days)
    minimum = minimum_training_days + folds * minimum_evaluation_days
    if len(unique_days) < minimum:
        raise Phase35ProtocolError(
            f"MS3-R needs at least {minimum} UTC days for rolling cross-fit; got {len(unique_days)}"
        )
    initial_days = minimum_training_days
    remaining = unique_days[initial_days:]
    groups = np.array_split(remaining, folds)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for group in groups:
        if len(group) < minimum_evaluation_days:
            continue
        first = group[0]
        train = np.flatnonzero(days < first)
        evaluate = np.flatnonzero(np.isin(days, group))
        if len(train) < minimum_rows or len(evaluate) < minimum_rows:
            raise Phase35ProtocolError("MS3-R rolling fold has too few rows")
        output.append((train, evaluate))
    if len(output) != folds:
        raise Phase35ProtocolError("MS3-R could not construct every frozen rolling fold")
    return output


@dataclass(frozen=True)
class RidgeState:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    coefficient: np.ndarray


def _solve_spd_conjugate_gradient(
    matrix: np.ndarray,
    vector: np.ndarray,
    *,
    tolerance: float = 1e-10,
    maximum_iterations: int | None = None,
) -> np.ndarray:
    """Solve a small SPD system without invoking platform LAPACK drivers.

    The Windows research environment loads both NumPy MKL and PyTorch's native
    runtime in the full test process.  Some combinations abort inside
    ``numpy.linalg.solve`` instead of raising an exception.  Ridge systems are
    symmetric positive definite, so deterministic conjugate gradients avoids
    that fragile driver boundary and is equally suitable here.
    """

    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)
    solution = np.zeros_like(vector)
    residual = vector - np.einsum("ij,j->i", matrix, solution, optimize=False)
    direction = residual.copy()
    squared = float(np.einsum("i,i->", residual, residual, optimize=False))
    if squared == 0.0:
        return solution
    initial = math.sqrt(squared)
    limit = maximum_iterations or max(32, matrix.shape[0] * 10)
    for _ in range(limit):
        product = np.einsum("ij,j->i", matrix, direction, optimize=False)
        denominator = float(np.einsum("i,i->", direction, product, optimize=False))
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise Phase35ProtocolError("MS3-R ridge system is not positive definite")
        step = squared / denominator
        solution = solution + step * direction
        residual = residual - step * product
        updated = float(np.einsum("i,i->", residual, residual, optimize=False))
        if math.sqrt(updated) <= tolerance * max(initial, 1.0):
            return solution
        direction = residual + (updated / squared) * direction
        squared = updated
    raise Phase35ProtocolError("MS3-R ridge conjugate-gradient solver did not converge")


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, epsilon: float) -> RidgeState:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[finite], y[finite]
    if len(y) < x.shape[1] + 2:
        raise Phase35ProtocolError("MS3-R ridge fit has insufficient finite rows")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > epsilon, scale, 1.0)
    normalized = (x - mean) / scale
    centered_y = y - y.mean()
    import torch

    x_tensor = torch.from_numpy(normalized)
    y_tensor = torch.from_numpy(centered_y)
    gram = x_tensor.T @ x_tensor
    gram = gram + float(alpha) * torch.eye(gram.shape[0], dtype=gram.dtype)
    coefficient = torch.linalg.solve(gram, x_tensor.T @ y_tensor).cpu().numpy()
    return RidgeState(mean, scale, float(y.mean()), coefficient)


def predict_ridge(state: RidgeState, x: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(x, dtype=float) - state.x_mean) / state.x_scale
    return state.y_mean + np.einsum(
        "ni,i->n", normalized, state.coefficient, optimize=False
    )


def crossfit_residuals(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.full(len(y), np.nan, dtype=float)
    fold_id = np.full(len(y), -1, dtype=np.int16)
    for index, (train, evaluate) in enumerate(splits):
        state = fit_ridge(x[train], y[train], alpha, epsilon)
        prediction = predict_ridge(state, x[evaluate])
        finite = np.isfinite(y[evaluate]) & np.isfinite(prediction)
        chosen = evaluate[finite]
        residual[chosen] = y[chosen] - prediction[finite]
        fold_id[chosen] = index
    return residual, fold_id


def crossfit_residual_matrix(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Residualize many targets with one ridge factorization per rolling fold."""

    import torch

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or len(y) != len(x):
        raise Phase35ProtocolError("MS3-R multi-output residualization shape mismatch")
    residual = np.full_like(y, np.nan, dtype=float)
    fold_id = np.full(len(y), -1, dtype=np.int16)
    for index, (train, evaluate) in enumerate(splits):
        if not np.isfinite(x[train]).all() or not np.isfinite(y[train]).all():
            raise Phase35ProtocolError("MS3-R training fold contains non-finite values")
        if not np.isfinite(x[evaluate]).all() or not np.isfinite(y[evaluate]).all():
            raise Phase35ProtocolError("MS3-R evaluation fold contains non-finite values")
        x_mean = x[train].mean(axis=0)
        x_scale = x[train].std(axis=0)
        x_scale = np.where(x_scale > epsilon, x_scale, 1.0)
        y_mean = y[train].mean(axis=0)
        x_train = torch.from_numpy((x[train] - x_mean) / x_scale)
        y_train = torch.from_numpy(y[train] - y_mean)
        gram = x_train.T @ x_train
        gram = gram + float(alpha) * torch.eye(gram.shape[0], dtype=gram.dtype)
        coefficient = torch.linalg.solve(gram, x_train.T @ y_train)
        x_evaluate = torch.from_numpy((x[evaluate] - x_mean) / x_scale)
        prediction = (x_evaluate @ coefficient).cpu().numpy() + y_mean
        residual[evaluate] = y[evaluate] - prediction
        fold_id[evaluate] = index
    return residual, fold_id


def _consecutive_anchor_mask(
    timestamps_ns: np.ndarray,
    anchors: np.ndarray,
    *,
    max_lag: int,
    max_horizon: int,
    step_seconds: float,
) -> np.ndarray:
    expected = int(step_seconds * 1_000_000_000)
    bad = np.diff(timestamps_ns) != expected
    prefix = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    starts = anchors - max_lag
    stops = anchors + max_horizon
    return (prefix[stops] - prefix[starts]) == 0


def analysis_anchors(caches: Mapping[str, Phase35Cache], config: Mapping[str, Any]) -> np.ndarray:
    cache = caches["A"]
    lo, hi = cache.split_bounds()["validation"]
    max_lag = max(int(v) for v in config["analysis"]["history_lags_steps"])
    max_horizon = max(int(v) for v in config["analysis"]["horizons_steps"])
    anchors = np.arange(lo + max_lag, hi - max_horizon, dtype=np.int64)
    consecutive = _consecutive_anchor_mask(
        cache.timestamps_ns,
        anchors,
        max_lag=max_lag,
        max_horizon=max_horizon,
        step_seconds=float(config["data_contract"]["step_seconds"]),
    )
    anchors = anchors[consecutive]
    valid = np.ones(len(anchors), dtype=bool)
    ranges = config["analysis"]["operating_ranges"]
    required = config["point_contract"]["conditioning_columns"]
    for side in SIDES:
        side_cache = caches[side]
        for column in required:
            values = side_cache.values[:, side_cache.index(column)]
            valid &= np.isfinite(values[anchors - max_lag]) & np.isfinite(values[anchors + max_horizon])
        for column, bounds in ranges.items():
            values = side_cache.values[:, side_cache.index(column)]
            valid &= (values[anchors - 1] >= float(bounds[0])) & (values[anchors - 1] <= float(bounds[1]))
    anchors = anchors[valid]
    if len(anchors) < int(config["analysis"]["minimum_rows_per_fold"]) * 2:
        raise Phase35ProtocolError("MS3-R has insufficient valid validation anchors")
    return anchors


def build_predictors(
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
        sp = cache.values[:, cache.index(SP_COLUMN)].astype(float)
        features.append(sp[anchors] - sp[anchors - 1])
    return np.stack(features, axis=1)


def _valve_delta(cache: Phase35Cache, anchors: np.ndarray) -> np.ndarray:
    valve = cache.values[:, cache.index(VALVE_COLUMN)].astype(float)
    return valve[anchors] - valve[anchors - 1]


def _point_series(cache: Phase35Cache, kind: str) -> np.ndarray:
    if kind == "local_drop":
        tin = cache.values[:, cache.index(TIN2_COLUMN)].astype(float)
        tout = cache.values[:, cache.index(TOUT2_COLUMN)].astype(float)
        return tin - tout
    columns = {"tout": TOUT2_COLUMN, "target": TARGET_COLUMN, "tin": TIN2_COLUMN}
    return cache.values[:, cache.index(columns[kind])].astype(float)


def _bootstrap_median(values: np.ndarray, *, samples: int, seed: int) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for start in range(0, samples, 512):
        stop = min(samples, start + 512)
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        estimates[start:stop] = np.median(values[indices], axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def residual_slope_summary(
    action: np.ndarray,
    outcome: np.ndarray,
    days: np.ndarray,
    *,
    orientation: float,
    minimum_day_rows: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    epsilon: float,
) -> dict[str, Any]:
    finite = np.isfinite(action) & np.isfinite(outcome)
    u, y, kept_days = action[finite], outcome[finite], days[finite]
    denominator = float(np.einsum("i,i->", u, u, optimize=False))
    numerator = float(np.einsum("i,i->", u, y, optimize=False))
    raw = float(numerator / denominator) if denominator > epsilon else None
    day_slopes: list[float] = []
    day_labels: list[str] = []
    for day in np.unique(kept_days):
        selected = kept_days == day
        if int(selected.sum()) < minimum_day_rows:
            continue
        daily_denominator = float(
            np.einsum("i,i->", u[selected], u[selected], optimize=False)
        )
        if daily_denominator <= epsilon:
            continue
        daily_numerator = float(
            np.einsum("i,i->", u[selected], y[selected], optimize=False)
        )
        day_slopes.append(float(daily_numerator / daily_denominator))
        day_labels.append(str(day))
    oriented = None if raw is None else float(orientation * raw)
    oriented_days = np.asarray(day_slopes, dtype=float) * float(orientation)
    u_std, y_std = float(np.std(u)), float(np.std(y))
    correlation = (
        float(np.mean((u - u.mean()) * (y - y.mean())) / (u_std * y_std))
        if len(u) > 2 and u_std > epsilon and y_std > epsilon
        else None
    )
    return {
        "sample_count": int(len(u)),
        "utc_day_count": int(len(day_slopes)),
        "utc_days": day_labels,
        "raw_coefficient": raw,
        "orientation": float(orientation),
        "oriented_coefficient": oriented,
        "oriented_day_median": float(np.median(oriented_days)) if len(oriented_days) else None,
        "oriented_day_bootstrap_ci95": _bootstrap_median(
            oriented_days,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "residual_correlation": correlation,
        "incremental_r2_diagnostic": None if correlation is None else float(correlation * correlation),
    }


def path_diagnostics(
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    innovations: Mapping[str, np.ndarray],
    outcome_residuals: Mapping[tuple[str, str, int, str], np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = config["analysis"]
    statistics = config["statistics"]
    epsilon = float(analysis["variance_epsilon"])
    days = _utc_days(caches["A"].timestamps_ns[anchors])
    output: dict[str, Any] = {}
    path_index = 0
    for action_side in SIDES:
        other = "B" if action_side == "A" else "A"
        output[action_side] = {}
        definitions = (
            ("correct_local_drop", action_side, "local_drop", 1.0),
            ("correct_tout", action_side, "tout", -1.0),
            ("correct_terminal", action_side, "target", -1.0),
            ("wrong_side_local_drop", other, "local_drop", 1.0),
            ("wrong_side_tout", other, "tout", -1.0),
            ("wrong_side_terminal", other, "target", -1.0),
            ("correct_upstream_tin_placebo", action_side, "tin", 1.0),
            ("wrong_side_upstream_tin_placebo", other, "tin", 1.0),
        )
        for name, outcome_side, kind, orientation in definitions:
            series = _point_series(caches[outcome_side], kind)
            output[action_side][name] = {}
            for horizon in (int(v) for v in analysis["horizons_steps"]):
                future_residual = outcome_residuals[(outcome_side, kind, horizon, "future")]
                past_residual = outcome_residuals[(outcome_side, kind, horizon, "lead")]
                seed = int(statistics["bootstrap_seed"]) + path_index * 100 + horizon
                positive = residual_slope_summary(
                    innovations[action_side], future_residual, days,
                    orientation=orientation,
                    minimum_day_rows=int(statistics["minimum_day_slope_rows"]),
                    bootstrap_samples=int(statistics["bootstrap_samples"]),
                    bootstrap_seed=seed,
                    epsilon=epsilon,
                )
                lead = residual_slope_summary(
                    innovations[action_side], past_residual, days,
                    orientation=orientation,
                    minimum_day_rows=int(statistics["minimum_day_slope_rows"]),
                    bootstrap_samples=int(statistics["bootstrap_samples"]),
                    bootstrap_seed=seed + 1,
                    epsilon=epsilon,
                )
                shifted_action = np.roll(innovations[action_side], int(analysis["day_shift_steps"]))
                shifted = residual_slope_summary(
                    shifted_action, future_residual, days,
                    orientation=orientation,
                    minimum_day_rows=int(statistics["minimum_day_slope_rows"]),
                    bootstrap_samples=int(statistics["bootstrap_samples"]),
                    bootstrap_seed=seed + 2,
                    epsilon=epsilon,
                )
                output[action_side][name][f"H{horizon}"] = {
                    "positive_lag": positive,
                    "action_lead_placebo": lead,
                    "day_shift_placebo": shifted,
                    "horizon_seconds": int(horizon * config["data_contract"]["step_seconds"]),
                    "outcome_side": outcome_side,
                    "outcome_kind": kind,
                }
            path_index += 1
    return output


def _finite_joint_innovations(innovations: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = np.asarray(innovations["A"], dtype=float), np.asarray(innovations["B"], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask], mask


def _hankel_spectrum(
    a: np.ndarray,
    b: np.ndarray,
    *,
    window: int,
    stride: int,
    maximum_rows: int,
    epsilon: float,
) -> dict[str, Any]:
    if len(a) < window + 1:
        return {"window_steps": window, "row_count": 0, "singular_values": []}
    av = np.lib.stride_tricks.sliding_window_view(a, window)[::stride]
    bv = np.lib.stride_tricks.sliding_window_view(b, window)[::stride]
    rows = np.concatenate((av, bv), axis=1)
    rows = rows[np.isfinite(rows).all(axis=1)]
    if len(rows) > maximum_rows:
        indices = np.linspace(0, len(rows) - 1, maximum_rows, dtype=np.int64)
        rows = rows[indices]
    rows = rows - rows.mean(axis=0, keepdims=True)
    scale = np.sqrt(max(len(rows) - 1, 1))
    if len(rows):
        import torch

        singular = (
            torch.linalg.svdvals(torch.from_numpy(rows / scale))
            .detach()
            .cpu()
            .numpy()
        )
    else:
        singular = np.asarray([])
    return {
        "window_steps": int(window),
        "row_count": int(len(rows)),
        "singular_values": singular.astype(float).tolist(),
        "effective_rank_at_relative_1e_3": int(np.sum(singular > singular[0] * 1e-3)) if len(singular) and singular[0] > epsilon else 0,
    }


def rank_diagnostics(
    innovations: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> dict[str, Any]:
    epsilon = float(config["analysis"]["variance_epsilon"])
    a, b, _ = _finite_joint_innovations(innovations)
    if len(a) < 3:
        raise Phase35ProtocolError("MS3-R has insufficient paired action innovations")
    scales = np.asarray([a.std(), b.std()], dtype=float)
    if np.any(scales <= epsilon):
        standardized = np.stack((a, b), axis=1)
    else:
        standardized = np.stack(((a - a.mean()) / scales[0], (b - b.mean()) / scales[1]), axis=1)
    centered = standardized - standardized.mean(axis=0, keepdims=True)
    covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False) / max(
        len(centered) - 1, 1
    )
    trace = float(covariance[0, 0] + covariance[1, 1])
    discriminant = math.sqrt(
        float((covariance[0, 0] - covariance[1, 1]) ** 2 + 4.0 * covariance[0, 1] ** 2)
    )
    eigenvalues = np.asarray(
        [(trace + discriminant) / 2.0, (trace - discriminant) / 2.0],
        dtype=float,
    )
    condition = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > epsilon else math.inf
    common = (standardized[:, 0] + standardized[:, 1]) / 2.0
    differential = (standardized[:, 0] - standardized[:, 1]) / 2.0
    rank_config = config["rank_analysis"]
    return {
        "paired_sample_count": int(len(standardized)),
        "raw_innovation_std": {"A": float(a.std()), "B": float(b.std())},
        "standardized_covariance": covariance.astype(float).tolist(),
        "standardized_correlation": float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])),
        "eigenvalues_descending": eigenvalues.astype(float).tolist(),
        "condition_number": condition if math.isfinite(condition) else None,
        "condition_number_is_infinite": not math.isfinite(condition),
        "common_energy": float(np.mean(common * common)),
        "differential_energy": float(np.mean(differential * differential)),
        "differential_to_common_energy_ratio": float(np.mean(differential * differential) / max(np.mean(common * common), epsilon)),
        "automatic_rank_pass": None,
        "hankel": [
            _hankel_spectrum(
                standardized[:, 0], standardized[:, 1],
                window=int(window),
                stride=int(rank_config["hankel_stride_steps"]),
                maximum_rows=int(rank_config["maximum_hankel_rows"]),
                epsilon=epsilon,
            )
            for window in rank_config["hankel_windows_steps"]
        ],
        "interpretation_boundary": "Condition number and spectra are diagnostics; Gate-A supervisor audit decides dual, common-only, or closed-loop-only support.",
    }


def run_gate1_analysis(
    caches: Mapping[str, Phase35Cache], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_ms3r_gate1_config(config)
    validate_aligned_caches(caches, config)
    anchors = analysis_anchors(caches, config)
    x = build_predictors(caches, anchors, config)
    times = caches["A"].timestamps_ns[anchors]
    splits = rolling_origin_splits(
        times,
        folds=int(config["analysis"]["rolling_folds"]),
        minimum_training_days=int(config["analysis"]["minimum_training_days"]),
        minimum_evaluation_days=int(config["analysis"]["minimum_evaluation_days"]),
        minimum_rows=int(config["analysis"]["minimum_rows_per_fold"]),
    )
    target_labels: list[tuple[Any, ...]] = []
    target_values: list[np.ndarray] = []
    for side in SIDES:
        target_labels.append(("action", side))
        target_values.append(_valve_delta(caches[side], anchors))
    for outcome_side in SIDES:
        for kind in ("local_drop", "tout", "target", "tin"):
            series = _point_series(caches[outcome_side], kind)
            for horizon in (int(v) for v in config["analysis"]["horizons_steps"]):
                target_labels.append(("outcome", outcome_side, kind, horizon, "future"))
                target_values.append(series[anchors + horizon] - series[anchors - 1])
                target_labels.append(("outcome", outcome_side, kind, horizon, "lead"))
                target_values.append(series[anchors - 1] - series[anchors - horizon - 1])
    target_matrix = np.stack(target_values, axis=1)
    residual_matrix, shared_fold_id = crossfit_residual_matrix(
        x,
        target_matrix,
        splits,
        alpha=float(config["analysis"]["ridge_alpha"]),
        epsilon=float(config["analysis"]["variance_epsilon"]),
    )
    label_index = {label: index for index, label in enumerate(target_labels)}
    innovations = {
        side: residual_matrix[:, label_index[("action", side)]] for side in SIDES
    }
    fold_ids = {side: shared_fold_id.copy() for side in SIDES}
    outcome_residuals = {
        (side, kind, horizon, timing): residual_matrix[
            :, label_index[("outcome", side, kind, horizon, timing)]
        ]
        for side in SIDES
        for kind in ("local_drop", "tout", "target", "tin")
        for horizon in (int(v) for v in config["analysis"]["horizons_steps"])
        for timing in ("future", "lead")
    }
    action_information: dict[str, Any] = {}
    for side in SIDES:
        raw_delta = _valve_delta(caches[side], anchors)
        evaluated = np.isfinite(innovations[side]) & np.isfinite(raw_delta)
        raw_eval = raw_delta[evaluated]
        residual_eval = innovations[side][evaluated]
        raw_variance = float(np.var(raw_eval))
        residual_variance = float(np.var(residual_eval))
        action_information[side] = {
            "crossfit_sample_count": int(evaluated.sum()),
            "raw_delta_std_pct": float(np.std(raw_eval)),
            "innovation_std_pct": float(np.std(residual_eval)),
            "past_history_and_current_sp_crossfit_r2": (
                float(1.0 - residual_variance / raw_variance)
                if raw_variance > float(config["analysis"]["variance_epsilon"])
                else None
            ),
            "mean_abs_innovation_pct": float(np.mean(np.abs(residual_eval))),
            "interpretation_boundary": "Predictability diagnoses closed-loop endogeneity; it is not a first-stage IV claim.",
        }
    paths = path_diagnostics(
        caches, anchors, innovations, outcome_residuals, config
    )
    rank = rank_diagnostics(innovations, config)
    summary = {
        "protocol_version": config["protocol_version"],
        "evidence_scope": config["evidence_scope"],
        "branch_semantics": {
            **branch_semantics_contract(config),
            "structural_probe": structural_branch_information_probe(),
        },
        "action_information_audit": action_information,
        "point_quality": {
            side: point_quality(caches[side], config["point_contract"]["conditioning_columns"])
            for side in SIDES
        },
        "analysis_support": {
            "candidate_anchor_count": int(len(anchors)),
            "crossfit_evaluated_count": int(np.sum(np.isfinite(innovations["A"]) & np.isfinite(innovations["B"]))),
            "rolling_fold_count": len(splits),
            "validation_bounds": list(caches["A"].split_bounds()["validation"]),
            "validation_time_start": str(np.datetime64(int(caches["A"].timestamps_ns[caches["A"].split_bounds()["validation"][0]]), "ns")),
            "validation_time_end": str(np.datetime64(int(caches["A"].timestamps_ns[caches["A"].split_bounds()["validation"][1] - 1]), "ns")),
        },
        "path_diagnostics": paths,
        "rank_diagnostics": rank,
        "automatic_scientific_pass": None,
        "test_accessed": False,
        "training_executed": False,
        "claim_boundary": "Validation-only closed-loop observational diagnostics; no causal, do(valve), open-loop plant, dual-MIMO, independent-test, MS4-release, or paper claim.",
    }
    arrays = {
        "anchors": anchors,
        "timestamps_ns": times,
        "innovation_A": innovations["A"],
        "innovation_B": innovations["B"],
        "fold_id_A": fold_ids["A"],
        "fold_id_B": fold_ids["B"],
    }
    return summary, arrays
