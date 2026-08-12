"""Real-data OOF trajectory calibration for the three RM3 response families."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from itertools import combinations
from typing import Any, Mapping

import numpy as np

from ..data import Phase35Cache, deterministic_anchor_subset
from ..schema import Phase35ProtocolError
from .gatec_data import extract_gatec_batch, paired_valid_anchors
from .rm3_contracts import RM3CalibrationSpec
from .rm3_orthogonal import orthogonal_trajectory_moments


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluate_x: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
) -> np.ndarray:
    center = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), epsilon)
    x_train = (train_x - center) / scale
    x_evaluate = (evaluate_x - center) / scale
    design = np.concatenate((np.ones((len(x_train), 1)), x_train), axis=1)
    evaluate_design = np.concatenate((np.ones((len(x_evaluate), 1)), x_evaluate), axis=1)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return evaluate_design @ coefficients


def _nuisance_features(batch: Any) -> np.ndarray:
    """Compact trainable boundary context; excludes future valve and temperatures."""
    history = np.asarray(batch.history, dtype=np.float64)
    last = history[:, -1]
    mean = history.mean(axis=1)
    slope = history[:, -1] - history[:, 0]
    future_sp = np.asarray(batch.future_sp, dtype=np.float64).reshape(len(history), -1)
    return np.concatenate((last, mean, slope, future_sp), axis=1)


def _trajectory_targets(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    valve_base = np.stack(
        [
            batch.history[:, -1, batch.history_feature_names.index(f"{side}::二级减温调节门阀位")]
            for side in ("A", "B")
        ], axis=1,
    )
    tin_base = np.stack(
        [
            batch.history[:, -1, batch.history_feature_names.index(f"{side}::二级减温器入口温度")]
            for side in ("A", "B")
        ], axis=1,
    )
    tout_base = np.stack(
        [
            batch.history[:, -1, batch.history_feature_names.index(f"{side}::二级减温器出口温度")]
            for side in ("A", "B")
        ], axis=1,
    )
    action = np.asarray(batch.logged_future_valve, dtype=np.float64) - valve_base[:, None]
    outcome = np.asarray(batch.local_drop_target, dtype=np.float64) - (tin_base - tout_base)[:, None]
    return action, outcome


def _nonnegative_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact active-set enumeration for the frozen three-column A1 basis."""
    if design.ndim != 2 or design.shape[1] != 3 or target.shape != (len(design),):
        raise Phase35ProtocolError("RM3 A1 NNLS requires an h×3 basis and h-vector target")
    best = np.zeros(3, dtype=float)
    best_error = float(np.dot(target, target))
    for size in range(1, 4):
        for active in combinations(range(3), size):
            coefficients = np.linalg.lstsq(design[:, active], target, rcond=None)[0]
            if np.any(coefficients < -1e-12):
                continue
            candidate = np.zeros(3, dtype=float)
            candidate[list(active)] = np.maximum(coefficients, 0.0)
            error = float(np.sum((design @ candidate - target) ** 2))
            if error < best_error:
                best, best_error = candidate, error
    return best


def a1_nonnegative_projection(matrices: np.ndarray, *, step_seconds: float) -> dict[str, Any]:
    horizon = len(matrices)
    time = (np.arange(horizon) + 1) * step_seconds
    taus = np.asarray((60.0, 180.0, 600.0))
    basis = 1.0 - np.exp(-time[:, None] / taus[None])
    coefficients = np.empty((3, 2, 2), dtype=float)
    fitted = np.empty_like(matrices)
    for i in range(2):
        for j in range(2):
            coefficients[:, i, j] = _nonnegative_least_squares(
                basis, matrices[:, i, j]
            )
            fitted[:, i, j] = basis @ coefficients[:, i, j]
    return {
        "fixed_tau_seconds": taus.tolist(),
        "nonnegative_coefficients": coefficients.tolist(),
        "trajectory_matrix": fitted.tolist(),
        "projection_rmse": float(np.sqrt(np.mean((fitted - matrices) ** 2))),
        "context_scheduling_identified": False,
        "solver": "exact_active_set_nnls_three_basis_columns",
        "note": "aggregate three-pole projection; scheduling requires downstream joint R-loss",
    }


def run_rm3_calibration(
    caches: Mapping[str, Phase35Cache],
    matrix: Mapping[str, Any],
    spec: RM3CalibrationSpec,
    *,
    output_dir: Path,
    provenance: Mapping[str, Any],
    train_anchor_limit: int = 16384,
    evaluation_anchor_limit: int = 4096,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"RM3 refuses existing calibration directory: {output_dir}")
    data, statistics = matrix["data_contract"], matrix["statistics"]
    n_rows = len(caches["A"].timestamps_ns)
    test_start = caches["A"].split_bounds()["test"][0]
    train_bounds = (int(n_rows * spec.train_fraction[0]), int(n_rows * spec.train_fraction[1]))
    evaluate_bounds = (
        int(n_rows * spec.validation_fraction[0]),
        min(int(n_rows * spec.validation_fraction[1]), test_start),
    )
    window, horizon = int(data["window_steps"]), int(spec.response_horizon_steps)
    train_pool = paired_valid_anchors(
        caches, "train", window=window, horizon=horizon, max_age_s=float(data["max_age_s"]),
        bounds_override=train_bounds,
    )
    evaluate_pool = paired_valid_anchors(
        caches, "validation", window=window, horizon=horizon,
        max_age_s=float(data["max_age_s"]), bounds_override=evaluate_bounds,
    )
    seed = 35600 + int(spec.fold_id[1:]) * 100 + spec.seed * 10 + horizon
    train_anchors = deterministic_anchor_subset(train_pool, train_anchor_limit, seed)
    evaluate_anchors = deterministic_anchor_subset(evaluate_pool, evaluation_anchor_limit, seed + 1)
    train_batch = extract_gatec_batch(caches, train_anchors, window=window, horizon=horizon, validate_pair=False)
    evaluate_batch = extract_gatec_batch(caches, evaluate_anchors, window=window, horizon=horizon, validate_pair=False)
    train_x, evaluate_x = _nuisance_features(train_batch), _nuisance_features(evaluate_batch)
    train_action, train_outcome = _trajectory_targets(train_batch)
    evaluate_action, evaluate_outcome = _trajectory_targets(evaluate_batch)
    alpha, epsilon = float(statistics["ridge_alpha"]), float(statistics["epsilon"])
    action_prediction = _ridge_fit_predict(
        train_x, train_action.reshape(len(train_x), -1), evaluate_x, alpha=alpha, epsilon=epsilon
    ).reshape(evaluate_action.shape)
    outcome_prediction = _ridge_fit_predict(
        train_x, train_outcome.reshape(len(train_x), -1), evaluate_x, alpha=alpha, epsilon=epsilon
    ).reshape(evaluate_outcome.shape)
    action_residual = evaluate_action - action_prediction
    outcome_residual = evaluate_outcome - outcome_prediction
    full = orthogonal_trajectory_moments(
        action_residual, outcome_residual, ridge_alpha=alpha, epsilon=epsilon,
        maximum_condition_number=float(statistics["maximum_input_condition_number"]),
        minimum_differential_to_common_energy=float(statistics["minimum_differential_to_common_energy"]),
    )
    common = orthogonal_trajectory_moments(
        action_residual, outcome_residual, ridge_alpha=alpha, epsilon=epsilon,
        maximum_condition_number=float(statistics["maximum_input_condition_number"]),
        minimum_differential_to_common_energy=float(statistics["minimum_differential_to_common_energy"]),
        common_only=True,
    )
    full_matrices = np.stack([item.matrix for item in full])
    common_matrices = np.stack([item.matrix for item in common])
    results = {
        "R0_linear_mimo": {
            "trajectory_matrix": full_matrices.tolist(),
            "independent_channels_supported_all_steps": bool(all(item.independent_channels_supported for item in full)),
        },
        "R1_a1_scheduled": a1_nonnegative_projection(
            full_matrices, step_seconds=float(data["step_seconds"])
        ),
        "R2_a1_common_only": {
            "trajectory_matrix": common_matrices.tolist(),
            "independent_channels_supported": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "orthogonal_residuals_validation.npz"
    _atomic_npz(arrays_path, {
        "train_anchors": train_anchors, "evaluation_anchors": evaluate_anchors,
        "timestamps_ns": caches["A"].timestamps_ns[evaluate_anchors],
        "action_residual": action_residual.astype(np.float32),
        "outcome_residual": outcome_residual.astype(np.float32),
        "r0_trajectory_matrix": full_matrices.astype(np.float64),
        "r2_common_trajectory_matrix": common_matrices.astype(np.float64),
    })
    payload = {
        "protocol_version": matrix["protocol_version"], "calibration_id": spec.calibration_id,
        "spec": spec.__dict__, "provenance": dict(provenance), "results": results,
        "full_prefix_trajectory": True, "test_accessed": False,
        "automatic_scientific_pass": None,
    }
    metrics_path = output_dir / "calibration_validation.json"
    _atomic_json(metrics_path, payload)
    _atomic_json(output_dir / "artifact_ledger.json", {
        arrays_path.name: _sha256(arrays_path), metrics_path.name: _sha256(metrics_path)
    })
    return {"calibration_id": spec.calibration_id, "status": "complete", "results": results}
