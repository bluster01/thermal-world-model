"""Zero-training AV0 replay helpers for legacy RM3/RM3-A checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
import torch

from ..schema import Phase35ProtocolError
from .rm3_prediction import RM3PredictionConfig
from .rm3_calibration import a1_nonnegative_projection
from .rm3_orthogonal import orthogonal_trajectory_moments
from .rm3av_model import RM3AVModelConfig, build_rm3av_model
from .rm3av_diagnostics import (
    build_assumption_ledger,
    build_manual_verdict_template,
    build_state_closure_audit,
)


ANCHOR_BY_LEGACY = {
    "P3_gatec_paired_free": "C00",
    "P4_gatec_a1_scheduled": "C01",
    "P5_hybrid_joint_latent": "C02",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_directories(root: Path) -> list[Path]:
    return sorted(
        directory for directory in root.iterdir()
        if directory.is_dir() and (directory / "artifact_ledger.json").is_file()
    )


def _verify_run(directory: Path) -> tuple[int, list[str]]:
    ledger = _read(directory / "artifact_ledger.json")
    errors = []
    for name, digest in ledger.items():
        path = directory / name
        if not path.is_file() or _sha(path) != digest:
            errors.append(f"{directory.name}/{name}")
    return int((directory / "checkpoint_best_validation.pt").is_file()), errors


def audit_reference_artifacts(rm3_prediction: Path, rm3a: Path) -> dict[str, Any]:
    rm3_dirs = _run_directories(rm3_prediction)
    rm3a_dirs = _run_directories(rm3a)
    errors: list[str] = []
    checkpoint_count = 0
    for directory in (*rm3_dirs, *rm3a_dirs):
        count, run_errors = _verify_run(directory)
        checkpoint_count += count
        errors.extend(run_errors)
    return {
        "rm3_prediction_run_count": len(rm3_dirs),
        "rm3a_run_count": len(rm3a_dirs),
        "checkpoint_count": checkpoint_count,
        "hash_error_count": len(errors),
        "hash_errors": errors,
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }


def audit_rm2_reference_artifacts(rm2_root: Path) -> dict[str, Any]:
    """Close all 54 RM2 ledgers, including checkpoints returned inside the archive."""

    root_ledger = _read(rm2_root / "artifact_ledger.json")
    root_errors = [
        name for name, digest in root_ledger.items()
        if not (rm2_root / name).is_file() or _sha(rm2_root / name) != digest
    ]
    archive_path = rm2_root / "checkpoints_validation.tar"
    archive_digests: dict[str, str] = {}
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            archive_digests[member.name.replace("\\", "/")] = hashlib.sha256(
                handle.read()
            ).hexdigest()
    run_errors = []
    checkpoint_count = 0
    directories = _run_directories(rm2_root)
    for directory in directories:
        ledger = _read(directory / "artifact_ledger.json")
        for name, digest in ledger.items():
            path = directory / name
            if path.is_file():
                observed = _sha(path)
            elif name == "checkpoint_best_validation.pt":
                observed = archive_digests.get(f"{directory.name}/{name}")
            else:
                observed = None
            if name == "checkpoint_best_validation.pt" and observed is not None:
                checkpoint_count += 1
            if observed != digest:
                run_errors.append(f"{directory.name}/{name}")
    return {
        "run_count": len(directories),
        "checkpoint_count": checkpoint_count,
        "root_hash_error_count": len(root_errors),
        "root_hash_errors": root_errors,
        "run_hash_error_count": len(run_errors),
        "run_hash_errors": run_errors,
        "checkpoint_archive_member_count": len(archive_digests),
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }


def load_legacy_checkpoint_as_rm3av(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("model_config", {})
    config = RM3PredictionConfig(**raw_config)
    if config.candidate_id not in ANCHOR_BY_LEGACY:
        raise Phase35ProtocolError(
            "full RM3-AV functional replay is restricted to legacy P3/P4/P5"
        )
    anchor = ANCHOR_BY_LEGACY[config.candidate_id]
    model = build_rm3av_model(
        RM3AVModelConfig(
            candidate_id=anchor,
            window=config.window,
            horizon=config.horizon,
            n_features=config.n_features,
            d_model=config.d_model,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        ),
        checkpoint["feature_names"],
    )
    model.base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, {
        "legacy_candidate_id": config.candidate_id,
        "rm3av_anchor_candidate_id": anchor,
        "legacy_parameters_loaded": True,
        "checkpoint_sha256": _sha(checkpoint_path),
        "test_accessed": False,
    }


def replay_loaded_model(
    model,
    history: torch.Tensor,
    future_sp: torch.Tensor,
    logged_valve: torch.Tensor,
    logged_tin: torch.Tensor,
    local_target: torch.Tensor,
    terminal_target: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        modes = model.diagnostic_forward(
            history,
            future_sp,
            logged_future_valve=logged_valve,
            logged_future_tin=logged_tin,
            local_target=local_target,
        )
        mode_metrics = {
            name: {
                "terminal_mae_c": float(
                    (value["terminal_prediction"] - terminal_target).abs().mean()
                ),
                "local_mae_c": float(
                    (value["local_drop_prediction"] - local_target).abs().mean()
                ),
            }
            for name, value in modes.items()
        }
        baseline = history[:, -1, model.valve_indices]
        context = (
            model._p5_context(history)
            if model.base_candidate_id == "P5_hybrid_joint_latent"
            else model.base.model.encoder(
                (history - model.base.model.history_center)
                / model.base.model.history_scale
            )
        )
        constant = baseline[:, None].expand_as(logged_valve)
        identity = model.explicit_response(context, constant, baseline)["effect"]
        changed_sp = future_sp.clone()
        changed_sp[:, 30:] += 10.0
        original = model(history, future_sp)["terminal_prediction"]
        changed = model(history, changed_sp)["terminal_prediction"]
        prefix_error = float((original[:, :30] - changed[:, :30]).abs().max())
    finite = all(
        torch.isfinite(value["terminal_prediction"]).all().item()
        and torch.isfinite(value["local_drop_prediction"]).all().item()
        for value in modes.values()
    )
    return {
        "mode_metrics": mode_metrics,
        "constant_action_identity_max_abs": float(identity.abs().max()),
        "prefix_causality_max_abs_before_change": prefix_error,
        "finite": bool(finite),
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }


def _legacy_descriptive_metrics(root: Path) -> dict[str, Any]:
    records = []
    for directory in _run_directories(root):
        manifest = _read(directory / "manifest.json")
        metrics = _read(directory / "metrics_validation.json")
        spec = manifest["run_spec"]
        records.append({
            "run_id": directory.name,
            "candidate_id": spec["candidate_id"],
            "base_candidate_id": spec.get("base_candidate_id", spec["candidate_id"]),
            "fold_id": spec["fold_id"],
            "seed": spec["seed"],
            "metrics": metrics["metrics"],
            "test_accessed": False,
        })
    return {"run_count": len(records), "records": records}


def _legacy_episode_diagnostics(root: Path) -> dict[str, Any]:
    records = []
    for directory in _run_directories(root):
        manifest = _read(directory / "manifest.json")
        spec = manifest["run_spec"]
        with np.load(directory / "episodes_validation.npz", allow_pickle=False) as arrays:
            available = set(arrays.files)
            target = np.asarray(arrays["terminal_target"], dtype=np.float64)
            prediction = np.asarray(arrays["terminal_prediction"], dtype=np.float64)
            # RM3 episode files do not store history-last terminal explicitly.  The
            # target's first sample is the only common output-domain baseline and is
            # labeled as such rather than misnamed an exact pre-horizon hold-last.
            persistence = np.broadcast_to(target[:, :1], target.shape)
            horizon_rows = {}
            for step in (6, 18, 60):
                pred_mae = float(np.mean(np.abs(prediction[:, :step] - target[:, :step])))
                baseline_mae = float(np.mean(np.abs(persistence[:, :step] - target[:, :step])))
                horizon_rows[str(step)] = {
                    "prediction_mae_c": pred_mae,
                    "first_target_hold_baseline_mae_c": baseline_mae,
                    "skill_vs_first_target_hold": (
                        float(1.0 - pred_mae / baseline_mae) if baseline_mae > 0.0 else None
                    ),
                }
            task_rows: dict[str, Any] = {"terminal": horizon_rows}
            if {"logged_valve", "valve_prediction"}.issubset(available):
                valve_target = np.asarray(arrays["logged_valve"], dtype=np.float64)
                valve_prediction = np.asarray(arrays["valve_prediction"], dtype=np.float64)
                valve_baseline = np.broadcast_to(valve_target[:, :1], valve_target.shape)
                task_rows["valve"] = {
                    "prediction_mae": float(np.mean(np.abs(valve_prediction - valve_target))),
                    "first_target_hold_baseline_mae": float(
                        np.mean(np.abs(valve_baseline - valve_target))
                    ),
                    "prediction_mean_abs_delta": float(
                        np.mean(np.abs(np.diff(valve_prediction, axis=1)))
                    ),
                    "target_mean_abs_delta": float(
                        np.mean(np.abs(np.diff(valve_target, axis=1)))
                    ),
                }
            for task, key in (("tin", "tin_prediction"), ("local", "local_prediction")):
                if key in available:
                    task_rows[task] = {
                        "prediction_available": True,
                        "target_not_stored_in_legacy_episode_file": True,
                        "mae_replay_status": "OUTPUT_TARGET_UNAVAILABLE",
                    }
        records.append({
            "run_id": directory.name,
            "candidate_id": spec["candidate_id"],
            "fold_id": spec["fold_id"],
            "seed": spec["seed"],
            "tasks": task_rows,
            "baseline_definition": "hold_first_stored_target_sample_across_horizon",
            "exact_history_last_persistence_available": False,
            "test_accessed": False,
        })
    return {
        "run_count": len(records),
        "records": records,
        "output_domain_qualified": True,
        "test_accessed": False,
    }


def build_calibration_corrections(calibration_root: Path) -> dict[str, Any]:
    records = []
    for directory in _run_directories(calibration_root):
        json_path = directory / "calibration_validation.json"
        npz_path = directory / "orthogonal_residuals_validation.npz"
        payload = _read(json_path)
        with np.load(npz_path, allow_pickle=False) as arrays:
            trajectory = np.asarray(arrays["r0_trajectory_matrix"], dtype=np.float64)
        corrected = a1_nonnegative_projection(trajectory, step_seconds=10.0)
        corrected_payload = {
            "results": {"R1_a1_scheduled": corrected},
            "source_json_sha256": _sha(json_path),
            "source_npz_sha256": _sha(npz_path),
            "algorithm": "exact_active_set_nnls_v1",
        }
        records.append({
            "calibration_id": directory.name,
            "source_json_sha256": _sha(json_path),
            "source_npz_sha256": _sha(npz_path),
            "supersedes_sha256": _sha(json_path),
            "source_projection_rmse": float(
                payload["results"]["R1_a1_scheduled"]["projection_rmse"]
            ),
            "corrected_projection_rmse": float(corrected["projection_rmse"]),
            "corrected_nonnegative_coefficients": corrected["nonnegative_coefficients"],
            "corrected_payload": corrected_payload,
            "corrected_payload_sha256": hashlib.sha256(
                json.dumps(
                    corrected_payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "algorithm": "exact_active_set_nnls_v1",
            "historical_file_overwritten": False,
        })
    return {
        "calibration_unit_count": len(records),
        "records": records,
        "historical_files_overwritten": False,
        "test_accessed": False,
    }


def _nonnegative_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    columns = design.shape[1]
    best = np.zeros(columns, dtype=np.float64)
    best_error = float(np.dot(target, target))
    from itertools import combinations

    for size in range(1, columns + 1):
        for active in combinations(range(columns), size):
            coefficients = np.linalg.lstsq(design[:, active], target, rcond=None)[0]
            if np.any(coefficients < -1e-12):
                continue
            candidate = np.zeros(columns, dtype=np.float64)
            candidate[list(active)] = np.maximum(coefficients, 0.0)
            error = float(np.sum((design @ candidate - target) ** 2))
            if error < best_error:
                best, best_error = candidate, error
    return best


def _shape_basis(shape: str, horizon: int, step_seconds: float, delay_steps: int = 0) -> np.ndarray:
    time = np.maximum(
        (np.arange(horizon, dtype=np.float64) + 1.0 - delay_steps) * step_seconds,
        0.0,
    )
    positive_time = np.maximum(time, step_seconds)
    if shape == "linear_ramp":
        basis = (time / max(horizon * step_seconds, step_seconds))[:, None]
    elif shape == "power_basis":
        normalized = positive_time / max(horizon * step_seconds, step_seconds)
        basis = np.stack((normalized**0.5, normalized, normalized**1.5), axis=1)
    else:
        taus = {
            "one_pole": (180.0,),
            "two_pole": (60.0, 180.0),
            "three_pole": (60.0, 180.0, 600.0),
            "three_pole_bounded_dead_time": (60.0, 180.0, 600.0),
        }[shape]
        basis = 1.0 - np.exp(-time[:, None] / np.asarray(taus)[None])
    basis[time <= 0.0] = 0.0
    return basis


def _fit_shape(
    matrices: np.ndarray, shape: str, *, step_seconds: float
) -> tuple[np.ndarray, dict[str, Any]]:
    delays = range(4) if shape == "three_pole_bounded_dead_time" else (0,)
    best_fitted: np.ndarray | None = None
    best_metadata: dict[str, Any] | None = None
    best_error = float("inf")
    for delay in delays:
        design = _shape_basis(shape, len(matrices), step_seconds, delay)
        fitted = np.empty_like(matrices, dtype=np.float64)
        coefficients = np.empty((design.shape[1], 2, 2), dtype=np.float64)
        for source in range(2):
            for target in range(2):
                coefficients[:, source, target] = _nonnegative_fit(
                    design, matrices[:, source, target]
                )
                fitted[:, source, target] = design @ coefficients[:, source, target]
        error = float(np.mean((fitted - matrices) ** 2))
        if error < best_error:
            best_error = error
            best_fitted = fitted
            best_metadata = {
                "selected_delay_steps": delay,
                "selected_delay_seconds": delay * step_seconds,
                "delay_at_grid_boundary": bool(shape.endswith("dead_time") and delay in {0, 3}),
                "nonnegative_coefficients": coefficients.tolist(),
                "train_rmse": float(np.sqrt(error)),
            }
    assert best_fitted is not None and best_metadata is not None
    return best_fitted, best_metadata


def blocked_shape_model_audit(
    npz_path: Path,
    *,
    step_seconds: float = 10.0,
    ridge_alpha: float = 1e-3,
    epsilon: float = 1e-10,
) -> dict[str, Any]:
    """Expanding-date OOS shape comparison on frozen validation residuals."""

    with np.load(npz_path, allow_pickle=False) as arrays:
        timestamps_ns = np.asarray(arrays["timestamps_ns"], dtype=np.int64)
        action = np.asarray(arrays["action_residual"], dtype=np.float64)
        outcome = np.asarray(arrays["outcome_residual"], dtype=np.float64)
    days = np.floor_divide(timestamps_ns, 86_400_000_000_000)
    unique_days = np.unique(days)
    if len(unique_days) < 8:
        raise Phase35ProtocolError("RM3-AV shape audit requires at least eight UTC days")
    first_evaluation_day = max(4, len(unique_days) // 3)
    evaluation_blocks = [
        block for block in np.array_split(unique_days[first_evaluation_day:], 3) if len(block)
    ]
    shapes = (
        "linear_ramp", "power_basis", "one_pole", "two_pole",
        "three_pole", "three_pole_bounded_dead_time",
    )
    records = []
    for split_index, held_days in enumerate(evaluation_blocks):
        train_days = unique_days[unique_days < held_days.min()]
        train, held = np.isin(days, train_days), np.isin(days, held_days)
        train_matrices = np.stack([
            item.matrix for item in orthogonal_trajectory_moments(
                action[train], outcome[train], ridge_alpha=ridge_alpha, epsilon=epsilon,
                maximum_condition_number=10000.0,
                minimum_differential_to_common_energy=0.05,
            )
        ])
        held_matrices = np.stack([
            item.matrix for item in orthogonal_trajectory_moments(
                action[held], outcome[held], ridge_alpha=ridge_alpha, epsilon=epsilon,
                maximum_condition_number=10000.0,
                minimum_differential_to_common_energy=0.05,
            )
        ])
        shape_rows = {}
        for shape in shapes:
            fitted, metadata = _fit_shape(
                train_matrices, shape, step_seconds=step_seconds
            )
            shape_rows[shape] = {
                **metadata,
                "held_rmse": float(np.sqrt(np.mean((fitted - held_matrices) ** 2))),
            }
        winner = min(shape_rows, key=lambda name: shape_rows[name]["held_rmse"])
        records.append({
            "split_index": split_index,
            "train_days": train_days.tolist(),
            "held_days": held_days.tolist(),
            "day_overlap_count": len(set(train_days.tolist()) & set(held_days.tolist())),
            "train_episode_count": int(train.sum()),
            "held_episode_count": int(held.sum()),
            "shape_metrics": shape_rows,
            "held_rmse_winner": winner,
        })
    aggregate = {
        shape: {
            "mean_held_rmse": float(np.mean([
                row["shape_metrics"][shape]["held_rmse"] for row in records
            ])),
            "winner_count": sum(row["held_rmse_winner"] == shape for row in records),
        }
        for shape in shapes
    }
    return {
        "source_npz_sha256": _sha(npz_path),
        "split_strategy": "expanding_UTC_day_train_then_contiguous_future_day_blocks",
        "split_count": len(records),
        "shape_families": list(shapes),
        "aggregate": aggregate,
        "records": records,
        "true_response_order_claim": False,
        "causal_identification_claim": False,
        "test_accessed": False,
    }


def build_blocked_shape_audits(calibration_root: Path) -> dict[str, Any]:
    records = []
    for directory in _run_directories(calibration_root):
        records.append({
            "calibration_id": directory.name,
            "audit": blocked_shape_model_audit(
                directory / "orthogonal_residuals_validation.npz"
            ),
        })
    return {
        "calibration_unit_count": len(records),
        "records": records,
        "cross_calibration_shape_summary": {
            shape: {
                "mean_held_rmse": float(np.mean([
                    row["audit"]["aggregate"][shape]["mean_held_rmse"]
                    for row in records
                ])),
                "total_split_wins": sum(
                    row["audit"]["aggregate"][shape]["winner_count"]
                    for row in records
                ),
            }
            for shape in (
                "linear_ramp", "power_basis", "one_pole", "two_pole",
                "three_pole", "three_pole_bounded_dead_time",
            )
        },
        "true_response_order_claim": False,
        "test_accessed": False,
    }


def build_av0_replay(
    rm3_root: Path,
    rm3a_root: Path,
    *,
    cache_a: Path | None,
    cache_b: Path | None,
    device: str = "cpu",
    anchor_count: int = 64,
) -> dict[str, Any]:
    rm2_root = rm3_root.parent / "ms3r_gatec_rm2"
    rm2_reference = audit_rm2_reference_artifacts(rm2_root)
    if (
        rm2_reference["run_count"] != 54
        or rm2_reference["checkpoint_count"] != 54
        or rm2_reference["root_hash_error_count"]
        or rm2_reference["run_hash_error_count"]
    ):
        raise Phase35ProtocolError("RM3-AV0 RM2 reference artifacts are not closed")
    reference = audit_reference_artifacts(rm3_root / "prediction", rm3a_root)
    if reference["hash_error_count"]:
        raise Phase35ProtocolError("RM3-AV0 reference artifact hashes are not closed")
    functional: dict[str, Any]
    if cache_a is None or cache_b is None:
        functional = {
            "status": "CACHE_REQUIRED_FOR_INPUT_REPLAY",
            "run_count": 0,
            "records": [],
        }
    else:
        from ..data import deterministic_anchor_subset, load_cache
        from .gatec_data import extract_gatec_batch, paired_valid_anchors

        caches = {"A": load_cache(cache_a), "B": load_cache(cache_b)}
        records = []
        for directory in (
            *_run_directories(rm3_root / "prediction"),
            *_run_directories(rm3a_root),
        ):
            checkpoint_path = directory / "checkpoint_best_validation.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            legacy_id = checkpoint["model_config"]["candidate_id"]
            if legacy_id not in ANCHOR_BY_LEGACY:
                records.append({
                    "run_id": directory.name,
                    "legacy_candidate_id": legacy_id,
                    "status": "OUTPUT_DOMAIN_DESCRIPTIVE_ONLY",
                    "reason": "candidate has no explicit local-response interface",
                })
                continue
            model, metadata = load_legacy_checkpoint_as_rm3av(checkpoint_path)
            model.to(device)
            manifest = _read(directory / "manifest.json")
            spec = manifest["run_spec"]
            fraction = tuple(float(value) for value in spec["validation_fraction"])
            n_rows = len(caches["A"].timestamps_ns)
            bounds = (int(n_rows * fraction[0]), int(n_rows * fraction[1]))
            anchors = paired_valid_anchors(
                caches,
                "validation",
                window=int(checkpoint["model_config"]["window"]),
                horizon=int(checkpoint["model_config"]["horizon"]),
                max_age_s=180.0,
                bounds_override=bounds,
            )
            selected = deterministic_anchor_subset(
                anchors, min(anchor_count, len(anchors)), 36600
            )
            batch = extract_gatec_batch(
                caches,
                selected,
                window=int(checkpoint["model_config"]["window"]),
                horizon=int(checkpoint["model_config"]["horizon"]),
                validate_pair=False,
            )
            history = torch.as_tensor(batch.history, dtype=torch.float32, device=device)
            future_sp = torch.as_tensor(batch.future_sp, dtype=torch.float32, device=device)
            replay = replay_loaded_model(
                model,
                history,
                future_sp,
                torch.as_tensor(batch.logged_future_valve, dtype=torch.float32, device=device),
                torch.as_tensor(batch.logged_future_tin, dtype=torch.float32, device=device),
                torch.as_tensor(batch.local_drop_target, dtype=torch.float32, device=device),
                torch.as_tensor(batch.terminal_target, dtype=torch.float32, device=device),
            )
            records.append({
                "run_id": directory.name,
                **metadata,
                "status": "FUNCTIONAL_REPLAY_COMPLETE",
                "anchor_count": len(selected),
                "replay": replay,
            })
        functional = {
            "status": "FUNCTIONAL_REPLAY_COMPLETE",
            "run_count": len(records),
            "records": records,
        }
    return {
        "protocol_version": "phase3.5-ms3r-rm3av0-v1",
        "scope": "zero_training_validation_replay_not_model_selection_not_identification",
        "zero_training": True,
        "reference_artifacts": {
            **reference,
            "rm2": rm2_reference,
        },
        "legacy_metrics": {
            "rm2": _legacy_descriptive_metrics(rm2_root),
            "rm3": _legacy_descriptive_metrics(rm3_root / "prediction"),
            "rm3a": _legacy_descriptive_metrics(rm3a_root),
        },
        "legacy_episode_diagnostics": {
            "rm3": _legacy_episode_diagnostics(rm3_root / "prediction"),
            "rm3a": _legacy_episode_diagnostics(rm3a_root),
        },
        "calibration_corrections": build_calibration_corrections(
            rm3_root / "calibration"
        ),
        "blocked_shape_model_audits": build_blocked_shape_audits(
            rm3_root / "calibration"
        ),
        "legacy_output_domain": {
            "P0_m7_oracle_valve": {
                "outputs": ["terminal", "valve_policy"],
                "functional_response_replay": False,
            },
            "P1_m7_predicted_valve": {
                "outputs": ["terminal", "valve_policy"],
                "functional_response_replay": False,
            },
            "P2_m9_future_sp": {
                "outputs": ["terminal"],
                "functional_response_replay": False,
            },
            "P3_P4_P5_and_rm3a": {
                "functional_response_replay": True,
                "requires_validation_cache": True,
            },
        },
        "functional_replay": functional,
        "state_closure": build_state_closure_audit(
            generated={"valve", "tin", "tout", "terminal"},
            declared_external={"sp", "load", "pressure", "feedwater", "coal", "steam"},
            required={"valve", "tin", "tout", "terminal", "sp", "load", "pressure", "feedwater", "coal", "steam"},
        ),
        "assumption_ledger": build_assumption_ledger(),
        "manual_audit_verdicts": build_manual_verdict_template(),
        "test_accessed": False,
        "automatic_scientific_pass": None,
        "claims": {
            "model_champion": False,
            "causal_identification": False,
            "arbitrary_do_valve": False,
            "state_closed_simulator": False,
        },
    }
