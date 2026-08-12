#!/usr/bin/env python3
"""Cache-free Supervisor replay for returned RM3 validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.multistep.rm3_calibration import a1_nonnegative_projection


DEFAULT_RESULTS = ROOT / "results/phase3_5/ms3r_rm3"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(results_root: Path) -> dict[str, Any]:
    status = _read(results_root / "matrix_execution_status.json")
    summary = _read(results_root / "summary_validation.json")
    available_verified = 0
    missing_checkpoints: list[dict[str, str]] = []
    integrity_errors: list[str] = []
    for ledger_path in results_root.rglob("artifact_ledger.json"):
        if ledger_path == results_root / "artifact_ledger.json":
            continue
        for name, expected in _read(ledger_path).items():
            path = ledger_path.parent / name
            if not path.is_file():
                if name == "checkpoint_best_validation.pt":
                    missing_checkpoints.append(
                        {"run_id": ledger_path.parent.name, "sha256": expected}
                    )
                else:
                    integrity_errors.append(f"missing:{path.relative_to(results_root)}")
                continue
            if _sha(path) != expected:
                integrity_errors.append(f"hash:{path.relative_to(results_root)}")
            else:
                available_verified += 1

    replay_error = 0.0
    run_metrics: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run_dir in sorted((results_root / "prediction").iterdir()):
        metrics = _read(run_dir / "metrics_validation.json")
        spec = _read(run_dir / "manifest.json")["run_spec"]
        with np.load(run_dir / "episodes_validation.npz", allow_pickle=False) as arrays:
            terminal_mae = float(
                np.mean(
                    np.abs(
                        arrays["terminal_prediction"].astype(float)
                        - arrays["terminal_target"].astype(float)
                    )
                )
            )
        replay_error = max(
            replay_error, abs(terminal_mae - float(metrics["metrics"]["terminal_mae_c"]))
        )
        run_metrics[(spec["candidate_id"], spec["fold_id"], int(spec["seed"]))] = metrics[
            "metrics"
        ]

    def contrast(left: str, right: str) -> dict[str, Any]:
        values = [
            run_metrics[(left, fold, seed)]["terminal_mae_c"]
            - run_metrics[(right, fold, seed)]["terminal_mae_c"]
            for fold in ("F0", "F1")
            for seed in (0, 1, 2)
        ]
        return {
            "left_minus_right_per_fold_seed_c": values,
            "mean_c": float(np.mean(values)),
            "left_better_count": int(np.sum(np.asarray(values) < 0)),
            "paired_run_count": len(values),
        }

    calibration: dict[tuple[str, int, int], np.ndarray] = {}
    corrected_projection = []
    independent_count = 0
    for directory in sorted((results_root / "calibration").iterdir()):
        payload = _read(directory / "calibration_validation.json")
        spec = payload["spec"]
        matrix = np.asarray(payload["results"]["R0_linear_mimo"]["trajectory_matrix"])
        calibration[(spec["fold_id"], int(spec["seed"]), int(spec["response_horizon_steps"]))] = matrix
        independent_count += int(
            payload["results"]["R0_linear_mimo"]["independent_channels_supported_all_steps"]
        )
        corrected = a1_nonnegative_projection(matrix, step_seconds=10.0)
        corrected_projection.append(
            {
                "calibration_id": spec["calibration_id"],
                "returned_posthoc_clip_rmse": payload["results"]["R1_a1_scheduled"][
                    "projection_rmse"
                ],
                "corrected_nnls_rmse": corrected["projection_rmse"],
                "maximum_coefficient": float(
                    np.max(np.asarray(corrected["nonnegative_coefficients"]))
                ),
            }
        )
    stability = []
    for horizon in (6, 18):
        means = {}
        standard_deviations = {}
        for fold in ("F0", "F1"):
            endpoints = np.stack(
                [calibration[(fold, seed, horizon)][-1] for seed in (0, 1, 2)]
            )
            means[fold] = endpoints.mean(axis=0)
            standard_deviations[fold] = endpoints.std(axis=0, ddof=1)
        stability.append(
            {
                "horizon_steps": horizon,
                "endpoint_matrix_mean": {key: value.tolist() for key, value in means.items()},
                "endpoint_matrix_sd_across_seeds": {
                    key: value.tolist() for key, value in standard_deviations.items()
                },
                "fold_difference_f1_minus_f0": (means["F1"] - means["F0"]).tolist(),
            }
        )
    return {
        "protocol_version": "phase3.5-ms3r-rm3-supervisor-replay-v1",
        "scope": "validation_only_cache_free_provisional_until_checkpoint_supplement",
        "execution": {
            "record_count": len(status["records"]),
            "complete_record_count": sum(row["status"] == "complete" for row in status["records"]),
            "prediction_run_count": summary["prediction_run_count"],
            "calibration_unit_count": summary["calibration_unit_count"],
            "test_accessed": summary["test_accessed"],
            "automatic_scientific_pass": summary["automatic_scientific_pass"],
        },
        "artifact_integrity": {
            "available_ledger_entries_verified": available_verified,
            "non_checkpoint_integrity_errors": integrity_errors,
            "missing_checkpoint_count": len(missing_checkpoints),
            "missing_checkpoints": missing_checkpoints,
            "checkpoint_audit_complete": not missing_checkpoints,
        },
        "metric_replay": {
            "prediction_run_count": len(run_metrics),
            "terminal_mae_max_absolute_replay_error": replay_error,
        },
        "prediction_contrasts": {
            "p4_a1_minus_p3_free": contrast(
                "P4_gatec_a1_scheduled", "P3_gatec_paired_free"
            ),
            "p5_hybrid_minus_p3_free": contrast(
                "P5_hybrid_joint_latent", "P3_gatec_paired_free"
            ),
            "p5_hybrid_minus_p4_a1": contrast(
                "P5_hybrid_joint_latent", "P4_gatec_a1_scheduled"
            ),
        },
        "response_calibration": {
            "independent_channel_rank_supported_unit_count": independent_count,
            "unit_count": len(calibration),
            "corrected_a1_nnls_projection": corrected_projection,
            "endpoint_stability": stability,
            "returned_r1_projection_invalid_due_to_posthoc_coefficient_clipping": True,
            "context_scheduling_identified": False,
        },
        "supervisor_status": "PROVISIONAL_ARTIFACT_INCOMPLETE_CHECKPOINT_SUPPLEMENT_REQUIRED",
        "claims": {
            "prediction_champion": False,
            "unique_plant_gain": False,
            "complete_physical_response": False,
            "arbitrary_do_valve": False,
            "test_evidence": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    output = Path(args.output).resolve() if args.output else root / "supervisor_replay_validation.json"
    payload = replay(root)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
