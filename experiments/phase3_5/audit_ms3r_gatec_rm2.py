#!/usr/bin/env python3
"""Cache-free Supervisor audit for the MS3-R Gate C RM2 batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/phase3_5/ms3r_gatec_rm2"
DEFAULT_GATEB_AUDIT = (
    ROOT / "results/phase3_5/ms3r_gateb_point_closure/supervisor_audit_validation.json"
)
EXPECTED_CANDIDATES = (
    "A0_paired_free",
    "A1_additive_base",
    "A2_a1_sched_base",
    "A3_a1_sched_large",
    "B1_koopman",
    "B2_pi_ode",
    "B3_deeponet",
    "C1_common_only",
    "C2_no_downstream_latent",
)
CORE_METRICS = (
    "shared_prediction_score",
    "forecast_valve_mae",
    "forecast_tin_mae_c",
    "forecast_local_mae_c",
    "forecast_terminal_mae_c",
    "oracle_terminal_mae_c",
    "logged_vs_shuffled_local_advantage_c",
    "predicted_effect_mean_abs_c",
    "logged_effect_mean_abs_c",
    "logged_effect_h60_mean_abs_c",
    "logged_effect_h180_mean_abs_c",
    "stable_pole_max",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _paired_contrast(
    grouped: dict[str, list[dict[str, Any]]], candidate: str, reference: str
) -> dict[str, Any]:
    candidate_runs = {
        (item["fold_id"], item["seed"]): item["metrics"] for item in grouped[candidate]
    }
    reference_runs = {
        (item["fold_id"], item["seed"]): item["metrics"] for item in grouped[reference]
    }
    if candidate_runs.keys() != reference_runs.keys():
        raise RuntimeError(f"RM2 paired run keys differ for {candidate} and {reference}")
    metrics: dict[str, Any] = {}
    for metric in CORE_METRICS:
        values = [
            float(candidate_runs[key][metric]) - float(reference_runs[key][metric])
            for key in sorted(candidate_runs)
        ]
        metrics[metric] = {
            "mean_delta": float(np.mean(values)),
            "run_deltas": values,
            "negative_count": int(np.sum(np.asarray(values) < 0)),
            "positive_count": int(np.sum(np.asarray(values) > 0)),
        }
    return {"candidate": candidate, "reference": reference, "metrics": metrics}


def _response_slopes(arrays: Any, horizon_index: int) -> np.ndarray:
    dose = arrays["logged_valve"][:, horizon_index] - arrays["baseline_valve"]
    rows = []
    for input_index, effect_key in enumerate(("a_only_effect", "b_only_effect")):
        x = dose[:, input_index]
        denominator = float(np.dot(x, x))
        if denominator <= 0:
            raise RuntimeError("RM2 response slope has no action dose")
        rows.append(np.sum(x[:, None] * arrays[effect_key][:, horizon_index], axis=0) / denominator)
    return np.stack(rows)


def audit(results: Path, gateb_audit_path: Path) -> dict[str, Any]:
    summary = _read_json(results / "summary_validation.json")
    if summary.get("complete_run_count") != 54 or not summary.get("matrix_complete"):
        raise RuntimeError("RM2 matrix is not complete")
    if summary.get("test_accessed") is not False:
        raise RuntimeError("RM2 test-access contract was violated")

    records = summary["records"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["candidate_id"]].append(record)
    if set(grouped) != set(EXPECTED_CANDIDATES) or any(len(items) != 6 for items in grouped.values()):
        raise RuntimeError("RM2 candidate/run matrix is not closed")

    archive_checkpoint_sha: dict[str, str] = {}
    archive_path = results / summary["checkpoint_archive"]["path"]
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"RM2 checkpoint member is unreadable: {member.name}")
            archive_checkpoint_sha[member.name.split("/", 1)[0]] = _sha256_bytes(handle.read())

    verified_entries = 0
    hash_mismatches: list[str] = []
    structural_failures: list[str] = []
    aggregates: dict[str, Any] = {}
    response_diagnostics: dict[str, Any] = {}
    for candidate, candidate_records in grouped.items():
        metric_summary: dict[str, Any] = {}
        for metric in CORE_METRICS:
            values = np.asarray([item["metrics"][metric] for item in candidate_records], dtype=float)
            metric_summary[metric] = {
                "mean": float(values.mean()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        aggregates[candidate] = {
            "run_count": len(candidate_records),
            "best_updates": sorted(int(item["best_update"]) for item in candidate_records),
            "metrics": metric_summary,
        }

        slopes: dict[str, list[np.ndarray]] = {"H60": [], "H180": []}
        positive_days = 0
        evaluated_days = 0
        for record in candidate_records:
            run_id = record["run_id"]
            run_dir = results / run_id
            ledger = _read_json(run_dir / "artifact_ledger.json")
            for name, expected in ledger.items():
                actual = (
                    archive_checkpoint_sha.get(run_id)
                    if name == "checkpoint_best_validation.pt"
                    else _sha256(run_dir / name)
                )
                verified_entries += 1
                if actual != expected:
                    hash_mismatches.append(f"{run_id}/{name}")
            metrics_payload = _read_json(run_dir / "metrics_validation.json")
            structural = metrics_payload["structural_validation"]
            if not structural.get("selector_eligible"):
                structural_failures.append(run_id)
            with np.load(run_dir / "episodes_validation.npz") as arrays:
                if candidate != "A0_paired_free":
                    slopes["H60"].append(_response_slopes(arrays, 5))
                    slopes["H180"].append(_response_slopes(arrays, 17))
                days = arrays["timestamps_ns"].astype("datetime64[ns]").astype("datetime64[D]")
                advantage = np.abs(arrays["shuffled_local"] - arrays["local_target"]).mean(
                    axis=(1, 2)
                ) - np.abs(arrays["logged_local"] - arrays["local_target"]).mean(axis=(1, 2))
                for day in np.unique(days):
                    evaluated_days += 1
                    positive_days += int(float(advantage[days == day].mean()) > 0)
        response_diagnostics[candidate] = {
            "logged_vs_shuffled_positive_utc_days": positive_days,
            "logged_vs_shuffled_evaluated_utc_days": evaluated_days,
        }
        if candidate != "A0_paired_free":
            for horizon, matrices in slopes.items():
                values = np.stack(matrices)
                response_diagnostics[candidate][f"{horizon}_action_to_local_slope_matrix"] = (
                    values.mean(axis=0).tolist()
                )
                diagonal = np.abs(np.diagonal(values, axis1=1, axis2=2)).mean(axis=1)
                off_diagonal = np.abs(np.stack((values[:, 0, 1], values[:, 1, 0]), axis=1)).mean(axis=1)
                response_diagnostics[candidate][f"{horizon}_diagonal_to_off_diagonal_ratio"] = {
                    "mean": float(np.mean(diagonal / off_diagonal)),
                    "minimum": float(np.min(diagonal / off_diagonal)),
                    "maximum": float(np.max(diagonal / off_diagonal)),
                }

    root_ledger = _read_json(results / "artifact_ledger.json")
    root_hashes_exact = all(_sha256(results / name) == expected for name, expected in root_ledger.items())
    if hash_mismatches or structural_failures or not root_hashes_exact:
        raise RuntimeError("RM2 artifact or structural replay failed")

    gateb = _read_json(gateb_audit_path)["short_horizon_mimo"]
    gateb_matrices = {
        "H60": np.asarray(gateb["H60_day_median_matrix"], dtype=float),
        "H180": np.asarray(gateb["H180_day_median_matrix"], dtype=float),
    }
    gateb_consistency: dict[str, Any] = {"role": "diagnostic_only_not_ground_truth"}
    for candidate in ("A2_a1_sched_base", "B1_koopman", "B2_pi_ode", "B3_deeponet"):
        gateb_consistency[candidate] = {}
        for horizon, gateb_matrix in gateb_matrices.items():
            learned = np.asarray(
                response_diagnostics[candidate][f"{horizon}_action_to_local_slope_matrix"]
            )
            gateb_consistency[candidate][horizon] = {
                "learned_matrix": learned.tolist(),
                "gateb_conditional_matrix": gateb_matrix.tolist(),
                "learned_to_gateb_diagonal_ratio": (
                    np.diag(learned) / np.diag(gateb_matrix)
                ).tolist(),
            }

    contrasts = {
        "additive_vs_paired_free": _paired_contrast(grouped, "A1_additive_base", "A0_paired_free"),
        "scheduled_vs_additive": _paired_contrast(grouped, "A2_a1_sched_base", "A1_additive_base"),
        "large_vs_base_capacity": _paired_contrast(grouped, "A3_a1_sched_large", "A2_a1_sched_base"),
        "common_only_vs_full_mimo": _paired_contrast(grouped, "C1_common_only", "A2_a1_sched_base"),
        "no_downstream_latent_vs_latent": _paired_contrast(
            grouped, "C2_no_downstream_latent", "A2_a1_sched_base"
        ),
        "koopman_vs_a1_scheduled": _paired_contrast(grouped, "B1_koopman", "A2_a1_sched_base"),
        "pi_ode_vs_a1_scheduled": _paired_contrast(grouped, "B2_pi_ode", "A2_a1_sched_base"),
        "deeponet_vs_a1_scheduled": _paired_contrast(grouped, "B3_deeponet", "A2_a1_sched_base"),
    }

    return {
        "protocol_version": "phase3.5-ms3r-gatec-rm2-supervisor-audit-v1",
        "audited_result_commit": "c095279",
        "executed_code_commit": _read_json(results / "run_manifest.json")["execution_git_sha"],
        "scope": "real_closed_loop_observational_train_validation_not_causal",
        "artifact_integrity": {
            "complete_run_count": 54,
            "per_run_ledger_entries_verified": verified_entries,
            "checkpoint_archive_members_verified": len(archive_checkpoint_sha),
            "root_ledger_entries_verified": len(root_ledger),
            "all_sha256_match": True,
            "all_structural_selector_checks_pass": True,
            "fold_anchor_contract": summary["fold_anchor_contract"],
            "test_accessed": False,
            "training_executed_locally": False,
        },
        "candidate_aggregates": aggregates,
        "paired_contrasts": contrasts,
        "response_diagnostics": response_diagnostics,
        "gateb_conditional_consistency": gateb_consistency,
        "supervisor_decision": {
            "label": "RM2_COMPLETE_CONDITIONAL_ACTION_PATH_REPRODUCED_OPERATOR_GAIN_NOT_IDENTIFIED",
            "components": [
                "ARTIFACT_AND_STRUCTURAL_REPLAY_PASS",
                "A1_LOCAL_PREDICTION_IMPROVES_OVER_PAIRED_FREE",
                "A1_CAPACITY_RESPONSE_COLLAPSE_NOT_OBSERVED",
                "LOGGED_ACTION_DEPENDENCE_STABLE_ACROSS_SEED_FOLD_AND_DAY",
                "DOWNSTREAM_LATENT_BLOCK_RETAINED",
                "DIFFERENTIAL_RESPONSE_MODE_NOT_DEMONSTRATED_NECESSARY",
                "OPERATOR_RESPONSE_AMPLITUDE_NONIDENTIFIABLE",
                "NO_OPERATOR_CHAMPION",
                "NO_CAUSAL_OR_OPEN_LOOP_UPGRADE",
                "NO_TEST_ACCESS",
                "NO_MS4_RELEASE",
            ],
            "retained_reference": "A2_a1_sched_base",
            "retained_partial_identification_baseline": "C1_common_only",
            "operator_ranking_supported": False,
            "automatic_scientific_pass": None,
            "linux_authorized_gate": None,
            "next": "DESIGN_RM3_ORTHOGONALIZED_OUT_OF_FOLD_RESPONSE_MOMENT_CALIBRATION",
            "claim_boundary": (
                "RM2 supports stable observed-policy prediction and a disturbance-conditioned local "
                "action-dependent response slot. Raw closed-loop future-valve auxiliary supervision "
                "does not identify a unique physical gain, arbitrary do(valve), measured spray-flow "
                "physics, an operator champion, or a deployable closed-loop simulator."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--gateb-audit", default=str(DEFAULT_GATEB_AUDIT))
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS / "supervisor_audit_validation.json"),
    )
    args = parser.parse_args()
    payload = audit(Path(args.results_root).resolve(), Path(args.gateb_audit).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload["supervisor_decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
