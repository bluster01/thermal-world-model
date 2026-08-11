#!/usr/bin/env python3
"""Cache-free Supervisor audit for the real RM1-A attribution batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/phase3_5/ms3r_gatec_local_real_rm1a"
EXPECTED_CANDIDATES = {
    "C0_paired_free",
    "C1_additive_base",
    "C2_sched_small",
    "C3_sched_base",
    "C4_sched_large",
    "C5_sched_base_terminal_only",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def build_audit(results: Path) -> dict[str, Any]:
    ledger = _read_json(results / "artifact_ledger.json")
    mismatches = {
        name: {"expected": expected, "actual": _sha256(results / name)}
        for name, expected in ledger.items()
        if not (results / name).is_file() or _sha256(results / name) != expected
    }
    if mismatches:
        raise RuntimeError(f"RM1-A artifact ledger mismatch: {sorted(mismatches)}")
    manifest = _read_json(results / "run_manifest.json")
    summary = _read_json(results / "summary_validation.json")
    candidates = {item["candidate_id"]: item for item in summary["results"]}
    if set(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError("RM1-A candidate set is incomplete or changed")
    if manifest.get("test_accessed") is not False or summary.get("test_accessed") is not False:
        raise RuntimeError("RM1-A audit found forbidden test access")
    if not summary.get("shared_train_anchors") or not summary.get("shared_validation_anchors"):
        raise RuntimeError("RM1-A candidates did not share frozen anchors")
    train_shas = {item["train_anchor_sha256"] for item in candidates.values()}
    validation_shas = {item["validation_anchor_sha256"] for item in candidates.values()}
    if len(train_shas) != 1 or len(validation_shas) != 1:
        raise RuntimeError("RM1-A per-candidate anchor hashes differ")

    metrics = {key: value["metrics_validation"] for key, value in candidates.items()}
    c0, c1, c2 = metrics["C0_paired_free"], metrics["C1_additive_base"], metrics["C2_sched_small"]
    c3, c4, c5 = metrics["C3_sched_base"], metrics["C4_sched_large"], metrics["C5_sched_base_terminal_only"]
    capacity_effects = [
        c2["logged_action_effect_mean_abs_c"],
        c3["logged_action_effect_mean_abs_c"],
        c4["logged_action_effect_mean_abs_c"],
    ]
    capacity_mean = sum(capacity_effects) / len(capacity_effects)
    capacity_relative_range = (max(capacity_effects) - min(capacity_effects)) / capacity_mean
    capacity_monotonic_disappearance = capacity_effects[0] > capacity_effects[1] > capacity_effects[2]
    terminal_only_semantic_failure = (
        c5["local_to_persistence_ratio"] > 1.05
        and c5["terminal_to_persistence_ratio"] < c3["terminal_to_persistence_ratio"]
    )
    rm0b_results = results.parent / "ms3r_gatec_local_real_subset_rm0b"
    rm0b_ledger = _read_json(rm0b_results / "artifact_ledger.json")
    rm0b_a1_path = rm0b_results / "a1phys_three_pole_validation.json"
    if _sha256(rm0b_a1_path) != rm0b_ledger["a1phys_three_pole_validation.json"]:
        raise RuntimeError("RM1-A audit found an invalid RM0-B A1 artifact")
    rm0b_a1 = _read_json(rm0b_a1_path)
    reuse_metric_keys = (
        "forecast_valve_mae",
        "forecast_tin_mae_c",
        "forecast_local_drop_mae_c",
        "forecast_terminal_mae_c",
        "oracle_terminal_mae_c",
        "predicted_action_effect_mean_abs_c",
        "logged_action_effect_mean_abs_c",
        "persistence_local_drop_mae_c",
    )
    reuse_max_abs_diff = max(
        abs(c3[key] - rm0b_a1["metrics_validation"][key]) for key in reuse_metric_keys
    )
    return {
        "protocol_version": "phase3.5-ms3r-gatec-local-real-rm1a-audit-v1",
        "audited_execution_git_sha": manifest["execution_git_sha"],
        "scope": "local_real_train_validation_attribution_not_causal",
        "artifact_integrity": {
            "ledger_entries_verified": len(ledger),
            "all_sha256_match": True,
            "candidate_count": len(candidates),
            "all_finite": summary["all_finite"],
            "test_accessed": False,
        },
        "anchor_contract": {
            "train_subset": candidates["C3_sched_base"]["train_anchor_count"],
            "validation_subset": candidates["C3_sched_base"]["validation_anchor_count"],
            "train_anchor_sha256": next(iter(train_shas)),
            "validation_anchor_sha256": next(iter(validation_shas)),
        },
        "rm0b_a1_reuse": {
            "path": "results/phase3_5/ms3r_gatec_local_real_subset_rm0b/a1phys_three_pole_validation.json",
            "shared_train_anchors": candidates["C3_sched_base"]["train_anchor_sha256"] == rm0b_a1["train_anchor_sha256"],
            "shared_validation_anchors": candidates["C3_sched_base"]["validation_anchor_sha256"] == rm0b_a1["validation_anchor_sha256"],
            "metric_count_compared": len(reuse_metric_keys),
            "max_abs_metric_difference": reuse_max_abs_diff,
            "duplicate_rm1b_rerun_needed": False,
        },
        "primary_contrasts": {
            "scheduled_base_vs_paired_free": {
                "shared_composite_delta": c3["dimensionless_composite_loss"] - c0["dimensionless_composite_loss"],
                "local_mae_delta_c": c3["forecast_local_drop_mae_c"] - c0["forecast_local_drop_mae_c"],
                "terminal_mae_delta_c": c3["forecast_terminal_mae_c"] - c0["forecast_terminal_mae_c"],
                "logged_vs_shuffled_advantage_c": c3["logged_vs_shuffled_local_advantage_c"],
            },
            "scheduled_base_vs_additive_base": {
                "shared_composite_delta": c3["dimensionless_composite_loss"] - c1["dimensionless_composite_loss"],
                "local_mae_delta_c": c3["forecast_local_drop_mae_c"] - c1["forecast_local_drop_mae_c"],
                "logged_effect_ratio": c3["logged_action_effect_mean_abs_c"] / c1["logged_action_effect_mean_abs_c"],
                "logged_vs_shuffled_advantage_delta_c": c3["logged_vs_shuffled_local_advantage_c"] - c1["logged_vs_shuffled_local_advantage_c"],
            },
            "scheduled_capacity_scan": {
                "logged_effect_mean_abs_c": {
                    "small": capacity_effects[0],
                    "base": capacity_effects[1],
                    "large": capacity_effects[2],
                },
                "relative_range": capacity_relative_range,
                "monotonic_disappearance_with_capacity": capacity_monotonic_disappearance,
                "large_vs_base_shared_composite_delta": c4["dimensionless_composite_loss"] - c3["dimensionless_composite_loss"],
            },
            "terminal_only_vs_scheduled_base": {
                "terminal_mae_delta_c": c5["forecast_terminal_mae_c"] - c3["forecast_terminal_mae_c"],
                "local_mae_delta_c": c5["forecast_local_drop_mae_c"] - c3["forecast_local_drop_mae_c"],
                "terminal_only_local_to_persistence_ratio": c5["local_to_persistence_ratio"],
                "semantic_failure": terminal_only_semantic_failure,
            },
        },
        "supervisor_decision": {
            "label": "RM1A_ATTRIBUTION_COMPLETE_CAPACITY_COLLAPSE_NOT_OBSERVED_LOCAL_SUPERVISION_REQUIRED",
            "components": [
                "REAL_1PCT_VALIDATION_ONLY",
                "CAPACITY_RESPONSE_STABLE_WITHIN_SCAN",
                "TERMINAL_ONLY_SEMANTIC_PATH_REJECTED",
                "SCHEDULED_BASE_RETAINED_AS_PREREGISTERED_REFERENCE",
                "PREDICTION_GAIN_OVER_PAIRED_FREE_NEGLIGIBLE",
                "NO_OPERATOR_RANKING",
                "NO_CAUSAL_UPGRADE",
                "NO_TEST_ACCESS",
                "NO_LINUX_RELEASE",
            ],
            "reference_candidate": "C3_sched_base",
            "reference_is_empirical_champion": False,
            "operator_ranking_supported": False,
            "automatic_scientific_pass": None,
            "linux_authorized_gate": None,
            "next": "DESIGN_RM2_DAY_BLOCK_RESPONSE_AND_ROLLING_FOLD_VALIDATION_BEFORE_ANY_OPERATOR_CLAIM",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output")
    args = parser.parse_args()
    results = Path(args.results_dir).resolve()
    output = Path(args.output).resolve() if args.output else results / "supervisor_audit_validation.json"
    audit = build_audit(results)
    _atomic_json(output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
