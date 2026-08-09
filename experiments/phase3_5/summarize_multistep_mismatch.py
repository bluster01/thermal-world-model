#!/usr/bin/env python3
"""Fail-closed aggregation for the frozen Phase 3.5-MS2 validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from multistep_mismatch import expand_runs, load_matrix


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_CONTRASTS = {
    "valve_nonlinear_r50": ("v_g2_monotone", "v_g2_identity"),
    "context_scheduled_2p": ("c_g2_scheduled", "c_g2_global"),
}


def _read_json(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(f"required MS2 artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _gate_metrics(metrics: dict, route: str) -> list[str]:
    diagnostics = metrics.get("structural_diagnostics", {})
    operator = diagnostics.get("operator", {})
    failures = []
    if diagnostics.get("reference_identity_max_error") != 0:
        failures.append("reference_identity")
    if diagnostics.get("future_action_leakage_max_error") != 0:
        failures.append("future_action_leakage")
    if not diagnostics.get("finite_effect") or not diagnostics.get("finite_state"):
        failures.append("non_finite_rollout")
    if diagnostics.get("post_change_sensitivity_max_c", 0) <= 1e-6:
        failures.append("no_post_change_sensitivity")
    if diagnostics.get("positive_step_terminal_effect_max_c", 0) >= 0:
        failures.append("positive_step_direction")
    if route in {"graybox", "koopman"}:
        spectral_radius = operator.get("spectral_radius")
        if spectral_radius is None or not 0 <= spectral_radius < 1:
            failures.append("spectral_radius")
    for key in (
        "effect_mae",
        "clean_effect_mae",
        "clean_effect_nmae",
        "direction_accuracy_clean_nonzero",
    ):
        if metrics.get(key) is None:
            failures.append(f"missing_{key}")
    return failures


def build_summary(matrix_path: Path, output_root: Path) -> dict:
    matrix = load_matrix(matrix_path)
    expected_runs = expand_runs(matrix)
    current_sha = _git_sha()
    records = []
    for run in expected_runs:
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        run_dir = output_root / f"ms2_{candidate_id}_s{seed}"
        manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics_validation.json")
        history = _read_json(run_dir / "history.json")
        checkpoint = run_dir / "checkpoint_best_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required MS2 checkpoint missing: {checkpoint}")
        if (run_dir / "metrics_test.json").exists() or (
            run_dir / "synthetic_test_access_ledger.json"
        ).exists():
            raise RuntimeError(f"unauthorized MS2 test artifact found: {run_dir}")
        expected_manifest = {
            "protocol_version": matrix["protocol_version"],
            "route_id": candidate_id,
            "seed": seed,
            "git_sha": current_sha,
            "test_accessed": False,
        }
        mismatches = [
            key for key, value in expected_manifest.items() if manifest.get(key) != value
        ]
        checkpoint_hash = _sha256(checkpoint)
        if manifest.get("checkpoint_sha256") != checkpoint_hash:
            mismatches.append("checkpoint_sha256")
        if not history or manifest.get("best_epoch", 0) > len(history):
            mismatches.append("history/best_epoch")
        if mismatches:
            raise RuntimeError(
                f"MS2 manifest mismatch for {candidate_id}/seed={seed}: "
                f"{sorted(set(mismatches))}"
            )
        gate_failures = _gate_metrics(metrics, run["route"])
        records.append({
            **run,
            "best_epoch": manifest["best_epoch"],
            "epochs_ran": len(history),
            "elapsed_seconds": manifest["elapsed_seconds"],
            "checkpoint_sha256": checkpoint_hash,
            "effect_mae": metrics["effect_mae"],
            "clean_effect_mae": metrics["clean_effect_mae"],
            "clean_effect_nmae": metrics["clean_effect_nmae"],
            "direction_accuracy_clean_nonzero": metrics["direction_accuracy_clean_nonzero"],
            "gate_failures": gate_failures,
        })

    candidates = {}
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        subset = [record for record in records if record["candidate_id"] == candidate_id]
        candidates[candidate_id] = {
            "regime_id": subset[0]["regime_id"],
            "route": subset[0]["route"],
            "seeds": [record["seed"] for record in subset],
            "clean_effect_nmae_mean": statistics.mean(
                record["clean_effect_nmae"] for record in subset
            ),
            "clean_effect_nmae_std": statistics.stdev(
                record["clean_effect_nmae"] for record in subset
            ),
            "clean_effect_mae_mean": statistics.mean(
                record["clean_effect_mae"] for record in subset
            ),
            "effect_mae_mean": statistics.mean(record["effect_mae"] for record in subset),
            "direction_accuracy_clean_nonzero_mean": statistics.mean(
                record["direction_accuracy_clean_nonzero"] for record in subset
            ),
            "all_structural_gates_pass": all(not record["gate_failures"] for record in subset),
            "epochs_ran": [record["epochs_ran"] for record in subset],
        }

    contrasts = {}
    for regime_id, (candidate_id, baseline_id) in PRIMARY_CONTRASTS.items():
        candidate_rows = {
            record["seed"]: record for record in records if record["candidate_id"] == candidate_id
        }
        baseline_rows = {
            record["seed"]: record for record in records if record["candidate_id"] == baseline_id
        }
        improvements = [
            (
                baseline_rows[seed]["clean_effect_nmae"]
                - candidate_rows[seed]["clean_effect_nmae"]
            ) / max(baseline_rows[seed]["clean_effect_nmae"], 1e-12)
            for seed in sorted(candidate_rows)
        ]
        contrasts[regime_id] = {
            "candidate_id": candidate_id,
            "baseline_id": baseline_id,
            "paired_seed_relative_improvement": improvements,
            "mean_relative_improvement": statistics.mean(improvements),
            "direction_consistent": all(value > 0 for value in improvements),
            "meets_20pct_screen": all(value > 0 for value in improvements)
            and statistics.mean(improvements) >= 0.20,
        }

    failures = [
        {
            "candidate_id": record["candidate_id"],
            "seed": record["seed"],
            "failures": record["gate_failures"],
        }
        for record in records if record["gate_failures"]
    ]
    return {
        "protocol_version": matrix["protocol_version"],
        "evidence_scope": matrix["evidence_scope"],
        "git_sha": current_sha,
        "split": "validation",
        "run_count": len(records),
        "test_accessed": False,
        "all_artifact_and_structural_gates_pass": not failures,
        "gate_failures": failures,
        "candidates": candidates,
        "primary_contrasts": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"),
    )
    parser.add_argument("--output-root", default="results/phase3_5/multistep_mismatch")
    parser.add_argument("--output")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    summary = build_summary(Path(args.matrix).resolve(), output_root)
    output = Path(args.output).resolve() if args.output else output_root / "summary_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2))
    if not summary["all_artifact_and_structural_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
