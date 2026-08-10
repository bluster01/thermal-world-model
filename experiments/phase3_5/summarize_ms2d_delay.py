#!/usr/bin/env python3
"""Fail-closed aggregation for Phase 3.5-MS2-D1 validation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.ms2d_delay import (  # noqa: E402
    FROZEN_EXECUTION_PATHS,
    _build_configs,
    _canonical,
    _sha256,
    _select,
    expand_runs,
    load_matrix,
)
from experiments.phase3_5.summarize_multistep_mismatch import (  # noqa: E402
    _gate_metrics,
)


NO_DELAY_ID = "d1_g2_no_delay"
LEARNED_DELAY_ID = "d1_g2_learned_delay"
ORACLE_DELAY_ID = "d1_g2_oracle_delay"


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required MS2-D1 artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _assert_code_equivalent(execution_sha: str) -> None:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{execution_sha}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"MS2-D1 execution commit unavailable: {execution_sha}")
    compared = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            execution_sha,
            "HEAD",
            "--",
            *FROZEN_EXECUTION_PATHS,
        ],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(
            f"MS2-D1 frozen execution code differs from {execution_sha}"
        )
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking MS2-D1 code equivalence")


def _delay_diagnostics(
    metrics: dict,
    *,
    dt_seconds: float,
    truth_delay_steps: int,
    neighborhood_mass_min: float,
) -> dict:
    operator = metrics.get("structural_diagnostics", {}).get("operator", {})
    expected_seconds = operator.get("expected_delay_seconds")
    weights = operator.get("delay_weights")
    if expected_seconds is None or not isinstance(weights, list) or not weights:
        raise RuntimeError("learned-delay run is missing delay diagnostics")
    numeric_weights = [float(value) for value in weights]
    if any(value < 0 for value in numeric_weights):
        raise RuntimeError("learned-delay weights must be non-negative")
    if abs(sum(numeric_weights) - 1.0) > 1e-5:
        raise RuntimeError("learned-delay weights must sum to one")
    expected_steps = float(expected_seconds) / dt_seconds
    weighted_steps = sum(index * value for index, value in enumerate(numeric_weights))
    if abs(expected_steps - weighted_steps) > 1e-5:
        raise RuntimeError("expected delay is inconsistent with delay weights")
    neighborhood_mass = sum(
        value
        for index, value in enumerate(numeric_weights)
        if abs(index - truth_delay_steps) <= 1
    )
    return {
        "expected_delay_seconds": float(expected_seconds),
        "expected_delay_steps": expected_steps,
        "truth_delay_steps": truth_delay_steps,
        "absolute_error_steps": abs(expected_steps - truth_delay_steps),
        "truth_plus_minus_one_step_mass": neighborhood_mass,
        "neighborhood_mass_min": neighborhood_mass_min,
        "delay_weights": numeric_weights,
    }


def build_summary(matrix_path: Path, output_root: Path) -> dict:
    matrix = load_matrix(matrix_path)
    current_sha = _git_sha()
    matrix_sha = _sha256(matrix_path)
    records = []
    execution_shas = set()
    for run in expand_runs(matrix):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        run_dir = output_root / f"ms2d_{candidate_id}_s{seed}"
        manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics_validation.json")
        history = _read_json(run_dir / "history.json")
        checkpoint = run_dir / "checkpoint_best_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"required MS2-D1 checkpoint missing: {checkpoint}"
            )
        unauthorized = [
            name
            for name in (
                "metrics_test.json",
                "episode_metrics_test.json",
                "synthetic_test_access_ledger.json",
            )
            if (run_dir / name).exists()
        ]
        expected_manifest = {
            "protocol_version": matrix["protocol_version"],
            "evidence_scope": matrix["evidence_scope"],
            "route_id": candidate_id,
            "seed": seed,
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_selector": "validation_effect_mae",
            "matrix_sha256": matrix_sha,
            "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
            "test_accessed": False,
            "test_authorized": False,
        }
        regime, candidate = _select(matrix, candidate_id)
        operator, training, synthetic, _ = _build_configs(
            matrix, regime, candidate, False
        )
        expected_synthetic = replace(
            synthetic, seed=synthetic.seed + seed * 1_000_003
        )
        expected_manifest.update(
            operator_config=operator.to_dict(),
            training_config=asdict(training),
            synthetic_spec=asdict(expected_synthetic),
            regime_id=run["regime_id"],
            candidate_role=run["role"],
        )
        mismatches = [
            key
            for key, value in expected_manifest.items()
            if _canonical(manifest.get(key)) != _canonical(value)
        ]
        execution_sha = manifest.get("git_sha")
        if not execution_sha:
            mismatches.append("git_sha")
        else:
            execution_shas.add(execution_sha)
        if not isinstance(history, list) or not history:
            mismatches.append("history")
        best_epoch = manifest.get("best_epoch")
        if not isinstance(best_epoch, int) or not 1 <= best_epoch <= len(history):
            mismatches.append("best_epoch")
        truth = metrics.get("truth", {})
        expected_truth = {
            "truth_regime": matrix["synthetic_defaults"]["truth_regime"],
            "truth_opening_map": matrix["synthetic_defaults"][
                "truth_opening_map"
            ],
            "input_delay_steps": matrix["synthetic_defaults"][
                "input_delay_steps"
            ],
            "input_delay_seconds": (
                matrix["synthetic_defaults"]["input_delay_steps"]
                * matrix["synthetic_defaults"]["dt_seconds"]
            ),
        }
        for key, value in expected_truth.items():
            if truth.get(key) != value:
                mismatches.append(f"truth_{key}")
        if unauthorized:
            mismatches.append(f"unauthorized_test={unauthorized}")
        if mismatches:
            raise RuntimeError(
                f"MS2-D1 artifact mismatch for {candidate_id}/seed={seed}: "
                f"{sorted(set(mismatches))}"
            )
        gate_failures = _gate_metrics(metrics, run["route"])
        records.append(
            {
                **run,
                "effect_mae": float(metrics["effect_mae"]),
                "clean_effect_mae": float(metrics["clean_effect_mae"]),
                "clean_effect_nmae": float(metrics["clean_effect_nmae"]),
                "direction_accuracy_clean_nonzero": float(
                    metrics["direction_accuracy_clean_nonzero"]
                ),
                "metrics": metrics,
                "gate_failures": gate_failures,
            }
        )

    if len(execution_shas) != 1:
        raise RuntimeError(
            f"MS2-D1 manifests contain multiple execution SHAs: "
            f"{sorted(execution_shas)}"
        )
    execution_sha = next(iter(execution_shas))
    _assert_code_equivalent(execution_sha)

    by_id_seed = {
        (record["candidate_id"], record["seed"]): record for record in records
    }
    candidates = {}
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        subset = [
            record for record in records if record["candidate_id"] == candidate_id
        ]
        candidates[candidate_id] = {
            "role": subset[0]["role"],
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
            "all_structural_gates_pass": all(
                not record["gate_failures"] for record in subset
            ),
        }

    structural_failures = [
        {
            "candidate_id": record["candidate_id"],
            "seed": record["seed"],
            "failures": record["gate_failures"],
        }
        for record in records
        if record["gate_failures"]
    ]
    artifact_structural_pass = not structural_failures

    oracle_threshold = float(matrix["gates"]["oracle_clean_nmae_max"])
    oracle_seed_results = []
    response_threshold = float(
        matrix["gates"]["learned_delay_relative_improvement_min"]
    )
    response_seed_results = []
    delay_error_threshold = float(
        matrix["gates"]["delay_identification_error_steps_max"]
    )
    neighborhood_mass_min = float(
        matrix["gates"]["delay_truth_neighborhood_mass_min"]
    )
    delay_seed_results = []
    dt_seconds = float(matrix["synthetic_defaults"]["dt_seconds"])
    truth_delay_steps = int(matrix["synthetic_defaults"]["input_delay_steps"])
    for seed in sorted(int(value) for value in matrix["seeds"]):
        oracle_nmae = by_id_seed[(ORACLE_DELAY_ID, seed)]["clean_effect_nmae"]
        oracle_seed_results.append(
            {
                "seed": seed,
                "clean_effect_nmae": oracle_nmae,
                "passes": oracle_nmae < oracle_threshold,
            }
        )
        no_delay_nmae = by_id_seed[(NO_DELAY_ID, seed)]["clean_effect_nmae"]
        learned_nmae = by_id_seed[(LEARNED_DELAY_ID, seed)]["clean_effect_nmae"]
        relative_improvement = (no_delay_nmae - learned_nmae) / max(
            no_delay_nmae, 1e-12
        )
        response_seed_results.append(
            {
                "seed": seed,
                "no_delay_clean_effect_nmae": no_delay_nmae,
                "learned_delay_clean_effect_nmae": learned_nmae,
                "relative_improvement": relative_improvement,
                "passes": relative_improvement >= response_threshold,
            }
        )
        diagnostic = _delay_diagnostics(
            by_id_seed[(LEARNED_DELAY_ID, seed)]["metrics"],
            dt_seconds=dt_seconds,
            truth_delay_steps=truth_delay_steps,
            neighborhood_mass_min=neighborhood_mass_min,
        )
        diagnostic.update(
            seed=seed,
            within_one_step=diagnostic["absolute_error_steps"]
            <= delay_error_threshold,
            concentrated_near_truth=(
                diagnostic["truth_plus_minus_one_step_mass"]
                >= neighborhood_mass_min
            ),
        )
        delay_seed_results.append(diagnostic)

    oracle_pass = all(result["passes"] for result in oracle_seed_results)
    response_pass = all(result["passes"] for result in response_seed_results)
    delay_diagnostic_pass = all(
        result["within_one_step"] and result["concentrated_near_truth"]
        for result in delay_seed_results
    )
    primary_gate_pass = artifact_structural_pass and oracle_pass and response_pass
    return {
        "protocol_version": matrix["protocol_version"],
        "evidence_scope": matrix["evidence_scope"],
        "split": "validation",
        "run_count": len(records),
        "test_accessed": False,
        "execution_git_sha": execution_sha,
        "aggregation_git_sha": current_sha,
        "all_artifact_and_structural_gates_pass": artifact_structural_pass,
        "structural_gate_failures": structural_failures,
        "candidates": candidates,
        "oracle_gate": {
            "candidate_id": ORACLE_DELAY_ID,
            "clean_effect_nmae_max": oracle_threshold,
            "seed_results": oracle_seed_results,
            "all_seeds_pass": oracle_pass,
        },
        "delay_response_gate": {
            "learned_delay_id": LEARNED_DELAY_ID,
            "no_delay_id": NO_DELAY_ID,
            "relative_improvement_min": response_threshold,
            "seed_results": response_seed_results,
            "all_seeds_pass": response_pass,
        },
        "delay_identification_diagnostic": {
            "candidate_id": LEARNED_DELAY_ID,
            "error_steps_max": delay_error_threshold,
            "truth_plus_minus_one_step_mass_min": neighborhood_mass_min,
            "seed_results": delay_seed_results,
            "all_seeds_within_one_step": all(
                result["within_one_step"] for result in delay_seed_results
            ),
            "all_seeds_concentrated_near_truth": all(
                result["concentrated_near_truth"] for result in delay_seed_results
            ),
            "all_seeds_pass": delay_diagnostic_pass,
            "interpretation": (
                "Parameter recovery is a separate diagnostic; response-gate PASS does "
                "not require a uniquely identifiable delay distribution."
            ),
        },
        "all_primary_gates_pass": primary_gate_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/ms2d_delay_matrix.json"),
    )
    parser.add_argument("--output-root", default="results/phase3_5/ms2d_delay")
    parser.add_argument("--output")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    summary = build_summary(Path(args.matrix).resolve(), output_root)
    output = (
        Path(args.output).resolve()
        if args.output
        else output_root / "summary_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    if not summary["all_primary_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
