#!/usr/bin/env python3
"""Fail-closed aggregation for Phase 3.5-MS2-D2 validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.ms2d_order import (  # noqa: E402
    _assert_no_test_artifacts,
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


TWO_POLE_ID = "d2_g2_two_pole"
THREE_POLE_ID = "d2_g3_three_pole"
ORACLE_ID = "d2_g3_oracle_structure"
DELAY_DIAGNOSTIC_ID = "d2_g2_delay_compensation"


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required MS2-D2 artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _replay_best_epoch(history: list[dict]) -> int:
    best_score = float("inf")
    best_epoch = 0
    for expected_epoch, record in enumerate(history, start=1):
        if not isinstance(record, dict) or record.get("epoch") != expected_epoch:
            raise RuntimeError("history epochs must be contiguous and one-indexed")
        score = record.get("validation_effect_mae")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise RuntimeError("history contains an invalid validation_effect_mae")
        score = float(score)
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = expected_epoch
    if best_epoch == 0:
        raise RuntimeError("history produced no validation-selected epoch")
    return best_epoch


def _assert_code_equivalent(execution_sha: str) -> None:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{execution_sha}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"MS2-D2 execution commit unavailable: {execution_sha}")
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
        raise RuntimeError(f"MS2-D2 frozen execution code differs from {execution_sha}")
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking MS2-D2 code equivalence")


def _tau_set_diagnostic(metrics: dict, truth_tau: list[float], threshold: float) -> dict:
    observed = metrics.get("structural_diagnostics", {}).get("operator", {}).get(
        "tau_seconds"
    )
    if not isinstance(observed, list) or len(observed) != len(truth_tau):
        raise RuntimeError("three-pole run is missing a three-value tau diagnostic")
    numeric = sorted(float(value) for value in observed)
    expected = sorted(float(value) for value in truth_tau)
    if min(numeric) <= 0:
        raise RuntimeError("reported time constants must be positive")
    log_mae = statistics.mean(
        abs(math.log(value / truth)) for value, truth in zip(numeric, expected)
    )
    return {
        "reported_tau_seconds_sorted": numeric,
        "truth_tau_seconds_sorted": expected,
        "permutation_invariant_log_mae": log_mae,
        "log_mae_max": threshold,
        "passes": log_mae <= threshold,
    }


def _no_true_delay_diagnostic(
    metrics: dict,
    *,
    dt_seconds: float,
    expected_steps_max: float,
    zero_mass_min: float,
) -> dict:
    operator = metrics.get("structural_diagnostics", {}).get("operator", {})
    expected_seconds = operator.get("expected_delay_seconds")
    weights = operator.get("delay_weights")
    if expected_seconds is None or not isinstance(weights, list) or not weights:
        raise RuntimeError("delay-compensation run is missing delay diagnostics")
    numeric = [float(value) for value in weights]
    if any(value < 0 for value in numeric) or abs(sum(numeric) - 1.0) > 1e-5:
        raise RuntimeError("delay-compensation weights must be a probability vector")
    expected_steps = float(expected_seconds) / dt_seconds
    weighted_steps = sum(index * value for index, value in enumerate(numeric))
    if abs(expected_steps - weighted_steps) > 1e-5:
        raise RuntimeError("expected delay is inconsistent with delay weights")
    return {
        "delay_weights": numeric,
        "expected_delay_seconds": float(expected_seconds),
        "expected_delay_steps": expected_steps,
        "expected_delay_steps_max": expected_steps_max,
        "zero_step_mass": numeric[0],
        "zero_step_mass_min": zero_mass_min,
        "passes": expected_steps <= expected_steps_max and numeric[0] >= zero_mass_min,
    }


def build_summary(matrix_path: Path, output_root: Path) -> dict:
    matrix = load_matrix(matrix_path)
    _assert_no_test_artifacts(output_root)
    current_sha = _git_sha()
    matrix_sha = _sha256(matrix_path)
    records = []
    execution_shas = set()
    required_environment = {
        "python",
        "torch",
        "cuda_runtime",
        "cuda_available",
        "device",
        "platform",
    }
    truth_tau = [float(value) for value in matrix["synthetic_defaults"]["tau_seconds"]]
    for run in expand_runs(matrix):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        run_dir = output_root / f"ms2o_{candidate_id}_s{seed}"
        manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics_validation.json")
        history = _read_json(run_dir / "history.json")
        checkpoint = run_dir / "checkpoint_best_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required MS2-D2 checkpoint missing: {checkpoint}")
        unauthorized = [
            name
            for name in (
                "metrics_test.json",
                "episode_metrics_test.json",
                "synthetic_test_access_ledger.json",
            )
            if (run_dir / name).exists()
        ]
        regime, candidate = _select(matrix, candidate_id)
        operator, training, synthetic, _ = _build_configs(
            matrix, regime, candidate, False
        )
        expected_synthetic = replace(
            synthetic, seed=synthetic.seed + seed * 1_000_003
        )
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
            "operator_config": operator.to_dict(),
            "training_config": asdict(training),
            "synthetic_spec": asdict(expected_synthetic),
            "regime_id": run["regime_id"],
            "candidate_role": run["role"],
        }
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
        environment = manifest.get("environment")
        if not isinstance(environment, dict) or not required_environment <= set(environment):
            mismatches.append("environment")
        elif (
            not isinstance(manifest.get("device"), str)
            or environment.get("device") != manifest.get("device")
        ):
            mismatches.append("environment_device")
        if not isinstance(history, list) or not history:
            mismatches.append("history")
        else:
            best_epoch = manifest.get("best_epoch")
            if not isinstance(best_epoch, int) or not 1 <= best_epoch <= len(history):
                mismatches.append("best_epoch")
            elif _replay_best_epoch(history) != best_epoch:
                mismatches.append("best_epoch_replay")
        truth = metrics.get("truth", {})
        expected_truth = {
            "truth_regime": "context_scheduled",
            "truth_opening_map": "equal_percentage_r50",
            "tau_seconds": truth_tau,
            "input_delay_steps": 0,
            "input_delay_seconds": 0.0,
        }
        for key, value in expected_truth.items():
            if _canonical(truth.get(key)) != _canonical(value):
                mismatches.append(f"truth_{key}")
        if unauthorized:
            mismatches.append(f"unauthorized_test={unauthorized}")
        if mismatches:
            raise RuntimeError(
                f"MS2-D2 artifact mismatch for {candidate_id}/seed={seed}: "
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
            f"MS2-D2 manifests contain multiple execution SHAs: {sorted(execution_shas)}"
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
    absolute_threshold = float(matrix["gates"]["order_aware_clean_nmae_max"])
    response_threshold = float(
        matrix["gates"]["order_aware_relative_improvement_min"]
    )
    tau_threshold = float(matrix["gates"]["tau_set_log_mae_max"])
    delay_steps_max = float(matrix["gates"]["no_true_delay_expected_steps_max"])
    zero_mass_min = float(matrix["gates"]["no_true_delay_zero_step_mass_min"])
    dt_seconds = float(matrix["synthetic_defaults"]["dt_seconds"])
    oracle_seed_results = []
    absolute_seed_results = []
    response_seed_results = []
    tau_seed_results = []
    oracle_tau_seed_results = []
    delay_seed_results = []
    for seed in sorted(int(value) for value in matrix["seeds"]):
        oracle_nmae = by_id_seed[(ORACLE_ID, seed)]["clean_effect_nmae"]
        oracle_seed_results.append(
            {
                "seed": seed,
                "clean_effect_nmae": oracle_nmae,
                "passes": oracle_nmae < oracle_threshold,
            }
        )
        two_nmae = by_id_seed[(TWO_POLE_ID, seed)]["clean_effect_nmae"]
        three_nmae = by_id_seed[(THREE_POLE_ID, seed)]["clean_effect_nmae"]
        absolute_seed_results.append(
            {
                "seed": seed,
                "clean_effect_nmae": three_nmae,
                "passes": three_nmae < absolute_threshold,
            }
        )
        relative_improvement = (two_nmae - three_nmae) / max(two_nmae, 1e-12)
        response_seed_results.append(
            {
                "seed": seed,
                "two_pole_clean_effect_nmae": two_nmae,
                "three_pole_clean_effect_nmae": three_nmae,
                "relative_improvement": relative_improvement,
                "passes": relative_improvement >= response_threshold,
            }
        )
        tau_result = _tau_set_diagnostic(
            by_id_seed[(THREE_POLE_ID, seed)]["metrics"], truth_tau, tau_threshold
        )
        tau_result["seed"] = seed
        tau_seed_results.append(tau_result)
        oracle_tau_result = _tau_set_diagnostic(
            by_id_seed[(ORACLE_ID, seed)]["metrics"], truth_tau, tau_threshold
        )
        oracle_tau_result["seed"] = seed
        oracle_tau_seed_results.append(oracle_tau_result)
        delay_result = _no_true_delay_diagnostic(
            by_id_seed[(DELAY_DIAGNOSTIC_ID, seed)]["metrics"],
            dt_seconds=dt_seconds,
            expected_steps_max=delay_steps_max,
            zero_mass_min=zero_mass_min,
        )
        delay_result["seed"] = seed
        delay_seed_results.append(delay_result)

    oracle_pass = all(result["passes"] for result in oracle_seed_results)
    absolute_pass = all(result["passes"] for result in absolute_seed_results)
    response_pass = all(result["passes"] for result in response_seed_results)
    primary_gate_pass = (
        artifact_structural_pass and oracle_pass and absolute_pass and response_pass
    )
    return {
        "protocol_version": matrix["protocol_version"],
        "evidence_scope": matrix["evidence_scope"],
        "split": "validation",
        "run_count": len(records),
        "test_accessed": False,
        "execution_git_sha": execution_sha,
        "aggregation_git_sha": current_sha,
        "matrix_sha256": matrix_sha,
        "all_artifact_and_structural_gates_pass": artifact_structural_pass,
        "structural_gate_failures": structural_failures,
        "candidates": candidates,
        "oracle_gate": {
            "candidate_id": ORACLE_ID,
            "clean_effect_nmae_max": oracle_threshold,
            "seed_results": oracle_seed_results,
            "all_seeds_pass": oracle_pass,
        },
        "order_aware_absolute_gate": {
            "candidate_id": THREE_POLE_ID,
            "clean_effect_nmae_max": absolute_threshold,
            "seed_results": absolute_seed_results,
            "all_seeds_pass": absolute_pass,
        },
        "order_aware_response_gate": {
            "order_aware_id": THREE_POLE_ID,
            "two_pole_id": TWO_POLE_ID,
            "relative_improvement_min": response_threshold,
            "seed_results": response_seed_results,
            "all_seeds_pass": response_pass,
        },
        "tau_recovery_diagnostic": {
            "candidate_results": {
                THREE_POLE_ID: tau_seed_results,
                ORACLE_ID: oracle_tau_seed_results,
            },
            "all_seeds_pass": all(
                result["passes"]
                for result in tau_seed_results + oracle_tau_seed_results
            ),
            "interpretation": (
                "Permutation-invariant parameter recovery is diagnostic only and "
                "does not enter the primary response gate."
            ),
        },
        "no_true_delay_diagnostic": {
            "candidate_id": DELAY_DIAGNOSTIC_ID,
            "seed_results": delay_seed_results,
            "all_seeds_pass": all(result["passes"] for result in delay_seed_results),
            "interpretation": (
                "This asks whether a misspecified two-pole model invents delay to "
                "compensate for an omitted pole; it cannot establish field delay."
            ),
        },
        "all_primary_gates_pass": primary_gate_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/ms2d_order_matrix.json"),
    )
    parser.add_argument("--output-root", default="results/phase3_5/ms2d_order")
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
