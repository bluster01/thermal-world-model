#!/usr/bin/env python3
"""Audit MS2-D1 synthetic test artifacts and apply paired-episode gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.ms2d_delay import expand_runs, load_matrix  # noqa: E402
from experiments.phase3_5.ms2d_delay_test import (  # noqa: E402
    DEFAULT_AUTHORIZATION,
    ROOT_LEDGER_NAME,
    RUN_LEDGER_NAME,
    _assert_pinned,
    _member_bytes,
    _read_json,
    _resolve_repo_path,
    _sha256,
    load_authorization,
)
NO_DELAY_ID = "d1_g2_no_delay"
LEARNED_DELAY_ID = "d1_g2_learned_delay"
ORACLE_DELAY_ID = "d1_g2_oracle_delay"


def _gate_metrics(metrics: dict[str, Any], route: str) -> list[str]:
    """Apply the frozen structural gates without legacy import-path state."""

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


def _delay_diagnostics(
    metrics: dict[str, Any],
    *,
    dt_seconds: float,
    truth_delay_steps: int,
    neighborhood_mass_min: float,
) -> dict[str, Any]:
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


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_strata(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
) -> dict[int, list[int]]:
    if not (
        len(candidate_error) == len(baseline_error) == len(profile_ids)
        and candidate_error
    ):
        raise ValueError("paired bootstrap arrays must have the same non-zero length")
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in (*candidate_error, *baseline_error)
    ):
        raise ValueError("paired bootstrap errors must be finite and non-negative")
    if any(not isinstance(value, int) for value in profile_ids):
        raise ValueError("paired bootstrap profile IDs must be integers")
    strata = {
        profile: [index for index, value in enumerate(profile_ids) if value == profile]
        for profile in sorted(set(profile_ids))
    }
    if not strata or any(not indices for indices in strata.values()):
        raise ValueError("paired bootstrap profile stratum is empty")
    return strata


def paired_stratified_bootstrap_relative_improvement(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired relative-MAE improvement within action profiles."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    strata = _bootstrap_strata(candidate_error, baseline_error, profile_ids)
    baseline_mean = statistics.mean(baseline_error)
    if baseline_mean <= 1e-12:
        raise ValueError("paired bootstrap baseline mean is effectively zero")
    candidate_mean = statistics.mean(candidate_error)
    observed = (baseline_mean - candidate_mean) / baseline_mean
    generator = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = []
        for indices in strata.values():
            sampled.extend(generator.choices(indices, k=len(indices)))
        sampled_baseline = statistics.mean(baseline_error[index] for index in sampled)
        if sampled_baseline <= 1e-12:
            raise ValueError("bootstrap draw has an effectively zero baseline mean")
        sampled_candidate = statistics.mean(candidate_error[index] for index in sampled)
        draws.append((sampled_baseline - sampled_candidate) / sampled_baseline)
    return {
        "observed": observed,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired_episode_stratified_by_action_profile",
        "episode_count": len(candidate_error),
        "profile_counts": {
            str(profile): len(indices) for profile, indices in strata.items()
        },
    }


def _assert_paired(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for key in ("episode_ids", "profile_ids", "trajectory_design_sha256"):
        if left.get(key) != right.get(key):
            raise RuntimeError(f"unpaired MS2-D1 test episodes for {label}: {key}")


def build_confirmatory_gates(
    episode_records: dict[tuple[str, int], dict[str, Any]],
    seeds: list[int],
    *,
    replicates: int,
    bootstrap_seed: int,
    response_ci_lower_min: float,
    oracle_nmae_max: float,
    delay_parameter_diagnostic_pass: bool,
) -> dict[str, Any]:
    """Build prespecified test gates; parameter recovery remains diagnostic."""

    response_results = []
    oracle_results = []
    for seed in sorted(int(value) for value in seeds):
        learned = episode_records[(LEARNED_DELAY_ID, seed)]
        no_delay = episode_records[(NO_DELAY_ID, seed)]
        oracle = episode_records[(ORACLE_DELAY_ID, seed)]
        _assert_paired(learned, no_delay, f"learned/no-delay/seed={seed}")
        _assert_paired(learned, oracle, f"learned/oracle/seed={seed}")
        response = paired_stratified_bootstrap_relative_improvement(
            learned["clean_effect_mae"],
            no_delay["clean_effect_mae"],
            learned["profile_ids"],
            replicates=replicates,
            seed=bootstrap_seed + seed,
        )
        response.update(
            seed=seed,
            ci_lower_meets_threshold=response["ci95"][0]
            >= response_ci_lower_min,
        )
        response_results.append(response)
        oracle_error = statistics.mean(oracle["clean_effect_mae"])
        oracle_scale = statistics.mean(oracle["clean_effect_scale"])
        if oracle_scale <= 1e-12:
            raise RuntimeError(f"oracle clean-effect scale is zero for seed={seed}")
        oracle_nmae = oracle_error / oracle_scale
        oracle_results.append(
            {
                "seed": seed,
                "clean_effect_nmae": oracle_nmae,
                "passes": oracle_nmae < oracle_nmae_max,
            }
        )
    oracle_gate = {
        "candidate_id": ORACLE_DELAY_ID,
        "clean_effect_nmae_max": oracle_nmae_max,
        "seed_results": oracle_results,
        "all_seeds_pass": all(result["passes"] for result in oracle_results),
    }
    response_gate = {
        "candidate_id": LEARNED_DELAY_ID,
        "baseline_id": NO_DELAY_ID,
        "ci_lower_relative_improvement_min": response_ci_lower_min,
        "seed_results": response_results,
        "all_seeds_pass": all(
            result["ci_lower_meets_threshold"] for result in response_results
        ),
    }
    return {
        "oracle_test": oracle_gate,
        "delay_response_test": response_gate,
        "delay_parameter_diagnostic_pass": delay_parameter_diagnostic_pass,
        "all_confirmatory_gates_pass": (
            oracle_gate["all_seeds_pass"] and response_gate["all_seeds_pass"]
        ),
    }


def _episode_integrity_failures(
    episodes: dict[str, Any], expected_count: int
) -> list[str]:
    failures = []
    if episodes.get("episode_ids") != list(range(expected_count)):
        failures.append("episode_ids")
    for key in (
        "profile_ids",
        "observed_effect_mae",
        "clean_effect_mae",
        "clean_effect_scale",
    ):
        values = episodes.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            failures.append(key)
            continue
        if key != "profile_ids" and any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            failures.append(f"{key}_values")
    profile_ids = episodes.get("profile_ids", [])
    if isinstance(profile_ids, list) and (
        set(profile_ids) != {0, 1, 2, 3, 4}
        or any(not isinstance(value, int) for value in profile_ids)
    ):
        failures.append("profile_coverage")
    trajectory_hash = episodes.get("trajectory_design_sha256")
    if not isinstance(trajectory_hash, str) or len(trajectory_hash) != 64:
        failures.append("trajectory_design_sha256")
    horizons = episodes.get("clean_horizon_absolute_error")
    if not isinstance(horizons, dict) or set(horizons) != {
        "H1",
        "H6",
        "H18",
        "H60",
    }:
        failures.append("clean_horizons")
    else:
        for key, values in horizons.items():
            if (
                not isinstance(values, list)
                or len(values) != expected_count
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    for value in values
                )
            ):
                failures.append(f"{key}_values")
    return failures


def _parameter_diagnostic(
    records: list[dict[str, Any]], matrix: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, Any]:
    gates = authorization["gates"]
    error_max = float(gates["delay_identification_error_steps_max"])
    mass_min = float(gates["delay_truth_neighborhood_mass_min"])
    dt_seconds = float(matrix["synthetic_defaults"]["dt_seconds"])
    truth_steps = int(matrix["synthetic_defaults"]["input_delay_steps"])
    results = []
    for record in records:
        if record["candidate_id"] != LEARNED_DELAY_ID:
            continue
        diagnostic = _delay_diagnostics(
            record["metrics"],
            dt_seconds=dt_seconds,
            truth_delay_steps=truth_steps,
            neighborhood_mass_min=mass_min,
        )
        diagnostic.update(
            seed=int(record["seed"]),
            within_one_step=diagnostic["absolute_error_steps"] <= error_max,
            concentrated_near_truth=diagnostic["truth_plus_minus_one_step_mass"]
            >= mass_min,
        )
        results.append(diagnostic)
    passed = all(
        result["within_one_step"] and result["concentrated_near_truth"]
        for result in results
    )
    return {
        "candidate_id": LEARNED_DELAY_ID,
        "error_steps_max": error_max,
        "truth_plus_minus_one_step_mass_min": mass_min,
        "seed_results": results,
        "all_seeds_pass": passed,
        "interpretation": (
            "Parameter recovery is diagnostic only and is not part of the response "
            "confirmation gate."
        ),
    }


def build_test_summary(authorization_path: str | Path) -> dict[str, Any]:
    authorization_path = Path(authorization_path).resolve()
    authorization = load_authorization(authorization_path)
    matrix_path = _resolve_repo_path(authorization["matrix"]["path"])
    validation_path = _resolve_repo_path(
        authorization["validation_summary"]["path"]
    )
    archive_path = _resolve_repo_path(authorization["checkpoint_archive"]["path"])
    _assert_pinned(matrix_path, authorization["matrix"]["sha256"], "matrix")
    _assert_pinned(
        validation_path,
        authorization["validation_summary"]["sha256"],
        "validation summary",
    )
    _assert_pinned(
        archive_path,
        authorization["checkpoint_archive"]["sha256"],
        "checkpoint archive",
    )
    matrix = load_matrix(matrix_path)
    if matrix["protocol_version"] != authorization["training_protocol_version"]:
        raise RuntimeError("training matrix protocol does not match test authorization")
    output_root = validation_path.parent
    root_ledger = _read_json(output_root / ROOT_LEDGER_NAME)
    if root_ledger.get("status") != "completed":
        raise RuntimeError("MS2-D1 matrix test ledger is not completed")
    expected_root = {
        "protocol_version": authorization["protocol_version"],
        "evidence_scope": authorization["evidence_scope"],
        "matrix_sha256": authorization["matrix"]["sha256"],
        "validation_summary_sha256": authorization["validation_summary"]["sha256"],
        "checkpoint_archive_sha256": authorization["checkpoint_archive"]["sha256"],
        "run_count": int(authorization["expected_run_count"]),
        "test_samples": int(authorization["test_samples"]),
        "authorization_sha256": _sha256(authorization_path),
    }
    root_mismatches = [
        key for key, value in expected_root.items() if root_ledger.get(key) != value
    ]
    runs = expand_runs(matrix)
    expected_completed = [
        {"candidate_id": run["candidate_id"], "seed": int(run["seed"])}
        for run in runs
    ]
    if root_ledger.get("completed_runs") != expected_completed:
        root_mismatches.append("completed_runs")
    environment = root_ledger.get("environment", {})
    for key in ("python", "torch", "cuda_runtime", "cuda_available", "device", "platform"):
        if key not in environment:
            root_mismatches.append(f"environment_{key}")
    if root_mismatches:
        raise RuntimeError(f"MS2-D1 matrix test ledger mismatch: {root_mismatches}")

    records = []
    episode_records: dict[tuple[str, int], dict[str, Any]] = {}
    expected_count = int(authorization["test_samples"])
    expected_members = {
        f"ms2d_{run['candidate_id']}_s{int(run['seed'])}/checkpoint_best_val.pt"
        for run in runs
    }
    with tarfile.open(archive_path, "r") as archive:
        observed_members = {
            member.name.replace("\\", "/")
            for member in archive.getmembers()
            if member.isfile()
        }
        if observed_members != expected_members:
            raise RuntimeError("MS2-D1 checkpoint archive members differ from frozen runs")
        for run in runs:
            candidate_id, seed = run["candidate_id"], int(run["seed"])
            run_dir = output_root / f"ms2d_{candidate_id}_s{seed}"
            manifest = _read_json(run_dir / "manifest.json")
            metrics = _read_json(run_dir / "metrics_test.json")
            episodes = _read_json(run_dir / "episode_metrics_test.json")
            ledger = _read_json(run_dir / RUN_LEDGER_NAME)
            member_name = f"{run_dir.name}/checkpoint_best_val.pt"
            _member_bytes(archive, member_name, manifest["checkpoint_sha256"])
            expected_manifest = {
                "protocol_version": matrix["protocol_version"],
                "route_id": candidate_id,
                "seed": seed,
                "test_accessed": True,
                "test_authorized": True,
                "test_access_note": "synthetic_delay_known_truth_only",
                "test_access_ledger": RUN_LEDGER_NAME,
                "test_episode_metrics": "episode_metrics_test.json",
            }
            mismatches = [
                key
                for key, value in expected_manifest.items()
                if manifest.get(key) != value
            ]
            expected_ledger = {
                "protocol_version": authorization["protocol_version"],
                "status": "completed",
                "evidence_scope": authorization["evidence_scope"],
                "candidate_id": candidate_id,
                "regime_id": run["regime_id"],
                "seed": seed,
                "training_git_sha": manifest.get("git_sha"),
                "evaluation_git_sha": root_ledger.get("evaluation_git_sha"),
                "checkpoint_archive_sha256": authorization["checkpoint_archive"]["sha256"],
                "checkpoint_member": member_name,
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "checkpoint_selector": "validation_effect_mae",
                "test_samples": expected_count,
                "trajectory_design_sha256": episodes.get("trajectory_design_sha256"),
            }
            mismatches.extend(
                f"ledger_{key}"
                for key, value in expected_ledger.items()
                if ledger.get(key) != value
            )
            mismatches.extend(_episode_integrity_failures(episodes, expected_count))
            if metrics.get("sample_count") != expected_count:
                mismatches.append("metrics_sample_count")
            if metrics.get("truth", {}).get("split") != "test":
                mismatches.append("metrics_truth_split")
            if mismatches:
                raise RuntimeError(
                    f"MS2-D1 test artifact mismatch for {candidate_id}/seed={seed}: "
                    f"{sorted(set(mismatches))}"
                )
            failures = _gate_metrics(metrics, run["route"])
            records.append(
                {
                    **run,
                    "metrics": metrics,
                    "effect_mae": float(metrics["effect_mae"]),
                    "clean_effect_mae": float(metrics["clean_effect_mae"]),
                    "clean_effect_nmae": float(metrics["clean_effect_nmae"]),
                    "gate_failures": failures,
                }
            )
            episode_records[(candidate_id, seed)] = episodes

    seed_hashes = []
    for seed in sorted(int(value) for value in matrix["seeds"]):
        hashes = {
            episode_records[(candidate["candidate_id"], seed)][
                "trajectory_design_sha256"
            ]
            for candidate in matrix["regimes"][0]["candidates"]
        }
        if len(hashes) != 1:
            raise RuntimeError(f"MS2-D1 candidates have unpaired trajectories seed={seed}")
        seed_hashes.append(next(iter(hashes)))
    if len(set(seed_hashes)) != len(seed_hashes):
        raise RuntimeError("MS2-D1 test seeds reuse the same trajectory design")

    candidates = {}
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        subset = [record for record in records if record["candidate_id"] == candidate_id]
        candidates[candidate_id] = {
            "role": subset[0]["role"],
            "route": subset[0]["route"],
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
            "all_structural_gates_pass": all(
                not record["gate_failures"] for record in subset
            ),
        }
    failures = [
        {
            "candidate_id": record["candidate_id"],
            "seed": record["seed"],
            "failures": record["gate_failures"],
        }
        for record in records
        if record["gate_failures"]
    ]
    parameter = _parameter_diagnostic(records, matrix, authorization)
    gates = authorization["gates"]
    confirmatory = build_confirmatory_gates(
        episode_records,
        [int(value) for value in matrix["seeds"]],
        replicates=int(authorization["bootstrap"]["replicates"]),
        bootstrap_seed=int(authorization["bootstrap"]["seed"]),
        response_ci_lower_min=float(gates["delay_response_ci_lower_min"]),
        oracle_nmae_max=float(gates["oracle_clean_nmae_max"]),
        delay_parameter_diagnostic_pass=parameter["all_seeds_pass"],
    )
    return {
        "protocol_version": authorization["protocol_version"],
        "evidence_scope": authorization["evidence_scope"],
        "split": "test",
        "run_count": len(records),
        "test_accessed": True,
        "frozen_validation_status": authorization["frozen_validation_status"],
        "all_artifact_and_structural_gates_pass": not failures,
        "gate_failures": failures,
        "candidates": candidates,
        "delay_identification_diagnostic": parameter,
        **confirmatory,
        "all_gates_pass": not failures
        and confirmatory["all_confirmatory_gates_pass"],
        "interpretation_rule": (
            "This one-shot known-truth synthetic test confirms or challenges the "
            "frozen D1 response result; it neither authorizes retraining nor supports "
            "field-causal or unique-delay-identification claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output")
    args = parser.parse_args()
    authorization_path = Path(args.authorization).resolve()
    if authorization_path != DEFAULT_AUTHORIZATION.resolve():
        raise SystemExit("formal MS2-D1 test summary requires repository authorization")
    summary = build_test_summary(authorization_path)
    authorization = load_authorization(authorization_path)
    output_root = _resolve_repo_path(
        authorization["validation_summary"]["path"]
    ).parent
    output = (
        Path(args.output).resolve()
        if args.output
        else output_root / "summary_test.json"
    )
    if output != (output_root / "summary_test.json").resolve():
        raise SystemExit("formal MS2-D1 test summary output path is frozen")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    if not summary["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
