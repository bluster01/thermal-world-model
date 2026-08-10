#!/usr/bin/env python3
"""Audit MS2-D2 synthetic test artifacts and apply paired-episode gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE_DIR = ROOT / "experiments/phase3_5"
for path in (ROOT, PHASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.phase3_5.ms2d_order import (  # noqa: E402
    FROZEN_EXECUTION_PATHS,
    expand_runs,
    load_matrix,
)
from experiments.phase3_5.ms2d_order_test import (  # noqa: E402
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


TWO_POLE_ID = "d2_g2_two_pole"
THREE_POLE_ID = "d2_g3_three_pole"
ORACLE_ID = "d2_g3_oracle_structure"
DELAY_DIAGNOSTIC_ID = "d2_g2_delay_compensation"
TEST_EXECUTION_PATHS = (
    "configs/phase3_5/ms2d_order_test_authorization.json",
    "experiments/phase3_5/ms2d_order_test.py",
    "experiments/phase3_5/summarize_ms2d_order_test.py",
    "experiments/phase3_5/ms2d_delay_test.py",
    "experiments/phase3_5/multistep_mismatch.py",
) + tuple(FROZEN_EXECUTION_PATHS)


def _gate_metrics(metrics: dict[str, Any], route: str) -> list[str]:
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
        radius = operator.get("spectral_radius")
        if radius is None or not 0 <= radius < 1:
            failures.append("spectral_radius")
    for key in (
        "effect_mae",
        "clean_effect_mae",
        "clean_effect_nmae",
        "direction_accuracy_clean_nonzero",
    ):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"invalid_{key}")
    return failures


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_stratified_bootstrap_relative_improvement(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not (
        candidate_error
        and len(candidate_error) == len(baseline_error) == len(profile_ids)
    ):
        raise ValueError("paired bootstrap arrays must have the same non-zero length")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in (*candidate_error, *baseline_error)
    ):
        raise ValueError("paired bootstrap errors must be finite and non-negative")
    if any(not isinstance(value, int) for value in profile_ids):
        raise ValueError("paired bootstrap profile IDs must be integers")
    strata = {
        profile: [i for i, observed in enumerate(profile_ids) if observed == profile]
        for profile in sorted(set(profile_ids))
    }
    if not strata or any(not indices for indices in strata.values()):
        raise ValueError("paired bootstrap profile stratum is empty")
    baseline_mean = statistics.fmean(baseline_error)
    if baseline_mean <= 1e-12:
        raise ValueError("paired bootstrap baseline mean is effectively zero")
    observed = (baseline_mean - statistics.fmean(candidate_error)) / baseline_mean
    generator = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = []
        for indices in strata.values():
            sampled.extend(generator.choices(indices, k=len(indices)))
        sampled_baseline = statistics.fmean(baseline_error[i] for i in sampled)
        if sampled_baseline <= 1e-12:
            raise ValueError("bootstrap draw has an effectively zero baseline mean")
        sampled_candidate = statistics.fmean(candidate_error[i] for i in sampled)
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
            raise RuntimeError(f"unpaired MS2-D2 test episodes for {label}: {key}")


def _episode_nmae(episodes: dict[str, Any], label: str) -> float:
    error = statistics.fmean(episodes["clean_effect_mae"])
    scale = statistics.fmean(episodes["clean_effect_scale"])
    if scale <= 1e-12:
        raise RuntimeError(f"{label} clean-effect scale is effectively zero")
    return error / scale


def build_confirmatory_gates(
    episode_records: dict[tuple[str, int], dict[str, Any]],
    seeds: list[int],
    *,
    replicates: int,
    bootstrap_seed: int,
    response_ci_lower_min: float,
    oracle_nmae_max: float,
    order_aware_nmae_max: float,
    tau_diagnostic_pass: bool,
    no_true_delay_diagnostic_pass: bool,
) -> dict[str, Any]:
    """Apply the three preregistered response gates; diagnostics stay nonblocking."""

    response_results = []
    oracle_results = []
    absolute_results = []
    for seed in sorted(int(value) for value in seeds):
        three = episode_records[(THREE_POLE_ID, seed)]
        two = episode_records[(TWO_POLE_ID, seed)]
        oracle = episode_records[(ORACLE_ID, seed)]
        _assert_paired(three, two, f"three/two/seed={seed}")
        _assert_paired(three, oracle, f"three/oracle/seed={seed}")
        response = paired_stratified_bootstrap_relative_improvement(
            three["clean_effect_mae"],
            two["clean_effect_mae"],
            three["profile_ids"],
            replicates=replicates,
            seed=bootstrap_seed + seed,
        )
        response.update(
            seed=seed,
            ci_lower_meets_threshold=response["ci95"][0]
            >= response_ci_lower_min,
        )
        response_results.append(response)
        oracle_nmae = _episode_nmae(oracle, f"oracle/seed={seed}")
        order_nmae = _episode_nmae(three, f"three-pole/seed={seed}")
        oracle_results.append(
            {
                "seed": seed,
                "clean_effect_nmae": oracle_nmae,
                "passes": oracle_nmae < oracle_nmae_max,
            }
        )
        absolute_results.append(
            {
                "seed": seed,
                "clean_effect_nmae": order_nmae,
                "passes": order_nmae < order_aware_nmae_max,
            }
        )
    oracle_gate = {
        "candidate_id": ORACLE_ID,
        "clean_effect_nmae_max": oracle_nmae_max,
        "seed_results": oracle_results,
        "all_seeds_pass": all(item["passes"] for item in oracle_results),
    }
    absolute_gate = {
        "candidate_id": THREE_POLE_ID,
        "clean_effect_nmae_max": order_aware_nmae_max,
        "seed_results": absolute_results,
        "all_seeds_pass": all(item["passes"] for item in absolute_results),
    }
    response_gate = {
        "order_aware_id": THREE_POLE_ID,
        "two_pole_id": TWO_POLE_ID,
        "ci_lower_relative_improvement_min": response_ci_lower_min,
        "seed_results": response_results,
        "all_seeds_pass": all(
            item["ci_lower_meets_threshold"] for item in response_results
        ),
    }
    return {
        "oracle_test": oracle_gate,
        "order_aware_absolute_test": absolute_gate,
        "order_aware_response_test": response_gate,
        "tau_recovery_diagnostic_pass": tau_diagnostic_pass,
        "no_true_delay_diagnostic_pass": no_true_delay_diagnostic_pass,
        "all_confirmatory_gates_pass": (
            oracle_gate["all_seeds_pass"]
            and absolute_gate["all_seeds_pass"]
            and response_gate["all_seeds_pass"]
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
            or not math.isfinite(float(value))
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
    if not isinstance(horizons, dict) or set(horizons) != {"H1", "H6", "H18", "H60"}:
        failures.append("clean_horizons")
    else:
        for key, values in horizons.items():
            if (
                not isinstance(values, list)
                or len(values) != expected_count
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                    for value in values
                )
            ):
                failures.append(f"{key}_values")
    return failures


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-7)


def _episode_metric_failures(
    episodes: dict[str, Any], metrics: dict[str, Any]
) -> list[str]:
    failures = []
    comparisons = {
        "effect_mae": statistics.fmean(episodes["observed_effect_mae"]),
        "clean_effect_mae": statistics.fmean(episodes["clean_effect_mae"]),
        "clean_effect_scale": statistics.fmean(episodes["clean_effect_scale"]),
    }
    for key, replayed in comparisons.items():
        if not _close(replayed, metrics.get(key)):
            failures.append(f"episode_replay_{key}")
    if comparisons["clean_effect_scale"] > 1e-12:
        replayed_nmae = (
            comparisons["clean_effect_mae"] / comparisons["clean_effect_scale"]
        )
        if not _close(replayed_nmae, metrics.get("clean_effect_nmae")):
            failures.append("episode_replay_clean_effect_nmae")
    for horizon, values in episodes["clean_horizon_absolute_error"].items():
        observed = metrics.get("clean_horizon_mae", {}).get(horizon)
        if not _close(statistics.fmean(values), observed):
            failures.append(f"episode_replay_{horizon}")
    return failures


def _tau_set_diagnostic(metrics: dict[str, Any], truth_tau: list[float], threshold: float) -> dict[str, Any]:
    observed = metrics.get("structural_diagnostics", {}).get("operator", {}).get(
        "tau_seconds"
    )
    if not isinstance(observed, list) or len(observed) != len(truth_tau):
        raise RuntimeError("three-pole run is missing a three-value tau diagnostic")
    numeric = sorted(float(value) for value in observed)
    expected = sorted(float(value) for value in truth_tau)
    if min(numeric) <= 0:
        raise RuntimeError("reported time constants must be positive")
    log_mae = statistics.fmean(
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
    metrics: dict[str, Any],
    *,
    dt_seconds: float,
    expected_steps_max: float,
    zero_mass_min: float,
) -> dict[str, Any]:
    operator = metrics.get("structural_diagnostics", {}).get("operator", {})
    expected_seconds = operator.get("expected_delay_seconds")
    weights = operator.get("delay_weights")
    if expected_seconds is None or not isinstance(weights, list) or not weights:
        raise RuntimeError("delay-compensation run is missing delay diagnostics")
    numeric = [float(value) for value in weights]
    if any(value < 0 for value in numeric) or abs(sum(numeric) - 1) > 1e-5:
        raise RuntimeError("delay-compensation weights must be a probability vector")
    expected_steps = float(expected_seconds) / dt_seconds
    if abs(expected_steps - sum(i * value for i, value in enumerate(numeric))) > 1e-5:
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


def _paired_heterogeneity_diagnostics(
    episode_records: dict[tuple[str, int], dict[str, Any]], seeds: list[int]
) -> dict[str, Any]:
    """Describe prespecified profile/horizon variation without adding gates."""

    profile_results = []
    horizon_results = []
    for seed in sorted(int(value) for value in seeds):
        three = episode_records[(THREE_POLE_ID, seed)]
        two = episode_records[(TWO_POLE_ID, seed)]
        _assert_paired(three, two, f"heterogeneity/seed={seed}")
        names = three.get("profile_names", [])
        for profile_id in sorted(set(three["profile_ids"])):
            indices = [
                index
                for index, value in enumerate(three["profile_ids"])
                if value == profile_id
            ]
            candidate = statistics.fmean(
                three["clean_effect_mae"][index] for index in indices
            )
            baseline = statistics.fmean(
                two["clean_effect_mae"][index] for index in indices
            )
            profile_results.append(
                {
                    "seed": seed,
                    "profile_id": profile_id,
                    "profile_name": (
                        names[profile_id]
                        if isinstance(names, list) and profile_id < len(names)
                        else None
                    ),
                    "episode_count": len(indices),
                    "three_pole_clean_effect_mae": candidate,
                    "two_pole_clean_effect_mae": baseline,
                    "relative_improvement": (
                        (baseline - candidate) / baseline
                        if baseline > 1e-12
                        else None
                    ),
                }
            )
        for horizon in sorted(
            three["clean_horizon_absolute_error"],
            key=lambda value: int(value[1:]),
        ):
            candidate = statistics.fmean(
                three["clean_horizon_absolute_error"][horizon]
            )
            baseline = statistics.fmean(
                two["clean_horizon_absolute_error"][horizon]
            )
            horizon_results.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "three_pole_clean_effect_mae": candidate,
                    "two_pole_clean_effect_mae": baseline,
                    "relative_improvement": (
                        (baseline - candidate) / baseline
                        if baseline > 1e-12
                        else None
                    ),
                }
            )
    return {
        "by_action_profile": profile_results,
        "by_horizon": horizon_results,
        "interpretation": (
            "Prespecified descriptive heterogeneity only; no profile or horizon "
            "creates an additional confirmatory gate."
        ),
    }


def _assert_test_code_equivalent(evaluation_sha: str) -> None:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{evaluation_sha}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"MS2-D2 evaluation commit unavailable: {evaluation_sha}")
    compared = subprocess.run(
        ["git", "diff", "--quiet", evaluation_sha, "HEAD", "--", *TEST_EXECUTION_PATHS],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(f"MS2-D2 test code differs from evaluation commit {evaluation_sha}")
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking MS2-D2 test code equivalence")


def build_test_summary(authorization_path: str | Path) -> dict[str, Any]:
    authorization_path = Path(authorization_path).resolve()
    authorization = load_authorization(authorization_path)
    matrix_path = _resolve_repo_path(authorization["matrix"]["path"])
    validation_path = _resolve_repo_path(authorization["validation_summary"]["path"])
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
    validation = _read_json(validation_path)
    output_root = validation_path.parent
    root_ledger = _read_json(output_root / ROOT_LEDGER_NAME)
    if root_ledger.get("status") != "completed":
        raise RuntimeError("MS2-D2 matrix test ledger is not completed")
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
    evaluation_sha = root_ledger.get("evaluation_git_sha")
    if not isinstance(evaluation_sha, str):
        root_mismatches.append("evaluation_git_sha")
    if root_mismatches:
        raise RuntimeError(f"MS2-D2 matrix test ledger mismatch: {root_mismatches}")
    _assert_test_code_equivalent(evaluation_sha)

    records = []
    episode_records: dict[tuple[str, int], dict[str, Any]] = {}
    expected_count = int(authorization["test_samples"])
    expected_members = {
        f"ms2o_{run['candidate_id']}_s{int(run['seed'])}/checkpoint_best_val.pt"
        for run in runs
    }
    with tarfile.open(archive_path, "r") as archive:
        observed_members = {
            member.name.replace("\\", "/")
            for member in archive.getmembers()
            if member.isfile()
        }
        if observed_members != expected_members:
            raise RuntimeError("MS2-D2 checkpoint archive members differ from frozen runs")
        for run in runs:
            candidate_id, seed = run["candidate_id"], int(run["seed"])
            run_dir = output_root / f"ms2o_{candidate_id}_s{seed}"
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
                "test_access_note": "synthetic_order_known_truth_only",
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
                "evaluation_git_sha": evaluation_sha,
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
            integrity = _episode_integrity_failures(episodes, expected_count)
            mismatches.extend(integrity)
            if not integrity:
                mismatches.extend(_episode_metric_failures(episodes, metrics))
            if metrics.get("sample_count") != expected_count:
                mismatches.append("metrics_sample_count")
            if metrics.get("truth", {}).get("split") != "test":
                mismatches.append("metrics_truth_split")
            if mismatches:
                raise RuntimeError(
                    f"MS2-D2 test artifact mismatch for {candidate_id}/seed={seed}: "
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
            raise RuntimeError(f"MS2-D2 candidates have unpaired trajectories seed={seed}")
        seed_hashes.append(next(iter(hashes)))
    if len(set(seed_hashes)) != len(seed_hashes):
        raise RuntimeError("MS2-D2 test seeds reuse the same trajectory design")

    candidates = {}
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        subset = [record for record in records if record["candidate_id"] == candidate_id]
        candidates[candidate_id] = {
            "role": subset[0]["role"],
            "route": subset[0]["route"],
            "clean_effect_nmae_mean": statistics.fmean(
                record["clean_effect_nmae"] for record in subset
            ),
            "clean_effect_nmae_std": statistics.stdev(
                record["clean_effect_nmae"] for record in subset
            ),
            "clean_effect_mae_mean": statistics.fmean(
                record["clean_effect_mae"] for record in subset
            ),
            "effect_mae_mean": statistics.fmean(record["effect_mae"] for record in subset),
            "all_structural_gates_pass": all(
                not record["gate_failures"] for record in subset
            ),
        }
    validation_candidates = validation.get("candidates", {})
    validation_to_test_drift = {}
    for candidate_id, test_values in candidates.items():
        validation_nmae = validation_candidates.get(candidate_id, {}).get(
            "clean_effect_nmae_mean"
        )
        if not isinstance(validation_nmae, (int, float)) or validation_nmae < 0:
            raise RuntimeError(
                f"validation summary lacks candidate NMAE for {candidate_id}"
            )
        test_nmae = test_values["clean_effect_nmae_mean"]
        validation_to_test_drift[candidate_id] = {
            "validation_clean_effect_nmae_mean": float(validation_nmae),
            "test_clean_effect_nmae_mean": test_nmae,
            "absolute_change": test_nmae - float(validation_nmae),
            "test_to_validation_ratio": (
                test_nmae / float(validation_nmae)
                if float(validation_nmae) > 1e-12
                else None
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
    by_id_seed = {
        (record["candidate_id"], int(record["seed"])): record for record in records
    }
    diagnostics = authorization["diagnostics"]
    truth_tau = [float(value) for value in matrix["synthetic_defaults"]["tau_seconds"]]
    tau_results = {THREE_POLE_ID: [], ORACLE_ID: []}
    delay_results = []
    for seed in sorted(int(value) for value in matrix["seeds"]):
        for candidate_id in (THREE_POLE_ID, ORACLE_ID):
            result = _tau_set_diagnostic(
                by_id_seed[(candidate_id, seed)]["metrics"],
                truth_tau,
                float(diagnostics["tau_set_log_mae_max"]),
            )
            result["seed"] = seed
            tau_results[candidate_id].append(result)
        delay = _no_true_delay_diagnostic(
            by_id_seed[(DELAY_DIAGNOSTIC_ID, seed)]["metrics"],
            dt_seconds=float(matrix["synthetic_defaults"]["dt_seconds"]),
            expected_steps_max=float(
                diagnostics["no_true_delay_expected_steps_max"]
            ),
            zero_mass_min=float(diagnostics["no_true_delay_zero_step_mass_min"]),
        )
        delay["seed"] = seed
        delay_results.append(delay)
    tau_pass = all(
        result["passes"] for values in tau_results.values() for result in values
    )
    delay_pass = all(result["passes"] for result in delay_results)
    gates = authorization["gates"]
    confirmatory = build_confirmatory_gates(
        episode_records,
        [int(value) for value in matrix["seeds"]],
        replicates=int(authorization["bootstrap"]["replicates"]),
        bootstrap_seed=int(authorization["bootstrap"]["seed"]),
        response_ci_lower_min=float(gates["order_aware_response_ci_lower_min"]),
        oracle_nmae_max=float(gates["oracle_clean_nmae_max"]),
        order_aware_nmae_max=float(gates["order_aware_clean_nmae_max"]),
        tau_diagnostic_pass=tau_pass,
        no_true_delay_diagnostic_pass=delay_pass,
    )
    heterogeneity = _paired_heterogeneity_diagnostics(
        episode_records, [int(value) for value in matrix["seeds"]]
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
        "validation_to_test_drift": validation_to_test_drift,
        "order_aware_heterogeneity_diagnostic": heterogeneity,
        "tau_recovery_diagnostic": {
            "candidate_results": tau_results,
            "all_seeds_pass": tau_pass,
            "interpretation": "Parameter recovery is diagnostic and nonblocking.",
        },
        "no_true_delay_diagnostic": {
            "candidate_id": DELAY_DIAGNOSTIC_ID,
            "seed_results": delay_results,
            "all_seeds_pass": delay_pass,
            "interpretation": (
                "This misspecification diagnostic cannot establish field delay and "
                "does not enter the confirmatory result."
            ),
        },
        **confirmatory,
        "all_gates_pass": not failures
        and confirmatory["all_confirmatory_gates_pass"],
        "interpretation_rule": (
            "This one-shot known-truth synthetic test confirms or challenges the "
            "frozen D2 order-response result. It neither authorizes retraining nor "
            "supports field-causal, unique-order, or world-model claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output")
    args = parser.parse_args()
    authorization_path = Path(args.authorization).resolve()
    if authorization_path != DEFAULT_AUTHORIZATION.resolve():
        raise SystemExit("formal MS2-D2 test summary requires repository authorization")
    summary = build_test_summary(authorization_path)
    authorization = load_authorization(authorization_path)
    output_root = _resolve_repo_path(
        authorization["validation_summary"]["path"]
    ).parent
    output = Path(args.output).resolve() if args.output else output_root / "summary_test.json"
    if output != (output_root / "summary_test.json").resolve():
        raise SystemExit("formal MS2-D2 test summary output path is frozen")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    if not summary["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
