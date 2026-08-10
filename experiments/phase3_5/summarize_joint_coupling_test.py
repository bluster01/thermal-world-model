#!/usr/bin/env python3
"""Audit MS2-J synthetic test artifacts and apply paired-episode gates."""

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
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.joint_coupling import expand_runs, load_matrix  # noqa: E402
from experiments.phase3_5.joint_coupling_test import (  # noqa: E402
    DEFAULT_AUTHORIZATION,
    ROOT_LEDGER_NAME,
    RUN_LEDGER_NAME,
    _member_bytes,
    _read_json,
    _resolve_repo_path,
    _sha256,
    _assert_pinned,
    load_authorization,
)
JOINT_ID = "j_g2_monotone_scheduled_joint"
STAGED_ID = "j_g2_monotone_scheduled_staged"
SINGLE_MODULE_IDS = (
    "j_g2_monotone_global",
    "j_g2_identity_scheduled",
)


def _gate_metrics(metrics: dict[str, Any], route: str) -> list[str]:
    """Apply the frozen MS2 structural gates without importing legacy CLI state."""

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


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_bootstrap_inputs(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
) -> dict[int, list[int]]:
    if not (
        len(candidate_error) == len(baseline_error) == len(profile_ids)
        and len(candidate_error) > 0
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


def _paired_stratified_bootstrap(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
    replicates: int,
    seed: int,
    statistic: Callable[[float, float], float],
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    strata = _validate_bootstrap_inputs(
        candidate_error, baseline_error, profile_ids
    )
    candidate_mean = statistics.mean(candidate_error)
    baseline_mean = statistics.mean(baseline_error)
    if baseline_mean <= 1e-12:
        raise ValueError("paired bootstrap baseline mean is effectively zero")
    observed = statistic(candidate_mean, baseline_mean)
    generator = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = []
        for indices in strata.values():
            sampled.extend(generator.choices(indices, k=len(indices)))
        boot_candidate = statistics.mean(candidate_error[index] for index in sampled)
        boot_baseline = statistics.mean(baseline_error[index] for index in sampled)
        if boot_baseline <= 1e-12:
            raise ValueError("bootstrap draw has an effectively zero baseline mean")
        draws.append(statistic(boot_candidate, boot_baseline))
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


def paired_stratified_bootstrap_relative_improvement(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    return _paired_stratified_bootstrap(
        candidate_error,
        baseline_error,
        profile_ids,
        replicates,
        seed,
        lambda candidate, baseline: (baseline - candidate) / baseline,
    )


def paired_stratified_bootstrap_ratio(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    return _paired_stratified_bootstrap(
        candidate_error,
        baseline_error,
        profile_ids,
        replicates,
        seed,
        lambda candidate, baseline: candidate / baseline,
    )


def _assert_paired(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for key in ("episode_ids", "profile_ids", "trajectory_design_sha256"):
        if left.get(key) != right.get(key):
            raise RuntimeError(f"unpaired MS2-J test episodes for {label}: {key}")


def build_confirmatory_gates(
    episode_records: dict[tuple[str, int], dict[str, Any]],
    seeds: list[int],
    replicates: int,
    bootstrap_seed: int,
    joint_improvement_min: float,
    staged_to_joint_ratio_max: float,
    staged_stage_a_improvement_min: float,
) -> dict[str, Any]:
    joint_seed_results = []
    staged_ratio_results = []
    staged_stage_a_results = []
    for seed in sorted(int(value) for value in seeds):
        joint = episode_records[(JOINT_ID, seed)]
        staged = episode_records[(STAGED_ID, seed)]
        stage_a = episode_records[(f"{STAGED_ID}:stage_a", seed)]
        joint_contrasts = {}
        for baseline_index, baseline_id in enumerate(SINGLE_MODULE_IDS):
            baseline = episode_records[(baseline_id, seed)]
            _assert_paired(joint, baseline, f"joint/{baseline_id}/seed={seed}")
            result = paired_stratified_bootstrap_relative_improvement(
                joint["clean_effect_mae"],
                baseline["clean_effect_mae"],
                joint["profile_ids"],
                replicates,
                bootstrap_seed + baseline_index * 10_000 + seed,
            )
            result["ci_lower_meets_threshold"] = (
                result["ci95"][0] >= joint_improvement_min
            )
            joint_contrasts[baseline_id] = result
        joint_seed_results.append(
            {
                "seed": seed,
                "contrasts": joint_contrasts,
                "all_contrasts_pass": all(
                    result["ci_lower_meets_threshold"]
                    for result in joint_contrasts.values()
                ),
            }
        )

        _assert_paired(staged, joint, f"staged/joint/seed={seed}")
        ratio = paired_stratified_bootstrap_ratio(
            staged["clean_effect_mae"],
            joint["clean_effect_mae"],
            staged["profile_ids"],
            replicates,
            bootstrap_seed + 20_000 + seed,
        )
        ratio["ci_upper_meets_noninferiority"] = (
            ratio["ci95"][1] <= staged_to_joint_ratio_max
        )
        ratio["seed"] = seed
        staged_ratio_results.append(ratio)

        _assert_paired(staged, stage_a, f"staged/stage_a/seed={seed}")
        improvement = paired_stratified_bootstrap_relative_improvement(
            staged["clean_effect_mae"],
            stage_a["clean_effect_mae"],
            staged["profile_ids"],
            replicates,
            bootstrap_seed + 30_000 + seed,
        )
        improvement["ci_lower_meets_threshold"] = (
            improvement["ci95"][0] >= staged_stage_a_improvement_min
        )
        improvement["seed"] = seed
        staged_stage_a_results.append(improvement)

    joint_gate = {
        "candidate_id": JOINT_ID,
        "baseline_ids": list(SINGLE_MODULE_IDS),
        "threshold": joint_improvement_min,
        "seed_results": joint_seed_results,
        "all_seeds_pass": all(
            result["all_contrasts_pass"] for result in joint_seed_results
        ),
        "multiplicity_note": (
            "intersection-union gate: both prespecified contrasts must pass per seed"
        ),
    }
    staged_ratio_gate = {
        "candidate_id": STAGED_ID,
        "baseline_id": JOINT_ID,
        "ratio_upper_bound": staged_to_joint_ratio_max,
        "seed_results": staged_ratio_results,
        "all_seeds_pass": all(
            result["ci_upper_meets_noninferiority"]
            for result in staged_ratio_results
        ),
    }
    staged_stage_a_gate = {
        "candidate_id": STAGED_ID,
        "baseline_id": f"{STAGED_ID}:stage_a",
        "threshold": staged_stage_a_improvement_min,
        "seed_results": staged_stage_a_results,
        "all_seeds_pass": all(
            result["ci_lower_meets_threshold"]
            for result in staged_stage_a_results
        ),
    }
    return {
        "joint_module_test": joint_gate,
        "staged_noninferiority_test": staged_ratio_gate,
        "staged_vs_stage_a_test": staged_stage_a_gate,
        "all_confirmatory_gates_pass": all(
            gate["all_seeds_pass"]
            for gate in (joint_gate, staged_ratio_gate, staged_stage_a_gate)
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


def build_test_summary(authorization_path: str | Path) -> dict[str, Any]:
    authorization = load_authorization(authorization_path)
    matrix_path = _resolve_repo_path(authorization["matrix"]["path"])
    validation_path = _resolve_repo_path(
        authorization["validation_summary"]["path"]
    )
    archive_path = _resolve_repo_path(
        authorization["checkpoint_archive"]["path"]
    )
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
    output_root = validation_path.parent
    root_ledger = _read_json(output_root / ROOT_LEDGER_NAME)
    if root_ledger.get("status") != "completed":
        raise RuntimeError("MS2-J matrix test ledger is not completed")
    expected_root = {
        "protocol_version": authorization["protocol_version"],
        "evidence_scope": authorization["evidence_scope"],
        "matrix_sha256": authorization["matrix"]["sha256"],
        "validation_summary_sha256": authorization["validation_summary"][
            "sha256"
        ],
        "checkpoint_archive_sha256": authorization["checkpoint_archive"][
            "sha256"
        ],
        "run_count": int(authorization["expected_run_count"]),
        "test_samples": int(authorization["test_samples"]),
        "authorization_sha256": _sha256(Path(authorization_path).resolve()),
    }
    root_mismatches = [
        key for key, value in expected_root.items() if root_ledger.get(key) != value
    ]
    expected_completed = [
        {"candidate_id": run["candidate_id"], "seed": int(run["seed"])}
        for run in expand_runs(matrix)
    ]
    if root_ledger.get("completed_runs") != expected_completed:
        root_mismatches.append("completed_runs")
    environment = root_ledger.get("environment", {})
    for key in (
        "python",
        "torch",
        "cuda_runtime",
        "cuda_available",
        "device",
        "platform",
    ):
        if key not in environment:
            root_mismatches.append(f"environment_{key}")
    if root_mismatches:
        raise RuntimeError(f"MS2-J matrix test ledger mismatch: {root_mismatches}")

    records = []
    episode_records: dict[tuple[str, int], dict[str, Any]] = {}
    expected_count = int(authorization["test_samples"])
    with tarfile.open(archive_path, "r") as archive:
        for run in expand_runs(matrix):
            candidate_id, seed = run["candidate_id"], int(run["seed"])
            run_dir = output_root / f"ms2j_{candidate_id}_s{seed}"
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
                "checkpoint_archive_sha256": authorization[
                    "checkpoint_archive"
                ]["sha256"],
                "checkpoint_member": member_name,
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "checkpoint_selector": manifest.get("checkpoint_selector"),
                "test_samples": expected_count,
                "trajectory_design_sha256": episodes.get(
                    "trajectory_design_sha256"
                ),
                "evaluation_git_sha": root_ledger.get("evaluation_git_sha"),
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
                    f"MS2-J test artifact mismatch for {candidate_id}/seed={seed}: "
                    f"{sorted(set(mismatches))}"
                )
            failures = _gate_metrics(metrics, run["route"])
            records.append(
                {
                    **run,
                    "effect_mae": metrics["effect_mae"],
                    "clean_effect_mae": metrics["clean_effect_mae"],
                    "clean_effect_nmae": metrics["clean_effect_nmae"],
                    "direction_accuracy_clean_nonzero": metrics[
                        "direction_accuracy_clean_nonzero"
                    ],
                    "gate_failures": failures,
                }
            )
            episode_records[(candidate_id, seed)] = episodes
            if run["training_mode"] == "staged":
                stage_metrics = _read_json(run_dir / "metrics_stage_a_test.json")
                stage_episodes = _read_json(
                    run_dir / "episode_metrics_stage_a_test.json"
                )
                stage_info = manifest.get("stage_checkpoints", {}).get(
                    "stage_a", {}
                )
                stage_member = f"{run_dir.name}/{stage_info.get('file', 'missing')}"
                _member_bytes(archive, stage_member, stage_info.get("sha256", ""))
                if manifest.get("test_stage_a_episode_metrics") != (
                    "episode_metrics_stage_a_test.json"
                ):
                    raise RuntimeError(
                        f"MS2-J staged manifest missing Stage-A test record seed={seed}"
                    )
                if ledger.get("stage_a_checkpoint_member") != stage_member or ledger.get(
                    "stage_a_checkpoint_sha256"
                ) != stage_info.get("sha256"):
                    raise RuntimeError(
                        f"MS2-J staged ledger checkpoint mismatch seed={seed}"
                    )
                stage_mismatches = _episode_integrity_failures(
                    stage_episodes, expected_count
                )
                if stage_metrics.get("clean_effect_mae") is None:
                    stage_mismatches.append("stage_a_clean_effect_mae")
                if stage_metrics.get("sample_count") != expected_count:
                    stage_mismatches.append("stage_a_sample_count")
                if stage_metrics.get("truth", {}).get("split") != "test":
                    stage_mismatches.append("stage_a_truth_split")
                stage_mismatches.extend(
                    f"stage_a_structural_{value}"
                    for value in _gate_metrics(stage_metrics, "graybox")
                )
                _assert_paired(
                    episodes, stage_episodes, f"staged/stage_a/seed={seed}"
                )
                if stage_mismatches:
                    raise RuntimeError(
                        f"MS2-J Stage-A test artifact mismatch seed={seed}: "
                        f"{stage_mismatches}"
                    )
                episode_records[(f"{STAGED_ID}:stage_a", seed)] = stage_episodes

    seed_design_hashes = []
    for seed in sorted(int(value) for value in matrix["seeds"]):
        hashes = {
            episode_records[(candidate["candidate_id"], seed)][
                "trajectory_design_sha256"
            ]
            for candidate in matrix["regimes"][0]["candidates"]
        }
        if len(hashes) != 1:
            raise RuntimeError(f"MS2-J candidates have unpaired trajectories seed={seed}")
        seed_design_hashes.append(next(iter(hashes)))
    if len(set(seed_design_hashes)) != len(seed_design_hashes):
        raise RuntimeError("MS2-J test seeds reuse the same trajectory design")

    candidates = {}
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        subset = [
            record for record in records if record["candidate_id"] == candidate_id
        ]
        candidates[candidate_id] = {
            "route": subset[0]["route"],
            "training_mode": subset[0]["training_mode"],
            "clean_effect_nmae_mean": statistics.mean(
                record["clean_effect_nmae"] for record in subset
            ),
            "clean_effect_nmae_std": statistics.stdev(
                record["clean_effect_nmae"] for record in subset
            ),
            "clean_effect_mae_mean": statistics.mean(
                record["clean_effect_mae"] for record in subset
            ),
            "effect_mae_mean": statistics.mean(
                record["effect_mae"] for record in subset
            ),
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
    gates = authorization["gates"]
    confirmatory = build_confirmatory_gates(
        episode_records,
        [int(value) for value in matrix["seeds"]],
        replicates=int(authorization["bootstrap"]["replicates"]),
        bootstrap_seed=int(authorization["bootstrap"]["seed"]),
        joint_improvement_min=float(gates["joint_relative_improvement_min"]),
        staged_to_joint_ratio_max=float(gates["staged_to_joint_ratio_max"]),
        staged_stage_a_improvement_min=float(
            gates["staged_relative_improvement_from_stage_a_min"]
        ),
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
        **confirmatory,
        "interpretation_rule": (
            "test confirms or challenges the frozen mixed validation result; "
            "it does not authorize retraining or field-causal claims"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output")
    args = parser.parse_args()
    authorization_path = Path(args.authorization).resolve()
    if authorization_path != DEFAULT_AUTHORIZATION.resolve():
        raise SystemExit("formal MS2-J test summary requires repository authorization")
    summary = build_test_summary(authorization_path)
    authorization = load_authorization(authorization_path)
    output_root = _resolve_repo_path(
        authorization["validation_summary"]["path"]
    ).parent
    output = Path(args.output).resolve() if args.output else output_root / "summary_test.json"
    if output != (output_root / "summary_test.json").resolve():
        raise SystemExit("formal MS2-J test summary output path is frozen")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    if not (
        summary["all_artifact_and_structural_gates_pass"]
        and summary["all_confirmatory_gates_pass"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
