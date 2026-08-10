#!/usr/bin/env python3
"""Audit MS2 synthetic test artifacts and run paired episode bootstrap gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.multistep_mismatch import expand_runs, load_matrix  # noqa: E402
from experiments.phase3_5.summarize_multistep_mismatch import (  # noqa: E402
    PRIMARY_CONTRASTS,
    _gate_metrics,
)


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required MS2 test artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_integrity_failures(episodes: dict, expected_count: int) -> list[str]:
    failures = []
    expected_ids = list(range(expected_count))
    if episodes.get("episode_ids") != expected_ids:
        failures.append("episode_ids")
    required_vectors = (
        "profile_ids",
        "observed_effect_mae",
        "clean_effect_mae",
        "clean_effect_scale",
    )
    for key in required_vectors:
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
    if not isinstance(horizons, dict) or set(horizons) != {"H1", "H6", "H18", "H60"}:
        failures.append("clean_horizons")
    else:
        for key, values in horizons.items():
            if not isinstance(values, list) or len(values) != expected_count or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in values
            ):
                failures.append(f"{key}_values")
    return failures


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_stratified_bootstrap(
    candidate_error: list[float],
    baseline_error: list[float],
    profile_ids: list[int],
    replicates: int,
    seed: int,
) -> dict:
    if not (
        len(candidate_error) == len(baseline_error) == len(profile_ids)
        and len(candidate_error) > 0
    ):
        raise ValueError("paired bootstrap arrays must have the same non-zero length")
    strata = {
        profile: [index for index, value in enumerate(profile_ids) if value == profile]
        for profile in sorted(set(profile_ids))
    }
    if any(not indices for indices in strata.values()):
        raise ValueError("bootstrap profile stratum is empty")
    baseline_mean = statistics.mean(baseline_error)
    candidate_mean = statistics.mean(candidate_error)
    observed = (baseline_mean - candidate_mean) / max(baseline_mean, 1e-12)
    generator = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = []
        for indices in strata.values():
            sampled.extend(generator.choices(indices, k=len(indices)))
        boot_baseline = statistics.mean(baseline_error[index] for index in sampled)
        boot_candidate = statistics.mean(candidate_error[index] for index in sampled)
        draws.append((boot_baseline - boot_candidate) / max(boot_baseline, 1e-12))
    return {
        "observed_relative_improvement": observed,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired_episode_stratified_by_action_profile",
        "episode_count": len(candidate_error),
        "profile_counts": {
            str(profile): len(indices) for profile, indices in strata.items()
        },
    }


def build_test_summary(
    matrix_path: Path,
    output_root: Path,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_810,
) -> dict:
    matrix = load_matrix(matrix_path)
    records = []
    episode_records = {}
    for run in expand_runs(matrix):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        run_dir = output_root / f"ms2_{candidate_id}_s{seed}"
        manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics_test.json")
        episodes = _read_json(run_dir / "episode_metrics_test.json")
        ledger = _read_json(run_dir / "synthetic_test_access_ledger.json")
        checkpoint = run_dir / "checkpoint_best_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen MS2 checkpoint missing: {checkpoint}")
        checkpoint_sha256 = _sha256(checkpoint)
        test_samples = int(matrix["synthetic_defaults"]["test_samples"])
        expected = {
            "protocol_version": matrix["protocol_version"],
            "route_id": candidate_id,
            "seed": seed,
            "test_accessed": True,
            "checkpoint_sha256": checkpoint_sha256,
            "test_access_ledger": "synthetic_test_access_ledger.json",
            "test_episode_metrics": "episode_metrics_test.json",
        }
        mismatches = [
            key for key, value in expected.items() if manifest.get(key) != value
        ]
        if ledger.get("status") != "completed":
            mismatches.append("ledger_status")
        ledger_expected = {
            "protocol_version": matrix["protocol_version"],
            "candidate_id": candidate_id,
            "regime_id": run["regime_id"],
            "seed": seed,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_selector": manifest.get("checkpoint_selector"),
            "test_samples": test_samples,
        }
        mismatches.extend(
            f"ledger_{key}"
            for key, value in ledger_expected.items()
            if ledger.get(key) != value
        )
        if ledger.get("training_git_sha") != manifest.get("git_sha"):
            mismatches.append("training_git_sha")
        if ledger.get("trajectory_design_sha256") != episodes.get(
            "trajectory_design_sha256"
        ):
            mismatches.append("trajectory_design_sha256")
        mismatches.extend(_episode_integrity_failures(episodes, test_samples))
        if mismatches:
            raise RuntimeError(
                f"MS2 test artifact mismatch for {candidate_id}/seed={seed}: "
                f"{sorted(set(mismatches))}"
            )
        gate_failures = _gate_metrics(metrics, run["route"])
        records.append({
            **run,
            "effect_mae": metrics["effect_mae"],
            "clean_effect_mae": metrics["clean_effect_mae"],
            "clean_effect_nmae": metrics["clean_effect_nmae"],
            "direction_accuracy_clean_nonzero": metrics[
                "direction_accuracy_clean_nonzero"
            ],
            "gate_failures": gate_failures,
        })
        episode_records[(candidate_id, seed)] = episodes

    for regime in matrix["regimes"]:
        candidate_ids = [candidate["candidate_id"] for candidate in regime["candidates"]]
        for seed in sorted(int(value) for value in matrix["seeds"]):
            design_hashes = {
                episode_records[(candidate_id, seed)]["trajectory_design_sha256"]
                for candidate_id in candidate_ids
            }
            if len(design_hashes) != 1:
                raise RuntimeError(
                    "MS2 candidates were not evaluated on the same test trajectories: "
                    f"{regime['regime_id']}/seed={seed}"
                )

    candidates = {}
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        subset = [record for record in records if record["candidate_id"] == candidate_id]
        candidates[candidate_id] = {
            "regime_id": subset[0]["regime_id"],
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
            "direction_accuracy_clean_nonzero_mean": statistics.mean(
                record["direction_accuracy_clean_nonzero"] for record in subset
            ),
            "all_structural_gates_pass": all(not record["gate_failures"] for record in subset),
        }

    primary_contrasts = {}
    for regime_index, (regime_id, (candidate_id, baseline_id)) in enumerate(
        PRIMARY_CONTRASTS.items()
    ):
        seed_results = []
        for seed in sorted(int(value) for value in matrix["seeds"]):
            candidate = episode_records[(candidate_id, seed)]
            baseline = episode_records[(baseline_id, seed)]
            for key in ("episode_ids", "profile_ids", "trajectory_design_sha256"):
                if candidate[key] != baseline[key]:
                    raise RuntimeError(
                        f"unpaired MS2 test episodes for {regime_id}/seed={seed}: {key}"
                    )
            result = paired_stratified_bootstrap(
                candidate["clean_effect_mae"],
                baseline["clean_effect_mae"],
                candidate["profile_ids"],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + regime_index * 1_000 + seed,
            )
            result["seed"] = seed
            result["ci_lower_exceeds_20pct"] = result["ci95"][0] >= 0.20
            seed_results.append(result)
        primary_contrasts[regime_id] = {
            "candidate_id": candidate_id,
            "baseline_id": baseline_id,
            "seed_results": seed_results,
            "all_seed_ci_lower_exceeds_20pct": all(
                result["ci_lower_exceeds_20pct"] for result in seed_results
            ),
        }

    failures = [
        {
            "candidate_id": record["candidate_id"],
            "seed": record["seed"],
            "failures": record["gate_failures"],
        }
        for record in records if record["gate_failures"]
    ]
    contrasts_pass = all(
        contrast["all_seed_ci_lower_exceeds_20pct"]
        for contrast in primary_contrasts.values()
    )
    return {
        "protocol_version": matrix["protocol_version"],
        "evidence_scope": "synthetic_mismatch_test_not_field_causality",
        "split": "test",
        "run_count": len(records),
        "test_accessed": True,
        "all_artifact_and_structural_gates_pass": not failures,
        "gate_failures": failures,
        "primary_contrasts_pass": contrasts_pass,
        "candidates": candidates,
        "primary_contrasts": primary_contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"),
    )
    parser.add_argument("--output-root", default="results/phase3_5/multistep_mismatch")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_810)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        raise SystemExit("formal MS2 test summary requires at least 1000 bootstrap replicates")
    output_root = Path(args.output_root).resolve()
    summary = build_test_summary(
        Path(args.matrix).resolve(),
        output_root,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output).resolve() if args.output else output_root / "summary_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    if not (
        summary["all_artifact_and_structural_gates_pass"]
        and summary["primary_contrasts_pass"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
