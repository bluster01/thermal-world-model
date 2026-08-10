#!/usr/bin/env python3
"""Fail-closed episode-level aggregation for Phase 3.5-MS2-D3 validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.ms2d_disturbance import (  # noqa: E402
    FROZEN_EXECUTION_PATHS,
    VALIDATION_EPISODES_NAME,
    _assert_no_test_artifacts,
    _build_configs,
    _canonical,
    _select,
    expand_runs,
    load_matrix,
)
from experiments.phase3_5.multistep_mismatch import _sha256  # noqa: E402
from experiments.phase3_5.summarize_ms2d_order_test import (  # noqa: E402
    _episode_integrity_failures,
    _episode_metric_failures,
    _gate_metrics,
    _no_true_delay_diagnostic,
    _tau_set_diagnostic,
    paired_stratified_bootstrap_relative_improvement,
)


TWO_POLE_ID = "d3_g2_two_pole"
THREE_POLE_ID = "d3_g3_three_pole"
ORACLE_ID = "d3_g3_oracle_structure"
DELAY_DIAGNOSTIC_ID = "d3_g2_delay_compensation"
DEFAULT_ARCHIVE_NAME = "checkpoints_validation.tar"


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required MS2-D3 artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _replay_best_epoch(history: list[dict[str, Any]]) -> int:
    best_score = float("inf")
    best_epoch = 0
    for expected_epoch, record in enumerate(history, start=1):
        if not isinstance(record, dict) or record.get("epoch") != expected_epoch:
            raise RuntimeError("history epochs must be contiguous and one-indexed")
        score = record.get("validation_effect_mae")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise RuntimeError("history contains invalid validation_effect_mae")
        if float(score) < best_score - 1e-8:
            best_score = float(score)
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
        raise RuntimeError(f"MS2-D3 execution commit unavailable: {execution_sha}")
    compared = subprocess.run(
        ["git", "diff", "--quiet", execution_sha, "HEAD", "--", *FROZEN_EXECUTION_PATHS],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(f"MS2-D3 frozen execution code differs from {execution_sha}")
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking MS2-D3 code equivalence")


def _assert_paired(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for key in ("episode_ids", "profile_ids", "trajectory_design_sha256"):
        if left.get(key) != right.get(key):
            raise RuntimeError(f"unpaired MS2-D3 validation episodes for {label}: {key}")


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
    robust_nmae_max: float,
    tau_diagnostic_pass: bool,
    no_true_delay_diagnostic_pass: bool,
) -> dict[str, Any]:
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
        robust_nmae = _episode_nmae(three, f"three-pole/seed={seed}")
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
                "clean_effect_nmae": robust_nmae,
                "passes": robust_nmae < robust_nmae_max,
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
        "clean_effect_nmae_max": robust_nmae_max,
        "seed_results": absolute_results,
        "all_seeds_pass": all(item["passes"] for item in absolute_results),
    }
    response_gate = {
        "disturbance_robust_id": THREE_POLE_ID,
        "two_pole_id": TWO_POLE_ID,
        "ci_lower_relative_improvement_min": response_ci_lower_min,
        "seed_results": response_results,
        "all_seeds_pass": all(
            item["ci_lower_meets_threshold"] for item in response_results
        ),
    }
    primary = (
        oracle_gate["all_seeds_pass"]
        and absolute_gate["all_seeds_pass"]
        and response_gate["all_seeds_pass"]
    )
    return {
        "oracle_gate": oracle_gate,
        "disturbance_robust_absolute_gate": absolute_gate,
        "disturbance_robust_response_gate": response_gate,
        "tau_recovery_diagnostic_pass": tau_diagnostic_pass,
        "no_true_delay_diagnostic_pass": no_true_delay_diagnostic_pass,
        "all_primary_gates_pass": primary,
    }


def _heterogeneity(
    episode_records: dict[tuple[str, int], dict[str, Any]], seeds: list[int]
) -> dict[str, Any]:
    profiles = []
    horizons = []
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
            profiles.append(
                {
                    "seed": seed,
                    "profile_id": profile_id,
                    "profile_name": (
                        names[profile_id]
                        if isinstance(names, list) and profile_id < len(names)
                        else None
                    ),
                    "episode_count": len(indices),
                    "relative_improvement": (
                        (baseline - candidate) / baseline
                        if baseline > 1e-12
                        else None
                    ),
                }
            )
        for horizon in ("H1", "H6", "H18", "H60"):
            candidate = statistics.fmean(
                three["clean_horizon_absolute_error"][horizon]
            )
            baseline = statistics.fmean(
                two["clean_horizon_absolute_error"][horizon]
            )
            horizons.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "relative_improvement": (
                        (baseline - candidate) / baseline
                        if baseline > 1e-12
                        else None
                    ),
                }
            )
    return {
        "by_action_profile": profiles,
        "by_horizon": horizons,
        "interpretation": "Descriptive only; no subgroup creates an additional gate.",
    }


def _write_deterministic_archive(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temporary, "w") as archive:
        for record in sorted(records, key=lambda item: (item["candidate_id"], item["seed"])):
            checkpoint: Path = record["checkpoint"]
            member_name = f"{record['run_dir'].name}/checkpoint_best_val.pt"
            info = tarfile.TarInfo(member_name)
            info.size = checkpoint.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with checkpoint.open("rb") as handle:
                archive.addfile(info, handle)
    os.replace(temporary, path)
    try:
        archive_path = str(path.resolve().relative_to(ROOT).as_posix())
    except ValueError:
        archive_path = str(path.resolve())
    return {
        "archive_path": archive_path,
        "archive_sha256": _sha256(path),
        "n_checkpoints": len(records),
    }


def build_summary(matrix_path: Path, output_root: Path) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    _assert_no_test_artifacts(output_root)
    current_sha = _git_sha()
    matrix_sha = _sha256(matrix_path)
    d2_reference_path = ROOT / matrix["d2_reference"]["path"]
    if _sha256(d2_reference_path) != matrix["d2_reference"]["sha256"]:
        raise RuntimeError("MS2-D3 D2 reference content pin changed")
    d2_reference = _read_json(d2_reference_path)
    runs = expand_runs(matrix)
    records = []
    episode_records: dict[tuple[str, int], dict[str, Any]] = {}
    execution_shas = set()
    expected_count = int(matrix["synthetic_defaults"]["validation_samples"])
    truth_tau = [float(value) for value in matrix["synthetic_defaults"]["tau_seconds"]]
    for run in runs:
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        run_dir = output_root / f"ms2d3_{candidate_id}_s{seed}"
        manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics_validation.json")
        history = _read_json(run_dir / "history.json")
        episodes = _read_json(run_dir / VALIDATION_EPISODES_NAME)
        checkpoint = run_dir / "checkpoint_best_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required MS2-D3 checkpoint missing: {checkpoint}")
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
        expected_spec = replace(synthetic, seed=synthetic.seed + seed * 1_000_003)
        expected_manifest = {
            "protocol_version": matrix["protocol_version"],
            "evidence_scope": matrix["evidence_scope"],
            "route_id": candidate_id,
            "seed": seed,
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_selector": "validation_effect_mae",
            "matrix_sha256": matrix_sha,
            "d2_reference_sha256": matrix["d2_reference"]["sha256"],
            "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
            "validation_episode_metrics": VALIDATION_EPISODES_NAME,
            "validation_trajectory_design_sha256": episodes.get(
                "trajectory_design_sha256"
            ),
            "test_accessed": False,
            "test_authorized": False,
            "operator_config": operator.to_dict(),
            "training_config": asdict(training),
            "synthetic_spec": asdict(expected_spec),
            "regime_id": run["regime_id"],
            "candidate_role": run["role"],
        }
        mismatches = [
            key
            for key, value in expected_manifest.items()
            if _canonical(manifest.get(key)) != _canonical(value)
        ]
        execution_sha = manifest.get("git_sha")
        if not isinstance(execution_sha, str):
            mismatches.append("git_sha")
        else:
            execution_shas.add(execution_sha)
        environment = manifest.get("environment")
        required_environment = {
            "python",
            "torch",
            "cuda_runtime",
            "cuda_available",
            "device",
            "platform",
        }
        if not isinstance(environment, dict) or not required_environment <= set(
            environment
        ):
            mismatches.append("environment")
        elif environment.get("device") != manifest.get("device"):
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
            "truth_regime": "disturbed_context_scheduled",
            "truth_opening_map": "equal_percentage_r50",
            "tau_seconds": truth_tau,
            "input_delay_steps": 0,
            "input_delay_seconds": 0.0,
            "disturbance_std": 0.03,
            "disturbance_tau_seconds": 120.0,
        }
        for key, value in expected_truth.items():
            if _canonical(truth.get(key)) != _canonical(value):
                mismatches.append(f"truth_{key}")
        expected_rho = math.exp(-10.0 / 120.0)
        if not math.isclose(
            float(truth.get("disturbance_rho", float("nan"))),
            expected_rho,
            rel_tol=1e-6,
        ):
            mismatches.append("truth_disturbance_rho")
        for key in (
            "disturbance_realized_mean",
            "disturbance_realized_std",
            "disturbance_realized_lag1_correlation",
        ):
            if not isinstance(truth.get(key), (int, float)) or not math.isfinite(
                float(truth[key])
            ):
                mismatches.append(f"truth_{key}")
        integrity = _episode_integrity_failures(episodes, expected_count)
        mismatches.extend(integrity)
        if not integrity:
            mismatches.extend(_episode_metric_failures(episodes, metrics))
        for key in ("colored_disturbance_mae", "colored_disturbance_mean"):
            values = episodes.get(key)
            if not isinstance(values, list) or len(values) != expected_count or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in (values or [])
            ):
                mismatches.append(key)
        if metrics.get("sample_count") != expected_count:
            mismatches.append("metrics_sample_count")
        if metrics.get("truth", {}).get("split") != "validation":
            mismatches.append("metrics_truth_split")
        if unauthorized:
            mismatches.append(f"unauthorized_test={unauthorized}")
        if mismatches:
            raise RuntimeError(
                f"MS2-D3 artifact mismatch for {candidate_id}/seed={seed}: "
                f"{sorted(set(mismatches))}"
            )
        failures = _gate_metrics(metrics, run["route"])
        records.append(
            {
                **run,
                "run_dir": run_dir,
                "checkpoint": checkpoint,
                "metrics": metrics,
                "effect_mae": float(metrics["effect_mae"]),
                "clean_effect_mae": float(metrics["clean_effect_mae"]),
                "clean_effect_nmae": float(metrics["clean_effect_nmae"]),
                "gate_failures": failures,
            }
        )
        episode_records[(candidate_id, seed)] = episodes
    if len(execution_shas) != 1:
        raise RuntimeError(f"MS2-D3 manifests use multiple execution SHAs: {execution_shas}")
    execution_sha = next(iter(execution_shas))
    _assert_code_equivalent(execution_sha)

    seed_hashes = []
    for seed in sorted(int(value) for value in matrix["seeds"]):
        hashes = {
            episode_records[(candidate["candidate_id"], seed)][
                "trajectory_design_sha256"
            ]
            for candidate in matrix["regimes"][0]["candidates"]
        }
        if len(hashes) != 1:
            raise RuntimeError(f"MS2-D3 candidates have unpaired trajectories seed={seed}")
        seed_hashes.append(next(iter(hashes)))
    if len(set(seed_hashes)) != len(seed_hashes):
        raise RuntimeError("MS2-D3 validation seeds reuse the same trajectory design")

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
            "effect_mae_mean": statistics.fmean(record["effect_mae"] for record in subset),
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
    by_id_seed = {
        (record["candidate_id"], int(record["seed"])): record for record in records
    }
    tau_results = {THREE_POLE_ID: [], ORACLE_ID: []}
    delay_results = []
    disturbance_results = []
    for seed in sorted(int(value) for value in matrix["seeds"]):
        for candidate_id in (THREE_POLE_ID, ORACLE_ID):
            result = _tau_set_diagnostic(
                by_id_seed[(candidate_id, seed)]["metrics"],
                truth_tau,
                float(matrix["gates"]["tau_set_log_mae_max"]),
            )
            result["seed"] = seed
            tau_results[candidate_id].append(result)
        delay = _no_true_delay_diagnostic(
            by_id_seed[(DELAY_DIAGNOSTIC_ID, seed)]["metrics"],
            dt_seconds=float(matrix["synthetic_defaults"]["dt_seconds"]),
            expected_steps_max=float(
                matrix["gates"]["no_true_delay_expected_steps_max"]
            ),
            zero_mass_min=float(
                matrix["gates"]["no_true_delay_zero_step_mass_min"]
            ),
        )
        delay["seed"] = seed
        delay_results.append(delay)
        truth = by_id_seed[(THREE_POLE_ID, seed)]["metrics"]["truth"]
        disturbance_results.append(
            {
                "seed": seed,
                "theoretical_std": truth["disturbance_std"],
                "theoretical_tau_seconds": truth["disturbance_tau_seconds"],
                "theoretical_rho": truth["disturbance_rho"],
                "realized_mean": truth["disturbance_realized_mean"],
                "realized_std": truth["disturbance_realized_std"],
                "realized_lag1_correlation": truth[
                    "disturbance_realized_lag1_correlation"
                ],
            }
        )
    tau_pass = all(
        result["passes"] for values in tau_results.values() for result in values
    )
    delay_pass = all(result["passes"] for result in delay_results)
    confirmatory = build_confirmatory_gates(
        episode_records,
        [int(value) for value in matrix["seeds"]],
        replicates=int(matrix["gates"]["bootstrap_replicates"]),
        bootstrap_seed=int(matrix["gates"]["bootstrap_seed"]),
        response_ci_lower_min=float(
            matrix["gates"]["disturbance_robust_ci_lower_min"]
        ),
        oracle_nmae_max=float(matrix["gates"]["oracle_clean_nmae_max"]),
        robust_nmae_max=float(
            matrix["gates"]["disturbance_robust_clean_nmae_max"]
        ),
        tau_diagnostic_pass=tau_pass,
        no_true_delay_diagnostic_pass=delay_pass,
    )
    d2_mapping = {
        "d3_g2_two_pole": "d2_g2_two_pole",
        "d3_g3_three_pole": "d2_g3_three_pole",
        "d3_g3_oracle_structure": "d2_g3_oracle_structure",
        "d3_g2_delay_compensation": "d2_g2_delay_compensation",
        "d3_k4_monotone": "d2_k4_monotone",
        "d3_pi_monotone": "d2_pi_monotone",
        "d3_deeponet": "d2_deeponet",
    }
    d2_drift = {}
    for d3_id, d2_id in d2_mapping.items():
        d2_nmae = float(d2_reference["candidates"][d2_id]["clean_effect_nmae_mean"])
        d3_nmae = candidates[d3_id]["clean_effect_nmae_mean"]
        d2_drift[d3_id] = {
            "d2_candidate_id": d2_id,
            "d2_test_clean_effect_nmae_mean": d2_nmae,
            "d3_validation_clean_effect_nmae_mean": d3_nmae,
            "d3_to_d2_ratio": d3_nmae / d2_nmae if d2_nmae > 1e-12 else None,
            "interpretation": "diagnostic_only_cross_gate_and_split",
        }
    archive = _write_deterministic_archive(
        records, output_root / DEFAULT_ARCHIVE_NAME
    )
    primary = not failures and confirmatory["all_primary_gates_pass"]
    return {
        "protocol_version": matrix["protocol_version"],
        "evidence_scope": matrix["evidence_scope"],
        "split": "validation",
        "run_count": len(records),
        "test_accessed": False,
        "execution_git_sha": execution_sha,
        "aggregation_git_sha": current_sha,
        "matrix_sha256": matrix_sha,
        "d2_reference": matrix["d2_reference"],
        "checkpoint_archive": archive,
        "all_artifact_and_structural_gates_pass": not failures,
        "structural_gate_failures": failures,
        "candidates": candidates,
        "d2_to_d3_drift_diagnostic": d2_drift,
        "disturbance_realization_diagnostic": {
            "seed_results": disturbance_results,
            "interpretation": "Generator provenance only; not a field-spectrum claim.",
        },
        "tau_recovery_diagnostic": {
            "candidate_results": tau_results,
            "all_seeds_pass": tau_pass,
            "interpretation": "Diagnostic only and not part of the primary gate.",
        },
        "no_true_delay_diagnostic": {
            "candidate_id": DELAY_DIAGNOSTIC_ID,
            "seed_results": delay_results,
            "all_seeds_pass": delay_pass,
            "interpretation": "Diagnostic only; it cannot establish field delay.",
        },
        "heterogeneity_diagnostic": _heterogeneity(
            episode_records, [int(value) for value in matrix["seeds"]]
        ),
        **confirmatory,
        "all_primary_gates_pass": primary,
        "interpretation_rule": (
            "A validation PASS is screening evidence for response recovery under "
            "one synthetic AR(1) output disturbance. It is not a test result, a "
            "field disturbance model, a state observer, or a simulator claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(ROOT / "configs/phase3_5/ms2d_disturbance_matrix.json"))
    parser.add_argument("--output-root", default="results/phase3_5/ms2d_disturbance")
    parser.add_argument("--output")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    summary = build_summary(Path(args.matrix).resolve(), output_root)
    output = Path(args.output).resolve() if args.output else output_root / "summary_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    if not summary["all_primary_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
