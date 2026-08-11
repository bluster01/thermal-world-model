#!/usr/bin/env python3
"""Fail-closed summary for the frozen Phase 3.5-MS3 validation matrix."""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.ms3_real_adaptation import (
    DEFAULT_MATRIX,
    FORBIDDEN_TEST_ARTIFACTS,
    FROZEN_EXECUTION_PATHS,
    PROTOCOL_VERSION,
    _sha256,
    expand_runs,
    load_matrix,
)
from src.phase35.multistep.training import _json_dump
from src.phase35.multistep.real_training import RealModelConfig


REQUIRED_RUN_FILES = (
    "manifest.json",
    "history.json",
    "checkpoint_best_val.pt",
    "metrics_validation.json",
    "episode_metrics_validation.json",
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required MS3 artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
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
        raise RuntimeError(f"MS3 execution commit unavailable: {execution_sha}")
    compared = subprocess.run(
        ["git", "diff", "--quiet", execution_sha, "HEAD", "--", *FROZEN_EXECUTION_PATHS],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(f"MS3 frozen execution code differs from {execution_sha}")
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking MS3 code equivalence")


def day_block_bootstrap(
    logged: np.ndarray,
    comparator: np.ndarray,
    days: np.ndarray,
    dynamic: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    logged = np.asarray(logged, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    days = np.asarray(days)
    dynamic = np.asarray(dynamic, dtype=bool)
    if not (logged.shape == comparator.shape == days.shape == dynamic.shape):
        raise ValueError("MS3 bootstrap arrays must have equal shapes")
    selected_days = np.unique(days[dynamic])
    if len(selected_days) < 2:
        return {
            "mean_improvement_c": None,
            "ci95": None,
            "day_count": int(len(selected_days)),
            "window_count": int(dynamic.sum()),
        }
    day_means = np.asarray(
        [np.mean(comparator[(days == day) & dynamic] - logged[(days == day) & dynamic]) for day in selected_days],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(day_means), size=(samples, len(day_means)))
    boot = day_means[draws].mean(axis=1)
    return {
        "mean_improvement_c": float(day_means.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "day_count": int(len(selected_days)),
        "window_count": int(dynamic.sum()),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "top_cluster": "UTC_day",
    }


def decide_ms3(
    records: dict[tuple[str, str, int], dict[str, Any]],
    seeds: list[int],
    gates: dict[str, Any],
) -> dict[str, Any]:
    seed_decisions = []
    for side_index, side in enumerate(("A", "B")):
        for seed in sorted(seeds):
            joint = records[(side, "ms3_joint_total", seed)]
            free = records[(side, "ms3_free_only", seed)]
            joint_metrics = joint["metrics"]
            free_metrics = free["metrics"]
            episodes = joint["episodes"]
            dynamic = np.asarray(episodes["dynamic_mask"], dtype=bool)
            days = np.asarray(episodes["utc_days"])
            logged = np.asarray(episodes["logged_mae_c"], dtype=float)
            baseline = np.asarray(episodes["baseline_action_mae_c"], dtype=float)
            shuffled = np.asarray(episodes["shuffled_action_mae_c"], dtype=float)
            base_seed = int(gates["day_block_bootstrap_seed"]) + side_index * 100 + seed * 10
            baseline_gain = day_block_bootstrap(
                logged,
                baseline,
                days,
                dynamic,
                samples=int(gates["day_block_bootstrap_samples"]),
                seed=base_seed + 1,
            )
            shuffled_gain = day_block_bootstrap(
                logged,
                shuffled,
                days,
                dynamic,
                samples=int(gates["day_block_bootstrap_samples"]),
                seed=base_seed + 2,
            )
            free_mae = float(free_metrics["logged_mae_c"])
            ratio = float(joint_metrics["logged_mae_c"]) / max(free_mae, 1e-12)
            support_pass = bool(
                joint_metrics["dynamic_support"]["window_count"]
                >= gates["min_dynamic_windows"]
                and joint_metrics["dynamic_support"]["day_count"]
                >= gates["min_dynamic_utc_days"]
            )
            effect_pass = bool(
                joint_metrics["dynamic_mean_abs_effect_c"] is not None
                and joint_metrics["dynamic_mean_abs_effect_c"]
                >= gates["min_dynamic_mean_abs_effect_c"]
                and joint_metrics["max_abs_effect_c"] <= gates["max_abs_effect_c"]
            )
            baseline_ci_pass = bool(
                baseline_gain["ci95"] is not None
                and baseline_gain["ci95"][0]
                > gates["logged_vs_baseline_ci_lower_min_c"]
            )
            shuffled_ci_pass = bool(
                shuffled_gain["ci95"] is not None
                and shuffled_gain["ci95"][0]
                > gates["logged_vs_shuffled_ci_lower_min_c"]
            )
            prediction_pass = ratio <= gates["joint_to_free_logged_mae_ratio_max"]
            passes = bool(
                support_pass
                and effect_pass
                and baseline_ci_pass
                and shuffled_ci_pass
                and prediction_pass
            )
            seed_decisions.append(
                {
                    "side": side,
                    "seed": seed,
                    "passes": passes,
                    "joint_to_free_logged_mae_ratio": ratio,
                    "prediction_noninferiority_pass": prediction_pass,
                    "dynamic_support_pass": support_pass,
                    "response_noncollapse_pass": effect_pass,
                    "logged_vs_baseline": baseline_gain,
                    "logged_vs_baseline_pass": baseline_ci_pass,
                    "logged_vs_shuffled": shuffled_gain,
                    "logged_vs_shuffled_pass": shuffled_ci_pass,
                }
            )
    side_decisions = []
    for side in ("A", "B"):
        selected = [item for item in seed_decisions if item["side"] == side]
        pass_count = sum(bool(item["passes"]) for item in selected)
        side_decisions.append(
            {
                "side": side,
                "successful_seed_count": pass_count,
                "required_seed_count": gates["min_successful_seeds_per_side"],
                "passes": pass_count >= gates["min_successful_seeds_per_side"],
            }
        )
    return {
        "seed_decisions": seed_decisions,
        "side_decisions": side_decisions,
        "observational_validation_pass": all(item["passes"] for item in side_decisions),
        "interpretation": "conditional_prediction_and_action_alignment_only_not_do_valve",
    }


def _write_deterministic_archive(output_root: Path, members: list[Path]) -> dict[str, Any]:
    archive = output_root / "checkpoints_validation.tar"
    with tarfile.open(archive, "w") as handle:
        for path in sorted(members, key=lambda item: str(item.relative_to(output_root))):
            data = path.read_bytes()
            name = str(path.relative_to(output_root)).replace("\\", "/")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            handle.addfile(info, io.BytesIO(data))
    return {
        "path": str(archive.relative_to(ROOT)).replace("\\", "/")
        if archive.is_relative_to(ROOT)
        else str(archive),
        "sha256": _sha256(archive),
        "members": [
            {
                "path": str(path.relative_to(output_root)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in sorted(members, key=lambda item: str(item.relative_to(output_root)))
        ],
    }


def build_summary(matrix_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    matrix_path = Path(matrix_path).resolve()
    output_root = Path(output_root).resolve()
    matrix = load_matrix(matrix_path)
    matrix_sha = _sha256(matrix_path)
    expected_operator = RealModelConfig(**matrix["model"]).operator_config(
        int(matrix["model"]["d_model"]) * 2
    ).to_dict()
    forbidden = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_TEST_ARTIFACTS
    )
    if forbidden:
        raise RuntimeError(f"MS3 summary refuses test artifacts: {forbidden}")
    artifact_failures: list[str] = []
    structural_failures: list[str] = []
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    execution_shas: set[str] = set()
    checkpoints: list[Path] = []
    run_summaries = []
    trajectory_pins: dict[tuple[str, int], set[str]] = {}
    tolerance = float(matrix["gates"]["structural_tolerance"])
    for run in expand_runs(matrix):
        run_id = f"{run['side']}_{run['candidate_id']}_s{run['seed']}"
        run_dir = output_root / run_id
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            artifact_failures.append(f"{run_id}:missing={missing}")
            continue
        manifest = _read_json(run_dir / "manifest.json")
        history = _read_json(run_dir / "history.json")
        metrics = _read_json(run_dir / "metrics_validation.json")
        episodes = _read_json(run_dir / "episode_metrics_validation.json")
        checkpoint = run_dir / "checkpoint_best_val.pt"
        checkpoints.append(checkpoint)
        expected_manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "side": run["side"],
            "seed": run["seed"],
            "mode": run["mode"],
            "candidate_role": run["role"],
            "matrix_sha256": matrix_sha,
            "model_config": matrix["model"],
            "training_config": matrix["training"],
            "operator_config": expected_operator,
            "feature_columns": matrix["data_contract"]["history_features"],
            "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
            "test_accessed": False,
            "test_authorized": False,
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                artifact_failures.append(f"{run_id}:{key}")
        if manifest.get("checkpoint_sha256") != _sha256(checkpoint):
            artifact_failures.append(f"{run_id}:checkpoint_sha256")
        if manifest.get("epochs_ran") != len(history):
            artifact_failures.append(f"{run_id}:epochs_ran")
        if not 1 <= int(manifest.get("best_epoch", 0)) <= len(history):
            artifact_failures.append(f"{run_id}:best_epoch")
        if len(episodes.get("anchors", [])) != metrics.get("sample_count"):
            artifact_failures.append(f"{run_id}:episode_count")
        if manifest.get("validation_trajectory_sha256") != episodes.get(
            "validation_trajectory_sha256"
        ):
            artifact_failures.append(f"{run_id}:trajectory_sha256")
        if manifest.get("cache_metadata", {}).get("source", {}).get("sha256") != matrix[
            "data_contract"
        ]["source_sha256"]:
            artifact_failures.append(f"{run_id}:source_sha256")
        side_contract = matrix["data_contract"]["side_mappings"][run["side"]]
        if manifest.get("cache_metadata", {}).get("control_loop") != side_contract[
            "control_loop"
        ]:
            artifact_failures.append(f"{run_id}:control_loop")
        if manifest.get("cache_metadata", {}).get("column_map") != side_contract[
            "column_map"
        ]:
            artifact_failures.append(f"{run_id}:column_map")
        if manifest.get("cache_metadata", {}).get("matrix_sha256") != matrix_sha:
            artifact_failures.append(f"{run_id}:cache_matrix_sha256")
        execution_shas.add(str(manifest.get("git_sha")))
        trajectory_pins.setdefault((run["side"], run["seed"]), set()).add(
            str(episodes.get("validation_trajectory_sha256"))
        )
        structural = metrics.get("structural_diagnostics", {})
        structural_pass = bool(
            structural.get("reference_identity_max_error", math.inf) <= tolerance
            and structural.get("free_future_action_leakage_max_error", math.inf) <= tolerance
            and structural.get("future_action_prefix_leakage_max_error", math.inf) <= tolerance
            and structural.get("positive_step_terminal_effect_max_c", math.inf) <= tolerance
            and structural.get("finite_prediction") is True
            and structural.get("finite_free") is True
            and structural.get("finite_effect") is True
            and structural.get("finite_state") is True
        )
        if run["mode"] == "free_only":
            structural_pass = structural_pass and metrics.get("max_abs_effect_c") == 0
            if any(float(item.get("response_gradient_norm", 0.0)) != 0.0 for item in history):
                structural_pass = False
        else:
            if not any(float(item.get("response_gradient_norm", 0.0)) > 0.0 for item in history):
                structural_pass = False
        if not structural_pass:
            structural_failures.append(run_id)
        records[(run["side"], run["candidate_id"], run["seed"])] = {
            "metrics": metrics,
            "episodes": episodes,
        }
        run_summaries.append(
            {
                "run_id": run_id,
                "side": run["side"],
                "seed": run["seed"],
                "candidate_id": run["candidate_id"],
                "logged_mae_c": metrics["logged_mae_c"],
                "dynamic_support": metrics["dynamic_support"],
                "dynamic_mean_abs_effect_c": metrics["dynamic_mean_abs_effect_c"],
                "structural_pass": structural_pass,
            }
        )
    for key, pins in trajectory_pins.items():
        if len(pins) != 1:
            artifact_failures.append(f"{key}:candidate_trajectory_mismatch")
    if len(execution_shas) != 1 or "None" in execution_shas:
        artifact_failures.append("execution_git_sha_not_unique")
    if len(records) != len(expand_runs(matrix)):
        decision = {
            "observational_validation_pass": False,
            "reason": "incomplete_run_matrix",
        }
    else:
        decision = decide_ms3(records, matrix["seeds"], matrix["gates"])
    artifact_pass = not artifact_failures and not structural_failures
    if execution_shas and len(execution_shas) == 1 and "None" not in execution_shas:
        _assert_code_equivalent(next(iter(execution_shas)))
    archive = _write_deterministic_archive(output_root, checkpoints) if len(checkpoints) == 12 else None
    all_pass = bool(artifact_pass and decision.get("observational_validation_pass"))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_scope": matrix["evidence_scope"],
        "split": "validation",
        "run_count": len(run_summaries),
        "expected_run_count": 12,
        "matrix_sha256": matrix_sha,
        "execution_git_sha": next(iter(execution_shas)) if len(execution_shas) == 1 else None,
        "summary_git_sha": _git_sha(),
        "artifact_gate": {
            "passes": artifact_pass,
            "artifact_failures": artifact_failures,
            "structural_failures": structural_failures,
        },
        "runs": run_summaries,
        "decision": decision,
        "checkpoint_archive": archive,
        "all_primary_gates_pass": all_pass,
        "next_gate": "ms4_closed_loop_validation" if all_pass else "ms3_diagnosis_no_retries",
        "test_accessed": False,
        "claim_boundary": (
            "Logged-action advantage is observational conditional-prediction evidence. "
            "It is not do(valve), open-loop plant identification, or closed-loop validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-root", default="results/phase3_5/ms3_real_adaptation")
    parser.add_argument("--output")
    args = parser.parse_args()
    summary = build_summary(args.matrix, args.output_root)
    output = Path(args.output) if args.output else Path(args.output_root) / "summary_validation.json"
    _json_dump(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_primary_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
