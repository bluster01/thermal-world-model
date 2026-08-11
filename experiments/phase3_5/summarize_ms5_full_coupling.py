#!/usr/bin/env python3
"""Fail-closed summary for the frozen Phase 3.5-MS5 validation matrix."""

from __future__ import annotations

import argparse
import json
import math
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

from experiments.phase3_5.ms5_full_coupling import (  # noqa: E402
    DEFAULT_MATRIX,
    FORBIDDEN_TEST_ARTIFACTS,
    FROZEN_EXECUTION_PATHS,
    PROTOCOL_VERSION,
    _canonical,
    _configs,
    _sha256,
    expand_runs,
    load_matrix,
)
from src.phase35.multistep.training import _json_dump  # noqa: E402


ORACLE_ID = "ms5_component_oracle"
FREE_ONLY_ID = "ms5_free_only"
JOINT_ID = "ms5_joint_total"
STAGED_ID = "ms5_staged_total"
REQUIRED_RUN_FILES = (
    "manifest.json",
    "history.json",
    "metrics_validation.json",
    "episode_metrics_validation.json",
    "checkpoint_best_val.pt",
)


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required MS5 artifact missing: {path}")
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
        raise RuntimeError(f"MS5 execution commit unavailable: {execution_sha}")
    compared = subprocess.run(
        ["git", "diff", "--quiet", execution_sha, "HEAD", "--", *FROZEN_EXECUTION_PATHS],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(f"MS5 frozen execution code differs from {execution_sha}")
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking MS5 code equivalence")


def _eligible(metrics: dict[str, Any], gates: dict[str, float]) -> bool:
    return bool(
        metrics["total_clean_nmae"] < gates["eligible_total_clean_nmae_max"]
        and metrics["free_clean_nmae"] < gates["eligible_free_clean_nmae_max"]
        and metrics["response_clean_nmae"]
        < gates["eligible_response_clean_nmae_max"]
        and gates["response_amplitude_ratio_min"]
        <= metrics["response_amplitude_ratio"]
        <= gates["response_amplitude_ratio_max"]
    )


def decide_strategy(
    records: dict[tuple[str, int], dict[str, Any]],
    seeds: list[int],
    gates: dict[str, float],
) -> dict[str, Any]:
    """Apply the frozen oracle-first, simplest-qualified strategy decision."""

    oracle_seed_results = []
    joint_seed_results = []
    staged_seed_results = []
    staged_ratios = []
    for seed in sorted(int(value) for value in seeds):
        oracle = records[(ORACLE_ID, seed)]
        joint = records[(JOINT_ID, seed)]
        staged = records[(STAGED_ID, seed)]
        oracle_pass = bool(
            oracle["total_clean_nmae"] < gates["oracle_total_clean_nmae_max"]
            and oracle["free_clean_nmae"] < gates["oracle_free_clean_nmae_max"]
            and oracle["response_clean_nmae"]
            < gates["oracle_response_clean_nmae_max"]
            and gates["response_amplitude_ratio_min"]
            <= oracle["response_amplitude_ratio"]
            <= gates["response_amplitude_ratio_max"]
        )
        oracle_seed_results.append({"seed": seed, "passes": oracle_pass, **oracle})
        joint_seed_results.append(
            {"seed": seed, "passes": _eligible(joint, gates), **joint}
        )
        staged_seed_results.append(
            {"seed": seed, "passes": _eligible(staged, gates), **staged}
        )
        denominator = float(joint["total_clean_nmae"])
        ratio = (
            float(staged["total_clean_nmae"]) / denominator
            if denominator > 1e-12
            else math.inf
        )
        staged_ratios.append(
            {
                "seed": seed,
                "staged_to_joint_total_clean_nmae_ratio": ratio,
                "passes": ratio <= gates["staged_total_mae_ratio_max"],
            }
        )

    oracle_pass = all(item["passes"] for item in oracle_seed_results)
    joint_pass = all(item["passes"] for item in joint_seed_results)
    staged_absolute_pass = all(item["passes"] for item in staged_seed_results)
    staged_ratio_pass = all(item["passes"] for item in staged_ratios)
    selected: str | None = None
    reason = "blocked_no_strategy_qualified"
    if not oracle_pass:
        reason = "blocked_positive_control_failed"
    elif joint_pass:
        selected = JOINT_ID
        reason = "joint_qualified_and_is_simplest"
    elif staged_absolute_pass and staged_ratio_pass:
        selected = STAGED_ID
        reason = "joint_failed_and_staged_qualified"
    return {
        "oracle_gate": {
            "seed_results": oracle_seed_results,
            "all_seeds_pass": oracle_pass,
        },
        "joint_gate": {
            "seed_results": joint_seed_results,
            "all_seeds_pass": joint_pass,
        },
        "staged_gate": {
            "seed_results": staged_seed_results,
            "absolute_all_seeds_pass": staged_absolute_pass,
            "ratio_seed_results": staged_ratios,
            "ratio_all_seeds_pass": staged_ratio_pass,
            "all_seeds_pass": staged_absolute_pass and staged_ratio_pass,
        },
        "selected_strategy": selected,
        "decision_reason": reason,
        "validation_strategy_pass": oracle_pass and selected is not None,
    }


def _replay_episode_metrics(episodes: dict[str, Any]) -> dict[str, float]:
    required = (
        "total_clean_mae",
        "total_clean_scale",
        "free_clean_mae",
        "free_clean_scale",
        "response_clean_mae",
        "response_clean_scale",
        "predicted_response_abs",
        "true_response_abs",
    )
    lengths = {len(episodes[key]) for key in required}
    lengths.update(
        {len(episodes["episode_ids"]), len(episodes["profile_ids"])}
    )
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise RuntimeError("MS5 episode arrays are empty or misaligned")

    def ratio(numerator: str, denominator: str) -> float:
        scale = statistics.fmean(float(value) for value in episodes[denominator])
        if scale <= 1e-12:
            raise RuntimeError(f"MS5 episode scale is zero: {denominator}")
        return statistics.fmean(float(value) for value in episodes[numerator]) / scale

    return {
        "total_clean_nmae": ratio("total_clean_mae", "total_clean_scale"),
        "free_clean_nmae": ratio("free_clean_mae", "free_clean_scale"),
        "response_clean_nmae": ratio(
            "response_clean_mae", "response_clean_scale"
        ),
        "response_amplitude_ratio": ratio(
            "predicted_response_abs", "true_response_abs"
        ),
    }


def _structural_failures(candidate_id: str, metrics: dict[str, Any]) -> list[str]:
    failures = []
    diagnostics = metrics.get("structural_diagnostics", {})
    exact_zero = (
        "reference_identity_max_error",
        "future_action_leakage_max_error",
        "free_future_action_leakage_max_error",
    )
    for key in exact_zero:
        value = diagnostics.get(key)
        if not isinstance(value, (int, float)) or abs(float(value)) > 1e-7:
            failures.append(f"{candidate_id}:{key}")
    for key in ("finite_effect", "finite_state", "finite_prediction", "finite_free"):
        if diagnostics.get(key) is not True:
            failures.append(f"{candidate_id}:{key}")
    terminal = diagnostics.get("positive_step_terminal_effect_max_c")
    if not isinstance(terminal, (int, float)) or float(terminal) > 1e-7:
        failures.append(f"{candidate_id}:positive_step_direction")
    sensitivity = diagnostics.get("post_change_sensitivity_max_c")
    if not isinstance(sensitivity, (int, float)):
        failures.append(f"{candidate_id}:post_change_sensitivity")
    elif candidate_id == FREE_ONLY_ID and abs(float(sensitivity)) > 1e-7:
        failures.append(f"{candidate_id}:negative_control_not_zero")
    elif candidate_id != FREE_ONLY_ID and float(sensitivity) <= 1e-6:
        failures.append(f"{candidate_id}:action_insensitive")
    return failures


def _stage_history_failures(
    candidate_id: str,
    mode: str,
    manifest: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[str]:
    failures = []
    expected = (
        [
            ("stage_a_free_hold", 80),
            ("stage_b_response_frozen_free", 140),
            ("stage_c_low_lr_joint", 80),
        ]
        if mode == "staged_total"
        else [(mode, 300)]
    )
    summaries = manifest.get("stage_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(expected):
        return [f"{candidate_id}:stage_summaries"]
    if manifest.get("epochs_ran") != len(history):
        failures.append(f"{candidate_id}:epochs_ran")
    cursor = 0
    final_selected_epoch = None
    for summary, (stage, epoch_cap) in zip(summaries, expected):
        if not isinstance(summary, dict):
            failures.append(f"{candidate_id}:{stage}:stage_summary")
            continue
        epochs_ran = summary.get("epochs_ran")
        best_stage_epoch = summary.get("best_stage_epoch")
        if (
            summary.get("stage") != stage
            or summary.get("epoch_cap") != epoch_cap
            or not isinstance(epochs_ran, int)
            or not 1 <= epochs_ran <= epoch_cap
            or not isinstance(best_stage_epoch, int)
            or not 0 <= best_stage_epoch <= epochs_ran
        ):
            failures.append(f"{candidate_id}:{stage}:stage_contract")
            continue
        records = history[cursor : cursor + epochs_ran]
        if len(records) != epochs_ran or any(
            record.get("phase") != stage
            or record.get("phase_epoch") != index
            for index, record in enumerate(records, start=1)
        ):
            failures.append(f"{candidate_id}:{stage}:history_partition")
        scores = [record.get("validation_total_noisy_mae") for record in records]
        if not all(
            isinstance(score, (int, float)) and math.isfinite(float(score))
            for score in scores
        ):
            failures.append(f"{candidate_id}:{stage}:validation_score")
        else:
            reported = summary.get("best_validation_total_noisy_mae")
            if not isinstance(reported, (int, float)) or not math.isfinite(
                float(reported)
            ):
                failures.append(f"{candidate_id}:{stage}:best_score")
            elif best_stage_epoch == 0:
                if float(reported) > min(float(score) for score in scores) + 1e-8:
                    failures.append(f"{candidate_id}:{stage}:initial_selector")
            elif not math.isclose(
                float(reported),
                float(scores[best_stage_epoch - 1]),
                rel_tol=1e-7,
                abs_tol=1e-8,
            ) or float(reported) > min(float(score) for score in scores) + 1e-8:
                failures.append(f"{candidate_id}:{stage}:best_epoch_replay")
        final_selected_epoch = cursor + best_stage_epoch
        cursor += epochs_ran
    if cursor != len(history):
        failures.append(f"{candidate_id}:unassigned_history")
    if manifest.get("best_epoch") != final_selected_epoch:
        failures.append(f"{candidate_id}:best_epoch")
    return failures


def _write_deterministic_archive(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    members = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for record in sorted(records, key=lambda item: (item["candidate_id"], item["seed"])):
            checkpoints = [Path(record["checkpoint_path"])] + [
                Path(item) for item in record["stage_checkpoint_paths"]
            ]
            for checkpoint in checkpoints:
                name = (
                    f"{record['candidate_id']}_s{record['seed']}/"
                    f"{checkpoint.name}"
                )
                info = archive.gettarinfo(str(checkpoint), arcname=name)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with checkpoint.open("rb") as handle:
                    archive.addfile(info, handle)
                members.append({"name": name, "sha256": _sha256(checkpoint)})
    return {"path": str(path), "sha256": _sha256(path), "members": members}


def build_summary(matrix_path: Path, output_root: Path) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    operator, full, synthetic, _ = _configs(matrix, False)
    matrix_sha = _sha256(matrix_path)
    run_records = []
    metrics_records: dict[tuple[str, int], dict[str, Any]] = {}
    trajectory_by_seed: dict[int, str] = {}
    execution_shas: set[str] = set()
    failures: list[str] = []
    found_test_artifacts = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_TEST_ARTIFACTS
    ) if output_root.exists() else []
    if found_test_artifacts:
        failures.append(f"premature_test_artifacts={found_test_artifacts}")

    candidates = {item["candidate_id"]: item for item in matrix["candidates"]}
    for run in expand_runs(matrix):
        candidate_id = run["candidate_id"]
        seed = int(run["seed"])
        run_dir = output_root / f"{candidate_id}_s{seed}"
        for name in REQUIRED_RUN_FILES:
            if not (run_dir / name).is_file():
                raise FileNotFoundError(f"required MS5 artifact missing: {run_dir / name}")
        manifest = _read_json(run_dir / "manifest.json")
        history = _read_json(run_dir / "history.json")
        metrics = _read_json(run_dir / "metrics_validation.json")
        episodes = _read_json(run_dir / "episode_metrics_validation.json")
        if not all(isinstance(item, dict) for item in (manifest, metrics, episodes)):
            raise RuntimeError(f"MS5 malformed JSON object in {run_dir}")
        if not isinstance(history, list) or not history:
            failures.append(f"{candidate_id}/s{seed}:history")
            history = []
        for expected_epoch, record in enumerate(history, start=1):
            if not isinstance(record, dict) or record.get("epoch") != expected_epoch:
                failures.append(f"{candidate_id}/s{seed}:history_contiguity")
                break
        failures.extend(
            f"{candidate_id}/s{seed}:{value.split(':', 1)[-1]}"
            for value in _stage_history_failures(
                candidate_id, run["mode"], manifest, history
            )
        )
        expected_spec = replace(synthetic, seed=synthetic.seed + seed * 1_000_003)
        expected_manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "evidence_scope": matrix["evidence_scope"],
            "route_id": candidate_id,
            "seed": seed,
            "training_mode": run["mode"],
            "candidate_role": candidates[candidate_id]["role"],
            "operator_config": operator.to_dict(),
            "full_training_config": asdict(full),
            "synthetic_spec": asdict(expected_spec),
            "checkpoint_selector": "validation_total_noisy_mae",
            "matrix_sha256": matrix_sha,
            "d3_reference_sha256": matrix["d3_reference"]["sha256"],
            "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
            "test_accessed": False,
            "test_authorized": False,
        }
        mismatches = [
            key
            for key, expected in expected_manifest.items()
            if _canonical(manifest.get(key)) != _canonical(expected)
        ]
        checkpoint = run_dir / "checkpoint_best_val.pt"
        if manifest.get("checkpoint_sha256") != _sha256(checkpoint):
            mismatches.append("checkpoint_sha256")
        best_epoch = manifest.get("best_epoch")
        if not isinstance(best_epoch, int) or not 0 <= best_epoch <= len(history):
            mismatches.append("best_epoch")
        execution_sha = manifest.get("git_sha")
        if not isinstance(execution_sha, str) or not execution_sha:
            mismatches.append("git_sha")
        else:
            execution_shas.add(execution_sha)
        environment = manifest.get("environment")
        if not isinstance(environment, dict) or environment.get("device") != manifest.get("device"):
            mismatches.append("environment")
        if mismatches:
            failures.append(f"{candidate_id}/s{seed}:manifest={sorted(set(mismatches))}")

        expected_stages = (
            [
                "stage_a_free_hold",
                "stage_b_response_frozen_free",
                "stage_c_low_lr_joint",
            ]
            if candidate_id == STAGED_ID
            else []
        )
        stage_checkpoints = manifest.get("stage_checkpoints")
        if not isinstance(stage_checkpoints, list) or [
            item.get("stage") for item in stage_checkpoints if isinstance(item, dict)
        ] != expected_stages:
            failures.append(f"{candidate_id}/s{seed}:stage_checkpoint_manifest")
        elif len(stage_checkpoints) != len(expected_stages):
            failures.append(f"{candidate_id}/s{seed}:stage_checkpoint_count")
        else:
            for item in stage_checkpoints:
                stage_path = run_dir / item.get("path", "")
                if not stage_path.is_file() or item.get("sha256") != _sha256(stage_path):
                    failures.append(
                        f"{candidate_id}/s{seed}:stage_checkpoint_hash"
                    )

        replay = _replay_episode_metrics(episodes)
        for key, value in replay.items():
            observed = metrics.get(key)
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), value, rel_tol=1e-6, abs_tol=1e-7
            ):
                failures.append(f"{candidate_id}/s{seed}:episode_replay_{key}")
        trajectory = episodes.get("trajectory_design_sha256")
        if episodes.get("episode_ids") != list(range(len(episodes["profile_ids"]))):
            failures.append(f"{candidate_id}/s{seed}:episode_ids")
        if trajectory != manifest.get("validation_trajectory_design_sha256"):
            failures.append(f"{candidate_id}/s{seed}:trajectory_manifest")
        if seed in trajectory_by_seed and trajectory_by_seed[seed] != trajectory:
            failures.append(f"seed={seed}:unpaired_candidate_trajectories")
        elif isinstance(trajectory, str):
            trajectory_by_seed[seed] = trajectory
        failures.extend(
            f"{candidate_id}/s{seed}:{value.split(':', 1)[-1]}"
            for value in _structural_failures(candidate_id, metrics)
        )
        if candidate_id == FREE_ONLY_ID:
            if abs(float(metrics.get("response_amplitude_ratio", math.inf))) > 1e-7:
                failures.append(f"{candidate_id}/s{seed}:response_not_zero")
            drift = metrics.get("parameter_drift", {}).get("response_l2")
            if not isinstance(drift, (int, float)) or abs(float(drift)) > 1e-7:
                failures.append(f"{candidate_id}/s{seed}:response_parameter_drift")
        metrics_records[(candidate_id, seed)] = {
            key: float(metrics[key])
            for key in (
                "total_clean_nmae",
                "free_clean_nmae",
                "response_clean_nmae",
                "response_amplitude_ratio",
            )
        }
        run_records.append(
            {
                "candidate_id": candidate_id,
                "mode": run["mode"],
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs_ran": manifest.get("epochs_ran"),
                "metrics": metrics_records[(candidate_id, seed)],
                "checkpoint_path": str(checkpoint),
                "stage_checkpoint_paths": [
                    str(run_dir / item["path"]) for item in stage_checkpoints
                ],
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )

    if len(execution_shas) != 1:
        failures.append(f"multiple_execution_shas={sorted(execution_shas)}")
    elif execution_shas:
        _assert_code_equivalent(next(iter(execution_shas)))
    if len(set(trajectory_by_seed.values())) != len(matrix["seeds"]):
        failures.append("validation_seed_trajectories_not_distinct")

    decision = decide_strategy(metrics_records, matrix["seeds"], matrix["gates"])
    archive = _write_deterministic_archive(
        run_records, output_root / "checkpoints_validation.tar"
    )
    try:
        archive["path"] = str(Path(archive["path"]).resolve().relative_to(ROOT))
    except ValueError:
        pass
    reported_runs = [
        {
            key: value
            for key, value in record.items()
            if key not in {"checkpoint_path", "stage_checkpoint_paths"}
        }
        for record in run_records
    ]
    try:
        reported_matrix_path = str(matrix_path.resolve().relative_to(ROOT))
    except ValueError:
        reported_matrix_path = str(matrix_path)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_scope": matrix["evidence_scope"],
        "summary_scope": "validation_only_no_test_by_budget_decision",
        "matrix_path": reported_matrix_path,
        "matrix_sha256": matrix_sha,
        "d3_reference": matrix["d3_reference"],
        "execution_git_sha": next(iter(execution_shas)) if len(execution_shas) == 1 else None,
        "summary_git_sha": _git_sha(),
        "run_count": len(run_records),
        "artifact_gate": {
            "all_artifact_and_structural_gates_pass": not failures,
            "failures": failures,
        },
        "strategy_decision": decision,
        "all_primary_gates_pass": not failures and decision["validation_strategy_pass"],
        "runs": reported_runs,
        "checkpoint_archive": archive,
        "test_accessed": False,
        "test_authorized": False,
        "next_gate": (
            "ms3_real_observational_validation"
            if not failures and decision["validation_strategy_pass"]
            else "blocked_for_supervisor_review"
        ),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-root", default="results/phase3_5/ms5_full_coupling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    output_root = Path(args.output_root).resolve()
    if matrix_path != DEFAULT_MATRIX.resolve():
        raise SystemExit("formal MS5 summary requires the frozen repository matrix")
    summary = build_summary(matrix_path, output_root)
    path = output_root / "summary_validation.json"
    _json_dump(path, summary)
    print(json.dumps(summary, indent=2))
    if not summary["all_primary_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
