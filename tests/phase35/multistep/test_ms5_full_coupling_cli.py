from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms5_full_coupling_matrix.json"
RUNNER = ROOT / "experiments/phase3_5/ms5_full_coupling.py"


def test_ms5_matrix_expands_to_12_validation_runs_without_test_access():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--matrix", str(MATRIX), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms5-v1"
    assert payload["run_count"] == 12
    assert payload["test_authorized"] is False
    assert {run["mode"] for run in payload["runs"]} == {
        "free_only",
        "joint_total",
        "staged_total",
        "component_oracle",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("synthetic", "action_context_coupling_pct"), 2.0),
        (("training", "stage_b_epochs"), 120),
        (("gates", "eligible_response_clean_nmae_max"), 0.20),
        (("candidates", 1, "mode"), "staged_total"),
    ],
)
def test_ms5_matrix_rejects_protocol_mutations(tmp_path, path, value):
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    target = matrix
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(matrix), encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, str(RUNNER), "--matrix", str(changed), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "frozen" in rejected.stderr.lower()


def test_ms5_cpu_smoke_writes_component_metrics(tmp_path):
    output_root = tmp_path / "ms5"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--candidate-id",
            "ms5_joint_total",
            "--seed",
            "0",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--smoke",
            "--execute",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    run = payload["runs"][0]
    run_dir = Path(run["output_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (run_dir / "metrics_validation.json").read_text(encoding="utf-8")
    )
    assert run_dir.name == "ms5_joint_total_s0"
    assert manifest["training_mode"] == "joint_total"
    assert manifest["test_accessed"] is False
    assert metrics["truth"]["truth_regime"] == "full_coupled_context_scheduled"
    assert "free_clean_nmae" in metrics
    assert "response_clean_nmae" in metrics
    assert not (run_dir / "metrics_test.json").exists()


def test_ms5_validation_refuses_premature_test_artifacts(tmp_path):
    output_root = tmp_path / "blocked"
    output_root.mkdir()
    (output_root / "summary_test.json").write_text("{}", encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--candidate-id",
            "ms5_joint_total",
            "--seed",
            "0",
            "--output-root",
            str(output_root),
            "--smoke",
            "--execute",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "refuses test artifacts" in rejected.stderr


def _metrics(total: float, free: float, response: float, amplitude: float) -> dict:
    return {
        "total_clean_nmae": total,
        "free_clean_nmae": free,
        "response_clean_nmae": response,
        "response_amplitude_ratio": amplitude,
    }


def _decision_fixture() -> dict[tuple[str, int], dict]:
    records = {}
    for seed in (0, 1, 2):
        records[("ms5_free_only", seed)] = _metrics(0.3, 0.08, 1.0, 0.0)
        records[("ms5_joint_total", seed)] = _metrics(0.07, 0.07, 0.10, 1.0)
        records[("ms5_staged_total", seed)] = _metrics(0.08, 0.08, 0.11, 1.0)
        records[("ms5_component_oracle", seed)] = _metrics(0.03, 0.03, 0.04, 1.0)
    return records


def test_ms5_decision_prefers_simpler_joint_when_it_qualifies():
    from experiments.phase3_5.summarize_ms5_full_coupling import decide_strategy
    from experiments.phase3_5.ms5_full_coupling import _expected_gates

    result = decide_strategy(_decision_fixture(), [0, 1, 2], _expected_gates())
    assert result["oracle_gate"]["all_seeds_pass"] is True
    assert result["selected_strategy"] == "ms5_joint_total"
    assert result["validation_strategy_pass"] is True


def test_ms5_decision_uses_staged_only_when_joint_fails_and_ratio_is_bounded():
    from experiments.phase3_5.summarize_ms5_full_coupling import decide_strategy
    from experiments.phase3_5.ms5_full_coupling import _expected_gates

    records = _decision_fixture()
    for seed in (0, 1, 2):
        records[("ms5_joint_total", seed)] = _metrics(0.095, 0.07, 0.18, 1.0)
        records[("ms5_staged_total", seed)] = _metrics(0.09, 0.08, 0.12, 1.0)
    result = decide_strategy(records, [0, 1, 2], _expected_gates())
    assert result["joint_gate"]["all_seeds_pass"] is False
    assert result["staged_gate"]["all_seeds_pass"] is True
    assert result["selected_strategy"] == "ms5_staged_total"


def test_ms5_decision_fails_closed_when_oracle_or_both_strategies_fail():
    from experiments.phase3_5.summarize_ms5_full_coupling import decide_strategy
    from experiments.phase3_5.ms5_full_coupling import _expected_gates

    records = _decision_fixture()
    records[("ms5_component_oracle", 0)] = _metrics(0.12, 0.03, 0.04, 1.0)
    result = decide_strategy(records, [0, 1, 2], _expected_gates())
    assert result["validation_strategy_pass"] is False
    assert result["selected_strategy"] is None

    records = _decision_fixture()
    records[("ms5_component_oracle", 1)]["response_amplitude_ratio"] = 1.21
    result = decide_strategy(records, [0, 1, 2], _expected_gates())
    assert result["oracle_gate"]["all_seeds_pass"] is False
    assert result["selected_strategy"] is None

    records = _decision_fixture()
    for seed in (0, 1, 2):
        records[("ms5_joint_total", seed)] = _metrics(0.12, 0.12, 0.20, 0.7)
        records[("ms5_staged_total", seed)] = _metrics(0.13, 0.12, 0.20, 0.7)
    result = decide_strategy(records, [0, 1, 2], _expected_gates())
    assert result["validation_strategy_pass"] is False
    assert result["selected_strategy"] is None


def _write_summary_fixture(output_root: Path) -> None:
    from experiments.phase3_5.ms5_full_coupling import (
        FROZEN_EXECUTION_PATHS,
        _configs,
        expand_runs,
        load_matrix,
    )

    matrix = load_matrix(MATRIX)
    operator, full, synthetic, _ = _configs(matrix, False)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    matrix_sha = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    values = {
        "ms5_free_only": _metrics(0.30, 0.08, 1.00, 0.00),
        "ms5_joint_total": _metrics(0.07, 0.07, 0.10, 1.00),
        "ms5_staged_total": _metrics(0.08, 0.08, 0.11, 1.00),
        "ms5_component_oracle": _metrics(0.03, 0.03, 0.04, 1.00),
    }
    candidates = {item["candidate_id"]: item for item in matrix["candidates"]}
    for run in expand_runs(matrix):
        candidate_id = run["candidate_id"]
        seed = run["seed"]
        run_dir = output_root / f"{candidate_id}_s{seed}"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint_best_val.pt"
        checkpoint.write_bytes(f"{candidate_id}-{seed}".encode())
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        trajectory_sha = hashlib.sha256(f"ms5-seed-{seed}".encode()).hexdigest()
        metric = values[candidate_id]
        response_scale = [1.0] * 8
        metrics = {
            **metric,
            "structural_diagnostics": {
                "reference_identity_max_error": 0.0,
                "future_action_leakage_max_error": 0.0,
                "free_future_action_leakage_max_error": 0.0,
                "post_change_sensitivity_max_c": (
                    0.0 if candidate_id == "ms5_free_only" else 0.1
                ),
                "positive_step_terminal_effect_max_c": 0.0,
                "finite_effect": True,
                "finite_state": True,
                "finite_prediction": True,
                "finite_free": True,
            },
            "parameter_drift": {
                "free_l2": 1.0,
                "response_l2": (
                    0.0 if candidate_id == "ms5_free_only" else 1.0
                ),
            },
        }
        episodes = {
            "episode_ids": list(range(8)),
            "profile_ids": [0, 1, 2, 3, 0, 1, 2, 3],
            "profile_names": ["hold", "step", "ramp", "pulse"],
            "trajectory_design_sha256": trajectory_sha,
            "total_clean_mae": [metric["total_clean_nmae"]] * 8,
            "total_clean_scale": [1.0] * 8,
            "free_clean_mae": [metric["free_clean_nmae"]] * 8,
            "free_clean_scale": [1.0] * 8,
            "response_clean_mae": [metric["response_clean_nmae"]] * 8,
            "response_clean_scale": response_scale,
            "predicted_response_abs": [metric["response_amplitude_ratio"]] * 8,
            "true_response_abs": response_scale,
        }
        manifest = {
            "protocol_version": matrix["protocol_version"],
            "evidence_scope": matrix["evidence_scope"],
            "route_id": candidate_id,
            "seed": seed,
            "training_mode": run["mode"],
            "candidate_role": candidates[candidate_id]["role"],
            "operator_config": operator.to_dict(),
            "full_training_config": asdict(full),
            "synthetic_spec": asdict(
                replace(synthetic, seed=synthetic.seed + seed * 1_000_003)
            ),
            "checkpoint_selector": "validation_total_noisy_mae",
            "matrix_sha256": matrix_sha,
            "d3_reference_sha256": matrix["d3_reference"]["sha256"],
            "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
            "checkpoint_sha256": checkpoint_sha,
            "validation_trajectory_design_sha256": trajectory_sha,
            "best_epoch": (3 if candidate_id == "ms5_staged_total" else 1),
            "epochs_ran": (3 if candidate_id == "ms5_staged_total" else 1),
            "git_sha": git_sha,
            "device": "cpu",
            "environment": {"device": "cpu"},
            "stage_checkpoints": [],
            "stage_summaries": [],
            "test_accessed": False,
            "test_authorized": False,
        }
        if candidate_id == "ms5_staged_total":
            for stage in (
                "stage_a_free_hold",
                "stage_b_response_frozen_free",
                "stage_c_low_lr_joint",
            ):
                stage_path = run_dir / f"checkpoint_{stage}.pt"
                stage_path.write_bytes(f"{candidate_id}-{seed}-{stage}".encode())
                manifest["stage_checkpoints"].append(
                    {
                        "stage": stage,
                        "path": stage_path.name,
                        "sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
                    }
                )
            stage_specs = [
                ("stage_a_free_hold", 80),
                ("stage_b_response_frozen_free", 140),
                ("stage_c_low_lr_joint", 80),
            ]
        else:
            stage_specs = [(run["mode"], 300)]
        history = []
        for global_epoch, (stage, epoch_cap) in enumerate(stage_specs, start=1):
            history.append(
                {
                    "epoch": global_epoch,
                    "phase": stage,
                    "phase_epoch": 1,
                    "validation_total_noisy_mae": 0.1,
                }
            )
            manifest["stage_summaries"].append(
                {
                    "stage": stage,
                    "epoch_cap": epoch_cap,
                    "epochs_ran": 1,
                    "best_stage_epoch": 1,
                    "best_validation_total_noisy_mae": 0.1,
                }
            )
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "history.json").write_text(
            json.dumps(history),
            encoding="utf-8",
        )
        (run_dir / "metrics_validation.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (run_dir / "episode_metrics_validation.json").write_text(
            json.dumps(episodes), encoding="utf-8"
        )


def test_ms5_summary_replays_all_12_artifacts_and_archives_checkpoints(tmp_path):
    from experiments.phase3_5.summarize_ms5_full_coupling import build_summary

    output_root = tmp_path / "ms5"
    _write_summary_fixture(output_root)
    summary = build_summary(MATRIX, output_root)
    assert summary["run_count"] == 12
    assert summary["artifact_gate"]["all_artifact_and_structural_gates_pass"] is True
    assert summary["strategy_decision"]["selected_strategy"] == "ms5_joint_total"
    assert summary["all_primary_gates_pass"] is True
    assert len(summary["checkpoint_archive"]["members"]) == 21
    assert (output_root / "checkpoints_validation.tar").is_file()
