from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from src.phase35.data import Phase35Cache, save_cache
from src.phase35.multistep.real_training import (
    RealModelConfig,
    RealTrainingConfig,
    build_real_model,
    shuffled_delta_paths,
    train_real_run,
)
from src.phase35.schema import MS3_HISTORY_FEATURES, TARGET_COLUMN, VALVE_COLUMN


def _cache(rows: int = 500, side: str = "A") -> Phase35Cache:
    columns = tuple(MS3_HISTORY_FEATURES)
    time = np.arange(rows, dtype=np.float32)
    values = np.zeros((rows, len(columns)), dtype=np.float32)
    for index in range(len(columns)):
        values[:, index] = 10.0 + index + 0.01 * time
    valve = 30.0 + 4.0 * np.sin(time / 12.0)
    target = 565.0 + 0.5 * np.sin(time / 20.0) - 0.03 * (valve - 30.0)
    values[:, columns.index(VALVE_COLUMN)] = valve
    values[:, columns.index(TARGET_COLUMN)] = target
    values[:, columns.index("机组负荷")] = 500.0
    values[:, columns.index("主蒸汽压力")] = 20.0
    values[:, columns.index("二级减温调节阀设定")] = 565.0
    return Phase35Cache(
        timestamps_ns=np.arange(rows, dtype=np.int64) * 10_000_000_000,
        values=values,
        ages_s=np.zeros_like(values),
        columns=columns,
        metadata={
            "protocol_version": "phase3.5-ms3-smoke",
            "side": side,
            "step_seconds": 10,
            "cross_pairing_frozen": True,
            "control_loop": f"{side}_smoke_cross_loop",
            "source": {"sha256": "smoke"},
        },
    )


def _model_config() -> RealModelConfig:
    return RealModelConfig(
        window=12,
        horizon=12,
        d_model=8,
        n_heads=2,
        dropout=0.0,
    )


def test_ms3_free_head_is_future_action_blind_and_reference_identity_is_exact():
    cache = _cache(80)
    model = build_real_model(_model_config(), cache.columns, "joint_total")
    history = torch.from_numpy(cache.values[:3, None, :].repeat(12, axis=1))
    baseline = torch.tensor([30.0, 35.0, 40.0])
    action = baseline[:, None].expand(3, 12).clone()
    identity = model(history, action, baseline)
    changed = action.clone()
    changed[:, 6:] += 5.0
    intervention = model(history, changed, baseline)
    torch.testing.assert_close(identity["effect"], torch.zeros_like(identity["effect"]), atol=0, rtol=0)
    torch.testing.assert_close(
        identity["free_prediction"], intervention["free_prediction"], atol=0, rtol=0
    )
    assert float(intervention["effect"][:, -1].max()) <= 0.0


def test_ms3_shuffled_action_preserves_own_baseline_and_avoids_fixed_points():
    baseline = np.asarray([21.0, 22.0, 23.0, 24.0], dtype=np.float32)
    deltas = np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0
    future = baseline[:, None] + deltas
    shuffled, design = shuffled_delta_paths(future, baseline, seed=7)
    assert shuffled.shape == future.shape
    assert design["fixed_point_count"] == 0
    assert design["permuted_group_count"] == 1
    assert np.all(np.isfinite(shuffled))


@pytest.mark.parametrize("mode", ["joint_total", "free_only"])
def test_ms3_real_training_cpu_smoke_writes_validation_only_artifacts(tmp_path, mode):
    cache = _cache()
    cache_path = tmp_path / "cache.npz"
    save_cache(cache, cache_path)
    output = tmp_path / mode
    result = train_real_run(
        cache=cache,
        cache_path=cache_path,
        feature_columns=cache.columns,
        model_config=_model_config(),
        training_config=RealTrainingConfig(
            batch_size=8,
            epochs=2,
            patience=2,
            steps_per_epoch=2,
            max_train_anchors=64,
            max_selector_anchors=16,
            max_validation_anchors=32,
            dynamic_dose_threshold_pct=0.1,
        ),
        side="A",
        seed=0,
        mode=mode,
        run_id=f"A_ms3_{mode}_s0",
        output_dir=output,
        device="cpu",
        protocol_version="phase3.5-ms3-smoke",
        matrix_sha256="smoke-matrix",
        repo_git_sha="smoke-sha",
    )
    assert result.checkpoint.is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (output / "metrics_validation.json").read_text(encoding="utf-8")
    )
    episodes = json.loads(
        (output / "episode_metrics_validation.json").read_text(encoding="utf-8")
    )
    assert manifest["test_accessed"] is False
    assert manifest["checkpoint_selector"] == "validation_logged_mae_c"
    assert manifest["selector_anchor_count"] == 16
    assert metrics["sample_count"] == 32
    assert len(episodes["anchors"]) == 32
    assert metrics["structural_diagnostics"]["reference_identity_max_error"] == 0
    if mode == "free_only":
        assert metrics["max_abs_effect_c"] == 0


def _decision_records(logged: float = 0.8) -> dict:
    records = {}
    days = [f"2026-01-{day:02d}" for day in range(1, 13)]
    for side in ("A", "B"):
        for seed in (0, 1, 2):
            records[(side, "ms3_joint_total", seed)] = {
                "metrics": {
                    "logged_mae_c": logged,
                    "dynamic_support": {"window_count": 1200, "day_count": 12},
                    "dynamic_mean_abs_effect_c": 0.1,
                    "max_abs_effect_c": 1.0,
                },
                "episodes": {
                    "dynamic_mask": [True] * 12,
                    "utc_days": days,
                    "logged_mae_c": [logged] * 12,
                    "baseline_action_mae_c": [1.0] * 12,
                    "shuffled_action_mae_c": [1.1] * 12,
                },
            }
            records[(side, "ms3_free_only", seed)] = {
                "metrics": {"logged_mae_c": 0.9},
                "episodes": {},
            }
    return records


def test_ms3_decision_requires_both_sides_and_two_of_three_seeds():
    from experiments.phase3_5.ms3_real_adaptation import load_matrix
    from experiments.phase3_5.summarize_ms3_real_adaptation import decide_ms3

    matrix = load_matrix("configs/phase3_5/ms3_real_adaptation_matrix.json")
    result = decide_ms3(_decision_records(), [0, 1, 2], matrix["gates"])
    assert result["observational_validation_pass"] is True
    assert all(item["successful_seed_count"] == 3 for item in result["side_decisions"])

    records = _decision_records(logged=1.2)
    failed = decide_ms3(records, [0, 1, 2], matrix["gates"])
    assert failed["observational_validation_pass"] is False


def test_ms3_summary_replays_all_artifacts_and_archives_checkpoints(tmp_path):
    from experiments.phase3_5.ms3_real_adaptation import (
        FROZEN_EXECUTION_PATHS,
        expand_runs,
        load_matrix,
    )
    from experiments.phase3_5.summarize_ms3_real_adaptation import build_summary

    root = Path(__file__).resolve().parents[3]
    matrix_path = root / "configs/phase3_5/ms3_real_adaptation_matrix.json"
    matrix = load_matrix(matrix_path)
    expected_operator = RealModelConfig(**matrix["model"]).operator_config(
        matrix["model"]["d_model"] * 2
    ).to_dict()
    matrix_sha = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    output_root = tmp_path / "ms3"
    days = [f"2026-01-{day:02d}" for day in range(1, 13)]
    for run in expand_runs(matrix):
        run_id = f"{run['side']}_{run['candidate_id']}_s{run['seed']}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint_best_val.pt"
        checkpoint.write_bytes(run_id.encode())
        joint = run["mode"] == "joint_total"
        trajectory_sha = hashlib.sha256(
            f"{run['side']}-{run['seed']}".encode()
        ).hexdigest()
        manifest = {
            "protocol_version": matrix["protocol_version"],
            "evidence_scope": matrix["evidence_scope"],
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
            "git_sha": git_sha,
            "test_accessed": False,
            "test_authorized": False,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "epochs_ran": 1,
            "best_epoch": 1,
            "validation_trajectory_sha256": trajectory_sha,
            "cache_metadata": {
                "source": {"sha256": matrix["data_contract"]["source_sha256"]},
                "control_loop": matrix["data_contract"]["side_mappings"][run["side"]]["control_loop"],
                "column_map": matrix["data_contract"]["side_mappings"][run["side"]]["column_map"],
                "matrix_sha256": matrix_sha,
                "timestamp_storage_unit": matrix["data_contract"]["timestamp_storage_unit"],
                "grid_start_ns": matrix["data_contract"]["grid_start_ns"],
                "grid_end_ns": matrix["data_contract"]["grid_end_ns"],
                "grid_rows": matrix["data_contract"]["source_rows"],
                "irregular_transition_count": matrix["data_contract"]["irregular_transition_count"],
                "max_transition_seconds": matrix["data_contract"]["max_transition_seconds"],
            },
            "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
        }
        metrics = {
            "sample_count": 12,
            "logged_mae_c": 0.8 if joint else 0.9,
            "dynamic_support": {"window_count": 1200, "day_count": 12},
            "dynamic_mean_abs_effect_c": 0.1 if joint else 0.0,
            "max_abs_effect_c": 1.0 if joint else 0.0,
            "structural_diagnostics": {
                "reference_identity_max_error": 0.0,
                "free_future_action_leakage_max_error": 0.0,
                "future_action_prefix_leakage_max_error": 0.0,
                "positive_step_terminal_effect_max_c": 0.0,
                "finite_prediction": True,
                "finite_free": True,
                "finite_effect": True,
                "finite_state": True,
            },
        }
        episodes = {
            "anchors": list(range(12)),
            "utc_days": days,
            "dynamic_mask": [True] * 12,
            "logged_mae_c": [0.8 if joint else 0.9] * 12,
            "baseline_action_mae_c": [1.0 if joint else 0.9] * 12,
            "shuffled_action_mae_c": [1.1 if joint else 0.9] * 12,
            "validation_trajectory_sha256": trajectory_sha,
        }
        history = [
            {
                "epoch": 1,
                "response_gradient_norm": 0.1 if joint else 0.0,
            }
        ]
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "metrics_validation.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "episode_metrics_validation.json").write_text(
            json.dumps(episodes), encoding="utf-8"
        )
        (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    summary = build_summary(matrix_path, output_root)
    assert summary["run_count"] == 12
    assert summary["artifact_gate"]["passes"] is True
    assert summary["decision"]["observational_validation_pass"] is True
    assert summary["all_primary_gates_pass"] is True
    assert len(summary["checkpoint_archive"]["members"]) == 12
