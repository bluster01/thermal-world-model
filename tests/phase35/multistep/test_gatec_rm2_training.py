from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.phase35.data import Phase35Cache
from src.phase35.multistep.gatec_rm2_contracts import rm2_run_specs
from src.phase35.multistep.gatec_rm2_training import rm2_run_bounds, run_rm2_training
from src.phase35.schema import MS3_HISTORY_FEATURES


ROOT = Path(__file__).resolve().parents[3]


def _caches(rows: int = 1000) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(71)
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(rows, dtype=np.int64) * 10_000_000_000
    shared = {
        name: base + rng.normal(0, 0.2, rows)
        for name, base in {
            "机组负荷": 360.0,
            "主蒸汽压力": 16.0,
            "主给水流量": 1000.0,
            "未校正总煤量": 160.0,
            "主蒸汽流量": 1750.0,
        }.items()
    }
    output = {}
    for side_index, side in enumerate(("A", "B")):
        values = np.zeros((rows, len(columns)), dtype=np.float32)
        for name, values_shared in shared.items():
            values[:, columns.index(name)] = values_shared
        valve = 30 + side_index * 3 + np.sin(np.arange(rows) / 17)
        tin = 551 + side_index + np.sin(np.arange(rows) / 31)
        values[:, columns.index("二级减温器入口温度")] = tin
        values[:, columns.index("二级减温器出口温度")] = tin - 10 - 0.04 * valve
        values[:, columns.index("末级过热器出口汽温")] = 540 + side_index + np.cos(np.arange(rows) / 43)
        values[:, columns.index("二级减温调节阀设定")] = 540 + side_index + np.sin(np.arange(rows) / 29)
        values[:, columns.index("二级减温调节门阀位")] = valve
        output[side] = Phase35Cache(
            timestamps_ns=timestamps.copy(),
            values=values,
            ages_s=np.zeros_like(values),
            columns=columns,
            metadata={"side": side, "step_seconds": 10, "source": {"sha256": "micro"}},
        )
    return output


def _micro_matrix() -> dict:
    matrix = json.loads(
        (ROOT / "configs/phase3_5/ms3r_gatec_rm2_matrix.json").read_text(encoding="utf-8")
    )
    matrix = copy.deepcopy(matrix)
    matrix["data_contract"].update({"window": 12, "horizon": 6, "max_age_s": 30.0})
    matrix["model"].update({"d_model": 8, "latent_dim": 6, "dropout": 0.0})
    matrix["training"].update(
        {
            "batch_size": 8,
            "optimizer_updates_cap": 3,
            "minimum_updates": 1,
            "evaluation_interval_updates": 1,
            "early_stopping_patience_evaluations": 2,
            "stats_anchor_count": 24,
            "selector_anchor_count": 12,
            "final_anchor_count": 16,
            "evaluation_batch_size": 8,
        }
    )
    return matrix


def test_rm2_training_writes_replayable_validation_only_artifacts(tmp_path: Path) -> None:
    matrix = _micro_matrix()
    spec = rm2_run_specs(matrix)[2]
    result = run_rm2_training(
        _caches(),
        matrix,
        spec,
        device="cpu",
        output_dir=tmp_path,
        provenance={"execution_git_sha": "test", "test_accessed": False},
    )
    assert result["status"] == "complete"
    required = {
        "manifest.json",
        "checkpoint_best_validation.pt",
        "metrics_validation.json",
        "episodes_validation.npz",
        "artifact_ledger.json",
    }
    assert required == {path.name for path in tmp_path.iterdir()}
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_accessed"] is False
    assert manifest["selector_reporting_disjoint"] is True
    assert manifest["train_anchor_pool_count"] > manifest["stats_anchor_count"]
    metrics = json.loads((tmp_path / "metrics_validation.json").read_text(encoding="utf-8"))
    assert metrics["optimizer_updates_completed"] <= 3
    assert metrics["metrics"]["finite"] is True
    assert metrics["structural_validation"]["constant_action_identity"] is True
    assert metrics["selector_eligible"] is True
    with np.load(tmp_path / "episodes_validation.npz", allow_pickle=False) as episodes:
        assert len(episodes["anchors"]) == 16
        assert episodes["forecast_terminal"].shape == (16, 6, 2)
        assert episodes["logged_effect"].shape == (16, 6, 2)
        assert episodes["a_only_effect"].shape == (16, 6, 2)
        assert episodes["b_only_effect"].shape == (16, 6, 2)
        assert episodes["timestamps_ns"].max() < _caches()["A"].timestamps_ns[800]
        assert episodes["shuffled_anchors"].shape == (16,)


def test_rm2_f1_caps_fractional_rounding_at_actual_test_start() -> None:
    matrix = _micro_matrix()
    spec = next(
        item for item in rm2_run_specs(matrix) if item.fold_id == "F1" and item.seed == 0
    )
    n_rows = 1003
    actual_test_start = int(n_rows * 0.6) + int(n_rows * 0.2)
    assert int(n_rows * 0.8) == actual_test_start + 1
    train_bounds, validation_bounds = rm2_run_bounds(
        n_rows, spec, actual_test_start=actual_test_start
    )
    assert train_bounds == (0, 702)
    assert validation_bounds == (702, actual_test_start)


@pytest.mark.parametrize(
    "candidate_id",
    [
        "A0_paired_free",
        "A1_additive_base",
        "A2_a1_sched_base",
        "A3_a1_sched_large",
        "B1_koopman",
        "B2_pi_ode",
        "B3_deeponet",
        "C1_common_only",
        "C2_no_downstream_latent",
    ],
)
def test_every_rm2_candidate_completes_local_micro_smoke(
    tmp_path: Path, candidate_id: str
) -> None:
    matrix = _micro_matrix()
    matrix["training"].update(
        {
            "optimizer_updates_cap": 1,
            "minimum_updates": 1,
            "early_stopping_patience_evaluations": 1,
        }
    )
    spec = next(
        item
        for item in rm2_run_specs(matrix)
        if item.candidate_id == candidate_id and item.fold_id == "F0" and item.seed == 0
    )
    output = tmp_path / candidate_id
    result = run_rm2_training(
        _caches(),
        matrix,
        spec,
        device="cpu",
        output_dir=output,
        provenance={"execution_git_sha": "micro", "test_accessed": False},
    )
    assert result["metrics"]["metrics"]["finite"] is True
    checkpoint = torch.load(
        output / "checkpoint_best_validation.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["model_config"]["response_coordinate_mode"] == spec.response_coordinate_mode
    assert checkpoint["model_config"]["downstream_mode"] == spec.downstream_mode
    if candidate_id == "C1_common_only":
        with np.load(output / "episodes_validation.npz", allow_pickle=False) as episodes:
            assert np.allclose(
                episodes["logged_effect"][..., 0], episodes["logged_effect"][..., 1]
            )
    if candidate_id == "C2_no_downstream_latent":
        assert checkpoint["model_config"]["downstream_mode"] == "direct_no_latent"
