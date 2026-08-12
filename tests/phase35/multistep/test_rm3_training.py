from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.phase35.data import Phase35Cache
from src.phase35.multistep.rm3_contracts import rm3_prediction_run_specs
from src.phase35.multistep.rm3_training import rm3_scope_loss, run_rm3_prediction_training
from src.phase35.schema import MS3_HISTORY_FEATURES, Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]


def _caches(rows: int = 1100) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(353)
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(rows, dtype=np.int64) * 10_000_000_000
    shared = {
        name: base + rng.normal(0, 0.1, rows)
        for name, base in {
            "机组负荷": 360.0, "主蒸汽压力": 16.0, "主给水流量": 1000.0,
            "未校正总煤量": 160.0, "主蒸汽流量": 1750.0,
        }.items()
    }
    result = {}
    time = np.arange(rows)
    for side_index, side in enumerate(("A", "B")):
        values = np.zeros((rows, len(columns)), dtype=np.float32)
        for name, series in shared.items():
            values[:, columns.index(name)] = series
        valve = 32 + side_index + 2 * np.sin(time / 21)
        tin = 550 + side_index + np.sin(time / 29)
        values[:, columns.index("二级减温器入口温度")] = tin
        values[:, columns.index("二级减温器出口温度")] = tin - 10 - 0.04 * valve
        values[:, columns.index("末级过热器出口汽温")] = 540 + side_index + np.cos(time / 41)
        values[:, columns.index("二级减温调节阀设定")] = 540 + side_index + np.sin(time / 31)
        values[:, columns.index("二级减温调节门阀位")] = valve
        result[side] = Phase35Cache(
            timestamps_ns=timestamps.copy(), values=values, ages_s=np.zeros_like(values),
            columns=columns, metadata={"side": side, "step_seconds": 10, "source": {"sha256": "micro"}},
        )
    return result


def _micro_matrix() -> dict:
    matrix = json.loads((ROOT / "configs/phase3_5/ms3r_rm3_matrix.json").read_text(encoding="utf-8"))
    matrix = copy.deepcopy(matrix)
    matrix["data_contract"].update({"window_steps": 16, "max_age_s": 30.0})
    matrix["model"].update({"d_model": 8, "latent_dim": 6, "dropout": 0.0})
    matrix["training"].update({
        "batch_size": 2, "optimizer_updates_cap": 1, "minimum_updates": 1,
        "evaluation_interval_updates": 1, "early_stopping_patience_evaluations": 1,
        "stats_anchor_count": 12, "selector_anchor_count": 4,
        "reporting_anchor_count": 5, "evaluation_batch_size": 2,
    })
    return matrix


def test_scope_loss_refuses_missing_multitask_outputs() -> None:
    target = torch.zeros(2, 60, 2)
    with pytest.raises(Phase35ProtocolError, match="lacks valve_prediction"):
        rm3_scope_loss(
            {"terminal_prediction": target},
            {key: target for key in ("valve", "tin", "local", "terminal")},
            output_scope="full_multitask",
            target_scales={key: 1.0 for key in ("valve", "tin", "local", "terminal")},
        )


@pytest.mark.parametrize("candidate_id", [
    "P0_m7_oracle_valve", "P1_m7_predicted_valve", "P2_m9_future_sp",
    "P3_gatec_paired_free", "P4_gatec_a1_scheduled", "P5_hybrid_joint_latent",
])
def test_rm3_prediction_training_writes_validation_only_artifacts(
    tmp_path: Path, candidate_id: str
) -> None:
    matrix = _micro_matrix()
    spec = next(
        item for item in rm3_prediction_run_specs(json.loads(
            (ROOT / "configs/phase3_5/ms3r_rm3_matrix.json").read_text(encoding="utf-8")
        ))
        if item.candidate_id == candidate_id and item.fold_id == "F0" and item.seed == 0
    )
    run_dir = tmp_path / spec.run_id
    result = run_rm3_prediction_training(
        _caches(), matrix, spec, device="cpu", output_dir=run_dir,
        provenance={"execution_git_sha": "micro", "test_accessed": False},
    )
    assert result["status"] == "complete"
    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json", "checkpoint_best_validation.pt", "metrics_validation.json",
        "episodes_validation.npz", "artifact_ledger.json",
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selector_reporting_disjoint"] is True
    assert manifest["test_accessed"] is False
    metrics = json.loads((run_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    assert metrics["output_scope"] == spec.output_scope
    assert np.isfinite(metrics["metrics"]["terminal_mae_c"])
    with pytest.raises(FileExistsError):
        run_rm3_prediction_training(
            _caches(), matrix, spec, device="cpu", output_dir=run_dir,
            provenance={"execution_git_sha": "micro", "test_accessed": False},
        )
