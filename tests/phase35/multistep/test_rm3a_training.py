from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.multistep.rm3a_contracts import rm3a_run_specs
from src.phase35.multistep.rm3a_training import rm3a_state_element_count, run_rm3a_training
from src.phase35.schema import MS3_HISTORY_FEATURES


ROOT = Path(__file__).resolve().parents[3]


def _caches(rows: int = 1100) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(355)
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(rows, dtype=np.int64) * 10_000_000_000
    shared = {name: base + rng.normal(0, 0.1, rows) for name, base in {
        "机组负荷": 360.0, "主蒸汽压力": 16.0, "主给水流量": 1000.0,
        "未校正总煤量": 160.0, "主蒸汽流量": 1750.0,
    }.items()}
    result = {}
    time = np.arange(rows)
    for side_index, side in enumerate(("A", "B")):
        values = np.zeros((rows, len(columns)), dtype=np.float32)
        for name, series in shared.items(): values[:, columns.index(name)] = series
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


def _matrix() -> dict:
    matrix = json.loads((ROOT / "configs/phase3_5/ms3r_rm3a_matrix.json").read_text(encoding="utf-8"))
    matrix = copy.deepcopy(matrix)
    matrix["data_contract"].update({"window_steps": 16, "max_age_s": 30.0})
    matrix["training"].update({
        "batch_size": 2, "optimizer_updates_cap": 1, "minimum_updates": 1,
        "evaluation_interval_updates": 1, "early_stopping_patience_evaluations": 1,
        "stats_anchor_count": 12, "selector_anchor_count": 4,
        "reporting_anchor_count": 5, "evaluation_batch_size": 2,
    })
    return matrix


def test_rm3a_capacity_points_are_bidirectionally_matched() -> None:
    frozen = json.loads((ROOT / "configs/phase3_5/ms3r_rm3a_matrix.json").read_text(encoding="utf-8"))
    specs = {spec.candidate_id: spec for spec in rm3a_run_specs(frozen)}
    for candidate in ("A0_p3_large", "A1_p4_large", "A2_p5_small", "A3_p5_local35", "A4_p5_local50"):
        spec = specs[candidate]
        assert rm3a_state_element_count(spec) == spec.state_elements_expected
    assert abs(specs["A0_p3_large"].state_elements_expected / 122301 - 1) < 0.04
    assert abs(specs["A1_p4_large"].state_elements_expected / 122301 - 1) < 0.04
    assert abs(specs["A2_p5_small"].state_elements_expected / 87258 - 1) < 0.05


@pytest.mark.parametrize(
    "candidate_id",
    ["A0_p3_large", "A1_p4_large", "A2_p5_small", "A3_p5_local35", "A4_p5_local50"],
)
def test_each_rm3a_candidate_completes_one_update_artifact_smoke(
    tmp_path: Path, candidate_id: str
) -> None:
    frozen = json.loads((ROOT / "configs/phase3_5/ms3r_rm3a_matrix.json").read_text(encoding="utf-8"))
    spec = next(
        item for item in rm3a_run_specs(frozen)
        if item.candidate_id == candidate_id and item.fold_id == "F0" and item.seed == 0
    )
    result = run_rm3a_training(
        _caches(), _matrix(), spec, device="cpu", output_dir=tmp_path / spec.run_id,
        provenance={"execution_git_sha": "micro", "test_accessed": False},
    )
    assert result["status"] == "complete"
    manifest = json.loads(
        (tmp_path / spec.run_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["architecture_candidate_id"] == spec.base_candidate_id
    assert manifest["component_loss_weights"] == spec.loss_weights
    assert manifest["checkpoint_selector"] == (
        "validation_full_multitask_declared_component_weighted_loss"
    )
    assert manifest["test_accessed"] is False
