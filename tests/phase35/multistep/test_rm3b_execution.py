from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import experiments.phase3_5.ms3r_rm3b_train as rm3b_train
from experiments.phase3_5.ms3r_rm3b_train import dry_run_payload, execute
from src.phase35.data import Phase35Cache
from src.phase35.multistep.rm3av_execution import run_rm3av_training
from src.phase35.multistep.rm3b_contracts import rm3b_run_specs
from src.phase35.schema import MS3_HISTORY_FEATURES


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_rm3b_matrix.json"


def _frozen_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _micro_matrix() -> dict:
    matrix = copy.deepcopy(_frozen_matrix())
    matrix["data_contract"].update({"window_steps": 16, "max_age_s": 30.0})
    matrix["model"].update({"d_model": 8, "latent_dim": 4, "dropout": 0.0})
    matrix["training"].update({
        "batch_size": 2, "default_optimizer_updates_cap": 1,
        "evaluation_interval_updates": 1,
        "stats_anchor_count": 20, "selector_anchor_count": 4,
        "reporting_anchor_count": 3, "diagnostic_anchor_count": 2,
        "evaluation_batch_size": 2,
    })
    return matrix


def _caches(rows: int = 60000) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(381)
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
            columns=columns,
            metadata={"side": side, "step_seconds": 10, "source": {"sha256": "micro"}},
        )
    return result


def test_rm3b_dry_run_closes_22_without_authorizing_training() -> None:
    payload = dry_run_payload(_frozen_matrix(), repo_root=ROOT)
    assert payload["candidate_count"] == 11
    assert payload["training_unit_count"] == 22
    assert payload["folds"] == ["F0", "F1"]
    assert payload["seeds"] == [0]
    assert payload["optimizer_updates_cap"] == [8000]
    assert payload["matrix_self_authorizing"] is False
    assert payload["rm3b2_authorized"] is False
    assert payload["automatic_scientific_pass"] is None


def test_rm3b_execute_refuses_a_different_authorized_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = json.loads(
        (ROOT / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    registry["experiments"]["ms3_r"]["decision"]["authorized_batch"] = "RM3-AV0+AV1"
    registry_path = tmp_path / "registry_other_batch.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(rm3b_train, "REGISTRY", registry_path)
    with pytest.raises(RuntimeError, match="Linux-authorized gate|ready_for_linux|authorized_batch"):
        execute(
            MATRIX_PATH, tmp_path / "missing_a.npz", tmp_path / "missing_b.npz",
            tmp_path / "output", ["cpu"], require_clean=False,
        )


def test_rm3b_current_registry_authorizes_only_rm3b1() -> None:
    rm3b_train._verify_registry()


@pytest.mark.parametrize("candidate_id", [f"B{index:02d}" for index in range(11)])
def test_every_rm3b_template_completes_one_update_and_preserves_dual_identity(
    tmp_path: Path, candidate_id: str
) -> None:
    frozen = _frozen_matrix()
    spec = next(
        item for item in rm3b_run_specs(frozen)
        if item.candidate_id == candidate_id and item.fold_id == "F0"
    )
    spec = replace(
        spec, optimizer_updates_cap=4000,
        train_fraction=(0.0, 0.4), validation_fraction=(0.4, 0.8),
    )
    result = run_rm3av_training(
        _caches(), _micro_matrix(), spec, device="cpu",
        output_dir=tmp_path / spec.run_id,
        provenance={"execution_git_sha": "micro", "test_accessed": False},
        template_candidate_id=spec.template_candidate_id,
    )
    assert result["status"] == "complete"
    run_dir = tmp_path / spec.run_id
    assert {path.name for path in run_dir.iterdir()} == set(
        frozen["execution_contract"]["required_run_artifacts"]
    )
    metrics = json.loads((run_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((run_dir / "diagnostics_validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["candidate_id"] == candidate_id
    assert metrics["template_candidate_id"] == spec.template_candidate_id
    assert metrics["optimizer_updates_completed"] == 1
    assert diagnostics["experiment_candidate_id"] == candidate_id
    assert diagnostics["template_candidate_id"] == spec.template_candidate_id
    assert manifest["run_spec"]["candidate_id"] == candidate_id
    assert manifest["run_spec"]["template_candidate_id"] == spec.template_candidate_id
    assert manifest["selector_reporting_utc_day_disjoint"] is True
    assert manifest["test_accessed"] is False
