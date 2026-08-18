from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import experiments.phase3_5.ms3r_rm3av_train as rm3av_train
from experiments.phase3_5.ms3r_rm3av_train import (
    _require_empty_output, dry_run_payload, execute,
)
from src.phase35.data import Phase35Cache
from src.phase35.multistep.rm3av_contracts import RM3AV_CANDIDATE_IDS, rm3av_run_specs
from src.phase35.multistep.rm3av_execution import (
    _candidate_pool, _split_validation, run_rm3av_training,
)
from src.phase35.schema import MS3_HISTORY_FEATURES
from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_rm3av_matrix.json"


def _caches(rows: int = 60000) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(357)
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


def _matrix() -> dict:
    matrix = copy.deepcopy(json.loads(MATRIX_PATH.read_text(encoding="utf-8")))
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


def test_rollout_and_nonrollout_candidates_share_the_same_h120_eligible_pool() -> None:
    caches = _caches()
    kwargs = {
        "split": "train", "bounds": (0, 10000), "window": 16,
        "horizon": 60, "max_age_s": 30.0,
    }
    ordinary = _candidate_pool(caches, candidate_id="C27", **kwargs)
    rollout = _candidate_pool(caches, candidate_id="C31", **kwargs)
    assert np.array_equal(ordinary, rollout)


def test_selector_and_reporting_are_chronological_disjoint_utc_day_blocks() -> None:
    timestamps = np.arange(5 * 8640, dtype=np.int64) * 10_000_000_000
    anchors = np.arange(100, len(timestamps) - 100, dtype=np.int64)
    selector, reporting, selector_days, reporting_days = _split_validation(
        anchors, timestamps, 100, 200, "F0"
    )
    assert len(selector) == 100 and len(reporting) == 200
    assert set(selector_days).isdisjoint(reporting_days)
    assert max(selector_days) < min(reporting_days)
    with pytest.raises(Phase35ProtocolError, match="at least two UTC days"):
        _split_validation(anchors[:1000], timestamps, 10, 20, "F0")


@pytest.mark.parametrize("candidate_id", RM3AV_CANDIDATE_IDS)
def test_every_candidate_training_path_writes_complete_validation_artifacts(
    tmp_path: Path, candidate_id: str
) -> None:
    frozen = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    spec = next(
        item for item in rm3av_run_specs(frozen)
        if item.candidate_id == candidate_id and item.fold_id == "F0"
    )
    # This is a one-update structural smoke even for the C28-C30 convergence controls.
    spec = replace(
        spec,
        optimizer_updates_cap=4000,
        train_fraction=(0.0, 0.4),
        validation_fraction=(0.4, 0.8),
    )
    result = run_rm3av_training(
        _caches(), _matrix(), spec, device="cpu", output_dir=tmp_path / spec.run_id,
        provenance={"execution_git_sha": "micro", "test_accessed": False},
    )
    assert result["status"] == "complete"
    run_dir = tmp_path / spec.run_id
    assert {path.name for path in run_dir.iterdir()} == set(
        frozen["execution_contract"]["required_run_artifacts"]
    )
    diagnostics = json.loads((run_dir / "diagnostics_validation.json").read_text(encoding="utf-8"))
    assert diagnostics["candidate_id"] == candidate_id
    assert diagnostics["test_accessed"] is False
    assert len(diagnostics["manual_audit_verdicts"]) == 33
    assert diagnostics["state_closure"]["state_closed_simulator"] is False
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selector_reporting_disjoint"] is True
    assert manifest["selector_reporting_utc_day_disjoint"] is True
    assert set(manifest["selector_utc_days"]).isdisjoint(manifest["reporting_utc_days"])
    assert manifest["maximum_attempts_per_run"] == 1
    assert manifest["checkpoint_selector"] == (
        "validation_full_multitask_common_four_task_loss"
    )
    assert manifest["early_stopping_enabled"] is False
    assert diagnostics["convergence"]["optimizer_updates_completed"] == 1
    assert "valve_trajectory" in diagnostics
    assert "response_trajectory" in diagnostics
    assert "finite_difference_response" in diagnostics
    assert diagnostics["finite_difference_response"]["perturbation_points_percent_valve"] == [
        -2.0, -1.0, -0.5, 0.5, 1.0, 2.0
    ]
    assert set(diagnostics["finite_difference_response"]["responses"]) == {
        "A_only", "B_only", "common", "differential"
    }
    assert diagnostics["branch_action_invariance"]["free_residual_action_invariant"] is True
    assert diagnostics["branch_action_invariance"]["terminal_bypass_action_invariant"] is True
    assert diagnostics["response_trajectory"]["ratio_is_descriptive_not_identified"] is True
    assert "terminal_strata" in diagnostics
    assert "training_graph" in diagnostics
    assert "raw_valve_change_rank" in diagnostics
    assert "residualized_valve_innovation" in diagnostics
    assert "mechanism_prediction_residual_dependence" in diagnostics
    assert "boundary_tin_placebo" in diagnostics
    if candidate_id == "C31":
        assert diagnostics["two_window_rollout"]["recursive_horizon_seconds"] == 1200
        assert diagnostics["two_window_rollout"]["unavailable_endpoints_seconds"] == [
            1800, 3600
        ]
        with np.load(run_dir / "episodes_validation.npz", allow_pickle=False) as arrays:
            assert "rollout_second_terminal_prediction" in arrays.files
    else:
        assert diagnostics["two_window_rollout"]["status"] == "not_applicable"


def test_dry_run_closes_64_units_without_authorizing_execution() -> None:
    payload = dry_run_payload(json.loads(MATRIX_PATH.read_text(encoding="utf-8")))
    assert payload["candidate_count"] == 32
    assert payload["training_unit_count"] == 64
    assert payload["folds"] == ["F0", "F1"]
    assert payload["seeds"] == [0]
    assert payload["matrix_self_authorizing"] is False
    assert payload["registry_authorization_required_for_execute"] is True
    assert payload["test_authorized"] is False


def test_execute_refuses_explicitly_unauthorized_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(
        (ROOT / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    registry["active_gate"] = "ms3_r"
    registry["linux_authorized_gate"] = None
    registry["experiments"]["ms3_r"]["status"] = "local_verified"
    registry["experiments"]["ms3_r"]["decision"]["authorized_batch"] = None
    registry_path = tmp_path / "experiment_registry_unauthorized.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(rm3av_train, "REGISTRY", registry_path)
    with pytest.raises(RuntimeError, match="linux_authorized_gate"):
        execute(MATRIX_PATH, tmp_path / "a", tmp_path / "b", tmp_path / "out", ["cpu"])


def test_registry_accepts_only_explicit_rm3av_batch_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(
        (ROOT / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    registry["active_gate"] = "ms3_r"
    registry["linux_authorized_gate"] = "ms3_r"
    registry["experiments"]["ms3_r"]["status"] = "ready_for_linux"
    registry["experiments"]["ms3_r"]["decision"]["authorized_batch"] = "RM3-AV0+AV1"
    registry_path = tmp_path / "experiment_registry_authorized.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(rm3av_train, "REGISTRY", registry_path)
    rm3av_train._verify_registry()


def test_execution_refuses_nonempty_output_root_instead_of_resuming(tmp_path: Path) -> None:
    output = tmp_path / "partial"
    output.mkdir()
    (output / "C00_F0_s0").mkdir()
    with pytest.raises(FileExistsError, match="non-empty output root"):
        _require_empty_output(output)
