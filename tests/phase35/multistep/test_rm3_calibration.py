from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.multistep.rm3_calibration import a1_nonnegative_projection, run_rm3_calibration
from src.phase35.multistep.rm3_contracts import rm3_calibration_specs
from src.phase35.schema import MS3_HISTORY_FEATURES


ROOT = Path(__file__).resolve().parents[3]


def _caches(rows: int = 1200) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(37)
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(rows, dtype=np.int64) * 10_000_000_000
    shared = {name: base + rng.normal(0, 0.1, rows) for name, base in {
        "机组负荷": 360.0, "主蒸汽压力": 16.0, "主给水流量": 1000.0,
        "未校正总煤量": 160.0, "主蒸汽流量": 1750.0,
    }.items()}
    output = {}
    time = np.arange(rows)
    innovations = rng.normal(0, 0.3, (rows, 2))
    for side_index, side in enumerate(("A", "B")):
        values = np.zeros((rows, len(columns)), dtype=np.float32)
        for name, series in shared.items():
            values[:, columns.index(name)] = series
        valve = 32 + side_index + np.sin(time / 23) + innovations[:, side_index]
        tin = 550 + side_index + np.sin(time / 31)
        local = 10 + 0.05 * valve
        values[:, columns.index("二级减温器入口温度")] = tin
        values[:, columns.index("二级减温器出口温度")] = tin - local
        values[:, columns.index("末级过热器出口汽温")] = 540 + side_index + np.cos(time / 43)
        values[:, columns.index("二级减温调节阀设定")] = 540 + side_index + np.sin(time / 29)
        values[:, columns.index("二级减温调节门阀位")] = valve
        output[side] = Phase35Cache(
            timestamps_ns=timestamps.copy(), values=values, ages_s=np.zeros_like(values),
            columns=columns, metadata={"side": side, "step_seconds": 10, "source": {"sha256": "micro"}},
        )
    return output


def _matrix() -> dict:
    matrix = json.loads((ROOT / "configs/phase3_5/ms3r_rm3_matrix.json").read_text(encoding="utf-8"))
    matrix = copy.deepcopy(matrix)
    matrix["data_contract"].update({"window_steps": 16, "max_age_s": 30.0})
    return matrix


def test_calibration_reports_all_three_families_from_full_prefix(tmp_path: Path) -> None:
    frozen = json.loads((ROOT / "configs/phase3_5/ms3r_rm3_matrix.json").read_text(encoding="utf-8"))
    spec = next(item for item in rm3_calibration_specs(frozen) if item.fold_id == "F0" and item.seed == 0 and item.response_horizon_steps == 6)
    result = run_rm3_calibration(
        _caches(), _matrix(), spec, output_dir=tmp_path,
        provenance={"execution_git_sha": "micro", "test_accessed": False},
        train_anchor_limit=100, evaluation_anchor_limit=80,
    )
    assert result["status"] == "complete"
    assert set(result["results"]) == set(spec.candidate_ids)
    payload = json.loads((tmp_path / "calibration_validation.json").read_text(encoding="utf-8"))
    assert payload["full_prefix_trajectory"] is True
    assert len(payload["results"]["R0_linear_mimo"]["trajectory_matrix"]) == 6
    assert payload["results"]["R1_a1_scheduled"]["context_scheduling_identified"] is False
    assert payload["test_accessed"] is False
    with pytest.raises(FileExistsError):
        run_rm3_calibration(
            _caches(), _matrix(), spec, output_dir=tmp_path,
            provenance={"execution_git_sha": "micro", "test_accessed": False},
        )


def test_a1_projection_uses_true_nnls_without_posthoc_clipping_explosion() -> None:
    time = (np.arange(6) + 1) * 10.0
    basis = 1.0 - np.exp(-time[:, None] / np.asarray([60.0, 180.0, 600.0])[None])
    # An unconstrained solution contains a negative coefficient. Clipping that
    # coefficient after least squares creates a large, invalid reconstruction.
    target = basis @ np.asarray([2.0, -5.0, 4.0])
    matrices = np.zeros((6, 2, 2), dtype=float)
    matrices[:, 0, 0] = target
    result = a1_nonnegative_projection(matrices, step_seconds=10.0)
    coefficients = np.asarray(result["nonnegative_coefficients"])
    fitted = np.asarray(result["trajectory_matrix"])
    assert np.all(coefficients >= 0)
    assert np.max(coefficients) < 10.0
    assert np.sqrt(np.mean((fitted[:, 0, 0] - target) ** 2)) < 0.1
