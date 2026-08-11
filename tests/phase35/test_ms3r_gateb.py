from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.ms3r_gateb import (
    daily_mimo_matrices,
    run_gateb_analysis,
    validate_ms3r_gateb_config,
)
from src.phase35.schema import MS3_HISTORY_FEATURES, Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/phase3_5/ms3r_gateb_point_closure.json"


def _config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["data_contract"].update({"source_sha256": "synthetic", "step_seconds": 3600})
    config["analysis"].update({
        "history_lags_steps": [1, 2],
        "rolling_folds": 2,
        "minimum_training_days": 10,
        "minimum_evaluation_days": 1,
        "minimum_rows_per_fold": 20,
        "minimum_day_rows": 12,
        "minimum_direction_day_rows": 4,
    })
    config["statistics"].update({"bootstrap_samples": 200, "minimum_utc_days": 5})
    return config


def _cache_pair(seed: int = 7) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(seed)
    n = 3600
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(n, dtype=np.int64) * 3_600_000_000_000
    dsp_a, dsp_b = rng.normal(0.0, 0.04, (2, n))
    noise_a, noise_b = rng.normal(0.0, 0.08, (2, n))
    dv_a, dv_b = 4.0 * dsp_a + noise_a, 3.0 * dsp_b + noise_b
    valve_a = 50.0 + np.cumsum(dv_a) * 0.05
    valve_b = 48.0 + np.cumsum(dv_b) * 0.05
    sp_a = 540.0 + np.cumsum(dsp_a) * 0.02
    sp_b = 542.0 + np.cumsum(dsp_b) * 0.02
    tin_a = 550.0 + rng.normal(0.0, 0.03, n)
    tin_b = 551.0 + rng.normal(0.0, 0.03, n)
    # Frozen delayed MIMO truth: both primary horizons receive the correct
    # current innovation, with only small cross-side response and no lead path.
    drop_a = 5.0 + rng.normal(0.0, 0.01, n)
    drop_b = 5.5 + rng.normal(0.0, 0.01, n)
    for lag in (6, 18):
        drop_a[lag:] += 3.0 * dv_a[:-lag] + 0.05 * dv_b[:-lag]
        drop_b[lag:] += 0.04 * dv_a[:-lag] + 2.5 * dv_b[:-lag]
    output: dict[str, Phase35Cache] = {}
    for side, valve, sp, tin, drop in (
        ("A", valve_a, sp_a, tin_a, drop_a),
        ("B", valve_b, sp_b, tin_b, drop_b),
    ):
        values = np.zeros((n, len(columns)), dtype=np.float32)
        defaults = {
            "机组负荷": 350.0,
            "主蒸汽压力": 15.0,
            "主给水流量": 1000.0,
            "未校正总煤量": 150.0,
            "主蒸汽流量": 1800.0,
            "末级过热器出口汽温": 540.0,
        }
        for name, value in defaults.items():
            values[:, columns.index(name)] = value
        values[:, columns.index("二级减温器入口温度")] = tin
        values[:, columns.index("二级减温器出口温度")] = tin - drop
        values[:, columns.index("二级减温调节阀设定")] = sp
        values[:, columns.index("二级减温调节门阀位")] = valve
        output[side] = Phase35Cache(
            timestamps_ns=timestamps.copy(), values=values, ages_s=np.zeros_like(values), columns=columns,
            metadata={"side": side, "step_seconds": 3600, "source": {"sha256": "synthetic"}},
        )
    return output


def test_gateb_config_rejects_test_and_causal_iv_promotion():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_ms3r_gateb_config(config)
    changed = copy.deepcopy(config)
    changed["data_contract"]["split"] = "test"
    with pytest.raises(Phase35ProtocolError, match="validation-only"):
        validate_ms3r_gateb_config(changed)
    changed = copy.deepcopy(config)
    changed["iv_contract"]["status"] = "causal"
    with pytest.raises(Phase35ProtocolError, match="feasibility-only"):
        validate_ms3r_gateb_config(changed)


def test_daily_mimo_recovers_known_matrix():
    rng = np.random.default_rng(3)
    action = rng.normal(size=(240, 2))
    truth = np.asarray([[1.2, 0.1], [0.05, 0.9]])
    outcome = np.einsum("ni,ij->nj", action, truth, optimize=False) + rng.normal(scale=0.01, size=(240, 2))
    days = np.repeat(np.arange(4), 60)
    _, matrices, counts = daily_mimo_matrices(
        action, outcome, days, minimum_rows=30, ridge_alpha=1e-8, epsilon=1e-12
    )
    assert np.all(counts == 60)
    assert np.nanmedian(matrices, axis=0) == pytest.approx(truth, abs=0.01)


def test_gateb_recovers_paired_specificity_and_clean_timing():
    summary, arrays = run_gateb_analysis(_cache_pair(), _config())
    for side in ("A", "B"):
        specificity = summary["paired_contrasts"]["specificity_family"][side]
        timing = summary["paired_contrasts"]["timing_family"][side]
        assert specificity["day_median"] > 0.2
        assert timing["day_median"] > 0.1
        assert specificity["simultaneous_interval"][0] > 0.0
        assert np.isfinite(arrays[f"specificity_{side}"]).sum() >= 5
        assert summary["iv_feasibility"]["sides"][side]["first_stage_partial_r2"] > 0.05
    assert summary["automatic_scientific_pass"] is None
    assert summary["test_accessed"] is False
    assert summary["training_executed"] is False
    assert summary["invariance"]["action_innovation_direction"]["A"]["opening"]["utc_day_count"] >= 5
    assert summary["mimo_response"]["diagnostic_point_maps"]["upstream_tin_placebo"]["horizons"]
    assert "target_mimo_future_H60" in arrays
    assert "mimo_future_H6" in arrays and "outcome_future_H60" in arrays
