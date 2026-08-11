from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.ms3d import (
    detect_ms3d_events,
    diagnosis_label,
    paired_day_contrasts,
    response_metric_keys,
    validate_ms3d_config,
)
from src.phase35.schema import MS3_HISTORY_FEATURES, Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/phase3_5/ms3d_asymmetry_diagnosis.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _cache(side: str, *, held: bool = True) -> Phase35Cache:
    columns = tuple(MS3_HISTORY_FEATURES)
    n = 900
    timestamps = np.arange(n, dtype=np.int64) * 10_000_000_000
    values = np.zeros((n, len(columns)), dtype=np.float32)
    defaults = {
        "机组负荷": 300.0,
        "主蒸汽压力": 15.0,
        "主给水流量": 1000.0,
        "未校正总煤量": 150.0,
        "主蒸汽流量": 1800.0,
        "二级减温器入口温度": 550.0,
        "二级减温器出口温度": 545.0,
        "末级过热器出口汽温": 540.0,
        "二级减温调节阀设定": 540.0,
        "二级减温调节门阀位": 50.0,
    }
    for column, value in defaults.items():
        values[:, columns.index(column)] = value
    onset = 620
    sp_index = columns.index("二级减温调节阀设定")
    valve_index = columns.index("二级减温调节门阀位")
    out_index = columns.index("二级减温器出口温度")
    target_index = columns.index("末级过热器出口汽温")
    values[onset:, sp_index] += 1.0
    if not held:
        values[onset + 10 :, sp_index] -= 1.0
    values[onset + 1 :, valve_index] -= 2.0
    values[onset + 1 :, out_index] += 1.0
    values[onset + 20 :, target_index] += 0.8
    metadata = {
        "side": side,
        "step_seconds": 10,
        "source": {"sha256": _config()["data_contract"]["source_sha256"]},
    }
    return Phase35Cache(
        timestamps_ns=timestamps,
        values=values,
        ages_s=np.zeros_like(values),
        columns=columns,
        metadata=metadata,
    )


def test_repository_ms3d_config_is_validation_only():
    validate_ms3d_config(_config())
    changed = copy.deepcopy(_config())
    changed["data_contract"]["split"] = "test"
    changed["data_contract"]["test_allowed"] = True
    with pytest.raises(Phase35ProtocolError, match="validation-only"):
        validate_ms3d_config(changed)


def test_detect_ms3d_event_preserves_expected_cascade_signs():
    config = _config()
    cache = _cache("A")
    events, funnel = detect_ms3d_events(cache, _cache("B"), "A", config)
    assert len(events) == 1
    event = events[0]
    assert event["primary_dual_steady"] is True
    assert event["clean_chain"] is True
    assert event["strict_600s_clean_chain"] is True
    assert event["h600_expected_valve_motion_pct"] == pytest.approx(2.0)
    assert event["h600_expected_local_drop_motion_c"] == pytest.approx(1.0)
    assert event["h600_expected_terminal_motion_c"] == pytest.approx(0.8, abs=1e-4)
    assert event["h600_local_drop_gain_c_per_pct"] == pytest.approx(0.5)
    assert funnel["accepted_primary_dual_steady"] == 1


def test_detect_ms3d_rejects_nonheld_sp_without_using_temperature_outcome():
    config = _config()
    events, funnel = detect_ms3d_events(
        _cache("A", held=False), _cache("B"), "A", config
    )
    # The initial upward step is rejected because it is withdrawn after 100 s.
    # The withdrawal is itself a valid, subsequently held downward step.
    assert len(events) == 1
    assert events[0]["direction"] == "sp_down"
    assert funnel["rejected_not_held_600s"] == 1


def test_paired_day_contrast_uses_common_utc_days_not_events_as_n():
    events = {
        "A": [
            {"utc_day": "2026-01-01", "metric": 1.0},
            {"utc_day": "2026-01-01", "metric": 3.0},
            {"utc_day": "2026-01-02", "metric": 2.0},
        ],
        "B": [
            {"utc_day": "2026-01-01", "metric": 4.0},
            {"utc_day": "2026-01-02", "metric": 5.0},
            {"utc_day": "2026-01-03", "metric": 100.0},
        ],
    }
    statistics = {
        "bootstrap_samples": 2000,
        "bootstrap_seed": 7,
        "diagnostic_block_lengths_days": [1, 2],
    }
    result = paired_day_contrasts(events, ["metric"], statistics)["metric"]
    assert result["paired_utc_day_count"] == 2
    assert result["paired_utc_days"] == ["2026-01-01", "2026-01-02"]
    assert result["median_difference"] == pytest.approx(2.5)


def test_diagnosis_label_does_not_call_zero_crossing_intervals_equivalence():
    keys = (
        "h180_local_drop_per_sp",
        "h300_local_drop_per_sp",
        "h180_local_drop_gain_c_per_pct",
        "h300_local_drop_gain_c_per_pct",
        "h600_terminal_per_sp",
    )
    paired = {key: {"ci95": [-0.1, 0.2]} for key in keys}
    checkpoint = {"B_to_A_abs_h600_effect_ratio_median": 4.0}
    result = diagnosis_label(paired, checkpoint, 3.0)
    assert result["label"] == "MODEL_A_RESPONSE_ABSORPTION_COMPATIBLE"
    assert "do not establish equivalence" in result["interpretation_boundary"]


def test_response_metric_contract_has_all_four_horizons():
    keys = response_metric_keys([60, 180, 300, 600])
    assert len(keys) == 20
    assert "h600_terminal_per_sp" in keys
