from __future__ import annotations

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.schema import MS3_HISTORY_FEATURES, Phase35ProtocolError


def _paired_caches(*, gap_after: int | None = None) -> dict[str, Phase35Cache]:
    n = 400
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(n, dtype=np.int64) * 10_000_000_000
    if gap_after is not None:
        timestamps[gap_after + 1 :] += 40_000_000_000
    shared = {
        "机组负荷": np.linspace(320.0, 420.0, n),
        "主蒸汽压力": np.linspace(14.0, 18.0, n),
        "主给水流量": np.linspace(900.0, 1100.0, n),
        "未校正总煤量": np.linspace(140.0, 180.0, n),
        "主蒸汽流量": np.linspace(1600.0, 1900.0, n),
    }
    output = {}
    for side_index, side in enumerate(("A", "B")):
        values = np.zeros((n, len(columns)), dtype=np.float32)
        for name, series in shared.items():
            values[:, columns.index(name)] = series
        tin = 550.0 + side_index + np.sin(np.arange(n) / 20.0)
        valve = 30.0 + 2 * side_index + np.sin(np.arange(n) / 8.0)
        values[:, columns.index("二级减温器入口温度")] = tin
        values[:, columns.index("二级减温器出口温度")] = tin - 5.0 - 0.02 * valve
        values[:, columns.index("末级过热器出口汽温")] = 540.0 + side_index + np.cos(np.arange(n) / 30.0)
        values[:, columns.index("二级减温调节阀设定")] = 540.0 + side_index + np.sin(np.arange(n) / 25.0)
        values[:, columns.index("二级减温调节门阀位")] = valve
        output[side] = Phase35Cache(
            timestamps_ns=timestamps.copy(),
            values=values,
            ages_s=np.zeros_like(values),
            columns=columns,
            metadata={"side": side, "step_seconds": 10, "source": {"sha256": "synthetic"}},
        )
    return output


def test_paired_gatec_batch_has_one_shared_block_and_two_side_blocks():
    from src.phase35.multistep.gatec_data import extract_gatec_batch, paired_valid_anchors

    caches = _paired_caches()
    anchors = paired_valid_anchors(caches, "validation", window=12, horizon=6, max_age_s=30.0)
    batch = extract_gatec_batch(caches, anchors[:8], window=12, horizon=6)
    assert batch.history.shape == (8, 12, 15)
    assert batch.future_sp.shape == (8, 6, 2)
    assert batch.logged_future_valve.shape == (8, 6, 2)
    assert batch.logged_future_tin.shape == (8, 6, 2)
    assert batch.local_drop_target.shape == (8, 6, 2)
    assert batch.tout_target.shape == (8, 6, 2)
    assert batch.terminal_target.shape == (8, 6, 2)
    assert len(batch.history_feature_names) == 15
    assert batch.history_feature_names.count("机组负荷") == 1


def test_forecast_inputs_do_not_contain_future_truth_or_logged_valve():
    from src.phase35.multistep.gatec_data import extract_gatec_batch, paired_valid_anchors

    caches = _paired_caches()
    anchors = paired_valid_anchors(caches, "validation", window=12, horizon=6, max_age_s=30.0)
    batch = extract_gatec_batch(caches, anchors[:4], window=12, horizon=6)
    forecast = batch.model_inputs("forecast_boundary")
    assert set(forecast) == {"history", "future_sp", "boundary_future", "boundary_mode"}
    assert forecast["boundary_future"] is None
    assert not any("valve" in key for key in forecast)
    with pytest.raises(Phase35ProtocolError, match="explicit audit permission"):
        batch.model_inputs("oracle_boundary")
    oracle = batch.model_inputs("oracle_boundary", allow_oracle=True)
    assert np.array_equal(oracle["boundary_future"], batch.logged_future_tin)
    scenario = np.full_like(batch.logged_future_tin, 555.0)
    supplied = batch.model_inputs("scenario_boundary", scenario_tin=scenario)
    assert np.array_equal(supplied["boundary_future"], scenario)


def test_paired_anchors_stay_inside_validation_and_do_not_cross_gap():
    from src.phase35.multistep.gatec_data import paired_valid_anchors

    caches = _paired_caches(gap_after=250)
    anchors = paired_valid_anchors(caches, "validation", window=12, horizon=6, max_age_s=30.0)
    lo, hi = caches["A"].split_bounds()["validation"]
    assert anchors.min() >= lo + 11
    assert anchors.max() + 6 < hi
    assert not np.any((anchors - 11 <= 250) & (anchors + 6 >= 251))


def test_paired_anchors_allow_train_but_refuse_test():
    from src.phase35.multistep.gatec_data import paired_valid_anchors

    caches = _paired_caches()
    train = paired_valid_anchors(
        caches, "train", window=12, horizon=6, max_age_s=30.0
    )
    lo, hi = caches["A"].split_bounds()["train"]
    assert train.min() >= lo + 11
    assert train.max() + 6 < hi
    with pytest.raises(Phase35ProtocolError, match="test split"):
        paired_valid_anchors(
            caches, "test", window=12, horizon=6, max_age_s=30.0
        )


def test_paired_data_rejects_misalignment_or_shared_point_drift():
    from src.phase35.multistep.gatec_data import paired_valid_anchors

    caches = _paired_caches()
    caches["B"].timestamps_ns += 1
    with pytest.raises(Phase35ProtocolError, match="timestamps"):
        paired_valid_anchors(caches, "validation", window=12, horizon=6, max_age_s=30.0)
    caches = _paired_caches()
    caches["B"].values[:, caches["B"].index("机组负荷")] += 1.0
    with pytest.raises(Phase35ProtocolError, match="shared feature"):
        paired_valid_anchors(caches, "validation", window=12, horizon=6, max_age_s=30.0)
