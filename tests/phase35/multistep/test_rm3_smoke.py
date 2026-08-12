from __future__ import annotations

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.multistep.rm3_prediction import PREDICTION_CANDIDATES
from src.phase35.multistep.rm3_smoke import run_rm3_prediction_micro_smoke
from src.phase35.schema import MS3_HISTORY_FEATURES


def _caches(rows: int = 900) -> dict[str, Phase35Cache]:
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(rows, dtype=np.int64) * 10_000_000_000
    shared = {
        "机组负荷": np.linspace(300.0, 450.0, rows),
        "主蒸汽压力": np.linspace(14.0, 19.0, rows),
        "主给水流量": np.linspace(850.0, 1150.0, rows),
        "未校正总煤量": np.linspace(130.0, 190.0, rows),
        "主蒸汽流量": np.linspace(1500.0, 2000.0, rows),
    }
    output = {}
    time = np.arange(rows)
    for side_index, side in enumerate(("A", "B")):
        values = np.zeros((rows, len(columns)), dtype=np.float32)
        for name, series in shared.items():
            values[:, columns.index(name)] = series
        tin = 520 + side_index + np.sin(time / 30)
        valve = 35 + side_index + 3 * np.sin(time / 20)
        values[:, columns.index("二级减温器入口温度")] = tin
        values[:, columns.index("二级减温器出口温度")] = tin - 12 - 0.04 * valve
        values[:, columns.index("末级过热器出口汽温")] = 540 + side_index + np.cos(time / 45)
        values[:, columns.index("二级减温调节阀设定")] = 540 + side_index + np.sin(time / 37)
        values[:, columns.index("二级减温调节门阀位")] = valve
        output[side] = Phase35Cache(
            timestamps_ns=timestamps.copy(),
            values=values,
            ages_s=np.zeros_like(values),
            columns=columns,
            metadata={"side": side, "step_seconds": 10, "source": {"sha256": "synthetic"}},
        )
    return output


@pytest.mark.parametrize("candidate", sorted(PREDICTION_CANDIDATES))
def test_rm3_all_prediction_candidates_complete_micro_backward(candidate: str) -> None:
    payload = run_rm3_prediction_micro_smoke(_caches(), candidate, anchor_count=2)
    assert payload["terminal_shape"] == [2, 60, 2]
    assert payload["finite_gradients"] is True
    assert payload["test_accessed"] is False
    assert payload["automatic_scientific_pass"] is None
