import numpy as np

from src.phase35.data import (
    Phase35Cache,
    causal_last_observation_grid,
    extract_windows,
    load_cache,
    save_cache,
    valid_window_anchors,
)
from src.phase35.schema import REQUIRED_COLUMNS, TARGET_COLUMN, VALVE_COLUMN


def _cache(n=40):
    cols = tuple(REQUIRED_COLUMNS)
    values = np.zeros((n, len(cols)), dtype=np.float32)
    ages = np.zeros_like(values)
    for j in range(len(cols)):
        values[:, j] = j + np.arange(n) * 0.1
    values[:, cols.index(VALVE_COLUMN)] = np.linspace(10, 20, n)
    values[:, cols.index(TARGET_COLUMN)] = np.linspace(565, 570, n)
    return Phase35Cache(
        timestamps_ns=np.arange(n, dtype=np.int64) * 10_000_000_000,
        values=values,
        ages_s=ages,
        columns=cols,
        metadata={"side": "A", "step_seconds": 10},
    )


def test_causal_grid_never_backfills_future_observation():
    sec = 1_000_000_000
    updates = {"x": (np.array([15, 31], dtype=np.int64) * sec, np.array([1.0, 2.0]))}
    grid, values, ages = causal_last_observation_grid(updates, ["x"], 0, 40 * sec, 10)
    assert np.isnan(values[0, 0]) and np.isnan(values[1, 0])
    assert values[2, 0] == 1.0 and ages[2, 0] == 5.0
    assert values[3, 0] == 1.0 and ages[3, 0] == 15.0
    assert values[4, 0] == 2.0 and ages[4, 0] == 9.0


def test_cache_roundtrip_and_chronological_windows(tmp_path):
    cache = _cache()
    path = tmp_path / "cache.npz"
    save_cache(cache, path)
    loaded = load_cache(path)
    np.testing.assert_allclose(loaded.values, cache.values)
    anchors = valid_window_anchors(
        loaded,
        split="train",
        feature_columns=loaded.columns[:3] + (TARGET_COLUMN,),
        target_column=TARGET_COLUMN,
        valve_column=VALVE_COLUMN,
        window=4,
        horizon=3,
    )
    batch = extract_windows(
        loaded, anchors[:2], loaded.columns[:3] + (TARGET_COLUMN,), TARGET_COLUMN, VALVE_COLUMN, 4, 3
    )
    assert batch["history"].shape == (2, 4, 4)
    assert np.all(batch["anchors"] < loaded.split_bounds()["train"][1])


def test_validation_history_stays_inside_split_and_rejects_stale_rows():
    cache = _cache(100)
    bounds = cache.split_bounds()
    anchors = valid_window_anchors(
        cache, "validation", (TARGET_COLUMN,), TARGET_COLUMN, VALVE_COLUMN, window=4, horizon=3
    )
    assert anchors[0] == bounds["validation"][0] + 3
    cache.ages_s[anchors[0] - 2, cache.index(TARGET_COLUMN)] = 999.0
    filtered = valid_window_anchors(
        cache, "validation", (TARGET_COLUMN,), TARGET_COLUMN, VALVE_COLUMN, window=4, horizon=3
    )
    assert anchors[0] not in filtered


def test_window_anchors_never_cross_an_irregular_timestamp_gap():
    cache = _cache(100)
    cache.timestamps_ns[65:] += 10_000_000_000
    anchors = valid_window_anchors(
        cache,
        "validation",
        (TARGET_COLUMN,),
        TARGET_COLUMN,
        VALVE_COLUMN,
        window=4,
        horizon=3,
    )
    assert len(anchors) > 0
    for anchor in anchors:
        start = anchor - 3
        stop = anchor + 3
        assert np.all(np.diff(cache.timestamps_ns[start : stop + 1]) == 10_000_000_000)
