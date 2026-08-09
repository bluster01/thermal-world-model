import numpy as np

from src.phase35.data import Phase35Cache
from src.phase35.events import (
    ValveEvent,
    detect_sp_execution_events,
    detect_valve_events,
    match_quiet_controls,
    matching_diagnostics,
    quiet_control_candidates,
)
from src.phase35.schema import Phase35ProtocolError
from src.phase35.schema import (
    LOAD_COLUMN,
    REQUIRED_COLUMNS,
    SP_COLUMN,
    TARGET_COLUMN,
    TOUT2_COLUMN,
    VALVE_COLUMN,
)


def _cache(n=600):
    cols = tuple(REQUIRED_COLUMNS)
    values = np.zeros((n, len(cols)), dtype=np.float32)
    ages = np.zeros_like(values)
    values[:, cols.index(LOAD_COLUMN)] = 500.0
    values[:, cols.index(TARGET_COLUMN)] = 565.0 + np.linspace(0, 1, n)
    values[:, cols.index(TOUT2_COLUMN)] = 550.0 + np.linspace(0, 1, n)
    values[:, cols.index(SP_COLUMN)] = 568.0
    values[:, cols.index(VALVE_COLUMN)] = 20.0
    return Phase35Cache(
        timestamps_ns=np.arange(n, dtype=np.int64) * 10_000_000_000,
        values=values,
        ages_s=ages,
        columns=cols,
        metadata={"side": "A", "step_seconds": 10},
    )


def test_detects_isolated_valve_event_and_respects_split():
    cache = _cache()
    vi = cache.index(VALVE_COLUMN)
    cache.values[70:, vi] += 3.0
    events = detect_valve_events(cache, "train", window=12, horizon=12, min_gap_seconds=60)
    assert len(events) == 1
    assert events[0].onset == 70 and events[0].direction == "open"
    assert all(e.onset < cache.split_bounds()["train"][1] for e in events)


def test_sp_events_classify_execution_from_feedback():
    cache = _cache()
    si, vi = cache.index(SP_COLUMN), cache.index(VALVE_COLUMN)
    cache.values[100:, si] += 2.0
    cache.values[220:, si] += 2.0
    cache.values[220:, vi] -= 2.0
    events = detect_sp_execution_events(cache, "train", window=12, horizon=60, min_gap_seconds=60)
    labels = {e.onset: e.execution for e in events}
    assert labels[100] == "no_execution"
    assert labels[220] == "executed"


def test_sp_no_execution_requires_valve_quiet_for_full_600_seconds():
    cache = _cache()
    si, vi = cache.index(SP_COLUMN), cache.index(VALVE_COLUMN)
    cache.values[100:, si] += 2.0
    cache.values[130:, vi] += 2.0  # delayed actuation after the 60 s fast window
    events = detect_sp_execution_events(cache, "train", window=12, horizon=60, min_gap_seconds=60)
    event = next(e for e in events if e.onset == 100)
    assert event.execution == "ambiguous"
    assert event.valve_max_60s == 0.0 and event.valve_max_600s == 2.0


def test_quiet_controls_and_matching_use_pretreatment_rows():
    cache = _cache(600)
    vi = cache.index(VALVE_COLUMN)
    cache.values[120:, vi] += 3.0
    events = detect_valve_events(cache, "train", window=12, horizon=12, min_gap_seconds=60)
    controls = quiet_control_candidates(cache, "train", window=12, horizon=12, stride_seconds=20)
    matches = match_quiet_controls(cache, events, controls, controls_per_event=2, min_time_separation_seconds=60)
    assert events[0].event_id in matches
    assert all(abs(c - events[0].anchor) >= 6 for c in matches[events[0].event_id])
    diagnostics = matching_diagnostics(cache, events, matches)
    assert diagnostics["status"] == "insufficient_events"  # one event cannot establish balance


def test_zero_variance_mean_imbalance_fails_closed():
    cache = _cache()
    event_anchors = [100, 120, 140]
    control_anchors = [300, 320, 340]
    li = cache.index(LOAD_COLUMN)
    cache.values[event_anchors, li] = 500.0
    cache.values[control_anchors, li] = 600.0
    events = [
        ValveEvent(f"event-{i}", anchor, anchor + 1, 2.0, "open", 20.0, "validation")
        for i, anchor in enumerate(event_anchors)
    ]
    matches = {event.event_id: [control] for event, control in zip(events, control_anchors)}
    diagnostics = matching_diagnostics(cache, events, matches, pretrend_seconds=10)
    assert diagnostics["status"] == "undefined_smd_zero_variance_imbalance"
    assert diagnostics["max_abs_smd"] is None
    assert "load" in diagnostics["zero_variance_imbalanced_covariates"]
    assert diagnostics["covariates"]["load"]["raw_mean_difference"] == -100.0


def test_matching_rejects_invalid_caliper_quantile():
    cache = _cache()
    event = ValveEvent("event", 100, 101, 2.0, "open", 20.0, "validation")
    controls = np.array([300, 320], dtype=np.int64)
    with np.testing.assert_raises(Phase35ProtocolError):
        match_quiet_controls(cache, [event], controls, controls_per_event=1, caliper_quantile=0.0)
