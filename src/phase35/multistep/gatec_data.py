"""Paired A/B causal windows for the MS3-R Gate C model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..data import Phase35Cache
from ..schema import (
    COAL_COLUMN,
    FEEDWATER_COLUMN,
    LOAD_COLUMN,
    PRESSURE_COLUMN,
    SP_COLUMN,
    STEAM_COLUMN,
    TARGET_COLUMN,
    TIN2_COLUMN,
    TOUT2_COLUMN,
    VALVE_COLUMN,
    Phase35ProtocolError,
)
from .gatec_contracts import BOUNDARY_MODES


SHARED_HISTORY_FEATURES = (
    LOAD_COLUMN,
    PRESSURE_COLUMN,
    FEEDWATER_COLUMN,
    COAL_COLUMN,
    STEAM_COLUMN,
)
SIDE_HISTORY_FEATURES = (TIN2_COLUMN, TOUT2_COLUMN, TARGET_COLUMN, SP_COLUMN, VALVE_COLUMN)
FUTURE_REQUIRED_FEATURES = (TIN2_COLUMN, TOUT2_COLUMN, TARGET_COLUMN, SP_COLUMN, VALVE_COLUMN)


def paired_history_feature_names() -> tuple[str, ...]:
    return (
        *SHARED_HISTORY_FEATURES,
        *(f"A::{name}" for name in SIDE_HISTORY_FEATURES),
        *(f"B::{name}" for name in SIDE_HISTORY_FEATURES),
    )


def _validate_pair(caches: Mapping[str, Phase35Cache], shared_tolerance: float) -> None:
    if set(caches) != {"A", "B"}:
        raise Phase35ProtocolError("Gate C requires exactly aligned A/B caches")
    a, b = caches["A"], caches["B"]
    if not np.array_equal(a.timestamps_ns, b.timestamps_ns):
        raise Phase35ProtocolError("Gate C A/B cache timestamps must align exactly")
    if a.split_bounds() != b.split_bounds():
        raise Phase35ProtocolError("Gate C A/B split bounds differ")
    for side, cache in caches.items():
        if cache.metadata.get("side") != side:
            raise Phase35ProtocolError(f"Gate C cache side mismatch for {side}")
        missing = set((*SHARED_HISTORY_FEATURES, *SIDE_HISTORY_FEATURES)) - set(cache.columns)
        if missing:
            raise Phase35ProtocolError(f"Gate C cache {side} is missing columns: {sorted(missing)}")
    for column in SHARED_HISTORY_FEATURES:
        av = a.values[:, a.index(column)].astype(float)
        bv = b.values[:, b.index(column)].astype(float)
        if not np.allclose(av, bv, rtol=0.0, atol=shared_tolerance, equal_nan=True):
            raise Phase35ProtocolError(f"Gate C shared feature drift for {column}")


def _range_has_bad(prefix: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    return (prefix[stops] - prefix[starts]) > 0


def paired_valid_anchors(
    caches: Mapping[str, Phase35Cache],
    split: str,
    *,
    window: int,
    horizon: int,
    max_age_s: float,
    shared_tolerance: float = 1e-5,
) -> np.ndarray:
    _validate_pair(caches, shared_tolerance)
    if split == "test":
        raise Phase35ProtocolError("Gate C paired windows prohibit the test split")
    if split not in {"train", "validation"}:
        raise Phase35ProtocolError(f"unknown Gate C split={split!r}")
    if min(window, horizon) < 1 or max_age_s <= 0:
        raise Phase35ProtocolError("Gate C window/horizon/age settings are invalid")
    cache = caches["A"]
    lo, hi = cache.split_bounds()[split]
    first, last = lo + window - 1, hi - horizon - 1
    if last < first:
        return np.empty(0, dtype=np.int64)
    anchors = np.arange(first, last + 1, dtype=np.int64)
    history_starts = anchors - window + 1
    future_stops = anchors + horizon + 1

    expected = int(round(float(cache.metadata.get("step_seconds", 10.0)) * 1e9))
    transition_bad = np.diff(cache.timestamps_ns) != expected
    transition_prefix = np.concatenate(([0], np.cumsum(transition_bad, dtype=np.int64)))
    valid = ~_range_has_bad(transition_prefix, history_starts, anchors + horizon)

    history_bad = np.zeros(len(cache.timestamps_ns), dtype=bool)
    future_bad = np.zeros(len(cache.timestamps_ns), dtype=bool)
    for side_cache in caches.values():
        history_indices = [side_cache.index(column) for column in SIDE_HISTORY_FEATURES]
        future_indices = [side_cache.index(column) for column in FUTURE_REQUIRED_FEATURES]
        history_bad |= (~np.isfinite(side_cache.values[:, history_indices])).any(axis=1)
        history_bad |= (side_cache.ages_s[:, history_indices] > max_age_s).any(axis=1)
        future_bad |= (~np.isfinite(side_cache.values[:, future_indices])).any(axis=1)
        future_bad |= (side_cache.ages_s[:, future_indices] > max_age_s).any(axis=1)
    shared_indices = [cache.index(column) for column in SHARED_HISTORY_FEATURES]
    history_bad |= (~np.isfinite(cache.values[:, shared_indices])).any(axis=1)
    history_bad |= (cache.ages_s[:, shared_indices] > max_age_s).any(axis=1)
    history_prefix = np.concatenate(([0], np.cumsum(history_bad, dtype=np.int64)))
    future_prefix = np.concatenate(([0], np.cumsum(future_bad, dtype=np.int64)))
    valid &= ~_range_has_bad(history_prefix, history_starts, anchors + 1)
    valid &= ~_range_has_bad(future_prefix, anchors + 1, future_stops)
    return anchors[valid]


@dataclass(frozen=True)
class PairedGateCBatch:
    anchors: np.ndarray
    history: np.ndarray
    history_feature_names: tuple[str, ...]
    future_sp: np.ndarray
    logged_future_valve: np.ndarray
    logged_future_tin: np.ndarray
    local_drop_target: np.ndarray
    tout_target: np.ndarray
    terminal_target: np.ndarray

    def model_inputs(
        self,
        boundary_mode: str,
        *,
        scenario_tin: np.ndarray | None = None,
        allow_oracle: bool = False,
    ) -> dict[str, Any]:
        if boundary_mode not in BOUNDARY_MODES:
            raise Phase35ProtocolError(f"unknown Gate C boundary mode={boundary_mode!r}")
        boundary: np.ndarray | None
        if boundary_mode == "forecast_boundary":
            if scenario_tin is not None:
                raise Phase35ProtocolError("forecast boundary cannot accept scenario Tin")
            boundary = None
        elif boundary_mode == "oracle_boundary":
            if not allow_oracle:
                raise Phase35ProtocolError("oracle boundary requires explicit audit permission")
            boundary = self.logged_future_tin.copy()
        else:
            if scenario_tin is None or scenario_tin.shape != self.logged_future_tin.shape:
                raise Phase35ProtocolError("scenario boundary requires explicit Tin with target shape")
            if not np.isfinite(scenario_tin).all():
                raise Phase35ProtocolError("scenario Tin must be finite")
            boundary = np.asarray(scenario_tin, dtype=np.float32).copy()
        return {
            "history": self.history,
            "future_sp": self.future_sp,
            "boundary_future": boundary,
            "boundary_mode": boundary_mode,
        }


def _window(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return values[rows]


def extract_gatec_batch(
    caches: Mapping[str, Phase35Cache],
    anchors: np.ndarray,
    *,
    window: int,
    horizon: int,
    shared_tolerance: float = 1e-5,
) -> PairedGateCBatch:
    _validate_pair(caches, shared_tolerance)
    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.ndim != 1 or len(anchors) == 0:
        raise Phase35ProtocolError("Gate C batch needs non-empty one-dimensional anchors")
    history_rows = anchors[:, None] - np.arange(window - 1, -1, -1, dtype=np.int64)[None, :]
    future_rows = anchors[:, None] + np.arange(1, horizon + 1, dtype=np.int64)[None, :]
    a = caches["A"]
    history_parts = [
        _window(a.values[:, a.index(column)].astype(np.float32), history_rows)
        for column in SHARED_HISTORY_FEATURES
    ]
    for side in ("A", "B"):
        cache = caches[side]
        history_parts.extend(
            _window(cache.values[:, cache.index(column)].astype(np.float32), history_rows)
            for column in SIDE_HISTORY_FEATURES
        )
    history = np.stack(history_parts, axis=2)

    def future(column: str) -> np.ndarray:
        return np.stack(
            [caches[side].values[:, caches[side].index(column)].astype(np.float32)[future_rows] for side in ("A", "B")],
            axis=2,
        )

    future_tin = future(TIN2_COLUMN)
    future_tout = future(TOUT2_COLUMN)
    return PairedGateCBatch(
        anchors=anchors.copy(),
        history=history,
        history_feature_names=paired_history_feature_names(),
        future_sp=future(SP_COLUMN),
        logged_future_valve=future(VALVE_COLUMN),
        logged_future_tin=future_tin,
        local_drop_target=future_tin - future_tout,
        tout_target=future_tout,
        terminal_target=future(TARGET_COLUMN),
    )
