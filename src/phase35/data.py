"""Causal reconstruction and chronological window access for Phase 3.5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .schema import DATE_COLUMN, Phase35ProtocolError, SplitSpec, validate_columns


NAT_INT = np.iinfo(np.int64).min


@dataclass
class Phase35Cache:
    timestamps_ns: np.ndarray
    values: np.ndarray
    ages_s: np.ndarray
    columns: tuple[str, ...]
    metadata: dict

    def __post_init__(self) -> None:
        if self.timestamps_ns.ndim != 1:
            raise Phase35ProtocolError("timestamps_ns must be one-dimensional")
        if self.values.shape != self.ages_s.shape:
            raise Phase35ProtocolError("cache values and ages must have identical shapes")
        if self.values.shape != (len(self.timestamps_ns), len(self.columns)):
            raise Phase35ProtocolError("cache matrix shape does not match timestamps/columns")
        if len(self.timestamps_ns) > 1 and np.any(np.diff(self.timestamps_ns) <= 0):
            raise Phase35ProtocolError("cache timestamps must be strictly increasing")
        if not self.columns or len(self.columns) != len(set(self.columns)):
            raise Phase35ProtocolError("cache columns must be non-empty and unique")

    def index(self, name: str) -> int:
        try:
            return self.columns.index(name)
        except ValueError as exc:
            raise Phase35ProtocolError(f"cache does not contain column={name!r}") from exc

    def split_bounds(self, split_spec: SplitSpec | None = None) -> dict[str, tuple[int, int]]:
        return (split_spec or SplitSpec()).bounds(len(self.timestamps_ns))


def causal_last_observation_grid(
    updates: Mapping[str, tuple[np.ndarray, np.ndarray]],
    columns: Sequence[str],
    start_ns: int,
    end_ns: int,
    step_seconds: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an inclusive regular grid using only observations at or before each grid time."""
    if step_seconds <= 0 or end_ns < start_ns:
        raise Phase35ProtocolError("invalid causal grid range")
    step_ns = int(step_seconds * 1_000_000_000)
    first = ((int(start_ns) + step_ns - 1) // step_ns) * step_ns
    last = (int(end_ns) // step_ns) * step_ns
    if last < first:
        raise Phase35ProtocolError("source span is shorter than one aligned grid interval")
    grid = np.arange(first, last + 1, step_ns, dtype=np.int64)
    values = np.full((len(grid), len(columns)), np.nan, dtype=np.float32)
    ages = np.full_like(values, np.inf, dtype=np.float32)
    for j, column in enumerate(columns):
        if column not in updates:
            continue
        ts, val = updates[column]
        ts = np.asarray(ts, dtype=np.int64)
        val = np.asarray(val, dtype=np.float32)
        if ts.shape != val.shape or ts.ndim != 1:
            raise Phase35ProtocolError(f"invalid updates for column={column!r}")
        finite = np.isfinite(val) & (ts != NAT_INT)
        ts, val = ts[finite], val[finite]
        if len(ts) == 0:
            continue
        if np.any(np.diff(ts) < 0):
            order = np.argsort(ts, kind="stable")
            ts, val = ts[order], val[order]
        pos = np.searchsorted(ts, grid, side="right") - 1
        ok = pos >= 0
        values[ok, j] = val[pos[ok]]
        ages[ok, j] = (grid[ok] - ts[pos[ok]]) / 1e9
    return grid, values, ages


def collect_sparse_updates(
    csv_path: str | Path,
    columns: Sequence[str],
    chunksize: int = 1_000_000,
    progress_every_rows: int = 5_000_000,
) -> tuple[int, int, int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Stream a sparse historian CSV and retain only non-null tag updates."""
    import pandas as pd

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    usecols = [DATE_COLUMN, *columns]
    t_parts: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    v_parts: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    n_rows = 0
    first_ns: int | None = None
    last_ns: int | None = None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        ts = pd.to_datetime(chunk[DATE_COLUMN], errors="coerce", utc=True).astype("int64").to_numpy()
        good_t = ts != NAT_INT
        if good_t.any():
            valid_t = ts[good_t]
            lo, hi = int(valid_t.min()), int(valid_t.max())
            first_ns = lo if first_ns is None else min(first_ns, lo)
            last_ns = hi if last_ns is None else max(last_ns, hi)
        for column in columns:
            val = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=np.float32)
            good = good_t & np.isfinite(val)
            if good.any():
                t_parts[column].append(ts[good].copy())
                v_parts[column].append(val[good].copy())
        n_rows += len(chunk)
        if progress_every_rows and n_rows % progress_every_rows == 0:
            print(json.dumps({"rows_scanned": n_rows}), flush=True)
    if first_ns is None or last_ns is None:
        raise Phase35ProtocolError("CSV contains no valid timestamps")
    updates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column in columns:
        if t_parts[column]:
            ts = np.concatenate(t_parts[column])
            val = np.concatenate(v_parts[column])
        else:
            ts = np.empty(0, dtype=np.int64)
            val = np.empty(0, dtype=np.float32)
        updates[column] = (ts, val)
    return n_rows, first_ns, last_ns, updates


def save_cache(cache: Phase35Cache, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        timestamps_ns=cache.timestamps_ns.astype(np.int64),
        values=cache.values.astype(np.float32),
        ages_s=cache.ages_s.astype(np.float32),
        columns=np.asarray(cache.columns, dtype="U"),
        metadata_json=np.asarray(json.dumps(cache.metadata, ensure_ascii=False, sort_keys=True), dtype="U"),
    )


def load_cache(path: str | Path) -> Phase35Cache:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata_json"].item()))
        return Phase35Cache(
            timestamps_ns=z["timestamps_ns"].astype(np.int64),
            values=z["values"].astype(np.float32),
            ages_s=z["ages_s"].astype(np.float32),
            columns=tuple(str(x) for x in z["columns"].tolist()),
            metadata=metadata,
        )


def deterministic_anchor_subset(anchors: np.ndarray, limit: int, seed: int) -> np.ndarray:
    anchors = np.asarray(anchors, dtype=np.int64)
    if limit <= 0 or len(anchors) <= limit:
        return anchors.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(anchors, size=limit, replace=False))


def valid_window_anchors(
    cache: Phase35Cache,
    split: str,
    feature_columns: Sequence[str],
    target_column: str,
    valve_column: str,
    window: int,
    horizon: int,
    max_age_s: float = 180.0,
    split_spec: SplitSpec | None = None,
) -> np.ndarray:
    bounds = cache.split_bounds(split_spec)
    if split not in bounds:
        raise Phase35ProtocolError(f"unknown split={split!r}")
    lo, hi = bounds[split]
    # A window must be wholly contained in its split.  In particular, the first
    # validation/test history may not borrow rows from the preceding split.
    first = lo + window - 1
    last = hi - horizon - 1
    if last < first:
        return np.empty(0, dtype=np.int64)
    anchors = np.arange(first, last + 1, dtype=np.int64)
    step_seconds = float(cache.metadata.get("step_seconds", 10.0))
    expected_step_ns = int(round(step_seconds * 1_000_000_000))
    transition_bad = np.diff(cache.timestamps_ns) != expected_step_ns
    transition_prefix = np.concatenate(
        ([0], np.cumsum(transition_bad, dtype=np.int64))
    )
    window_start = anchors - window + 1
    future_end = anchors + horizon
    contiguous_ok = (
        transition_prefix[future_end] - transition_prefix[window_start]
    ) == 0
    fi = [cache.index(c) for c in feature_columns]
    ti, vi = cache.index(target_column), cache.index(valve_column)
    history_bad = (~np.isfinite(cache.values[:, fi])).any(axis=1)
    history_bad |= (cache.ages_s[:, fi] > max_age_s).any(axis=1)
    prefix = np.concatenate(([0], np.cumsum(history_bad, dtype=np.int64)))
    history_bad_count = prefix[anchors + 1] - prefix[anchors - window + 1]
    endpoint_ok = (history_bad_count == 0) & contiguous_ok
    endpoint_ok &= np.isfinite(cache.values[anchors, vi])
    endpoint_ok &= cache.ages_s[anchors, vi] <= max_age_s
    # All future target/action points must be observed causally and not excessively stale.
    future_ok = np.ones(len(anchors), dtype=bool)
    for k in range(1, horizon + 1):
        rows = anchors + k
        future_ok &= np.isfinite(cache.values[rows, ti]) & np.isfinite(cache.values[rows, vi])
        future_ok &= (cache.ages_s[rows, ti] <= max_age_s) & (cache.ages_s[rows, vi] <= max_age_s)
    return anchors[endpoint_ok & future_ok]


def extract_windows(
    cache: Phase35Cache,
    anchors: Iterable[int],
    feature_columns: Sequence[str],
    target_column: str,
    valve_column: str,
    window: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    anchors = np.asarray(list(anchors), dtype=np.int64)
    fi = [cache.index(c) for c in feature_columns]
    ti, vi = cache.index(target_column), cache.index(valve_column)
    history = np.stack([cache.values[a - window + 1:a + 1, fi] for a in anchors]).astype(np.float32)
    future_target = np.stack([cache.values[a + 1:a + horizon + 1, ti] for a in anchors]).astype(np.float32)
    future_valve = np.stack([cache.values[a + 1:a + horizon + 1, vi] for a in anchors]).astype(np.float32)
    return {
        "history": history,
        "baseline_valve": cache.values[anchors, vi].astype(np.float32),
        "future_valve": future_valve,
        "target": future_target,
        "anchors": anchors,
    }
