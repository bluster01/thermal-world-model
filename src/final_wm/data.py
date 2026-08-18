"""D0 data pipeline: discovery, quality gates, canonical record, windows.

Two-phase contract (Linux executes both; it never edits code or mapping):

1. `discover_dataset` scans a data root and returns a schema/quality
   discovery report (bounded size; no bulk data leaves the machine).
   The channel mapping is frozen locally from that report.
2. `build_canonical` applies the frozen mapping, resamples to the 10 s
   grid, runs the fail-closed quality gates from the discrimination
   matrix §1.2, and writes the single canonical record every later unit
   reads.

The canonical record is one `.npz` (float32 arrays in registry order,
int64 timestamps, int8 split id) plus a `meta.json`.  Split ids:
0=train, 1=validation, 2=test (locked; the window sampler refuses it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BOUNDARY_ELEMENTS,
    FinalWMProtocolError,
    OBSERVATION_ELEMENTS,
)
from src.final_wm.model import HistoryWindow

CANONICAL_VERSION = 1
SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2  # locked

SUPPORTED_SUFFIXES = (".csv", ".parquet")


# ---------------------------------------------------------------------------
# D0a: discovery
# ---------------------------------------------------------------------------

def discover_dataset(root: str | Path, max_rows_per_file: int = 200_000) -> dict:
    """Bounded schema/quality discovery over CSV/parquet exports."""
    import pandas as pd

    root = Path(root)
    if not root.exists():
        raise FinalWMProtocolError(f"data root does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FinalWMProtocolError(f"no CSV/parquet files under {root}")
    report = {"root": str(root), "files": []}
    for path in files:
        rel = str(path.relative_to(root))
        entry = {"file": rel, "size_bytes": path.stat().st_size}
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path, nrows=max_rows_per_file)
            entry["n_rows_sampled"] = int(len(frame))
            entry["columns"] = [str(c) for c in frame.columns]
            numeric = frame.select_dtypes(include="number")
            entry["numeric_stats"] = {
                str(col): {
                    "min": float(numeric[col].min()),
                    "max": float(numeric[col].max()),
                    "mean": float(numeric[col].mean()),
                    "nan_ratio": float(numeric[col].isna().mean()),
                }
                for col in numeric.columns
            }
            entry["head"] = frame.head(3).astype(str).to_dict(orient="records")
        except Exception as exc:  # report, do not hide
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["files"].append(entry)
    return report


# ---------------------------------------------------------------------------
# Channel mapping (frozen locally after D0a)
# ---------------------------------------------------------------------------

def load_channel_mapping(path: str | Path) -> dict:
    """Validate a frozen channel mapping against the registries.

    Schema:
    {
      "time_columns": {"relative/file.csv": "timestamp_column", ...},
      "channels": {
        "steam_flow": {"file": "relative/file.csv", "column": "...", "unit_scale": 1.0, "unit_offset": 0.0},
        ...  # every BOUNDARY/ACTION/OBSERVATION element exactly once
      }
    }
    """
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    channels = mapping.get("channels", {})
    required = set(BOUNDARY_ELEMENTS) | set(ACTION_ELEMENTS) | set(OBSERVATION_ELEMENTS)
    missing = required - set(channels)
    extra = set(channels) - required
    if missing or extra:
        raise FinalWMProtocolError(f"channel mapping mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    for name, spec in channels.items():
        if "file" not in spec or "column" not in spec:
            raise FinalWMProtocolError(f"channel {name} lacks file/column")
        if float(spec.get("unit_scale", 1.0)) == 0.0:
            raise FinalWMProtocolError(f"channel {name} has zero unit_scale")
    time_columns = mapping.get("time_columns", {})
    used_files = {spec["file"] for spec in channels.values()}
    if not used_files <= set(time_columns):
        raise FinalWMProtocolError("every file used by channels needs a time column entry")
    return mapping


# ---------------------------------------------------------------------------
# D0b: canonical record build with fail-closed quality gates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityGateConfig:
    dt_seconds: float = 10.0
    max_gap_ratio: float = 0.01
    max_stuck_ratio: float = 0.05
    stuck_minutes: float = 30.0
    min_valve_active_ratio: float = 0.60
    min_days: float = 30.0
    val_fraction: float = 0.15
    test_fraction: float = 0.10  # reserved, locked


class QualityReport(NamedTuple):
    gap_ratio: float
    stuck_ratio: dict
    valve_active_ratio: float
    days: float


def _stuck_ratio(values: np.ndarray, min_run: int) -> float:
    """Fraction of samples in zero-variance runs of length >= min_run."""
    if len(values) < min_run:
        return 0.0
    change = np.abs(np.diff(values)) > 1e-12
    # index of last change up to each point
    idx = np.arange(len(values))
    last_change = np.maximum.accumulate(np.where(np.concatenate(([True], change)), idx, 0))
    run_length = idx - last_change + 1
    return float((run_length >= min_run).mean())


def _read_frame(root: Path, rel: str):
    import pandas as pd

    path = root / rel
    if not path.exists():
        raise FinalWMProtocolError(f"mapping references missing file: {rel}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_canonical(
    root: str | Path,
    mapping_path: str | Path,
    out_path: str | Path,
    *,
    gates: QualityGateConfig = QualityGateConfig(),
) -> QualityReport:
    """Build the canonical record; raises (fail-closed) on any gate breach."""
    import pandas as pd

    root = Path(root)
    mapping = load_channel_mapping(mapping_path)
    dt = pd.Timedelta(seconds=gates.dt_seconds)

    frames: dict[str, "pd.DataFrame"] = {}
    for rel, time_col in mapping["time_columns"].items():
        frame = _read_frame(root, rel)
        if time_col not in frame.columns:
            raise FinalWMProtocolError(f"time column {time_col} missing in {rel}")
        frame[time_col] = pd.to_datetime(frame[time_col])
        frames[rel] = frame.set_index(time_col).sort_index()

    # Common 10 s grid spanning the intersection of all sources.
    lo = max(f.index.min() for f in frames.values())
    hi = min(f.index.max() for f in frames.values())
    if not lo < hi:
        raise FinalWMProtocolError("source time ranges do not overlap")
    grid = pd.date_range(lo, hi, freq=dt)
    limit = 3  # interpolate at most 30 s gaps

    series: dict[str, np.ndarray] = {}
    for name in list(BOUNDARY_ELEMENTS) + list(ACTION_ELEMENTS) + list(OBSERVATION_ELEMENTS):
        spec = mapping["channels"][name]
        frame = frames[spec["file"]]
        if spec["column"] not in frame.columns:
            raise FinalWMProtocolError(f"column {spec['column']} missing in {spec['file']}")
        resampled = frame[spec["column"]].reindex(frame.index.union(grid)).sort_index()
        resampled = resampled.interpolate(method="time", limit=limit).reindex(grid)
        values = resampled.to_numpy(dtype=np.float64)
        values = values * float(spec.get("unit_scale", 1.0)) + float(spec.get("unit_offset", 0.0))
        series[name] = values

    timestamps = (grid.asi8 // 10**9).astype(np.int64)
    boundary = np.stack([series[n] for n in BOUNDARY_ELEMENTS], axis=1)
    actions = np.stack([series[n] for n in ACTION_ELEMENTS], axis=1)
    obs = np.stack([series[n] for n in OBSERVATION_ELEMENTS], axis=1)
    valid = np.isfinite(boundary).all(axis=1) & np.isfinite(actions).all(axis=1) & np.isfinite(obs).all(axis=1)
    gap_ratio = 1.0 - float(valid.mean())

    boundary = np.nan_to_num(boundary, nan=0.0)
    actions = np.nan_to_num(actions, nan=0.0)
    obs = np.nan_to_num(obs, nan=0.0)

    min_run = int(gates.stuck_minutes * 60.0 / gates.dt_seconds)
    stuck = {
        name: _stuck_ratio(series[name], min_run)
        for name in list(BOUNDARY_ELEMENTS) + list(ACTION_ELEMENTS) + list(OBSERVATION_ELEMENTS)
    }
    v1, v2 = actions[:, 0], actions[:, 1]
    valve_active = float(((v1 > 0.02) & (v1 < 0.98) & (v2 > 0.02) & (v2 < 0.98)).mean())
    days = float((timestamps[-1] - timestamps[0]) / 86400.0)
    report = QualityReport(gap_ratio=gap_ratio, stuck_ratio=stuck, valve_active_ratio=valve_active, days=days)

    breaches = []
    if gap_ratio > gates.max_gap_ratio:
        breaches.append(f"gap_ratio {gap_ratio:.4f} > {gates.max_gap_ratio}")
    worst_stuck = max(stuck.values())
    if worst_stuck > gates.max_stuck_ratio:
        breaches.append(f"stuck_ratio {worst_stuck:.4f} > {gates.max_stuck_ratio}")
    if valve_active < gates.min_valve_active_ratio:
        breaches.append(f"valve_active_ratio {valve_active:.3f} < {gates.min_valve_active_ratio}")
    if days < gates.min_days:
        breaches.append(f"days {days:.1f} < {gates.min_days}")
    if breaches:
        raise FinalWMProtocolError("D0 quality gates breached: " + "; ".join(breaches))

    n = len(timestamps)
    n_test = int(n * gates.test_fraction)
    n_val = int(n * gates.val_fraction)
    split = np.zeros(n, dtype=np.int8)
    split[n - n_val - n_test : n - n_test] = SPLIT_VAL
    split[n - n_test :] = SPLIT_TEST

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        boundary=boundary.astype(np.float32),
        actions=actions.astype(np.float32),
        obs=obs.astype(np.float32),
        valid=valid,
        timestamps=timestamps,
        split=split,
    )
    meta = {
        "version": CANONICAL_VERSION,
        "dt_seconds": gates.dt_seconds,
        "n_samples": n,
        "start_epoch": int(timestamps[0]),
        "end_epoch": int(timestamps[-1]),
        "quality": {
            "gap_ratio": gap_ratio,
            "stuck_ratio": stuck,
            "valve_active_ratio": valve_active,
            "days": days,
        },
        "splits": {"train": [0, n - n_val - n_test], "val": [n - n_val - n_test, n - n_test], "test": "LOCKED"},
        "mapping_path": str(mapping_path),
        "test_locked": True,
    }
    (out_path.parent / (out_path.stem + "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# Runtime: canonical record loading and window sampling
# ---------------------------------------------------------------------------

class CanonicalRecord:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FinalWMProtocolError(f"canonical record missing: {path}")
        arrays = np.load(path)
        self.boundary = torch.from_numpy(arrays["boundary"].astype(np.float32))
        self.actions = torch.from_numpy(arrays["actions"].astype(np.float32))
        self.obs = torch.from_numpy(arrays["obs"].astype(np.float32))
        self.timestamps = torch.from_numpy(arrays["timestamps"].astype(np.int64))
        self.split = torch.from_numpy(arrays["split"].astype(np.int64))
        self.n = int(self.boundary.shape[0])
        if self.boundary.shape[1] != len(BOUNDARY_ELEMENTS):
            raise FinalWMProtocolError("canonical record boundary width mismatch")
        if self.actions.shape[1] != len(ACTION_ELEMENTS) or self.obs.shape[1] != len(OBSERVATION_ELEMENTS):
            raise FinalWMProtocolError("canonical record channel width mismatch")

    def split_runs(self, split_id: int) -> list[tuple[int, int]]:
        """Contiguous [start, end) index runs belonging to a split."""
        if split_id == SPLIT_TEST:
            raise FinalWMProtocolError("test split is locked and cannot be read")
        mask = (self.split == split_id).numpy()
        runs: list[tuple[int, int]] = []
        start = None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs


class WindowBatch(NamedTuple):
    history: HistoryWindow
    future_boundary: torch.Tensor   # (B, H, 7)
    future_actions: torch.Tensor    # (B, H, 2)
    future_obs: torch.Tensor        # (B, H, 5)
    day_ids: torch.Tensor           # (B,) UTC-day of the first future step


def sample_windows(
    record: CanonicalRecord,
    split_id: int,
    batch_size: int,
    history_steps: int,
    horizon: int,
    generator: torch.Generator,
) -> WindowBatch:
    """Uniform windows fully contained in one contiguous run of the split."""
    span = history_steps + horizon
    runs = [(s, e) for s, e in record.split_runs(split_id) if e - s >= span]
    if not runs:
        raise FinalWMProtocolError(f"no contiguous run of length >= {span} in split {split_id}")
    starts = []
    for _ in range(batch_size):
        s, e = runs[int(torch.randint(len(runs), (1,), generator=generator))]
        starts.append(int(torch.randint(s, e - span + 1, (1,), generator=generator)) + history_steps)
    idx = torch.tensor(starts, dtype=torch.long)  # first future index
    hist_off = torch.arange(-history_steps, 0)
    fut_off = torch.arange(0, horizon)
    hist_idx = idx[:, None] + hist_off[None, :]
    fut_idx = idx[:, None] + fut_off[None, :]
    history = HistoryWindow(
        obs=record.obs[hist_idx],
        actions=record.actions[hist_idx],
        boundary=record.boundary[hist_idx],
    )
    day_ids = torch.div(record.timestamps[idx], 86400, rounding_mode="floor")
    return WindowBatch(
        history=history,
        future_boundary=record.boundary[fut_idx],
        future_actions=record.actions[fut_idx],
        future_obs=record.obs[fut_idx],
        day_ids=day_ids,
    )
