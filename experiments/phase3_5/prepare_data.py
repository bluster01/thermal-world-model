#!/usr/bin/env python3
"""Prepare a causal 10-second Phase 3.5 cache from an asynchronous historian CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.phase35.data import (
    Phase35Cache,
    causal_last_observation_grid,
    collect_sparse_updates,
    save_cache,
)
from src.phase35.schema import REQUIRED_COLUMNS, validate_columns


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="A/B raw sparse CSV")
    parser.add_argument("--output", required=True, help="output .npz cache")
    parser.add_argument("--side", required=True, choices=["A", "B"])
    parser.add_argument("--step-seconds", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    columns = tuple(REQUIRED_COLUMNS)
    validate_columns(columns)
    source = Path(args.input).resolve()
    n_rows, first_ns, last_ns, updates = collect_sparse_updates(
        source, columns, chunksize=args.chunksize
    )
    grid, values, ages = causal_last_observation_grid(
        updates, columns, first_ns, last_ns, args.step_seconds
    )
    quality = {}
    for column, (timestamps, observed) in updates.items():
        gaps = np.diff(timestamps) / 1e9 if len(timestamps) > 1 else np.empty(0)
        quality[column] = {
            "updates": int(len(observed)),
            "update_gap_median_s": float(np.median(gaps)) if len(gaps) else None,
            "update_gap_p90_s": float(np.quantile(gaps, 0.9)) if len(gaps) else None,
            "grid_age_p90_s": float(np.quantile(ages[np.isfinite(ages[:, columns.index(column)]), columns.index(column)], 0.9))
            if np.isfinite(ages[:, columns.index(column)]).any() else None,
        }
    metadata = {
        "protocol_version": "phase3.5-v1",
        "side": args.side,
        "step_seconds": args.step_seconds,
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "sha256": sha256_file(source),
            "raw_rows": int(n_rows),
        },
        "grid_start_ns": int(grid[0]),
        "grid_end_ns": int(grid[-1]),
        "grid_rows": int(len(grid)),
        "reconstruction": "causal_last_observation_carried_forward",
        "quality": quality,
    }
    cache = Phase35Cache(grid, values, ages, columns, metadata)
    save_cache(cache, args.output)
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(tmp, manifest_path)
    print(json.dumps({"cache": str(Path(args.output).resolve()), "grid_rows": len(grid)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
