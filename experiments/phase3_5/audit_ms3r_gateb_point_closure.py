#!/usr/bin/env python3
"""Replay MS3-R Gate B daily MIMO matrices and paired contrasts without caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.phase35.ms3r_gateb import daily_mimo_matrices, paired_path_contrasts


def replay(config: dict, arrays: np.lib.npyio.NpzFile) -> dict:
    days = arrays["timestamps_ns"].astype("datetime64[ns]").astype("datetime64[D]")
    day_order = arrays["utc_day_order"].astype("datetime64[D]")
    action = np.stack((arrays["innovation_A"], arrays["innovation_B"]), axis=1)
    future, lead = {}, {}
    maximum_matrix_error = 0.0
    for horizon in sorted({*config["point_contract"]["primary_horizons_steps"], *config["point_contract"]["diagnostic_horizons_steps"]}):
        _, future[horizon], _ = daily_mimo_matrices(
            action, arrays[f"outcome_future_H{horizon}"], days,
            minimum_rows=int(config["analysis"]["minimum_day_rows"]),
            ridge_alpha=float(config["analysis"]["daily_mimo_ridge_alpha"]),
            epsilon=float(config["analysis"]["variance_epsilon"]), day_order=day_order,
        )
        _, lead[horizon], _ = daily_mimo_matrices(
            action, arrays[f"outcome_lead_H{horizon}"], days,
            minimum_rows=int(config["analysis"]["minimum_day_rows"]),
            ridge_alpha=float(config["analysis"]["daily_mimo_ridge_alpha"]),
            epsilon=float(config["analysis"]["variance_epsilon"]), day_order=day_order,
        )
        for name, recomputed in (("future", future[horizon]), ("lead", lead[horizon])):
            stored = arrays[f"mimo_{name}_H{horizon}"]
            finite = np.isfinite(stored) & np.isfinite(recomputed)
            if finite.any():
                maximum_matrix_error = max(maximum_matrix_error, float(np.max(np.abs(stored[finite] - recomputed[finite]))))
            if not np.array_equal(np.isnan(stored), np.isnan(recomputed)):
                raise RuntimeError(f"MS3-R Gate B replay NaN pattern mismatch for {name} H{horizon}")
    contrasts, contrast_arrays = paired_path_contrasts(future, lead, config)
    maximum_contrast_error = max(
        float(np.nanmax(np.abs(contrast_arrays[name] - arrays[name]))) for name in contrast_arrays
    )
    return {
        "protocol_version": config["protocol_version"],
        "maximum_daily_matrix_error": maximum_matrix_error,
        "maximum_paired_contrast_error": maximum_contrast_error,
        "paired_contrasts_recomputed": contrasts,
        "cache_accessed": False,
        "training_executed": False,
        "scientific_decision": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/phase3_5/ms3r_gateb_point_closure.json"))
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    with np.load(args.arrays, allow_pickle=False) as arrays:
        report = replay(config, arrays)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
