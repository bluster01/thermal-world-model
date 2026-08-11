#!/usr/bin/env python3
"""Independently audit committed MS3-D event and day-level artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/phase3_5/ms3d_asymmetry_diagnosis"
DEFAULT_OUTPUT = "supervisor_replay_validation.json"
PHYSICAL_KEYS = (
    "h180_local_drop_per_sp",
    "h300_local_drop_per_sp",
    "h180_local_drop_gain_c_per_pct",
    "h300_local_drop_gain_c_per_pct",
    "h600_terminal_per_sp",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_bootstrap_median_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    block_length: int = 1,
) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    if block_length == 1:
        indices = rng.integers(len(values), size=(samples, len(values)))
    else:
        block_count = math.ceil(len(values) / block_length)
        starts = rng.integers(len(values), size=(samples, block_count))
        offsets = np.arange(block_length, dtype=np.int64)
        indices = (starts[..., None] + offsets) % len(values)
        indices = indices.reshape(samples, -1)[:, : len(values)]
    estimates = np.median(values[indices], axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _paired_differences(
    events: list[dict[str, Any]], key: str, direction: str | None = None
) -> tuple[list[str], np.ndarray]:
    daily: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"A": [], "B": []}
    )
    for event in events:
        if not event["primary_dual_steady"]:
            continue
        if direction is not None and event["direction"] != direction:
            continue
        value = event.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        daily[event["utc_day"]][event["side"]].append(float(value))
    common = sorted(
        day for day, sides in daily.items() if sides["A"] and sides["B"]
    )
    differences = np.asarray(
        [
            np.median(daily[day]["B"]) - np.median(daily[day]["A"])
            for day in common
        ],
        dtype=float,
    )
    return common, differences


def run_audit(results_root: Path) -> dict[str, Any]:
    summary_path = results_root / "summary_validation.json"
    manifest_path = results_root / "run_manifest.json"
    events_path = results_root / "event_manifest_validation.jsonl"
    metrics_path = results_root / "event_metrics_validation.csv"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    config_path = ROOT / manifest["config_path"]
    config = _read_json(config_path)
    events = _load_events(events_path)
    metric_keys = list(summary["paired_primary_day_contrasts"])
    samples = int(config["statistics"]["bootstrap_samples"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    block_lengths = [
        int(value) for value in config["statistics"]["diagnostic_block_lengths_days"]
    ]

    contract_checks = {
        "summary_manifest_exact": summary["manifest"] == manifest,
        "config_sha256_exact": _sha256(config_path) == manifest["config_sha256"],
        "validation_only": manifest["split"] == "validation",
        "test_not_accessed": summary["test_accessed"] is False
        and manifest["test_accessed"] is False,
        "training_not_executed": summary["training_executed"] is False
        and manifest["training_executed"] is False,
        "unique_event_ids": len({event["event_id"] for event in events}) == len(events),
        "all_events_validation": all(event["split"] == "validation" for event in events),
        "sp_dose_bounds": all(
            1.0 <= float(event["abs_delta_sp_c"]) <= 3.0 for event in events
        ),
        "primary_layer_flags_exact": all(
            bool(event["primary_dual_steady"])
            == bool(event["stable_60s"] and event["stable_180s"])
            for event in events
        ),
    }
    for side in ("A", "B"):
        times = sorted(
            int(event["timestamp_ns"]) for event in events if event["side"] == side
        )
        contract_checks[f"{side}_same_side_gap_ge_600s"] = bool(
            len(times) < 2 or np.min(np.diff(times)) >= 600_000_000_000
        )

    max_error = 0.0
    recomputed: dict[str, Any] = {}
    for index, key in enumerate(metric_keys):
        days, differences = _paired_differences(events, key)
        point = float(np.median(differences)) if len(differences) else None
        ci = independent_bootstrap_median_ci(
            differences, samples=samples, seed=base_seed + index * 100
        )
        blocks = {
            str(block): independent_bootstrap_median_ci(
                differences,
                samples=samples,
                seed=base_seed + index * 100 + block,
                block_length=block,
            )
            for block in block_lengths
            if block <= len(differences)
        }
        stored = summary["paired_primary_day_contrasts"][key]
        candidates: list[float] = []
        if point is not None and stored["median_difference"] is not None:
            candidates.append(abs(point - float(stored["median_difference"])))
        if ci is not None:
            candidates.extend(abs(a - b) for a, b in zip(ci, stored["ci95"]))
        for block, interval in blocks.items():
            candidates.extend(
                abs(a - b)
                for a, b in zip(
                    interval, stored["diagnostic_circular_block_ci95"][block]
                )
            )
        if candidates:
            max_error = max(max_error, max(candidates))
        recomputed[key] = {
            "paired_utc_days": days,
            "median_B_minus_A": point,
            "ci95": ci,
            "diagnostic_circular_block_ci95": blocks,
        }

    counts: dict[str, Any] = {}
    for side in ("A", "B"):
        side_events = [event for event in events if event["side"] == side]
        primary = [event for event in side_events if event["primary_dual_steady"]]
        counts[side] = {
            "all_held_operating_events": len(side_events),
            "primary_events": len(primary),
            "primary_utc_days": len({event["utc_day"] for event in primary}),
            "primary_sp_up": sum(event["direction"] == "sp_up" for event in primary),
            "primary_sp_down": sum(event["direction"] == "sp_down" for event in primary),
            "primary_other_loop_quiet": sum(event["other_loop_quiet"] for event in primary),
            "clean_chain_events": sum(event["clean_chain"] for event in side_events),
            "strict_600s_clean_chain_events": sum(
                event["strict_600s_clean_chain"] is True for event in side_events
            ),
        }
        stored = summary["layers"]["primary_dual_steady"][side]
        contract_checks[f"{side}_headline_counts_exact"] = bool(
            counts[side]["primary_events"] == stored["event_count"]
            and counts[side]["primary_utc_days"] == stored["utc_day_count"]
        )

    field_intervals = {key: recomputed[key]["ci95"] for key in PHYSICAL_KEYS}
    direction_stratified: dict[str, Any] = {}
    for direction_index, direction in enumerate(("sp_up", "sp_down")):
        direction_stratified[direction] = {
            "physical_valve_direction": (
                "expected_valve_closing" if direction == "sp_up" else "expected_valve_opening"
            ),
            "metrics": {},
        }
        for index, key in enumerate(metric_keys):
            days, differences = _paired_differences(events, key, direction)
            direction_stratified[direction]["metrics"][key] = {
                "paired_utc_day_count": len(days),
                "paired_utc_days": days,
                "median_B_minus_A": (
                    float(np.median(differences)) if len(differences) else None
                ),
                "ci95": independent_bootstrap_median_ci(
                    differences,
                    samples=samples,
                    seed=base_seed + 50_000 + direction_index * 10_000 + index * 100,
                ),
                "diagnostic_only": True,
            }
    no_field_lower_positive = all(
        interval is None or interval[0] <= 0 for interval in field_intervals.values()
    )
    model_ratio = float(
        summary["checkpoint_reference"]["B_to_A_abs_h600_effect_ratio_median"]
    )
    expected_label = (
        "MODEL_A_RESPONSE_ATTENUATION_EXCEEDS_FIELD_EVIDENCE"
        if model_ratio
        >= float(config["checkpoint_reference"]["minimum_large_model_side_ratio"])
        and no_field_lower_positive
        else "INCONCLUSIVE_ASYMMETRY_DIAGNOSIS"
    )
    contract_checks["supervisor_label_recomputed"] = (
        summary["supervisor_diagnostic"]["label"] == expected_label
    )
    passes = bool(all(contract_checks.values()) and max_error <= 1e-12)
    return {
        "protocol_version": config["protocol_version"],
        "artifact_sha256": {
            "run_manifest": _sha256(manifest_path),
            "event_manifest_validation": _sha256(events_path),
            "event_metrics_validation": _sha256(metrics_path),
            "summary_validation": _sha256(summary_path),
        },
        "contract_checks": contract_checks,
        "event_counts": counts,
        "paired_primary_day_recomputation": recomputed,
        "max_numeric_recomputation_error": max_error,
        "checkpoint_B_to_A_abs_h600_effect_ratio_median": model_ratio,
        "field_physical_contrast_ci95": field_intervals,
        "direction_stratified_paired_day_diagnostic": direction_stratified,
        "supervisor_label": expected_label,
        "passes": passes,
        "independent_unit": "UTC_day",
        "test_accessed": False,
        "claim_boundary": (
            "Artifact and day-level replay only; no equivalence, causal, open-loop, "
            "MS3 revision, or MS4 release claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else results_root / DEFAULT_OUTPUT
    )
    audit = run_audit(results_root)
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passes": audit["passes"],
                "max_numeric_recomputation_error": audit[
                    "max_numeric_recomputation_error"
                ],
                "supervisor_label": audit["supervisor_label"],
                "test_accessed": audit["test_accessed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not audit["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
