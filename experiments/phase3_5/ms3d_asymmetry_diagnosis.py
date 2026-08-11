#!/usr/bin/env python3
"""Run the frozen local validation-only Phase 3.5-MS3-D diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.ms3_real_adaptation import _validate_cache, load_matrix  # noqa: E402
from src.phase35.data import load_cache  # noqa: E402
from src.phase35.ms3d import (  # noqa: E402
    detect_ms3d_events,
    diagnosis_label,
    paired_day_contrasts,
    response_metric_keys,
    summarize_side_layer,
    validate_ms3d_config,
)
from src.phase35.multistep.training import _json_dump, _sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/phase3_5/ms3d_asymmetry_diagnosis.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3d_asymmetry_diagnosis"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _git_status_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    return [line[3:].strip() for line in output.splitlines() if line.strip()]


def _three_pole_unit_step_factor(taus_seconds: list[float]) -> float:
    states = [0.0] * len(taus_seconds)
    for _ in range(60):
        stage = 1.0
        for index, tau in enumerate(taus_seconds):
            decay = math.exp(-10.0 / tau)
            states[index] = decay * states[index] + (1.0 - decay) * stage
            stage = states[index]
    return states[-1]


def _checkpoint_reference(reference: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    candidate = config["checkpoint_reference"]["candidate_id"]
    step = float(config["checkpoint_reference"]["standardized_raw_valve_step_pct"])
    by_side: dict[str, list[dict[str, float]]] = {"A": [], "B": []}
    for run in reference["runs"]:
        if run["candidate_id"] != candidate:
            continue
        median_effect = float(
            run["standardized_step_diagnostics"]["plus5_effect_c"]["H60"]["median"]
        )
        diagnostics = run["standardized_step_diagnostics"]
        taus = [
            float(diagnostics["tau_seconds"][f"pole_{index}"]["median"])
            for index in (1, 2, 3)
        ]
        by_side[run["side"]].append(
            {
                "seed": int(run["seed"]),
                "h600_effect_c": median_effect,
                "raw_valve_gain_c_per_pct": median_effect / step,
                "effective_opening_delta_pct": float(
                    diagnostics["effective_opening_delta_plus5_pct"]["median"]
                ),
                "scheduled_gain_c_per_effective_pct": float(
                    diagnostics["scheduled_gain_c_per_effective_pct"]["median"]
                ),
                "tau_seconds": taus,
                "median_tau_unit_step_factor_h600": _three_pole_unit_step_factor(
                    taus
                ),
            }
        )
    if any(len(values) != 3 for values in by_side.values()):
        raise RuntimeError("MS3-D checkpoint reference does not contain 3 joint seeds per side")
    ratios = []
    decomposition = []
    for seed in sorted(item["seed"] for item in by_side["A"]):
        a = next(item for item in by_side["A"] if item["seed"] == seed)
        b = next(item for item in by_side["B"] if item["seed"] == seed)
        effect_ratio = abs(b["h600_effect_c"]) / abs(a["h600_effect_c"])
        ratios.append(effect_ratio)
        opening_ratio = (
            b["effective_opening_delta_pct"] / a["effective_opening_delta_pct"]
        )
        gain_ratio = abs(b["scheduled_gain_c_per_effective_pct"]) / abs(
            a["scheduled_gain_c_per_effective_pct"]
        )
        dynamics_ratio = (
            b["median_tau_unit_step_factor_h600"]
            / a["median_tau_unit_step_factor_h600"]
        )
        decomposition.append(
            {
                "seed": seed,
                "observed_abs_effect_ratio": effect_ratio,
                "effective_opening_ratio": opening_ratio,
                "scheduled_gain_ratio": gain_ratio,
                "median_tau_dynamics_ratio": dynamics_ratio,
                "factorized_ratio_approximation": (
                    opening_ratio * gain_ratio * dynamics_ratio
                ),
                "approximation_boundary": (
                    "Product of marginal medians; context-level covariance is not retained."
                ),
            }
        )
    return {
        "standardized_raw_valve_step_pct": step,
        "by_side": by_side,
        "matched_seed_B_to_A_abs_h600_effect_ratios": ratios,
        "B_to_A_abs_h600_effect_ratio_median": float(np.median(ratios)),
        "matched_seed_factor_decomposition": decomposition,
        "claim_boundary": "checkpoint_constant-step_diagnostic_not_field_intervention",
    }


def _flatten_event(event: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in event.items():
        if isinstance(value, dict):
            for child, child_value in value.items():
                row[f"{key}_{child}"] = child_value
        else:
            row[key] = value
    return row


def _write_event_artifacts(output: Path, events: list[dict[str, Any]]) -> None:
    jsonl = output / "event_manifest_validation.jsonl"
    with jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    rows = [_flatten_event(event) for event in events]
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "event_metrics_validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    config_path: Path,
    cache_paths: dict[str, Path],
    output: Path,
    require_clean: bool,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_ms3d_config(config)
    if require_clean and _git_status_paths():
        raise RuntimeError("MS3-D requires a clean committed worktree before execution")
    parent_path = ROOT / config["data_contract"]["parent_matrix"]
    if _sha256(parent_path) != config["data_contract"]["parent_matrix_sha256"]:
        raise RuntimeError("MS3-D parent MS3 matrix hash changed")
    parent = load_matrix(parent_path)
    reference_path = ROOT / config["checkpoint_reference"]["path"]
    if _sha256(reference_path) != config["checkpoint_reference"]["sha256"]:
        raise RuntimeError("MS3-D checkpoint replay reference hash changed")
    caches = {side: load_cache(path) for side, path in cache_paths.items()}
    parent_sha = _sha256(parent_path)
    for side in ("A", "B"):
        _validate_cache(caches[side], side, parent, parent_sha)
        if caches[side].metadata["source"]["sha256"] != config["data_contract"]["source_sha256"]:
            raise RuntimeError(f"MS3-D {side} cache source hash changed")

    events_by_side: dict[str, list[dict[str, Any]]] = {}
    funnels: dict[str, dict[str, int]] = {}
    for side in ("A", "B"):
        events_by_side[side], funnels[side] = detect_ms3d_events(
            caches[side], caches["B" if side == "A" else "A"], side, config
        )
    events = sorted(
        events_by_side["A"] + events_by_side["B"],
        key=lambda event: (event["timestamp_ns"], event["side"]),
    )
    metric_keys = response_metric_keys(config["response_contract"]["horizons_seconds"])
    primary = {
        side: [event for event in events_by_side[side] if event["primary_dual_steady"]]
        for side in ("A", "B")
    }
    clean = {
        side: [event for event in events_by_side[side] if event["clean_chain"]]
        for side in ("A", "B")
    }
    dynamic = {
        side: [event for event in events_by_side[side] if not event["primary_dual_steady"]]
        for side in ("A", "B")
    }
    paired = paired_day_contrasts(primary, metric_keys, config["statistics"])
    checkpoint = _checkpoint_reference(_read_json(reference_path), config)
    decision = diagnosis_label(
        paired,
        checkpoint,
        float(config["checkpoint_reference"]["minimum_large_model_side_ratio"]),
    )
    bounds = caches["A"].split_bounds()["validation"]
    manifest = {
        "protocol_version": config["protocol_version"],
        "execution_git_sha": _git_sha(),
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "parent_matrix_sha256": parent_sha,
        "checkpoint_reference_sha256": _sha256(reference_path),
        "source_sha256": config["data_contract"]["source_sha256"],
        "cache_ids": {side: cache_paths[side].name for side in ("A", "B")},
        "cache_location_scope": "external_nonversioned_local_cache",
        "split": "validation",
        "split_bounds": list(bounds),
        "validation_time_start": str(np.datetime64(int(caches["A"].timestamps_ns[bounds[0]]), "ns")),
        "validation_time_end": str(np.datetime64(int(caches["A"].timestamps_ns[bounds[1] - 1]), "ns")),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "test_accessed": False,
        "training_executed": False,
    }
    summary = {
        "protocol_version": config["protocol_version"],
        "evidence_scope": config["evidence_scope"],
        "manifest": manifest,
        "event_funnels": funnels,
        "layers": {
            "primary_dual_steady": {
                side: summarize_side_layer(primary[side], metric_keys)
                for side in ("A", "B")
            },
            "clean_chain": {
                side: summarize_side_layer(clean[side], metric_keys)
                for side in ("A", "B")
            },
            "dynamic_secondary": {
                side: summarize_side_layer(dynamic[side], metric_keys)
                for side in ("A", "B")
            },
        },
        "paired_primary_day_contrasts": paired,
        "checkpoint_reference": checkpoint,
        "supervisor_diagnostic": decision,
        "test_accessed": False,
        "training_executed": False,
        "claim_boundary": (
            "Observational validation-only cascade diagnosis; no equivalence, do(valve), "
            "open-loop gain, independent test, MS3 revision, or MS4 release claim."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_event_artifacts(output, events)
    _json_dump(output / "run_manifest.json", manifest)
    _json_dump(output / "summary_validation.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-a", required=True)
    parser.add_argument("--cache-b", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        config_path=Path(args.config).resolve(),
        cache_paths={"A": Path(args.cache_a).resolve(), "B": Path(args.cache_b).resolve()},
        output=Path(args.output_dir).resolve(),
        require_clean=not args.allow_dirty,
    )
    print(
        json.dumps(
            {
                "protocol_version": summary["protocol_version"],
                "primary_events": {
                    side: summary["layers"]["primary_dual_steady"][side]["event_count"]
                    for side in ("A", "B")
                },
                "supervisor_diagnostic": summary["supervisor_diagnostic"],
                "test_accessed": summary["test_accessed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
