#!/usr/bin/env python3
"""Q32-S: paired shared-disturbance incremental closed-loop probe.

This is an inference-only solution test.  It deliberately emits raw diagnostics
without selecting a winner or writing a scientific verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


q33 = import_file(ROOT / "33_qnav_residual_feedback_probe.py", "q34_q33")
q32 = q33.q32
DEVICE = q32.DEVICE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(revision: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT
    )


def sha256_git(revision: str, path: str) -> str:
    return hashlib.sha256(git_bytes(revision, path)).hexdigest()


def json_dump(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["status"] != "ready_for_linux":
        raise ValueError("Q32-S is not ready_for_linux")
    if config["probe"]["states"] != ["wet", "dry"]:
        raise ValueError("Q32-S state order changed")
    if config["probe"]["modes"] != ["physical", "live", "shared"]:
        raise ValueError("Q32-S modes changed")
    if set(config["folds"]) != {"F0", "F1"}:
        raise ValueError("Q32-S folds changed")
    if float(config["probe"]["setpoint_delta_c"]) != 0.5:
        raise ValueError("Q32-S intervention changed")
    return config


def verify_parent(config: dict[str, Any]):
    parent = config["parent"]
    revision = parent["results_commit"]
    for key in ("manifest", "summary"):
        path = parent[f"{key}_path"]
        expected = parent[f"{key}_sha256_git"]
        if sha256_git(revision, path) != expected:
            raise RuntimeError(f"Q32-R parent {key} Git content changed")
        changed = subprocess.run(
            ["git", "diff", "--quiet", revision, "--", path], cwd=ROOT
        )
        if changed.returncode != 0:
            raise RuntimeError(f"Q32-R parent {key} differs from {revision}")
    for fold, item in parent["checkpoints"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Q32 h_now checkpoint changed: {fold}")


def load_development_data(csv_path: Path, config: dict[str, Any]):
    use_columns = list(dict.fromkeys(q32.E0_COLS + q32.OUTPUTS + [q32.POWER_COLUMN]))
    start = int(config["data"]["window_start"])
    rows = int(config["data"]["development_rows"])
    frame = pd.read_csv(
        csv_path,
        usecols=use_columns,
        dtype=np.float32,
        skiprows=range(1, start + 1),
        nrows=rows,
    ).ffill().bfill().reset_index(drop=True)
    if len(frame) != rows:
        raise ValueError(f"expected {rows} development rows, got {len(frame)}")
    exo, targets = q32.prepare_arrays(frame)
    power = frame[q32.POWER_COLUMN].to_numpy(np.float32)
    digest = hashlib.sha256(
        exo.tobytes() + targets.tobytes() + power.tobytes()
    ).hexdigest()
    return exo, targets, power, digest


def warm_state(model, residual, row, observed, steps: int, enabled: bool):
    constant = torch.tensor(row, device=DEVICE)[None, None, :]
    state = q32.initialize(model, row, observed)
    candidate = {"injection": "h_only", "residual_reads_w": False}
    with torch.no_grad():
        for _ in range(steps):
            output = q32.integrate(
                model,
                residual,
                constant,
                *state,
                1,
                candidate["injection"],
                candidate["residual_reads_w"],
                residual_enabled=enabled,
            )
            state = output[1:]
    return q32.clone_state(state)


def run_deviation_loop(
    model,
    residual,
    row: np.ndarray,
    observed: np.ndarray,
    kw: float,
    power_mw: float,
    config: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    probe = config["probe"]
    sample_seconds = int(config["data"]["sample_seconds"])
    steps = int(probe["control_steps"])
    rate = float(probe["valve_rate_fraction_per_step"])
    target = float(probe["setpoint_delta_c"])
    enabled = mode != "physical"
    initial = warm_state(
        model, residual, row, observed, int(probe["warm_steps"]), enabled
    )
    base_state = q32.clone_state(initial)
    loop_state = q32.clone_state(initial)
    constant = torch.tensor(row, device=DEVICE)[None, None, :]
    baseline = np.empty(steps, dtype=np.float64)
    controlled = np.empty(steps, dtype=np.float64)
    valve_history = np.empty(steps, dtype=np.float64)
    valve0 = float(row[6])
    w0 = float(row[8])
    valve = valve0
    integral = 0.0
    candidate = {"injection": "h_only", "residual_reads_w": False}

    if mode == "shared":
        base_residual = q33.RecordResidual(residual)
        loop_residual = q33.ReplayResidual(base_residual.outputs)
    else:
        base_residual = residual
        loop_residual = residual

    with torch.no_grad():
        for index in range(steps):
            base_output = q32.integrate(
                model,
                base_residual,
                constant,
                *base_state,
                1,
                candidate["injection"],
                candidate["residual_reads_w"],
                residual_enabled=enabled,
            )
            baseline[index] = float(base_output[0][0, 0, 4])
            base_state = base_output[1:]

            scenario = constant.clone()
            scenario[0, 0, 6] = valve
            scenario[0, 0, 8] = max(0.0, w0 * (1.0 + kw * (valve - valve0)))
            loop_output = q32.integrate(
                model,
                loop_residual,
                scenario,
                *loop_state,
                1,
                candidate["injection"],
                candidate["residual_reads_w"],
                residual_enabled=enabled,
            )
            controlled[index] = float(loop_output[0][0, 0, 4])
            loop_state = loop_output[1:]

            delta = controlled[index] - baseline[index]
            error = delta - target
            kp, ti = q32.pi_parameters(-error, float(power_mw))
            integral += error * sample_seconds
            command = float(
                np.clip(valve0 + kp * error + kp / ti * integral, 0.0, 1.0)
            )
            valve = float(
                np.clip(valve + np.clip(command - valve, -rate, rate), 0.0, 1.0)
            )
            valve_history[index] = valve

    if mode == "shared":
        loop_residual.assert_consumed()
    delta_trace = controlled - baseline
    tail = min(60, steps)
    variation_tail = min(120, steps)
    achieved = float(np.mean(delta_trace[-tail:]))
    valve_diff = np.diff(valve_history)
    nonzero = valve_diff[np.abs(valve_diff) > 1e-8]
    reversals = int(np.sum(nonzero[1:] * nonzero[:-1] < 0)) if len(nonzero) > 1 else 0
    return {
        "setpoint_delta_c": target,
        "achieved_delta_c": achieved,
        "tracking_error_c": float(abs(achieved - target)),
        "tail_std_delta_c": float(np.std(delta_trace[-variation_tail:])),
        "baseline_tail_drift_c": float(np.mean(baseline[-tail:]) - baseline[0]),
        "valve_initial": valve0,
        "valve_min": float(np.min(valve_history)),
        "valve_max": float(np.max(valve_history)),
        "valve_saturation_fraction": float(
            np.mean((valve_history <= 1e-6) | (valve_history >= 1.0 - 1e-6))
        ),
        "valve_reversals": reversals,
        "max_valve_move_per_step": float(np.max(np.abs(valve_diff))) if len(valve_diff) else 0.0,
        "controller_power_mw": float(power_mw),
        "residual_calls_shared": int(loop_residual.cursor) if mode == "shared" else None,
    }


def summarize(points: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["probe"]["diagnostic_thresholds"]
    result: dict[str, Any] = {}
    for mode in config["probe"]["modes"]:
        metrics = [point["modes"][mode] for point in points]
        tracking = np.array([item["tracking_error_c"] for item in metrics])
        tail_std = np.array([item["tail_std_delta_c"] for item in metrics])
        saturation = np.array([item["valve_saturation_fraction"] for item in metrics])
        result[mode] = {
            "n": len(metrics),
            "median_tracking_error_c": float(np.median(tracking)),
            "tracking_error_q90_c": float(np.quantile(tracking, 0.9)),
            "median_tail_std_delta_c": float(np.median(tail_std)),
            "median_valve_saturation_fraction": float(np.median(saturation)),
            "median_valve_reversals": float(
                np.median([item["valve_reversals"] for item in metrics])
            ),
            "tracking_within_threshold_fraction": float(
                np.mean(tracking <= float(thresholds["tracking_error_c"]))
            ),
            "stable_within_threshold_fraction": float(
                np.mean(tail_std <= float(thresholds["tail_std_delta_c"]))
            ),
            "not_saturated_fraction": float(
                np.mean(saturation < float(thresholds["valve_saturation_fraction"]))
            ),
        }
    return result


def execute(args, config: dict[str, Any], config_path: Path):
    verify_parent(config)
    output = Path(args.output).resolve()
    summary_path = output / "summary_development.json"
    if summary_path.exists():
        raise RuntimeError(f"Q32-S summary already exists: {summary_path}")
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    exo, targets, power, data_hash = load_development_data(csv_path, config)
    model = q32.load_evap_model(ROOT / "out" / "model_e0_evap_seed0.pt")
    folds: dict[str, Any] = {}

    for fold_id, fold in config["folds"].items():
        residual = q33.load_residual(
            ROOT / config["parent"]["checkpoints"][fold_id]["path"]
        )
        coupling = q32.estimate_w_coupling(exo, fold["training"])
        state_results: dict[str, Any] = {}
        for state in config["probe"]["states"]:
            kw = coupling[state]
            if kw is None:
                raise RuntimeError(f"{state} W coupling unavailable: {fold_id}")
            lo, hi = fold["evaluation"]
            indices = q33.select_quantile_points(
                exo,
                int(lo),
                int(hi),
                int(config["probe"]["points_per_state_per_fold"]),
                state=state,
            )
            expected = int(config["probe"]["points_per_state_per_fold"])
            if len(indices) != expected:
                raise RuntimeError(
                    f"expected {expected} {state} points in {fold_id}, got {len(indices)}"
                )
            points = []
            for index in indices:
                print(f"[Q32-S] {fold_id} state={state} row={index}", flush=True)
                modes = {
                    mode: run_deviation_loop(
                        model,
                        residual,
                        exo[index],
                        targets[index],
                        float(kw),
                        float(power[index]),
                        config,
                        mode,
                    )
                    for mode in config["probe"]["modes"]
                }
                points.append(
                    {
                        "row": index,
                        "state": state,
                        "pressure_mpa": float(exo[index, 2]),
                        "valve2_fraction": float(exo[index, 6]),
                        "spray_w": float(exo[index, 8]),
                        "power_mw": float(power[index]),
                        "modes": modes,
                    }
                )
            state_results[state] = {
                "training_only_w_coupling": float(kw),
                "point_count": len(points),
                "points": points,
                "aggregate": summarize(points, config),
            }
        folds[fold_id] = {"states": state_results}

    payload = {
        "experiment": config["experiment"],
        "status": "results_returned",
        "diagnostic_thresholds": config["probe"]["diagnostic_thresholds"],
        "folds": folds,
        "scientific_verdict": None,
    }
    json_dump(summary_path, payload)
    manifest = {
        "experiment": config["experiment"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__)),
        "parent_results_commit": config["parent"]["results_commit"],
        "parent_manifest_sha256_git": config["parent"]["manifest_sha256_git"],
        "parent_summary_sha256_git": config["parent"]["summary_sha256_git"],
        "data_path": str(csv_path),
        "development_arrays_sha256": data_hash,
        "development_rows_loaded": int(config["data"]["development_rows"]),
        "reserved_rows_loaded": False,
        "training_performed": False,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    }
    json_dump(output / "manifest.json", manifest)
    point_count = sum(
        state["point_count"]
        for fold in folds.values()
        for state in fold["states"].values()
    )
    print(json.dumps({"summary": str(summary_path), "points": point_count}, indent=2))


def dry_run(config: dict[str, Any], config_path: Path, args):
    verify_parent(config)
    payload = {
        "experiment": config["experiment"],
        "status": config["status"],
        "config_sha256": sha256_file(config_path),
        "folds": list(config["folds"]),
        "states": config["probe"]["states"],
        "modes": config["probe"]["modes"],
        "points": (
            len(config["folds"])
            * len(config["probe"]["states"])
            * int(config["probe"]["points_per_state_per_fold"])
        ),
        "training": False,
        "scientific_verdict": None,
        "linux_command": config["linux_boundary"]["command"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "qnav_shared_disturbance_loop.json"),
    )
    parser.add_argument("--csv", default=q32.t02.CSV)
    parser.add_argument(
        "--output", default=str(ROOT / "out" / "qnav_shared_disturbance_loop")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.dry_run:
        dry_run(config, config_path, args)
    else:
        execute(args, config, config_path)


if __name__ == "__main__":
    main()
