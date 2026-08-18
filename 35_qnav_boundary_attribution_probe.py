#!/usr/bin/env python3
"""Q32-T: inference-only object/controller/initialization attribution probe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


q34 = import_file(ROOT / "34_qnav_shared_disturbance_loop.py", "q35_q34")
q33 = q34.q33
q32 = q34.q32
DEVICE = q32.DEVICE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["status"] != "ready_for_linux":
        raise ValueError("Q32-T is not ready_for_linux")
    if config["object_panel"]["residual_modes"] != ["physical", "live", "shared"]:
        raise ValueError("Q32-T object residual modes changed")
    if config["controller_panel"]["residual_modes"] != ["physical", "live", "shared"]:
        raise ValueError("Q32-T controller residual modes changed")
    expected_controllers = [
        "aw_only",
        "deadband_aw",
        "lpf_aw",
        "deadband_lpf_aw",
    ]
    if config["controller_panel"]["controllers"] != expected_controllers:
        raise ValueError("Q32-T controller set changed")
    if config["initialization_panel"]["residual_modes"] != ["physical", "live"]:
        raise ValueError("Q32-T initialization modes changed")
    if set(config["folds"]) != {"F0", "F1"}:
        raise ValueError("Q32-T folds changed")
    return config


def load_parent_results(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent"]
    revision = parent["results_commit"]
    for key in ("manifest", "summary"):
        path = parent[f"{key}_path"]
        expected = parent[f"{key}_sha256_git"]
        content = q34.git_bytes(revision, path)
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"Q32-S parent {key} Git content changed")
        if subprocess.run(
            ["git", "diff", "--quiet", revision, "--", path], cwd=ROOT
        ).returncode != 0:
            raise RuntimeError(f"Q32-S parent {key} differs from {revision}")
    for fold, item in parent["checkpoints"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Q32 h_now checkpoint changed: {fold}")
    manifest = json.loads(q34.git_bytes(revision, parent["manifest_path"]))
    if manifest["development_arrays_sha256"] != parent["development_arrays_sha256"]:
        raise RuntimeError("Q32-S development data identity changed")
    summary = json.loads(q34.git_bytes(revision, parent["summary_path"]))
    if summary["status"] != "results_returned" or summary["scientific_verdict"] is not None:
        raise RuntimeError("Q32-S parent status changed")
    count = sum(
        len(state["points"])
        for fold in summary["folds"].values()
        for state in fold["states"].values()
    )
    if count != 16:
        raise RuntimeError(f"Q32-T expected 16 frozen parent points, got {count}")
    return summary


def residual_enabled(mode: str) -> bool:
    return mode != "physical"


def object_response(
    model,
    residual,
    row: np.ndarray,
    observed: np.ndarray,
    kw: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    panel = config["object_panel"]
    candidate = {"injection": "h_only", "residual_reads_w": False}
    result: dict[str, Any] = {}
    for valve_delta in panel["valve_step_fractions"]:
        delta_key = f"{float(valve_delta):+.3f}"
        result[delta_key] = {}
        for mode in panel["residual_modes"]:
            run_mode = "replay" if mode == "shared" else mode
            delta = q33.run_pair(
                model,
                residual,
                candidate,
                row,
                observed,
                int(panel["steps"]),
                float(valve_delta),
                1.0 + float(kw) * float(valve_delta),
                run_mode,
            )
            metrics = q32.response_metrics(
                delta, int(config["data"]["sample_seconds"])
            )
            steady = float(metrics["steady_main_c"])
            metrics.update(
                {
                    "valve_step_fraction": float(valve_delta),
                    "w_multiplier": 1.0 + float(kw) * float(valve_delta),
                    "direction_correct": bool(steady * float(valve_delta) < 0.0),
                    "local_gain_c_per_valve_fraction": steady / float(valve_delta),
                }
            )
            result[delta_key][mode] = metrics
    return result


def controller_flags(controller: str) -> tuple[bool, bool]:
    return "deadband" in controller, "lpf" in controller


def controller_command(
    measurement: float,
    target: float,
    filtered: float,
    integral: float,
    valve0: float,
    power_mw: float,
    controller: str,
    config: dict[str, Any],
) -> tuple[float, float, float, bool, bool]:
    panel = config["controller_panel"]
    sample_seconds = float(config["data"]["sample_seconds"])
    use_deadband, use_lpf = controller_flags(controller)
    if use_lpf:
        alpha = sample_seconds / (float(panel["lpf_tau_seconds"]) + sample_seconds)
        filtered = filtered + alpha * (measurement - filtered)
    else:
        filtered = measurement
    error = filtered - target
    in_deadband = use_deadband and abs(error) <= float(panel["deadband_c"])
    control_error = 0.0 if in_deadband else error
    proposed_integral = integral + control_error * sample_seconds
    kp, ti = q32.pi_parameters(-control_error, float(power_mw))
    raw = valve0 + kp * control_error + kp / ti * proposed_integral
    hold_integral = bool(
        (raw < 0.0 and control_error < 0.0)
        or (raw > 1.0 and control_error > 0.0)
    )
    if hold_integral:
        proposed_integral = integral
        raw = valve0 + kp * control_error + kp / ti * proposed_integral
    command = float(np.clip(raw, 0.0, 1.0))
    return command, float(proposed_integral), float(filtered), hold_integral, in_deadband


def run_controller(
    model,
    residual,
    row: np.ndarray,
    kw: float,
    power_mw: float,
    initial_state,
    config: dict[str, Any],
    residual_mode: str,
    controller: str,
) -> dict[str, Any]:
    panel = config["controller_panel"]
    steps = int(panel["control_steps"])
    sample_seconds = int(config["data"]["sample_seconds"])
    target = float(panel["setpoint_delta_c"])
    rate = float(panel["valve_rate_fraction_per_step"])
    enabled = residual_enabled(residual_mode)
    base_state = q32.clone_state(initial_state)
    loop_state = q32.clone_state(initial_state)
    constant = torch.tensor(row, device=DEVICE)[None, None, :]
    baseline = np.empty(steps, dtype=np.float64)
    controlled = np.empty(steps, dtype=np.float64)
    valve_history = np.empty(steps, dtype=np.float64)
    valve0 = float(row[6])
    w0 = float(row[8])
    valve = valve0
    integral = 0.0
    filtered = 0.0
    hold_count = 0
    deadband_count = 0
    candidate = {"injection": "h_only", "residual_reads_w": False}
    if residual_mode == "shared":
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
            command, integral, filtered, held, in_deadband = controller_command(
                delta,
                target,
                filtered,
                integral,
                valve0,
                power_mw,
                controller,
                config,
            )
            hold_count += int(held)
            deadband_count += int(in_deadband)
            valve = float(
                np.clip(valve + np.clip(command - valve, -rate, rate), 0.0, 1.0)
            )
            valve_history[index] = valve

    if residual_mode == "shared":
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
        "valve_min": float(np.min(valve_history)),
        "valve_max": float(np.max(valve_history)),
        "valve_saturation_fraction": float(
            np.mean((valve_history <= 1e-6) | (valve_history >= 1.0 - 1e-6))
        ),
        "valve_reversals": reversals,
        "max_valve_move_per_step": float(np.max(np.abs(valve_diff))) if len(valve_diff) else 0.0,
        "antiwindup_hold_fraction": hold_count / steps,
        "deadband_fraction": deadband_count / steps,
        "final_filtered_delta_c": filtered,
        "residual_calls_shared": int(loop_residual.cursor) if residual_mode == "shared" else None,
    }


def initialization_diagnostics(
    model,
    residual,
    exo: np.ndarray,
    targets: np.ndarray,
    index: int,
    config: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    panel = config["initialization_panel"]
    enabled = residual_enabled(mode)
    candidate = {"injection": "h_only", "residual_reads_w": False}

    one = torch.tensor(exo[index], device=DEVICE)[None, None, :]
    if index + 1 < len(targets):
        state = q32.initialize(model, exo[index], targets[index])
        with torch.no_grad():
            output = q32.integrate(
                model,
                residual,
                one,
                *state,
                1,
                candidate["injection"],
                candidate["residual_reads_w"],
                residual_enabled=enabled,
            )[0]
        one_step_prediction = float(output[0, -1, 4])
        one_step_error = one_step_prediction - float(targets[index + 1, 4])
    else:
        one_step_prediction = None
        one_step_error = None

    history_steps = int(panel["history_steps"])
    start = index - history_steps
    if start < 0:
        raise RuntimeError(f"insufficient logged history for row {index}")
    state = q32.initialize(model, exo[start], targets[start])
    history = torch.tensor(exo[start:index], device=DEVICE)[None, :, :]
    with torch.no_grad():
        output = q32.integrate(
            model,
            residual,
            history,
            *state,
            history_steps,
            candidate["injection"],
            candidate["residual_reads_w"],
            residual_enabled=enabled,
        )[0]
    history_prediction = float(output[0, -1, 4])
    history_error = history_prediction - float(targets[index, 4])

    horizons = [int(value) for value in panel["constant_warm_horizons"]]
    state = q32.initialize(model, exo[index], targets[index])
    constant = one
    equilibrium_offsets: dict[str, float] = {}
    current_output = None
    with torch.no_grad():
        for step in range(1, max(horizons) + 1):
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
            current_output = float(output[0][0, -1, 4])
            state = output[1:]
            if step in horizons:
                equilibrium_offsets[str(step)] = current_output - float(targets[index, 4])
    return {
        "one_step_prediction_c": one_step_prediction,
        "one_step_error_c": one_step_error,
        "history_steps": history_steps,
        "history_prediction_c": history_prediction,
        "history_error_c": history_error,
        "constant_warm_offsets_c": equilibrium_offsets,
    }


def aggregate_points(points: list[dict[str, Any]], config: dict[str, Any]):
    object_aggregate: dict[str, Any] = {}
    for delta in config["object_panel"]["valve_step_fractions"]:
        key = f"{float(delta):+.3f}"
        object_aggregate[key] = {}
        for mode in config["object_panel"]["residual_modes"]:
            values = [point["object"][key][mode] for point in points]
            taus = [item["tau63_seconds"] for item in values if item["tau63_seconds"] is not None]
            object_aggregate[key][mode] = {
                "n": len(values),
                "direction_correct_fraction": float(
                    np.mean([item["direction_correct"] for item in values])
                ),
                "median_steady_main_c": float(
                    np.median([item["steady_main_c"] for item in values])
                ),
                "median_local_gain_c_per_valve_fraction": float(
                    np.median([item["local_gain_c_per_valve_fraction"] for item in values])
                ),
                "median_tau63_seconds": float(np.median(taus)) if taus else None,
            }

    controller_aggregate: dict[str, Any] = {}
    for residual_mode in config["controller_panel"]["residual_modes"]:
        controller_aggregate[residual_mode] = {}
        for controller in config["controller_panel"]["controllers"]:
            values = [point["controllers"][residual_mode][controller] for point in points]
            controller_aggregate[residual_mode][controller] = {
                "n": len(values),
                "median_tracking_error_c": float(
                    np.median([item["tracking_error_c"] for item in values])
                ),
                "median_tail_std_delta_c": float(
                    np.median([item["tail_std_delta_c"] for item in values])
                ),
                "median_valve_saturation_fraction": float(
                    np.median([item["valve_saturation_fraction"] for item in values])
                ),
                "median_valve_reversals": float(
                    np.median([item["valve_reversals"] for item in values])
                ),
                "median_antiwindup_hold_fraction": float(
                    np.median([item["antiwindup_hold_fraction"] for item in values])
                ),
            }

    initialization_aggregate: dict[str, Any] = {}
    for mode in config["initialization_panel"]["residual_modes"]:
        values = [point["initialization"][mode] for point in points]
        one_step_errors = [
            abs(item["one_step_error_c"])
            for item in values
            if item["one_step_error_c"] is not None
        ]
        initialization_aggregate[mode] = {
            "n": len(values),
            "one_step_n": len(one_step_errors),
            "median_abs_one_step_error_c": float(
                np.median(one_step_errors)
            ) if one_step_errors else None,
            "median_abs_history_error_c": float(
                np.median([abs(item["history_error_c"]) for item in values])
            ),
            "median_abs_constant_warm_offsets_c": {
                str(horizon): float(
                    np.median(
                        [
                            abs(item["constant_warm_offsets_c"][str(horizon)])
                            for item in values
                        ]
                    )
                )
                for horizon in config["initialization_panel"]["constant_warm_horizons"]
            },
        }
    return {
        "object": object_aggregate,
        "controllers": controller_aggregate,
        "initialization": initialization_aggregate,
    }


def execute(args, config: dict[str, Any], config_path: Path):
    parent = load_parent_results(config)
    output = Path(args.output).resolve()
    summary_path = output / "summary_development.json"
    if summary_path.exists():
        raise RuntimeError(f"Q32-T summary already exists: {summary_path}")
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    exo, targets, power, data_hash = q34.load_development_data(csv_path, config)
    if data_hash != config["parent"]["development_arrays_sha256"]:
        raise RuntimeError("Q32-T development arrays differ from Q32-S")
    model = q32.load_evap_model(ROOT / "out" / "model_e0_evap_seed0.pt")
    folds: dict[str, Any] = {}

    for fold_id, fold in config["folds"].items():
        residual = q33.load_residual(
            ROOT / config["parent"]["checkpoints"][fold_id]["path"]
        )
        coupling = q32.estimate_w_coupling(exo, fold["training"])
        states: dict[str, Any] = {}
        for state_name, parent_state in parent["folds"][fold_id]["states"].items():
            kw = coupling[state_name]
            if kw is None:
                raise RuntimeError(f"{state_name} W coupling unavailable: {fold_id}")
            points = []
            for frozen_point in parent_state["points"]:
                index = int(frozen_point["row"])
                print(f"[Q32-T] {fold_id} state={state_name} row={index}", flush=True)
                if (float(exo[index, 2]) <= q32.P_CRIT) != (state_name == "wet"):
                    raise RuntimeError(f"parent point state changed: {fold_id}/{index}")
                object_result = object_response(
                    model, residual, exo[index], targets[index], float(kw), config
                )
                controllers: dict[str, Any] = {}
                for residual_mode in config["controller_panel"]["residual_modes"]:
                    initial = q34.warm_state(
                        model,
                        residual,
                        exo[index],
                        targets[index],
                        int(config["controller_panel"]["warm_steps"]),
                        residual_enabled(residual_mode),
                    )
                    controllers[residual_mode] = {
                        controller: run_controller(
                            model,
                            residual,
                            exo[index],
                            float(kw),
                            float(power[index]),
                            initial,
                            config,
                            residual_mode,
                            controller,
                        )
                        for controller in config["controller_panel"]["controllers"]
                    }
                initialization = {
                    mode: initialization_diagnostics(
                        model, residual, exo, targets, index, config, mode
                    )
                    for mode in config["initialization_panel"]["residual_modes"]
                }
                points.append(
                    {
                        "row": index,
                        "state": state_name,
                        "pressure_mpa": float(exo[index, 2]),
                        "valve2_fraction": float(exo[index, 6]),
                        "spray_w": float(exo[index, 8]),
                        "power_mw": float(power[index]),
                        "object": object_result,
                        "controllers": controllers,
                        "initialization": initialization,
                    }
                )
            states[state_name] = {
                "training_only_w_coupling": float(kw),
                "point_count": len(points),
                "parent_raw_controller_aggregate": parent_state["aggregate"],
                "points": points,
                "aggregate": aggregate_points(points, config),
            }
        folds[fold_id] = {"states": states}

    payload = {
        "experiment": config["experiment"],
        "status": "results_returned",
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
        "development_arrays_sha256": data_hash,
        "development_rows_loaded": int(config["data"]["development_rows"]),
        "reserved_rows_loaded": False,
        "training_performed": False,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    }
    json_dump(output / "manifest.json", manifest)
    print(json.dumps({"summary": str(summary_path), "points": 16}, indent=2))


def dry_run(config: dict[str, Any], config_path: Path):
    parent = load_parent_results(config)
    point_count = sum(
        len(state["points"])
        for fold in parent["folds"].values()
        for state in fold["states"].values()
    )
    payload = {
        "experiment": config["experiment"],
        "status": config["status"],
        "config_sha256": sha256_file(config_path),
        "points": point_count,
        "panels": ["object", "controller", "initialization"],
        "training": False,
        "scientific_verdict": None,
        "linux_command": config["linux_boundary"]["command"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "qnav_boundary_attribution_probe.json"),
    )
    parser.add_argument("--csv", default=q32.t02.CSV)
    parser.add_argument(
        "--output", default=str(ROOT / "out" / "qnav_boundary_attribution_probe")
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
        dry_run(config, config_path)
    else:
        execute(args, config, config_path)


if __name__ == "__main__":
    main()
