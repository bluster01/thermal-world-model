#!/usr/bin/env python3
"""Q32-R: inference-only probe for residual-mediated action feedback."""
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


q32 = import_file(ROOT / "32_qnav_first_principles.py", "q33_q32")
DEVICE = q32.DEVICE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_git_file(revision: str, path: str) -> str:
    """Hash canonical Git bytes so text EOL conversion cannot break preflight."""
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT
    )
    return hashlib.sha256(content).hexdigest()


def json_dump(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


class RecordResidual:
    def __init__(self, base):
        self.base = base
        self.features: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        output = self.base(features)
        self.features.append(features.detach().clone())
        self.outputs.append(output.detach().clone())
        return output


class ReplayResidual:
    def __init__(self, outputs: list[torch.Tensor]):
        self.outputs = outputs
        self.cursor = 0

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.cursor >= len(self.outputs):
            raise RuntimeError("residual replay exhausted")
        output = self.outputs[self.cursor].to(features)
        self.cursor += 1
        if output.shape[0] != features.shape[0]:
            raise RuntimeError("residual replay batch changed")
        return output

    def assert_consumed(self):
        if self.cursor != len(self.outputs):
            raise RuntimeError("residual replay was not fully consumed")


class ScaledResidual:
    def __init__(self, base, scale: float):
        self.base = base
        self.scale = scale

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * self.base(features)


class FreezeFeaturesResidual:
    def __init__(self, base, baseline: list[torch.Tensor], feature_slice: slice):
        self.base = base
        self.baseline = baseline
        self.feature_slice = feature_slice
        self.cursor = 0

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.cursor >= len(self.baseline):
            raise RuntimeError("feature replay exhausted")
        frozen = features.clone()
        reference = self.baseline[self.cursor].to(features)
        frozen[:, self.feature_slice] = reference[:, self.feature_slice]
        self.cursor += 1
        return self.base(frozen)

    def assert_consumed(self):
        if self.cursor != len(self.baseline):
            raise RuntimeError("feature replay was not fully consumed")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected_modes = {"physical", "live", "replay", "half", "freeze_ts", "freeze_tm"}
    if config["status"] != "ready_for_linux":
        raise ValueError("Q32-R is not ready_for_linux")
    if set(config["probe"]["modes"]) != expected_modes:
        raise ValueError("Q32-R modes changed")
    if set(config["folds"]) != {"F0", "F1"}:
        raise ValueError("Q32-R folds changed")
    if config["probe"]["state"] != "dry":
        raise ValueError("Q32-R is frozen to dry points")
    return config


def verify_parent(config: dict[str, Any]):
    parent = config["parent"]
    manifest_path = parent["manifest_path"]
    if (
        sha256_git_file(parent["results_commit"], manifest_path)
        != parent["manifest_sha256"]
    ):
        raise RuntimeError("Q32 parent manifest changed")
    if subprocess.run(
        ["git", "diff", "--quiet", parent["results_commit"], "--", manifest_path],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("Q32 parent manifest differs from frozen result commit")
    for fold, item in parent["checkpoints"].items():
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Q32 h_now checkpoint changed: {fold}")


def load_development_data(csv_path: Path, config: dict[str, Any]):
    data = config["data"]
    use_columns = list(dict.fromkeys(q32.E0_COLS + q32.OUTPUTS))
    start = int(data["window_start"])
    rows = int(data["development_rows"])
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
    digest = hashlib.sha256(exo.tobytes() + targets.tobytes()).hexdigest()
    return exo, targets, digest


def select_quantile_points(
    exo: np.ndarray,
    lo: int,
    hi: int,
    count: int,
    state: str = "dry",
) -> list[int]:
    indices = np.arange(lo, hi)
    pressure = exo[indices, 2]
    mask = pressure > q32.P_CRIT if state == "dry" else pressure <= q32.P_CRIT
    valve = exo[indices, 6]
    eligible = indices[mask & (valve >= 0.0) & (valve <= 0.95)]
    if not len(eligible):
        return []
    positions = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    return np.unique(eligible[positions]).astype(int).tolist()


def load_residual(checkpoint: Path):
    residual = q32.r09.ResMLP(10, q32.r09.Q_SCALE).to(DEVICE)
    residual.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    residual.eval()
    for parameter in residual.parameters():
        parameter.requires_grad_(False)
    return residual


def run_pair(
    model,
    residual,
    candidate: dict[str, Any],
    row: np.ndarray,
    observed: np.ndarray,
    steps: int,
    valve_delta: float,
    w_multiplier: float,
    mode: str,
):
    if mode == "physical":
        baseline = q32.run_constant(
            model, residual, candidate, row, observed, steps, 0.0, 1.0, False
        )
        changed = q32.run_constant(
            model, residual, candidate, row, observed, steps,
            valve_delta, w_multiplier, False,
        )
    elif mode == "live":
        baseline = q32.run_constant(
            model, residual, candidate, row, observed, steps, 0.0, 1.0, True
        )
        changed = q32.run_constant(
            model, residual, candidate, row, observed, steps,
            valve_delta, w_multiplier, True,
        )
    elif mode == "half":
        scaled = ScaledResidual(residual, 0.5)
        baseline = q32.run_constant(
            model, scaled, candidate, row, observed, steps, 0.0, 1.0, True
        )
        changed = q32.run_constant(
            model, scaled, candidate, row, observed, steps,
            valve_delta, w_multiplier, True,
        )
    else:
        recorder = RecordResidual(residual)
        baseline = q32.run_constant(
            model, recorder, candidate, row, observed, steps, 0.0, 1.0, True
        )
        if mode == "replay":
            probe = ReplayResidual(recorder.outputs)
        elif mode == "freeze_ts":
            probe = FreezeFeaturesResidual(residual, recorder.features, slice(0, 3))
        elif mode == "freeze_tm":
            probe = FreezeFeaturesResidual(residual, recorder.features, slice(3, 6))
        else:
            raise ValueError(f"unknown probe mode: {mode}")
        changed = q32.run_constant(
            model, probe, candidate, row, observed, steps,
            valve_delta, w_multiplier, True,
        )
        probe.assert_consumed()
    return changed - baseline


def probe_point(
    model,
    residual,
    row: np.ndarray,
    observed: np.ndarray,
    kw: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    probe = config["probe"]
    steps = int(probe["steps"])
    valve_delta = float(probe["valve_step_fraction"])
    candidate = {"injection": "h_only", "residual_reads_w": False}
    multipliers = {
        "valve_only": 1.0,
        "coupled": 1.0 + kw * valve_delta,
    }
    results: dict[str, Any] = {}
    for path in probe["paths"]:
        results[path] = {}
        for mode in probe["modes"]:
            delta = run_pair(
                model, residual, candidate, row, observed, steps,
                valve_delta, multipliers[path], mode,
            )
            results[path][mode] = q32.response_metrics(
                delta, int(config["data"]["sample_seconds"])
            )
        results[path]["residual_mediated_steady_c"] = (
            results[path]["live"]["steady_main_c"]
            - results[path]["physical"]["steady_main_c"]
        )
    return results


def summarize(points: list[dict[str, Any]], config: dict[str, Any]):
    summary: dict[str, Any] = {}
    for path in config["probe"]["paths"]:
        summary[path] = {}
        physical = np.array(
            [point["responses"][path]["physical"]["steady_main_c"] for point in points]
        )
        for mode in config["probe"]["modes"]:
            gains = np.array(
                [point["responses"][path][mode]["steady_main_c"] for point in points]
            )
            taus = [
                point["responses"][path][mode]["tau63_seconds"] for point in points
                if point["responses"][path][mode]["tau63_seconds"] is not None
            ]
            summary[path][mode] = {
                "n": int(len(gains)),
                "negative_gain_fraction": float(np.mean(gains < 0.0)),
                "median_gain_c": float(np.median(gains)),
                "gain_q10_c": float(np.quantile(gains, 0.1)),
                "gain_q90_c": float(np.quantile(gains, 0.9)),
                "median_tau63_seconds": float(np.median(taus)) if taus else None,
                "median_delta_vs_physical_c": float(np.median(gains - physical)),
            }
    return summary


def execute(args, config: dict[str, Any], config_path: Path):
    verify_parent(config)
    output = Path(args.output).resolve()
    summary_path = output / "summary_development.json"
    if summary_path.exists():
        raise RuntimeError(f"Q32-R summary already exists: {summary_path}")
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    exo, targets, development_hash = load_development_data(csv_path, config)
    model = q32.load_evap_model(ROOT / "out" / "model_e0_evap_seed0.pt")
    all_folds: dict[str, Any] = {}

    for fold_id, fold in config["folds"].items():
        checkpoint = ROOT / config["parent"]["checkpoints"][fold_id]["path"]
        residual = load_residual(checkpoint)
        kw = q32.estimate_w_coupling(exo, fold["training"])["dry"]
        if kw is None:
            raise RuntimeError(f"dry W coupling unavailable: {fold_id}")
        lo, hi = fold["evaluation"]
        indices = select_quantile_points(
            exo, int(lo), int(hi), int(config["probe"]["points_per_fold"])
        )
        if not indices:
            raise RuntimeError(f"no eligible dry points: {fold_id}")
        points = []
        for index in indices:
            print(f"[Q32-R] {fold_id} row={index}", flush=True)
            points.append(
                {
                    "row": index,
                    "pressure_mpa": float(exo[index, 2]),
                    "valve2_fraction": float(exo[index, 6]),
                    "spray_w": float(exo[index, 8]),
                    "responses": probe_point(
                        model, residual, exo[index], targets[index], float(kw), config
                    ),
                }
            )
        all_folds[fold_id] = {
            "training_only_w_coupling": float(kw),
            "point_count": len(points),
            "points": points,
            "aggregate": summarize(points, config),
        }

    payload = {
        "experiment": config["experiment"],
        "status": "results_returned",
        "folds": all_folds,
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
        "parent_manifest_sha256": config["parent"]["manifest_sha256"],
        "data_path": str(csv_path),
        "development_arrays_sha256": development_hash,
        "development_rows_loaded": int(config["data"]["development_rows"]),
        "reserved_rows_loaded": False,
        "training_performed": False,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    }
    json_dump(output / "manifest.json", manifest)
    point_count = sum(fold["point_count"] for fold in all_folds.values())
    print(json.dumps({"summary": str(summary_path), "points": point_count}, indent=2))


def dry_run(config: dict[str, Any], config_path: Path, args):
    verify_parent(config)
    payload = {
        "experiment": config["experiment"],
        "status": config["status"],
        "config_sha256": sha256_file(config_path),
        "folds": list(config["folds"]),
        "max_points": len(config["folds"]) * int(config["probe"]["points_per_fold"]),
        "modes": config["probe"]["modes"],
        "training": False,
        "scientific_verdict": None,
        "linux_command": (
            f"python 33_qnav_residual_feedback_probe.py --execute "
            f"--csv {args.csv} --output {args.output}"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "qnav_residual_feedback_probe.json"),
    )
    parser.add_argument("--csv", default=q32.t02.CSV)
    parser.add_argument(
        "--output", default=str(ROOT / "out" / "qnav_residual_feedback_probe")
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
