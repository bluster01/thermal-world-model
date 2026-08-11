#!/usr/bin/env python3
"""Run the explicitly authorized local 1/100 Gate C real-data smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.phase35.data import load_cache
from src.phase35.multistep.gatec_real_smoke import (
    GateCRealSmokeConfig,
    run_gatec_real_subset_smoke,
)


DEFAULT_CONFIG = ROOT / "configs/phase3_5/ms3r_gatec_local_real_subset.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3r_gatec_local_real_subset"
PINNED_CODE = (
    "src/phase35/multistep/gatec_contracts.py",
    "src/phase35/multistep/gatec_data.py",
    "src/phase35/multistep/gatec_model.py",
    "src/phase35/multistep/gatec_training.py",
    "src/phase35/multistep/gatec_real_smoke.py",
    "experiments/phase3_5/ms3r_gatec_real_subset.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty_paths() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line.strip()]


def validate_config(config: dict[str, Any], config_path: Path) -> None:
    if config.get("protocol_version") != "phase3.5-ms3r-gatec-local-real-subset-v1":
        raise RuntimeError("unsupported Gate C local real-subset protocol")
    data = config.get("data_contract", {})
    if data.get("allowed_splits") != ["train", "validation"] or data.get("test_allowed") is not False:
        raise RuntimeError("Gate C local real subset must be train/validation only")
    if int(data.get("fraction_denominator", 0)) not in {10, 100}:
        raise RuntimeError("Gate C local real subset must be 1/10 or 1/100")
    execution = config.get("execution_contract", {})
    if execution.get("local_real_smoke_authorized") is not True:
        raise RuntimeError("Gate C local real smoke is not authorized")
    if execution.get("linux_authorized") is not False or execution.get("real_full_matrix_authorized") is not False:
        raise RuntimeError("Gate C local smoke cannot authorize Linux/full real training")
    parent = config["parent_gatec_matrix"]
    parent_path = ROOT / parent["path"]
    if _sha256(parent_path) != parent["sha256"]:
        raise RuntimeError("Gate C parent matrix pin changed")
    if len(set(config.get("routes", []))) != 4 or len(config.get("seeds", [])) != 1:
        raise RuntimeError("Gate C local real route/seed matrix changed")
    if config_path.resolve() == parent_path.resolve():
        raise RuntimeError("Gate C local smoke config cannot be the parent matrix")


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": config["protocol_version"],
        "scope": config["scope"],
        "routes": config["routes"],
        "seeds": config["seeds"],
        "fraction_denominator": config["data_contract"]["fraction_denominator"],
        "allowed_splits": config["data_contract"]["allowed_splits"],
        "test_allowed": config["data_contract"]["test_allowed"],
        "local_real_smoke_authorized": True,
        "linux_authorized": False,
        "real_full_matrix_authorized": False,
        "automatic_scientific_pass": None,
    }


def run(
    *,
    config_path: Path,
    cache_paths: dict[str, Path],
    output: Path,
    device: str,
    require_clean: bool,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_config(config, config_path)
    if require_clean and _dirty_paths():
        raise RuntimeError("Gate C local real subset requires a clean committed worktree")
    caches = {side: load_cache(path) for side, path in cache_paths.items()}
    expected_source = config["data_contract"]["source_sha256"]
    for side in ("A", "B"):
        if caches[side].metadata.get("side") != side:
            raise RuntimeError(f"Gate C cache side mismatch for {side}")
        if caches[side].metadata.get("source", {}).get("sha256") != expected_source:
            raise RuntimeError(f"Gate C cache source pin mismatch for {side}")
    manifest = {
        "protocol_version": config["protocol_version"],
        "execution_git_sha": _git_sha(),
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "parent_gatec_matrix": config["parent_gatec_matrix"],
        "source_sha256": expected_source,
        "cache_paths": {side: str(path) for side, path in cache_paths.items()},
        "cache_sha256": {side: _sha256(path) for side, path in cache_paths.items()},
        "code_sha256": {path: _sha256(ROOT / path) for path in PINNED_CODE},
        "device": device,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "fraction_denominator": config["data_contract"]["fraction_denominator"],
        "splits_accessed": ["train", "validation"],
        "test_accessed": False,
        "linux_authorized": False,
        "real_full_matrix_authorized": False,
        "maximum_attempts_per_route": 1,
        "automatic_scientific_pass": None,
    }
    _atomic_json(output / "run_manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for route in config["routes"]:
        smoke = GateCRealSmokeConfig(
            route=route,
            fraction_denominator=int(config["data_contract"]["fraction_denominator"]),
            seed=int(config["seeds"][0]),
            window=int(config["data_contract"]["window"]),
            horizon=int(config["data_contract"]["horizon"]),
            d_model=int(config["model"]["d_model"]),
            latent_dim=int(config["model"]["latent_dim"]),
            dropout=float(config["model"]["dropout"]),
            batch_size=int(config["training"]["batch_size"]),
            optimizer_updates=int(config["training"]["optimizer_updates"]),
            validation_batch_size=int(config["training"]["validation_batch_size"]),
            max_validation_anchors=int(config["training"]["max_validation_anchors"]),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            gradient_clip=float(config["training"]["gradient_clip"]),
            max_age_s=float(config["data_contract"]["max_age_s"]),
        )
        result = run_gatec_real_subset_smoke(caches, smoke, device=device)
        _atomic_json(output / f"{route}_validation.json", result)
        results.append(result)
    summary = {
        "protocol_version": config["protocol_version"],
        "scope": config["scope"],
        "results": results,
        "route_count": len(results),
        "all_finite": all(item["metrics_validation"]["finite"] for item in results),
        "selector_eligible_routes": [
            item["route"] for item in results if item["selector_eligible"]
        ],
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }
    _atomic_json(output / "summary_validation.json", summary)
    required = [
        name
        for name in config["execution_contract"]["required_artifacts"]
        if name != "artifact_ledger.json"
    ]
    ledger = {name: _sha256(output / name) for name in required}
    _atomic_json(output / "artifact_ledger.json", ledger)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    validate_config(config, config_path)
    if args.dry_run:
        payload = dry_run(config)
    else:
        if not args.cache_a or not args.cache_b:
            raise SystemExit("Gate C local real subset requires --cache-a and --cache-b")
        payload = run(
            config_path=config_path,
            cache_paths={"A": Path(args.cache_a).resolve(), "B": Path(args.cache_b).resolve()},
            output=Path(args.output_dir).resolve(),
            device=args.device,
            require_clean=not args.allow_dirty,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
