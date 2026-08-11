#!/usr/bin/env python3
"""Run the local validation-only RM1-A attribution matrix on real data."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.phase3_5.ms3r_gatec_real_subset import (
    _atomic_json,
    _dirty_paths,
    _git_sha,
    _read_json,
    _sha256,
)
from src.phase35.data import load_cache
from src.phase35.multistep.gatec_contracts import validate_gatec_matrix
from src.phase35.multistep.gatec_real_smoke import (
    GateCRealSmokeConfig,
    run_gatec_real_subset_smoke,
)


DEFAULT_CONFIG = ROOT / "configs/phase3_5/ms3r_gatec_local_real_rm1a.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3r_gatec_local_real_rm1a"
PINNED_CODE = (
    "src/phase35/multistep/gatec_contracts.py",
    "src/phase35/multistep/gatec_data.py",
    "src/phase35/multistep/gatec_model.py",
    "src/phase35/multistep/gatec_training.py",
    "src/phase35/multistep/gatec_real_smoke.py",
    "experiments/phase3_5/ms3r_gatec_real_attribution.py",
)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != "phase3.5-ms3r-gatec-local-real-rm1a-v1":
        raise RuntimeError("unsupported Gate C RM1-A protocol")
    data = config.get("data_contract", {})
    if data.get("allowed_splits") != ["train", "validation"]:
        raise RuntimeError("Gate C RM1-A must use train/validation only")
    if data.get("test_allowed") is not False or int(data.get("fraction_denominator", 0)) != 100:
        raise RuntimeError("Gate C RM1-A is frozen to real 1/100 with no test access")
    execution = config.get("execution_contract", {})
    if execution.get("local_real_attribution_authorized") is not True:
        raise RuntimeError("Gate C RM1-A local attribution is not authorized")
    if execution.get("linux_authorized") is not False or execution.get("test_authorized") is not False:
        raise RuntimeError("Gate C RM1-A cannot authorize Linux or test")

    matrix_pin = config["parent_gatec_matrix"]
    matrix_path = ROOT / matrix_pin["path"]
    if _sha256(matrix_path) != matrix_pin["sha256"]:
        raise RuntimeError("Gate C RM1-A parent matrix pin changed")
    matrix = _read_json(matrix_path)
    validate_gatec_matrix(matrix)
    if config.get("candidates") != matrix["rm1_attribution"]:
        raise RuntimeError("Gate C RM1-A candidates differ from the frozen parent matrix")

    audit_pin = config["parent_rm0b_audit"]
    audit_path = ROOT / audit_pin["path"]
    if _sha256(audit_path) != audit_pin["sha256"]:
        raise RuntimeError("Gate C RM1-A parent RM0-B audit pin changed")
    audit = _read_json(audit_path)
    if audit.get("supervisor_decision", {}).get("label") != audit_pin["required_label"]:
        raise RuntimeError("Gate C RM1-A parent RM0-B decision changed")
    if len(config["candidates"]) != 6 or config.get("seeds") != [0]:
        raise RuntimeError("Gate C RM1-A candidate/seed matrix changed")


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": config["protocol_version"],
        "batch_id": config["batch_id"],
        "candidate_ids": [item["candidate_id"] for item in config["candidates"]],
        "fraction_denominator": config["data_contract"]["fraction_denominator"],
        "allowed_splits": config["data_contract"]["allowed_splits"],
        "test_authorized": False,
        "linux_authorized": False,
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
    validate_config(config)
    if require_clean and _dirty_paths():
        raise RuntimeError("Gate C RM1-A requires a clean committed worktree")
    caches = {side: load_cache(path) for side, path in cache_paths.items()}
    expected_source = config["data_contract"]["source_sha256"]
    for side in ("A", "B"):
        if caches[side].metadata.get("side") != side:
            raise RuntimeError(f"Gate C RM1-A cache side mismatch for {side}")
        if caches[side].metadata.get("source", {}).get("sha256") != expected_source:
            raise RuntimeError(f"Gate C RM1-A cache source pin mismatch for {side}")

    manifest = {
        "protocol_version": config["protocol_version"],
        "batch_id": config["batch_id"],
        "execution_git_sha": _git_sha(),
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "parent_gatec_matrix": config["parent_gatec_matrix"],
        "parent_rm0b_audit": config["parent_rm0b_audit"],
        "source_sha256": expected_source,
        "cache_paths": {side: str(path) for side, path in cache_paths.items()},
        "cache_sha256": {side: _sha256(path) for side, path in cache_paths.items()},
        "code_sha256": {path: _sha256(ROOT / path) for path in PINNED_CODE},
        "device": device,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "splits_accessed": ["train", "validation"],
        "test_accessed": False,
        "linux_authorized": False,
        "maximum_attempts_per_candidate": 1,
        "automatic_scientific_pass": None,
    }
    _atomic_json(output / "run_manifest.json", manifest)

    results: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        smoke = GateCRealSmokeConfig(
            route=candidate["response_route"],
            candidate_id=candidate["candidate_id"],
            residual_capacity=candidate["residual_capacity"],
            response_scheduling=candidate["response_scheduling"],
            local_supervision=candidate["local_supervision"],
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
        _atomic_json(output / f"{candidate['candidate_id']}_validation.json", result)
        results.append(result)

    train_shas = {item["train_anchor_sha256"] for item in results}
    validation_shas = {item["validation_anchor_sha256"] for item in results}
    summary = {
        "protocol_version": config["protocol_version"],
        "scope": config["scope"],
        "results": results,
        "candidate_count": len(results),
        "all_finite": all(item["metrics_validation"]["finite"] for item in results),
        "shared_train_anchors": len(train_shas) == 1,
        "shared_validation_anchors": len(validation_shas) == 1,
        "selector_eligible_candidates": [
            item["candidate_id"] for item in results if item["selector_eligible"]
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
    _atomic_json(output / "artifact_ledger.json", {name: _sha256(output / name) for name in required})
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    validate_config(config)
    if args.dry_run:
        payload = dry_run(config)
    else:
        if not args.cache_a or not args.cache_b:
            raise SystemExit("Gate C RM1-A requires --cache-a and --cache-b")
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
