#!/usr/bin/env python3
"""Run frozen MS3-R Gate A point and input-identifiability diagnostics."""

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

from src.phase35.data import load_cache
from src.phase35.ms3r import run_gate1_analysis, validate_ms3r_gate1_config


DEFAULT_CONFIG = ROOT / "configs/phase3_5/ms3r_gate1_point_identifiability.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3r_gate1_point_identifiability"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _dirty_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def run(
    *,
    config_path: Path,
    cache_paths: dict[str, Path],
    output: Path,
    require_clean: bool,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_ms3r_gate1_config(config)
    if require_clean and _dirty_paths():
        raise RuntimeError("MS3-R Gate A requires a clean committed worktree")
    parent = (ROOT / config["parent_matrix"]["path"]).resolve()
    parent_sha = _sha256(parent)
    if parent_sha != config["parent_matrix"]["sha256"]:
        raise RuntimeError("MS3-R parent matrix SHA changed")
    caches = {side: load_cache(cache_paths[side]) for side in ("A", "B")}
    for side in ("A", "B"):
        cached_parent = caches[side].metadata.get("matrix_sha256")
        if cached_parent is not None and cached_parent != parent_sha:
            raise RuntimeError(f"MS3-R {side} cache was built from another parent matrix")

    analysis, arrays = run_gate1_analysis(caches, config)
    validation_bounds = caches["A"].split_bounds()["validation"]
    manifest = {
        "protocol_version": config["protocol_version"],
        "batch_id": config["execution_contract"]["batch_id"],
        "execution_git_sha": _git_sha(),
        "config_path": _relative(config_path),
        "config_sha256": _sha256(config_path),
        "parent_matrix_path": _relative(parent),
        "parent_matrix_sha256": parent_sha,
        "source_sha256": config["data_contract"]["source_sha256"],
        "cache_paths": {side: str(cache_paths[side]) for side in ("A", "B")},
        "cache_sha256": {side: _sha256(cache_paths[side]) for side in ("A", "B")},
        "split": "validation",
        "split_bounds": list(validation_bounds),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "test_accessed": False,
        "training_executed": False,
        "automatic_scientific_pass": None,
        "execution_budget": {
            key: config["execution_contract"][key]
            for key in (
                "maximum_attempts",
                "expected_wall_clock_minutes",
                "hard_wall_clock_seconds",
                "cpu_threads_max",
                "gpu_required",
                "expected_peak_rss_gib_max",
                "automatic_retry_allowed",
            )
        },
    }
    summary = dict(analysis)
    summary["manifest"] = manifest
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "run_manifest.json", manifest)
    _atomic_json(output / "branch_semantics.json", analysis["branch_semantics"])
    _atomic_json(output / "point_quality_validation.json", analysis["point_quality"])
    _atomic_json(output / "path_diagnostics_validation.json", analysis["path_diagnostics"])
    _atomic_json(output / "rank_diagnostics_validation.json", analysis["rank_diagnostics"])
    _atomic_npz(output / "analysis_arrays_validation.npz", arrays)
    _atomic_json(output / "summary_validation.json", summary)
    artifact_names = (
        "run_manifest.json",
        "branch_semantics.json",
        "point_quality_validation.json",
        "path_diagnostics_validation.json",
        "rank_diagnostics_validation.json",
        "analysis_arrays_validation.npz",
        "summary_validation.json",
    )
    artifact_ledger = {name: _sha256(output / name) for name in artifact_names}
    _atomic_json(output / "artifact_ledger.json", artifact_ledger)
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
                "crossfit_evaluated_count": summary["analysis_support"]["crossfit_evaluated_count"],
                "automatic_scientific_pass": summary["automatic_scientific_pass"],
                "test_accessed": summary["test_accessed"],
                "next_action": "return_artifacts_for_local_gate_a_audit",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
