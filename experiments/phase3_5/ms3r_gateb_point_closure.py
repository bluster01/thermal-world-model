#!/usr/bin/env python3
"""Run or finalize the frozen MS3-R Gate B validation batch."""

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
from src.phase35.ms3r_gateb import run_gateb_analysis, validate_ms3r_gateb_config


DEFAULT_CONFIG = ROOT / "configs/phase3_5/ms3r_gateb_point_closure.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3r_gateb_point_closure"


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


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty_paths() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [line for line in output.splitlines() if line.strip()]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _verify_parent(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent_gate_a"]
    config_path = ROOT / parent["config_path"]
    audit_path = ROOT / parent["audit_path"]
    if _sha256(config_path) != parent["config_sha256"]:
        raise RuntimeError("MS3-R Gate A config pin changed")
    if _sha256(audit_path) != parent["audit_sha256"]:
        raise RuntimeError("MS3-R Gate A audit pin changed")
    audit = _read_json(audit_path)
    if audit.get("supervisor_decision", {}).get("label") != parent["required_label"]:
        raise RuntimeError("MS3-R Gate A supervisor label is not the required conditional pass")
    return {
        "gate_a_config_path": parent["config_path"],
        "gate_a_config_sha256": parent["config_sha256"],
        "gate_a_audit_path": parent["audit_path"],
        "gate_a_audit_sha256": parent["audit_sha256"],
        "gate_a_required_label": parent["required_label"],
    }


def run(
    *, config_path: Path, cache_paths: dict[str, Path], output: Path, require_clean: bool
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_ms3r_gateb_config(config)
    if require_clean and _dirty_paths():
        raise RuntimeError("MS3-R Gate B requires a clean committed worktree")
    parent = _verify_parent(config)
    caches = {side: load_cache(cache_paths[side]) for side in ("A", "B")}
    summary, arrays = run_gateb_analysis(caches, config)
    manifest = {
        "protocol_version": config["protocol_version"],
        "batch_id": config["execution_contract"]["batch_id"],
        "execution_git_sha": _git_sha(),
        "config_path": _relative(config_path),
        "config_sha256": _sha256(config_path),
        **parent,
        "source_sha256": config["data_contract"]["source_sha256"],
        "cache_paths": {side: str(cache_paths[side]) for side in ("A", "B")},
        "cache_sha256": {side: _sha256(cache_paths[side]) for side in ("A", "B")},
        "split": "validation",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "test_accessed": False,
        "training_executed": False,
        "automatic_scientific_pass": None,
        "maximum_attempts": 1,
        "resource_capture_command": config["execution_contract"]["resource_capture_command"],
    }
    summary = {**summary, "manifest": manifest}
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "run_manifest.json", manifest)
    _atomic_json(output / "paired_contrasts_validation.json", summary["paired_contrasts"])
    _atomic_json(output / "mimo_response_validation.json", summary["mimo_response"])
    _atomic_json(output / "invariance_validation.json", summary["invariance"])
    _atomic_json(output / "iv_feasibility_validation.json", summary["iv_feasibility"])
    _atomic_npz(output / "replay_arrays_validation.npz", arrays)
    _atomic_json(output / "summary_validation.json", summary)
    return summary


def finalize(*, config_path: Path, output: Path) -> dict[str, str]:
    """Hash every returned artifact after the shell has closed stdout/stderr."""

    config = _read_json(config_path)
    validate_ms3r_gateb_config(config)
    required = [name for name in config["execution_contract"]["required_artifacts"] if name != "artifact_ledger.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"cannot finalize MS3-R Gate B; missing artifacts: {missing}")
    ledger = {name: _sha256(output / name) for name in required}
    _atomic_json(output / "artifact_ledger.json", ledger)
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output = Path(args.output_dir).resolve()
    if args.finalize_only:
        ledger = finalize(config_path=config_path, output=output)
        print(json.dumps({"finalized_artifact_count": len(ledger)}, indent=2))
        return
    if not args.cache_a or not args.cache_b:
        raise SystemExit("--cache-a and --cache-b are required unless --finalize-only is used")
    summary = run(
        config_path=config_path,
        cache_paths={"A": Path(args.cache_a).resolve(), "B": Path(args.cache_b).resolve()},
        output=output,
        require_clean=not args.allow_dirty,
    )
    print(json.dumps({
        "protocol_version": summary["protocol_version"],
        "independent_utc_day_count": summary["analysis_support"]["independent_utc_day_count"],
        "automatic_scientific_pass": None,
        "test_accessed": False,
        "next_action": "finalize_artifacts_then_return_for_single_local_supervisor_audit",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
