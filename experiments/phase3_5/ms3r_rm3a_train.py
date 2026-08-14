#!/usr/bin/env python3
"""Frozen RM3-A 30-run validation executor; remains unauthorized by default."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.data import load_cache
from src.phase35.multistep.rm3_reporting import build_root_artifact_ledger, file_sha256
from src.phase35.multistep.rm3a_contracts import rm3a_run_specs, validate_rm3a_matrix
from src.phase35.multistep.rm3a_reporting import summarize_rm3a
from src.phase35.multistep.rm3a_training import run_rm3a_training


MATRIX = ROOT / "configs/phase3_5/ms3r_rm3a_matrix.json"
OUTPUT = ROOT / "results/phase3_5/ms3r_rm3a"
REFERENCE = ROOT / "results/phase3_5/ms3r_rm3/prediction"
REGISTRY = ROOT / "configs/phase3_5/experiment_registry.json"
PINNED_CODE = (
    "src/phase35/multistep/rm3_prediction.py",
    "src/phase35/multistep/rm3_training.py",
    "src/phase35/multistep/rm3a_contracts.py",
    "src/phase35/multistep/rm3a_training.py",
    "src/phase35/multistep/rm3a_reporting.py",
    "experiments/phase3_5/ms3r_rm3a_train.py",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _verify_registry() -> None:
    registry = _read(REGISTRY)
    if registry.get("active_gate") != "ms3_r" or registry.get("linux_authorized_gate") != "ms3_r":
        raise RuntimeError("RM3-A requires active and linux_authorized gate ms3_r")
    experiment = registry["experiments"]["ms3_r"]
    if experiment.get("status") != "ready_for_linux":
        raise RuntimeError("RM3-A requires ms3_r.status=ready_for_linux")
    decision = experiment.get("decision", {})
    if not isinstance(decision, dict) or decision.get("authorized_batch") != "RM3-A":
        raise RuntimeError("RM3-A requires decision.authorized_batch=RM3-A")


def _dirty_paths(output: Path) -> list[str]:
    raw = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    try:
        allowed = str(output.resolve().relative_to(ROOT)).replace("\\", "/").rstrip("/") + "/"
    except ValueError:
        allowed = None
    return [
        line for line in raw.splitlines()
        if line.strip() and (
            allowed is None or not line[3:].replace("\\", "/").startswith(allowed)
        )
    ]


def dry_run_payload(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_rm3a_matrix(matrix, repo_root=ROOT)
    specs = rm3a_run_specs(matrix)
    return {
        "protocol_version": matrix["protocol_version"],
        "new_candidate_count": len({spec.candidate_id for spec in specs}),
        "new_run_count": len(specs),
        "reused_reference_candidate_count": len(matrix["reference_candidates"]),
        "reused_reference_run_count": len(matrix["reference_candidates"]) * 6,
        "new_run_ids": [spec.run_id for spec in specs],
        "matrix_self_authorizing": matrix["execution_contract"]["linux_authorized"],
        "registry_authorization_required_for_execute": True,
        "test_authorized": False,
        "automatic_scientific_pass": None,
    }


def _worker(device: str, specs: list[Any], matrix_path: str, cache_a: str, cache_b: str, output: str, provenance: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = _read(Path(matrix_path))
    caches = {"A": load_cache(cache_a), "B": load_cache(cache_b)}
    records = []
    for spec in specs:
        directory = Path(output) / spec.run_id
        if directory.exists():
            records.append({"run_id": spec.run_id, "status": "refused_existing"})
            continue
        try:
            records.append(run_rm3a_training(caches, matrix, spec, device=device, output_dir=directory, provenance={**provenance, "device": device}))
        except Exception as exc:
            directory.mkdir(parents=True, exist_ok=True)
            record = {"run_id": spec.run_id, "status": "failed", "exception_type": type(exc).__name__, "message": str(exc)}
            _atomic(directory / "failure.json", record)
            records.append(record)
    return records


def execute(matrix_path: Path, cache_a: Path, cache_b: Path, output: Path, devices: list[str], *, require_clean: bool = True) -> list[dict[str, Any]]:
    matrix = _read(matrix_path)
    validate_rm3a_matrix(matrix, repo_root=ROOT)
    _verify_registry()
    if require_clean and _dirty_paths(output):
        raise RuntimeError("RM3-A requires a clean worktree")
    if not devices:
        raise RuntimeError("RM3-A needs a device pool")
    caches = {"A": load_cache(cache_a), "B": load_cache(cache_b)}
    expected = matrix["data_contract"]["source_sha256"]
    for side in ("A", "B"):
        if caches[side].metadata.get("side") != side or caches[side].metadata.get("source", {}).get("sha256") != expected:
            raise RuntimeError(f"RM3-A cache pin mismatch for {side}")
    specs = rm3a_run_specs(matrix)
    provenance = {
        "execution_git_sha": _git_sha(), "matrix_sha256": file_sha256(matrix_path),
        "cache_sha256": {"A": file_sha256(cache_a), "B": file_sha256(cache_b)},
        "code_sha256": {path: file_sha256(ROOT / path) for path in PINNED_CODE},
        "parent_rm3_audit": matrix["parent_rm3_audit"], "test_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic(output / "run_manifest.json", {
        **provenance, "devices": devices, "new_run_ids": [spec.run_id for spec in specs],
        "reused_reference_runs": 18, "maximum_attempts_per_run": 1,
        "hermes_role": "execute_30_frozen_new_runs_and_return_all_checkpoints",
        "automatic_scientific_pass": None,
    })
    partitions = [specs[index::len(devices)] for index in range(len(devices))]
    records = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_worker, device, part, str(matrix_path), str(cache_a), str(cache_b), str(output), provenance) for device, part in zip(devices, partitions) if part]
        for future in as_completed(futures): records.extend(future.result())
    records.sort(key=lambda row: row["run_id"])
    complete = len(records) == 30 and all(row["status"] == "complete" for row in records)
    _atomic(output / "matrix_execution_status.json", {"records": records, "all_complete": complete, "test_accessed": False})
    if complete:
        summary = summarize_rm3a(
            output, REFERENCE, specs,
            required_artifacts=matrix["execution_contract"]["required_run_artifacts"],
            reference_candidates=matrix["reference_candidates"],
        )
        _atomic(output / "summary_validation.json", summary)
        _atomic(output / "artifact_ledger.json", build_root_artifact_ledger(output, matrix["execution_contract"]["required_root_artifacts"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(MATRIX))
    parser.add_argument("--output-root", default=str(OUTPUT))
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--devices", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix = _read(Path(args.matrix).resolve())
    if args.dry_run:
        print(json.dumps(dry_run_payload(matrix), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not args.cache_a or not args.cache_b: parser.error("--execute requires both caches")
    records = execute(Path(args.matrix).resolve(), Path(args.cache_a).resolve(), Path(args.cache_b).resolve(), Path(args.output_root).resolve(), [item.strip() for item in args.devices.split(",") if item.strip()])
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
