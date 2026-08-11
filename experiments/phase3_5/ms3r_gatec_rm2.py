#!/usr/bin/env python3
"""Execute the frozen 54-run Gate C RM2 matrix on Hermes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.phase35.data import load_cache
from src.phase35.multistep.gatec_rm2_contracts import (
    RM2RunSpec,
    partition_rm2_runs,
    rm2_run_specs,
    validate_rm2_matrix,
)
from src.phase35.multistep.gatec_rm2_training import run_rm2_training


DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms3r_gatec_rm2_matrix.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3r_gatec_rm2"
REGISTRY = ROOT / "configs/phase3_5/experiment_registry.json"
PINNED_CODE = (
    "src/phase35/multistep/gatec_contracts.py",
    "src/phase35/multistep/gatec_data.py",
    "src/phase35/multistep/gatec_model.py",
    "src/phase35/multistep/gatec_training.py",
    "src/phase35/multistep/gatec_rm2_contracts.py",
    "src/phase35/multistep/gatec_rm2_training.py",
    "experiments/phase3_5/ms3r_gatec_rm2.py",
    "experiments/phase3_5/summarize_ms3r_gatec_rm2.py",
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


def _dirty_paths(*, allowed_output_root: Path | None = None) -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    allowed_prefix: str | None = None
    if allowed_output_root is not None:
        try:
            allowed_prefix = (
                str(allowed_output_root.resolve().relative_to(ROOT)).replace("\\", "/").rstrip("/")
                + "/"
            )
        except ValueError as exc:
            raise RuntimeError("RM2 output root must stay inside the repository") from exc
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        relative = line[3:].replace("\\", "/")
        if allowed_prefix is not None and relative.startswith(allowed_prefix):
            continue
        dirty.append(line)
    return dirty


def _verify_parent(matrix: dict[str, Any]) -> dict[str, str]:
    parent = matrix["parent_rm1a_audit"]
    path = ROOT / parent["path"]
    if _sha256(path) != parent["sha256"]:
        raise RuntimeError("RM2 parent RM1-A audit pin changed")
    payload = _read_json(path)
    if payload.get("supervisor_decision", {}).get("label") != parent["required_label"]:
        raise RuntimeError("RM2 parent RM1-A decision changed")
    return dict(parent)


def _verify_registry(registry_path: Path = REGISTRY) -> None:
    registry = _read_json(registry_path)
    experiment = registry.get("experiments", {}).get("ms3_r", {})
    if registry.get("active_gate") != "ms3_r":
        raise RuntimeError("RM2 webhook requires active_gate=ms3_r")
    if registry.get("linux_authorized_gate") != "ms3_r":
        raise RuntimeError("RM2 webhook requires linux_authorized_gate=ms3_r")
    if experiment.get("status") != "ready_for_linux":
        raise RuntimeError("RM2 webhook requires ms3_r.status=ready_for_linux")


def _verify_complete_run(run_dir: Path, required: Iterable[str]) -> bool:
    ledger_path = run_dir / "artifact_ledger.json"
    if not ledger_path.is_file():
        return False
    try:
        ledger = _read_json(ledger_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = set(required) - {"artifact_ledger.json"}
    if set(ledger) != expected:
        return False
    return all((run_dir / name).is_file() and _sha256(run_dir / name) == digest for name, digest in ledger.items())


def dry_run_payload(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_rm2_matrix(matrix)
    _verify_parent(matrix)
    specs = rm2_run_specs(matrix)
    return {
        "protocol_version": matrix["protocol_version"],
        "batch_id": matrix["batch_id"],
        "unique_candidate_count": len({spec.candidate_id for spec in specs}),
        "run_count": len(specs),
        "group_run_counts": {
            group: sum(spec.group == group for spec in specs) for group in ("A", "B", "C")
        },
        "seeds": sorted({spec.seed for spec in specs}),
        "folds": sorted({spec.fold_id for spec in specs}),
        "test_authorized": False,
        "linux_authorized": True,
        "automatic_scientific_pass": None,
    }


def _worker(
    *,
    device: str,
    specs: list[RM2RunSpec],
    matrix_path: str,
    cache_a: str,
    cache_b: str,
    output_root: str,
    provenance: dict[str, Any],
    skip_complete: bool,
) -> list[dict[str, Any]]:
    matrix = _read_json(Path(matrix_path))
    caches = {"A": load_cache(cache_a), "B": load_cache(cache_b)}
    required = matrix["execution_contract"]["required_run_artifacts"]
    records: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = Path(output_root) / spec.run_id
        if _verify_complete_run(run_dir, required):
            if skip_complete:
                records.append({"run_id": spec.run_id, "status": "skipped_complete"})
                continue
            records.append({"run_id": spec.run_id, "status": "refused_existing_complete"})
            continue
        if run_dir.exists():
            records.append({"run_id": spec.run_id, "status": "refused_existing_incomplete"})
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            run_dir / "attempt_started.json",
            {"run_id": spec.run_id, "attempt": 1, "device": device, "provenance": provenance},
        )
        try:
            record = run_rm2_training(
                caches,
                matrix,
                spec,
                device=device,
                output_dir=run_dir,
                provenance={**provenance, "device": device},
            )
        except Exception as exc:  # machine batch must preserve independent failures
            record = {
                "run_id": spec.run_id,
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            _atomic_json(run_dir / "failure.json", record)
        records.append(record)
    return records


def execute_matrix(
    *,
    matrix_path: Path,
    cache_paths: dict[str, Path],
    output_root: Path,
    devices: list[str],
    groups: set[str],
    skip_complete: bool,
    require_clean: bool,
) -> list[dict[str, Any]]:
    matrix = _read_json(matrix_path)
    validate_rm2_matrix(matrix)
    parent = _verify_parent(matrix)
    _verify_registry()
    if require_clean and _dirty_paths(allowed_output_root=output_root):
        raise RuntimeError("RM2 Hermes execution requires a clean worktree")
    if not devices:
        raise RuntimeError("RM2 execution needs a non-empty device pool")
    caches = {side: load_cache(path) for side, path in cache_paths.items()}
    expected_source = matrix["data_contract"]["source_sha256"]
    for side in ("A", "B"):
        if caches[side].metadata.get("side") != side:
            raise RuntimeError(f"RM2 cache side mismatch for {side}")
        if caches[side].metadata.get("source", {}).get("sha256") != expected_source:
            raise RuntimeError(f"RM2 cache source pin mismatch for {side}")
    specs = [spec for spec in rm2_run_specs(matrix) if spec.group in groups]
    if groups == {"A", "B", "C"} and len(specs) != 54:
        raise RuntimeError("RM2 complete matrix expansion changed")
    provenance = {
        "execution_git_sha": _git_sha(),
        "matrix_path": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "matrix_sha256": _sha256(matrix_path),
        "parent_rm1a_audit": parent,
        "cache_sha256": {side: _sha256(path) for side, path in cache_paths.items()},
        "code_sha256": {path: _sha256(ROOT / path) for path in PINNED_CODE},
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "test_accessed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_root / "run_manifest.json",
        {
            **provenance,
            "batch_id": matrix["batch_id"],
            "devices": devices,
            "groups": sorted(groups),
            "run_ids": [spec.run_id for spec in specs],
            "maximum_attempts_per_run": 1,
            "automatic_scientific_pass": None,
        },
    )
    partitions = [part for part in partition_rm2_runs(specs, len(devices)) if part]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=len(partitions), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(
                _worker,
                device=devices[index],
                specs=partition,
                matrix_path=str(matrix_path),
                cache_a=str(cache_paths["A"]),
                cache_b=str(cache_paths["B"]),
                output_root=str(output_root),
                provenance=provenance,
                skip_complete=skip_complete,
            )
            for index, partition in enumerate(partitions)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda item: item["run_id"])
    _atomic_json(output_root / "matrix_execution_status.json", {"records": records})
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--list-runs", action="store_true")
    actions.add_argument("--execute-matrix", action="store_true")
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--groups", default="A,B,C")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix = _read_json(matrix_path)
    validate_rm2_matrix(matrix)
    if args.dry_run:
        payload: Any = dry_run_payload(matrix)
    elif args.list_runs:
        payload = [asdict(spec) for spec in rm2_run_specs(matrix)]
    else:
        if not args.cache_a or not args.cache_b:
            raise SystemExit("RM2 execution requires --cache-a and --cache-b")
        groups = {value.strip() for value in args.groups.split(",") if value.strip()}
        if not groups or not groups <= {"A", "B", "C"}:
            raise SystemExit("RM2 groups must be a subset of A,B,C")
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        payload = execute_matrix(
            matrix_path=matrix_path,
            cache_paths={"A": Path(args.cache_a).resolve(), "B": Path(args.cache_b).resolve()},
            output_root=Path(args.output_root).resolve(),
            devices=devices,
            groups=groups,
            skip_complete=args.skip_complete,
            require_clean=not args.allow_dirty,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
