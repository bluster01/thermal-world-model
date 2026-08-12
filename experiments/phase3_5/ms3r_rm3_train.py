#!/usr/bin/env python3
"""Frozen RM3 train/validation executor; registry authorization is separate from science."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.phase35.data import load_cache
from src.phase35.multistep.rm3_calibration import run_rm3_calibration
from src.phase35.multistep.rm3_contracts import (
    rm3_calibration_specs,
    rm3_prediction_run_specs,
    validate_rm3_matrix,
)
from src.phase35.multistep.rm3_reporting import (
    build_root_artifact_ledger,
    file_sha256,
    summarize_rm3_predictions,
)
from src.phase35.multistep.rm3_training import run_rm3_prediction_training


DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms3r_rm3_matrix.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3r_rm3"
REGISTRY = ROOT / "configs/phase3_5/experiment_registry.json"
PINNED_CODE = (
    "src/phase35/multistep/rm3_contracts.py",
    "src/phase35/multistep/rm3_prediction.py",
    "src/phase35/multistep/rm3_joint_model.py",
    "src/phase35/multistep/rm3_orthogonal.py",
    "src/phase35/multistep/rm3_training.py",
    "src/phase35/multistep/rm3_calibration.py",
    "src/phase35/multistep/rm3_reporting.py",
    "experiments/phase3_5/ms3r_rm3_train.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _verify_registry() -> None:
    registry = _read_json(REGISTRY)
    experiment = registry.get("experiments", {}).get("ms3_r", {})
    if registry.get("active_gate") != "ms3_r" or registry.get("linux_authorized_gate") != "ms3_r":
        raise RuntimeError("RM3 Hermes requires active and linux_authorized gate ms3_r")
    if experiment.get("status") != "ready_for_linux":
        raise RuntimeError("RM3 Hermes requires ms3_r.status=ready_for_linux")


def _verify_parent(matrix: dict[str, Any]) -> dict[str, str]:
    parent = matrix["parent_rm2_audit"]
    path = ROOT / parent["path"]
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if parent.get("hash_mode") != "utf8_text_normalized_lf" or digest != parent["sha256"]:
        raise RuntimeError("RM3 parent RM2 audit hash changed")
    payload = _read_json(path)
    if payload.get("supervisor_decision", {}).get("label") != parent["required_label"]:
        raise RuntimeError("RM3 parent RM2 decision changed")
    return dict(parent)


def _dirty_paths(output_root: Path) -> list[str]:
    raw = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    try:
        allowed = str(output_root.resolve().relative_to(ROOT)).replace("\\", "/").rstrip("/") + "/"
    except ValueError as exc:
        raise RuntimeError("RM3 output root must stay inside repository") from exc
    return [line for line in raw.splitlines() if line.strip() and not line[3:].replace("\\", "/").startswith(allowed)]


def _verify_complete(directory: Path, required: Iterable[str]) -> bool:
    ledger_path = directory / "artifact_ledger.json"
    if not ledger_path.is_file():
        return False
    try:
        ledger = _read_json(ledger_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = set(required) - {"artifact_ledger.json"}
    return set(ledger) == expected and all(
        (directory / name).is_file() and file_sha256(directory / name) == digest
        for name, digest in ledger.items()
    )


def dry_run_payload(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_rm3_matrix(matrix)
    _verify_parent(matrix)
    predictions, calibrations = rm3_prediction_run_specs(matrix), rm3_calibration_specs(matrix)
    return {
        "protocol_version": matrix["protocol_version"],
        "prediction_run_count": len(predictions), "calibration_unit_count": len(calibrations),
        "total_run_count": len(predictions) + len(calibrations),
        "prediction_ids": [item.run_id for item in predictions],
        "calibration_ids": [item.calibration_id for item in calibrations],
        "calibration_candidates_per_unit": list(calibrations[0].candidate_ids),
        "matrix_self_authorizing": matrix["execution_contract"]["linux_authorized"],
        "registry_authorization_required_for_execute": True,
        "test_authorized": False, "automatic_scientific_pass": None,
    }


def _worker(
    *, device: str, jobs: list[tuple[str, dict[str, Any]]], matrix_path: str,
    cache_a: str, cache_b: str, output_root: str, provenance: dict[str, Any], skip_complete: bool,
) -> list[dict[str, Any]]:
    matrix = _read_json(Path(matrix_path))
    caches = {"A": load_cache(cache_a), "B": load_cache(cache_b)}
    predictions = {item.run_id: item for item in rm3_prediction_run_specs(matrix)}
    calibrations = {item.calibration_id: item for item in rm3_calibration_specs(matrix)}
    records = []
    for kind, raw in jobs:
        job_id = raw["job_id"]
        required = matrix["execution_contract"][
            "required_prediction_artifacts" if kind == "prediction" else "required_calibration_artifacts"
        ]
        directory = Path(output_root) / ("prediction" if kind == "prediction" else "calibration") / job_id
        if _verify_complete(directory, required):
            records.append({"job_id": job_id, "kind": kind, "status": "skipped_complete" if skip_complete else "refused_existing_complete"})
            continue
        if directory.exists():
            records.append({"job_id": job_id, "kind": kind, "status": "refused_existing_incomplete"})
            continue
        try:
            if kind == "prediction":
                record = run_rm3_prediction_training(
                    caches, matrix, predictions[job_id], device=device, output_dir=directory,
                    provenance={**provenance, "device": device},
                )
            else:
                record = run_rm3_calibration(
                    caches, matrix, calibrations[job_id], output_dir=directory,
                    provenance={**provenance, "device": device},
                )
            record["kind"] = kind
        except Exception as exc:
            directory.mkdir(parents=True, exist_ok=True)
            record = {"job_id": job_id, "kind": kind, "status": "failed", "exception_type": type(exc).__name__, "message": str(exc)}
            _atomic_json(directory / "failure.json", record)
        records.append(record)
    return records


def execute_matrix(
    *, matrix_path: Path, cache_paths: dict[str, Path], output_root: Path,
    devices: list[str], skip_complete: bool, require_clean: bool,
) -> list[dict[str, Any]]:
    matrix = _read_json(matrix_path)
    validate_rm3_matrix(matrix)
    parent = _verify_parent(matrix)
    _verify_registry()
    if require_clean and _dirty_paths(output_root):
        raise RuntimeError("RM3 Hermes requires a clean worktree")
    if not devices:
        raise RuntimeError("RM3 needs a non-empty device pool")
    caches = {side: load_cache(path) for side, path in cache_paths.items()}
    expected_source = matrix["data_contract"]["source_sha256"]
    for side in ("A", "B"):
        if caches[side].metadata.get("side") != side or caches[side].metadata.get("source", {}).get("sha256") != expected_source:
            raise RuntimeError(f"RM3 cache pin mismatch for {side}")
    prediction_specs, calibration_specs = rm3_prediction_run_specs(matrix), rm3_calibration_specs(matrix)
    jobs = [
        *(('prediction', {"job_id": item.run_id}) for item in prediction_specs),
        *(('calibration', {"job_id": item.calibration_id}) for item in calibration_specs),
    ]
    provenance = {
        "execution_git_sha": _git_sha(), "matrix_sha256": file_sha256(matrix_path),
        "parent_rm2_audit": parent,
        "cache_sha256": {side: file_sha256(path) for side, path in cache_paths.items()},
        "code_sha256": {path: file_sha256(ROOT / path) for path in PINNED_CODE},
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "test_accessed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "run_manifest.json", {
        **provenance, "devices": devices, "job_ids": [raw["job_id"] for _, raw in jobs],
        "prediction_run_count": 36, "calibration_unit_count": 12,
        "maximum_attempts_per_run": 1, "hermes_role": "execute_frozen_jobs_and_return_artifacts_only",
        "scientific_decision_by_hermes": False, "automatic_scientific_pass": None,
    })
    partitions = [jobs[index::len(devices)] for index in range(len(devices))]
    records = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(
            _worker, device=devices[index], jobs=partition, matrix_path=str(matrix_path),
            cache_a=str(cache_paths["A"]), cache_b=str(cache_paths["B"]),
            output_root=str(output_root), provenance=provenance, skip_complete=skip_complete,
        ) for index, partition in enumerate(partitions) if partition]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: (row["kind"], row.get("run_id", row.get("job_id", ""))))
    _atomic_json(output_root / "matrix_execution_status.json", {
        "records": records, "complete_or_skipped": all(row["status"] in {"complete", "skipped_complete"} for row in records),
        "test_accessed": False, "automatic_scientific_pass": None,
    })
    if all(row["status"] in {"complete", "skipped_complete"} for row in records):
        required_calibration = matrix["execution_contract"]["required_calibration_artifacts"]
        for spec in calibration_specs:
            directory = output_root / "calibration" / spec.calibration_id
            if not _verify_complete(directory, required_calibration):
                raise RuntimeError(f"RM3 calibration ledger failed verification: {spec.calibration_id}")
        summary = summarize_rm3_predictions(
            output_root / "prediction", prediction_specs,
            required_artifacts=matrix["execution_contract"]["required_prediction_artifacts"],
        )
        summary["calibration_unit_count"] = len(calibration_specs)
        summary["scientific_decision"] = None
        _atomic_json(output_root / "summary_validation.json", summary)
        _atomic_json(output_root / "artifact_ledger.json", build_root_artifact_ledger(
            output_root, matrix["execution_contract"]["required_root_artifacts"]
        ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix = _read_json(matrix_path)
    if args.dry_run:
        print(json.dumps(dry_run_payload(matrix), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not args.cache_a or not args.cache_b:
        parser.error("--execute requires --cache-a and --cache-b")
    records = execute_matrix(
        matrix_path=matrix_path, cache_paths={"A": Path(args.cache_a).resolve(), "B": Path(args.cache_b).resolve()},
        output_root=Path(args.output_root).resolve(), devices=[item.strip() for item in args.devices.split(",") if item.strip()],
        skip_complete=args.skip_complete, require_clean=not args.allow_dirty,
    )
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
