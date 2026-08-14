#!/usr/bin/env python3
"""Frozen RM3-B1 22-unit executor; dry-run is local, training needs exact registry authorization."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.data import load_cache
from src.phase35.multistep.rm3_reporting import build_root_artifact_ledger, file_sha256
from src.phase35.multistep.rm3av_execution import run_rm3av_training
from src.phase35.multistep.rm3b_contracts import rm3b_run_specs, validate_rm3b_matrix


MATRIX = ROOT / "configs/phase3_5/ms3r_rm3b_matrix.json"
OUTPUT = ROOT / "results/phase3_5/ms3r_rm3b"
REGISTRY = ROOT / "configs/phase3_5/experiment_registry.json"
PINNED_CODE = (
    "src/phase35/data.py",
    "src/phase35/schema.py",
    "src/phase35/multistep/gatec_data.py",
    "src/phase35/multistep/rm3b_contracts.py",
    "src/phase35/multistep/rm3av_contracts.py",
    "src/phase35/multistep/rm3av_model.py",
    "src/phase35/multistep/rm3av_training.py",
    "src/phase35/multistep/rm3av_diagnostics.py",
    "src/phase35/multistep/rm3av_execution.py",
    "src/phase35/multistep/rm3_prediction.py",
    "src/phase35/multistep/rm3_joint_model.py",
    "src/phase35/multistep/rm3_training.py",
    "src/phase35/multistep/rm3_reporting.py",
    "src/phase35/multistep/gatec_model.py",
    "experiments/phase3_5/ms3r_rm3b_train.py",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, path)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_registry() -> None:
    registry = _read(REGISTRY)
    if registry.get("active_gate") != "ms3_r" or registry.get("linux_authorized_gate") != "ms3_r":
        raise RuntimeError("RM3-B1 requires active and Linux-authorized gate ms3_r")
    experiment = registry.get("experiments", {}).get("ms3_r", {})
    if experiment.get("status") != "ready_for_linux":
        raise RuntimeError("RM3-B1 requires ms3_r.status=ready_for_linux")
    if experiment.get("decision", {}).get("authorized_batch") != "RM3-B1":
        raise RuntimeError("RM3-B1 registry authorized_batch must equal RM3-B1")


def _dirty_paths(output: Path) -> list[str]:
    raw = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    try:
        allowed = str(output.resolve().relative_to(ROOT)).replace("\\", "/").rstrip("/") + "/"
    except ValueError:
        allowed = ""
    return [
        line for line in raw.splitlines()
        if line.strip() and (not allowed or not line[3:].replace("\\", "/").startswith(allowed))
    ]


def _require_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "RM3-B1 refuses a non-empty output root; return partial artifacts instead of retrying"
        )


def dry_run_payload(matrix: dict[str, Any], *, repo_root: Path = ROOT) -> dict[str, Any]:
    validate_rm3b_matrix(matrix, repo_root=repo_root)
    specs = rm3b_run_specs(matrix)
    return {
        "protocol_version": matrix["protocol_version"],
        "candidate_count": len({spec.candidate_id for spec in specs}),
        "training_unit_count": len(specs),
        "folds": sorted({spec.fold_id for spec in specs}),
        "seeds": sorted({spec.seed for spec in specs}),
        "optimizer_updates_cap": sorted({spec.optimizer_updates_cap for spec in specs}),
        "pairs": matrix["pair_contract"]["pairs"],
        "run_ids": [spec.run_id for spec in specs],
        "matrix_self_authorizing": matrix["execution_contract"]["linux_authorized"],
        "registry_authorization_required_for_execute": True,
        "test_authorized": False,
        "rm3b2_authorized": False,
        "automatic_scientific_pass": None,
    }


def _worker(
    device: str,
    specs: list[Any],
    matrix_path: str,
    cache_a: str,
    cache_b: str,
    output: str,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    matrix = _read(Path(matrix_path))
    caches = {"A": load_cache(cache_a), "B": load_cache(cache_b)}
    records = []
    for spec in specs:
        directory = Path(output) / spec.run_id
        if directory.exists():
            records.append({"run_id": spec.run_id, "status": "refused_existing"})
            continue
        try:
            records.append(
                run_rm3av_training(
                    caches, matrix, spec, device=device, output_dir=directory,
                    provenance={**provenance, "device": device},
                    template_candidate_id=spec.template_candidate_id,
                )
            )
        except Exception as exc:
            directory.mkdir(parents=True, exist_ok=True)
            record = {
                "run_id": spec.run_id, "status": "failed", "attempt": 1,
                "exception_type": type(exc).__name__, "message": str(exc),
            }
            _atomic(directory / "failure.json", record)
            records.append(record)
            break
    return records


def _verify_run(output: Path, spec: Any, required: list[str]) -> dict[str, Any]:
    run_dir = output / spec.run_id
    if {path.name for path in run_dir.iterdir()} != set(required):
        raise RuntimeError(f"RM3-B1 artifact set changed for {spec.run_id}")
    ledger = _read(run_dir / "artifact_ledger.json")
    if set(ledger) != set(required) - {"artifact_ledger.json"}:
        raise RuntimeError(f"RM3-B1 ledger fields changed for {spec.run_id}")
    for name, digest in ledger.items():
        if file_sha256(run_dir / name) != digest:
            raise RuntimeError(f"RM3-B1 ledger hash mismatch for {spec.run_id}/{name}")
    manifest = _read(run_dir / "manifest.json")
    metrics = _read(run_dir / "metrics_validation.json")
    diagnostics = _read(run_dir / "diagnostics_validation.json")
    if any(item.get("test_accessed") is not False for item in (manifest, metrics, diagnostics)):
        raise RuntimeError(f"RM3-B1 validation artifact accessed test for {spec.run_id}")
    if manifest.get("run_id") != spec.run_id or metrics.get("candidate_id") != spec.candidate_id:
        raise RuntimeError(f"RM3-B1 run identity drift for {spec.run_id}")
    if metrics.get("template_candidate_id") != spec.template_candidate_id:
        raise RuntimeError(f"RM3-B1 template identity drift for {spec.run_id}")
    return {
        "run_id": spec.run_id, "candidate_id": spec.candidate_id,
        "template_candidate_id": spec.template_candidate_id,
        "pair_anchor_id": spec.pair_anchor_id, "fold_id": spec.fold_id,
        "terminal_mae_c": float(metrics["metrics"]["terminal_mae_c"]),
        "local_mae_c": float(metrics["metrics"]["local_mae_c"]),
        "valve_mae": float(metrics["metrics"]["valve_mae"]),
        "tin_mae_c": float(metrics["metrics"]["tin_mae_c"]),
        "optimizer_updates_completed": int(metrics["optimizer_updates_completed"]),
    }


def _summarize(output: Path, specs: tuple[Any, ...], required: list[str]) -> dict[str, Any]:
    records = [_verify_run(output, spec, required) for spec in specs]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_candidate.setdefault(record["candidate_id"], []).append(record)
    candidate_summary = []
    for candidate_id, values in sorted(by_candidate.items()):
        candidate_summary.append({
            "candidate_id": candidate_id,
            "template_candidate_id": values[0]["template_candidate_id"],
            "fold_count": len(values),
            "terminal_mae_c_mean": float(np.mean([value["terminal_mae_c"] for value in values])),
            "local_mae_c_mean": float(np.mean([value["local_mae_c"] for value in values])),
            "valve_mae_mean": float(np.mean([value["valve_mae"] for value in values])),
            "tin_mae_c_mean": float(np.mean([value["tin_mae_c"] for value in values])),
        })
    return {
        "training_unit_count": len(records), "all_runs_complete": len(records) == 22,
        "candidate_summary": candidate_summary, "pair_verdicts": None,
        "composite_ranking": None, "model_champion": None,
        "manual_supervisor_verdict_required": True,
        "test_accessed": False, "automatic_scientific_pass": None,
        "records": records,
    }


def execute(
    matrix_path: Path,
    cache_a: Path,
    cache_b: Path,
    output: Path,
    devices: list[str],
    *,
    require_clean: bool = True,
) -> list[dict[str, Any]]:
    matrix = _read(matrix_path)
    validate_rm3b_matrix(matrix, repo_root=ROOT)
    _verify_registry()
    if require_clean and _dirty_paths(output):
        raise RuntimeError("RM3-B1 requires a clean worktree outside its output root")
    _require_empty_output(output)
    if not devices:
        raise RuntimeError("RM3-B1 requires a non-empty device pool")
    caches = {"A": load_cache(cache_a), "B": load_cache(cache_b)}
    expected = matrix["data_contract"]["source_sha256"]
    for side in ("A", "B"):
        if caches[side].metadata.get("side") != side or caches[side].metadata.get("source", {}).get("sha256") != expected:
            raise RuntimeError(f"RM3-B1 cache pin mismatch for {side}")
    specs = rm3b_run_specs(matrix)
    provenance = {
        "execution_git_sha": _git_sha(), "matrix_sha256": file_sha256(matrix_path),
        "cache_sha256": {"A": file_sha256(cache_a), "B": file_sha256(cache_b)},
        "code_sha256": {path: file_sha256(ROOT / path) for path in PINNED_CODE},
        "parent_supervisor_decision": matrix["parent_supervisor_decision"],
        "parent_design": matrix["parent_design"], "test_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic(output / "run_manifest.json", {
        **provenance, "devices": devices, "run_ids": [spec.run_id for spec in specs],
        "maximum_attempts_per_run": 1,
        "hermes_role": "execute_frozen_22_units_once_and_return_raw_artifacts",
        "hermes_may_edit_code_or_configs": False, "hermes_may_retry_or_tune": False,
        "hermes_may_access_test": False, "hermes_may_assign_pair_verdicts": False,
        "hermes_may_generate_rm3b2": False, "automatic_scientific_pass": None,
    })
    partitions = [list(specs[index::len(devices)]) for index in range(len(devices))]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(
                _worker, device, partition, str(matrix_path), str(cache_a), str(cache_b),
                str(output), provenance,
            )
            for device, partition in zip(devices, partitions) if partition
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: row["run_id"])
    complete = len(records) == 22 and all(row["status"] == "complete" for row in records)
    _atomic(output / "matrix_execution_status.json", {
        "records": records, "all_complete": complete, "test_accessed": False,
        "automatic_scientific_pass": None,
    })
    if complete:
        _atomic(
            output / "summary_validation.json",
            _summarize(output, specs, matrix["execution_contract"]["required_run_artifacts"]),
        )
        _atomic(
            output / "artifact_ledger.json",
            build_root_artifact_ledger(output, matrix["execution_contract"]["required_root_artifacts"]),
        )
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
    matrix_path = Path(args.matrix).resolve()
    if args.dry_run:
        print(json.dumps(dry_run_payload(_read(matrix_path)), ensure_ascii=False, indent=2))
        return
    if not args.cache_a or not args.cache_b:
        parser.error("--execute requires --cache-a and --cache-b")
    records = execute(
        matrix_path, Path(args.cache_a).resolve(), Path(args.cache_b).resolve(),
        Path(args.output_root).resolve(),
        [item.strip() for item in args.devices.split(",") if item.strip()],
    )
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2))
    if len(records) != 22 or any(record["status"] != "complete" for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
