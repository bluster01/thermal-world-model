#!/usr/bin/env python3
"""Verify and summarize the frozen Gate C RM2 machine batch without judging it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms3r_gatec_rm2_matrix.json"
DEFAULT_RESULTS = ROOT / "results/phase3_5/ms3r_gatec_rm2"

from src.phase35.multistep.gatec_rm2_contracts import rm2_run_specs, validate_rm2_matrix


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def summarize(matrix: dict[str, Any], results: Path) -> dict[str, Any]:
    validate_rm2_matrix(matrix)
    root_manifest_path = results / "run_manifest.json"
    execution_status_path = results / "matrix_execution_status.json"
    if not root_manifest_path.is_file() or not execution_status_path.is_file():
        raise RuntimeError("RM2 root machine manifests are incomplete")
    expected_specs = {spec.run_id: spec for spec in rm2_run_specs(matrix)}
    root_manifest = _read_json(root_manifest_path)
    execution_status = _read_json(execution_status_path)
    if set(root_manifest.get("run_ids", [])) != set(expected_specs):
        raise RuntimeError("RM2 root manifest run IDs changed")
    status_records = execution_status.get("records", [])
    if len(status_records) != 54 or {item.get("run_id") for item in status_records} != set(expected_specs):
        raise RuntimeError("RM2 matrix execution status is incomplete or duplicated")
    required = set(matrix["execution_contract"]["required_run_artifacts"])
    records: list[dict[str, Any]] = []
    complete_checkpoints: list[tuple[str, Path]] = []
    for spec in rm2_run_specs(matrix):
        run_dir = results / spec.run_id
        missing = sorted(name for name in required if not (run_dir / name).is_file())
        hash_mismatches: list[str] = []
        if not missing:
            ledger = _read_json(run_dir / "artifact_ledger.json")
            expected_ledger = required - {"artifact_ledger.json"}
            if set(ledger) != expected_ledger:
                hash_mismatches.append("ledger_fields")
            else:
                hash_mismatches.extend(
                    name
                    for name, digest in ledger.items()
                    if _sha256(run_dir / name) != digest
                )
        complete = not missing and not hash_mismatches
        if complete:
            manifest = _read_json(run_dir / "manifest.json")
            metrics = _read_json(run_dir / "metrics_validation.json")
            if manifest.get("test_accessed") is not False:
                raise RuntimeError(f"RM2 run reports forbidden test access: {spec.run_id}")
            if manifest.get("run_spec", {}).get("run_id") != spec.run_id:
                raise RuntimeError(f"RM2 run-spec identity changed: {spec.run_id}")
            if manifest.get("selector_reporting_disjoint") is not True:
                raise RuntimeError(f"RM2 selector/report anchors overlap: {spec.run_id}")
            complete_checkpoints.append((spec.run_id, run_dir / "checkpoint_best_validation.pt"))
            records.append(
                {
                    "run_id": spec.run_id,
                    "group": spec.group,
                    "candidate_id": spec.candidate_id,
                    "fold_id": spec.fold_id,
                    "seed": spec.seed,
                    "complete": True,
                    "best_update": metrics["best_update"],
                    "optimizer_updates_completed": metrics["optimizer_updates_completed"],
                    "metrics": metrics["metrics"],
                }
            )
        else:
            records.append(
                {
                    "run_id": spec.run_id,
                    "group": spec.group,
                    "candidate_id": spec.candidate_id,
                    "fold_id": spec.fold_id,
                    "seed": spec.seed,
                    "complete": False,
                    "missing": missing,
                    "hash_mismatches": hash_mismatches,
                    "failure_recorded": (run_dir / "failure.json").is_file(),
                }
            )
    archive_path = results / "checkpoints_validation.tar"
    temporary_archive = archive_path.with_suffix(".tar.tmp")
    with tarfile.open(temporary_archive, "w") as archive:
        for run_id, checkpoint in complete_checkpoints:
            archive.add(checkpoint, arcname=f"{run_id}/checkpoint_best_validation.pt")
    os.replace(temporary_archive, archive_path)
    complete_count = sum(record["complete"] for record in records)
    fold_anchor_contract: dict[str, Any] = {}
    for fold_id in ("F0", "F1"):
        fold_manifests = [
            _read_json(results / spec.run_id / "manifest.json")
            for spec in expected_specs.values()
            if spec.fold_id == fold_id and (results / spec.run_id / "manifest.json").is_file()
        ]
        fold_anchor_contract[fold_id] = {
            "manifest_count": len(fold_manifests),
            "stats_sha_count": len({item["stats_anchor_sha256"] for item in fold_manifests}),
            "selector_sha_count": len({item["selector_anchor_sha256"] for item in fold_manifests}),
            "final_sha_count": len({item["final_anchor_sha256"] for item in fold_manifests}),
        }
        if complete_count == 54 and any(
            fold_anchor_contract[fold_id][key] != 1
            for key in ("stats_sha_count", "selector_sha_count", "final_sha_count")
        ):
            raise RuntimeError(f"RM2 {fold_id} candidate/seed anchors drifted")
    payload = {
        "protocol_version": matrix["protocol_version"],
        "scope": matrix["scope"],
        "expected_run_count": 54,
        "complete_run_count": complete_count,
        "incomplete_run_count": 54 - complete_count,
        "matrix_complete": complete_count == 54,
        "fold_anchor_contract": fold_anchor_contract,
        "records": records,
        "checkpoint_archive": {
            "path": "checkpoints_validation.tar",
            "member_count": len(complete_checkpoints),
            "sha256": _sha256(archive_path),
        },
        "test_accessed": False,
        "automatic_scientific_pass": None,
        "supervisor_decision": None,
    }
    _atomic_json(results / "summary_validation.json", payload)
    root_artifacts = {
        name: _sha256(results / name)
        for name in (
            "run_manifest.json",
            "matrix_execution_status.json",
            "summary_validation.json",
            "checkpoints_validation.tar",
        )
        if (results / name).is_file()
    }
    _atomic_json(results / "artifact_ledger.json", root_artifacts)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    matrix = _read_json(Path(args.matrix).resolve())
    payload = summarize(matrix, Path(args.results_root).resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    if not payload["matrix_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
