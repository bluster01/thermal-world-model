"""Integrity-first RM3 validation reporting without cross-scope composite ranking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import Phase35ProtocolError
from .rm3_contracts import RM3PredictionRunSpec


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_rm3_prediction_run(
    run_dir: Path, required_artifacts: Sequence[str]
) -> dict[str, Any]:
    required = set(required_artifacts)
    if not required.issubset({path.name for path in run_dir.iterdir()}):
        raise Phase35ProtocolError(f"RM3 run artifacts are incomplete: {run_dir.name}")
    ledger = _read_json(run_dir / "artifact_ledger.json")
    expected = required - {"artifact_ledger.json"}
    if set(ledger) != expected:
        raise Phase35ProtocolError(f"RM3 artifact ledger fields changed: {run_dir.name}")
    for name, digest in ledger.items():
        if file_sha256(run_dir / name) != digest:
            raise Phase35ProtocolError(f"RM3 artifact hash mismatch: {run_dir.name}/{name}")
    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "metrics_validation.json")
    if manifest.get("test_accessed") is not False or metrics.get("test_accessed") is not False:
        raise Phase35ProtocolError(f"RM3 validation run accessed test: {run_dir.name}")
    if manifest.get("selector_reporting_disjoint") is not True:
        raise Phase35ProtocolError(f"RM3 selector/reporting anchors overlap: {run_dir.name}")
    if manifest.get("run_id") != run_dir.name or metrics.get("run_id") != run_dir.name:
        raise Phase35ProtocolError(f"RM3 run identity drift: {run_dir.name}")
    return {"manifest": manifest, "metrics": metrics}


def summarize_rm3_predictions(
    output_root: Path,
    specs: Sequence[RM3PredictionRunSpec],
    *,
    required_artifacts: Sequence[str],
) -> dict[str, Any]:
    expected = {spec.run_id: spec for spec in specs}
    by_scope: dict[str, list[dict[str, Any]]] = {}
    candidate_values: dict[str, list[float]] = {}
    records = []
    for run_id, spec in expected.items():
        payload = verify_rm3_prediction_run(output_root / run_id, required_artifacts)
        manifest, metrics = payload["manifest"], payload["metrics"]
        if manifest["run_spec"]["candidate_id"] != spec.candidate_id:
            raise Phase35ProtocolError(f"RM3 candidate drift: {run_id}")
        if metrics.get("output_scope") != spec.output_scope:
            raise Phase35ProtocolError(f"RM3 output scope drift: {run_id}")
        terminal = float(metrics["metrics"]["terminal_mae_c"])
        if not np.isfinite(terminal):
            raise Phase35ProtocolError(f"RM3 non-finite terminal metric: {run_id}")
        candidate_values.setdefault(spec.candidate_id, []).append(terminal)
        records.append({
            "run_id": run_id, "candidate_id": spec.candidate_id,
            "output_scope": spec.output_scope, "fold_id": spec.fold_id,
            "seed": spec.seed, "terminal_mae_c": terminal,
        })
    candidate_summary = []
    for candidate_id, values in sorted(candidate_values.items()):
        spec = next(item for item in specs if item.candidate_id == candidate_id)
        row = {
            "candidate_id": candidate_id, "output_scope": spec.output_scope,
            "run_count": len(values), "terminal_mae_c_mean": float(np.mean(values)),
            "terminal_mae_c_std_across_runs": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        candidate_summary.append(row)
        by_scope.setdefault(spec.output_scope, []).append(row)
    scope_leaderboards = {
        scope: sorted(rows, key=lambda row: (row["terminal_mae_c_mean"], row["candidate_id"]))
        for scope, rows in by_scope.items()
    }
    return {
        "prediction_run_count": len(records),
        "all_runs_complete": len(records) == len(expected),
        "common_descriptive_metric": "terminal_mae_c",
        "candidate_summary": candidate_summary,
        "scope_qualified_leaderboards": scope_leaderboards,
        "cross_output_scope_composite_ranking": None,
        "test_accessed": False,
        "automatic_scientific_pass": None,
        "records": records,
    }


def build_root_artifact_ledger(
    output_root: Path, required_root_artifacts: Sequence[str]
) -> dict[str, str]:
    names = set(required_root_artifacts) - {"artifact_ledger.json"}
    missing = [name for name in sorted(names) if not (output_root / name).is_file()]
    if missing:
        raise Phase35ProtocolError(f"RM3 root artifacts missing: {missing}")
    return {name: file_sha256(output_root / name) for name in sorted(names)}
