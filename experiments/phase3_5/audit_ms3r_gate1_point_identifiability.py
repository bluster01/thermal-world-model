#!/usr/bin/env python3
"""Deterministically replay MS3-R Gate A artifacts before supervisor interpretation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.phase35.data import load_cache
from src.phase35.ms3r import run_gate1_analysis


DEFAULT_RESULTS = ROOT / "results/phase3_5/ms3r_gate1_point_identifiability"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def run_audit(
    results_root: Path, cache_paths: dict[str, Path]
) -> dict[str, Any]:
    manifest = _read_json(results_root / "run_manifest.json")
    summary = _read_json(results_root / "summary_validation.json")
    ledger = _read_json(results_root / "artifact_ledger.json")
    config_path = (ROOT / manifest["config_path"]).resolve()
    if not config_path.is_file():
        # Unit tests may place the frozen config outside the repository.
        config_path = Path(manifest["config_path"]).resolve()
    config = _read_json(config_path)
    artifact_hashes_exact = all(
        (results_root / name).is_file() and _sha256(results_root / name) == digest
        for name, digest in ledger.items()
    )
    caches = {side: load_cache(cache_paths[side]) for side in ("A", "B")}
    replay, arrays = run_gate1_analysis(caches, config)
    stored_arrays = np.load(results_root / "analysis_arrays_validation.npz")
    array_checks = {
        name: bool(name in stored_arrays.files and np.array_equal(stored_arrays[name], value, equal_nan=True))
        for name, value in arrays.items()
    }
    stored_arrays.close()
    replay_fields = (
        "protocol_version",
        "evidence_scope",
        "branch_semantics",
        "action_information_audit",
        "point_quality",
        "analysis_support",
        "path_diagnostics",
        "rank_diagnostics",
        "automatic_scientific_pass",
        "test_accessed",
        "training_executed",
        "claim_boundary",
    )
    field_checks = {field: summary.get(field) == replay.get(field) for field in replay_fields}
    contract_checks = {
        "summary_manifest_exact": summary.get("manifest") == manifest,
        "config_sha256_exact": _sha256(config_path) == manifest["config_sha256"],
        "validation_only": manifest.get("split") == "validation",
        "test_not_accessed": manifest.get("test_accessed") is False and summary.get("test_accessed") is False,
        "training_not_executed": manifest.get("training_executed") is False and summary.get("training_executed") is False,
        "no_automatic_scientific_pass": manifest.get("automatic_scientific_pass") is None and summary.get("automatic_scientific_pass") is None,
        "artifact_hashes_exact": artifact_hashes_exact,
        "all_arrays_exact": all(array_checks.values()),
        "all_replay_fields_exact": all(field_checks.values()),
    }
    passes = bool(all(contract_checks.values()))
    return {
        "protocol_version": config["protocol_version"],
        "contract_checks": contract_checks,
        "array_checks": array_checks,
        "replay_field_checks": field_checks,
        "passes": passes,
        "test_accessed": False,
        "scientific_decision": None,
        "claim_boundary": "Deterministic artifact replay only; the supervisor must separately interpret timing, placebo, excitation, and rank support.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cache-a", required=True)
    parser.add_argument("--cache-b", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = Path(args.results_root).resolve()
    output = Path(args.output).resolve() if args.output else results / "local_replay_validation.json"
    audit = run_audit(
        results,
        {"A": Path(args.cache_a).resolve(), "B": Path(args.cache_b).resolve()},
    )
    _atomic_json(output, audit)
    print(
        json.dumps(
            {
                "output": str(output),
                "passes": audit["passes"],
                "scientific_decision": None,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not audit["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
