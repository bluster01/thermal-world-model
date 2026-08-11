#!/usr/bin/env python3
"""Inspect Gate C matrices or run a local known-truth smoke control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.phase35.multistep.gatec_contracts import (
    RESPONSE_ROUTES,
    gatec_run_specs,
    validate_gatec_matrix,
)
from src.phase35.multistep.gatec_synthetic import (
    assert_independent_channel_support,
    evaluate_synthetic_controls,
    generate_gatec_known_truth,
    recover_local_gain,
    train_synthetic_response_operator,
)


DEFAULT_CONFIG = ROOT / "configs/phase3_5/ms3r_gatec_model_matrix.json"
DEFAULT_OUTPUT = ROOT / "results/phase3_5/ms3r_gatec_model_screen_local"
PINNED_SOURCE_FILES = (
    "src/phase35/multistep/gatec_contracts.py",
    "src/phase35/multistep/gatec_data.py",
    "src/phase35/multistep/gatec_model.py",
    "src/phase35/multistep/gatec_training.py",
    "src/phase35/multistep/gatec_synthetic.py",
    "experiments/phase3_5/ms3r_gatec_model_screen.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _verify_parent(matrix: dict[str, Any]) -> dict[str, str]:
    parent = matrix["parent_gate_b"]
    path = ROOT / parent["path"]
    actual = _sha256(path)
    if actual != parent["sha256"]:
        raise RuntimeError("Gate C parent Gate B audit pin changed")
    payload = _read_json(path)
    if payload.get("supervisor_decision", {}).get("label") != parent["required_label"]:
        raise RuntimeError("Gate C parent Gate B label changed")
    return {"path": parent["path"], "sha256": actual, "label": parent["required_label"]}


def dry_run_payload(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_gatec_matrix(matrix)
    _verify_parent(matrix)
    attribution = [asdict(spec) for spec in gatec_run_specs(matrix, "rm1_attribution")]
    operator = [asdict(spec) for spec in gatec_run_specs(matrix, "rm1_operator")]
    training = matrix["training"]
    per_run = int(training["epochs_cap"]) * int(training["steps_per_epoch"])
    run_count = len(attribution) + len(operator)
    return {
        "protocol_version": matrix["protocol_version"],
        "scope": "local_contract_and_synthetic_only",
        "split": matrix["data_contract"]["split"],
        "test_allowed": matrix["data_contract"]["test_allowed"],
        "boundary_modes": matrix["information_flow"]["boundary_modes"],
        "primary_boundary_mode": matrix["selector"]["primary_boundary_mode"],
        "rm0_structural_adapters": sorted(RESPONSE_ROUTES),
        "rm1_attribution": attribution,
        "rm1_operator": operator,
        "rm1_training_run_count": run_count,
        "budget": {
            "optimizer_updates_per_run_cap": per_run,
            "rm1_optimizer_updates_cap": per_run * run_count,
            "epochs_cap": int(training["epochs_cap"]),
            "steps_per_epoch": int(training["steps_per_epoch"]),
            "seed_count": len(matrix["data_contract"]["seeds"]),
            "fold_count": 1,
        },
        "linux_authorized": matrix["execution_contract"]["linux_authorized"],
        "real_training_authorized": matrix["execution_contract"]["real_training_authorized"],
        "automatic_scientific_pass": None,
    }


def _candidate(matrix: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        candidate
        for stage in ("rm1_attribution", "rm1_operator")
        for candidate in matrix[stage]
        if candidate["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or ambiguous Gate C candidate={candidate_id!r}")
    return matches[0]


def run_synthetic_smoke(
    *, matrix_path: Path, matrix: dict[str, Any], candidate_id: str, output: Path
) -> dict[str, Any]:
    validate_gatec_matrix(matrix)
    parent = _verify_parent(matrix)
    candidate = _candidate(matrix, candidate_id)
    batch = generate_gatec_known_truth(seed=1701, n_episodes=48, horizon=60)
    rank = assert_independent_channel_support(batch.opening_dose)
    recovery = recover_local_gain(batch.opening_dose, batch.local_effect)
    numerator = float(np.sqrt(np.sum((recovery.gain - batch.true_gain) ** 2)))
    denominator = float(np.sqrt(np.sum(batch.true_gain**2)))
    structural = evaluate_synthetic_controls()
    diagnostics = {
        "scope": "known_truth_component_recovery_not_real_data",
        "candidate_id": candidate_id,
        "operator_route": candidate["response_route"],
        "known_truth_relative_gain_error": numerator / denominator,
        "known_truth_decay_error": abs(recovery.decay - batch.true_decay),
        "input_condition_number": rank.condition_number,
        "differential_energy_ratio": rank.differential_energy_ratio,
        "independent_channels_supported": rank.independent_channels_supported,
        "structural": asdict(structural),
        "selector_eligible": structural.eligible,
        "automatic_scientific_pass": None,
    }
    manifest = {
        "protocol_version": matrix["protocol_version"],
        "scope": "local_synthetic_smoke_only",
        "candidate_id": candidate_id,
        "operator_route": candidate["response_route"],
        "boundary_mode": matrix["selector"]["primary_boundary_mode"],
        "execution_git_sha": _git_sha(),
        "config_path": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(matrix_path),
        "source_sha256": matrix["data_contract"]["source_sha256"],
        "cache_sha256": None,
        "cache_role": "not_loaded_synthetic_control",
        "parent_gate_b": parent,
        "code_sha256": {path: _sha256(ROOT / path) for path in PINNED_SOURCE_FILES},
        "split": "synthetic",
        "test_accessed": False,
        "real_training_executed": False,
        "linux_authorized": False,
        "selector_eligible": structural.eligible,
        "resource_contract": matrix["training"],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "automatic_scientific_pass": None,
    }
    _atomic_json(output / "run_manifest.json", manifest)
    _atomic_json(output / "synthetic_diagnostics.json", diagnostics)
    _atomic_json(
        output / "artifact_ledger.json",
        {
            "run_manifest.json": _sha256(output / "run_manifest.json"),
            "synthetic_diagnostics.json": _sha256(output / "synthetic_diagnostics.json"),
        },
    )
    return {"manifest": manifest, "diagnostics": diagnostics}


def run_operator_smoke_all(
    *,
    matrix_path: Path,
    matrix: dict[str, Any],
    output: Path,
    seeds: list[int],
    steps: int,
) -> dict[str, Any]:
    validate_gatec_matrix(matrix)
    parent = _verify_parent(matrix)
    if not seeds or steps < 1:
        raise RuntimeError("Gate C operator smoke budget is invalid")
    batch = generate_gatec_known_truth(seed=31, n_episodes=40, horizon=36)
    results = [
        asdict(
            train_synthetic_response_operator(
                route=route,
                batch=batch,
                seed=seed,
                steps=steps,
                learning_rate=0.03,
            )
        )
        for seed in seeds
        for route in sorted(RESPONSE_ROUTES - {"none"})
    ]
    payload = {
        "protocol_version": matrix["protocol_version"],
        "scope": "local_route_specific_known_truth_training",
        "execution_git_sha": _git_sha(),
        "config_path": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(matrix_path),
        "source_sha256": matrix["data_contract"]["source_sha256"],
        "parent_gate_b": parent,
        "code_sha256": {path: _sha256(ROOT / path) for path in PINNED_SOURCE_FILES},
        "synthetic_contract": {
            "generator_seed": 31,
            "n_episodes": 40,
            "horizon_steps": 36,
            "train_fraction": 0.75,
            "operator_seeds": seeds,
            "optimizer_steps": steps,
            "learning_rate": 0.03,
        },
        "results": results,
        "test_accessed": False,
        "real_training_executed": False,
        "linux_authorized": False,
        "automatic_scientific_pass": None,
    }
    artifact = output / "operator_recovery_local.json"
    _atomic_json(artifact, payload)
    _atomic_json(
        output / "operator_artifact_ledger.json",
        {"operator_recovery_local.json": _sha256(artifact)},
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--synthetic-smoke", metavar="CANDIDATE_ID")
    actions.add_argument("--operator-smoke-all", action="store_true")
    actions.add_argument("--real-run", metavar="CANDIDATE_ID")
    parser.add_argument("--operator-seeds", default="7")
    parser.add_argument("--operator-steps", type=int, default=140)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.config).resolve()
    matrix = _read_json(matrix_path)
    if args.real_run:
        validate_gatec_matrix(matrix)
        if not matrix["execution_contract"]["real_training_authorized"]:
            raise SystemExit("Gate C real training is not authorized")
        raise SystemExit("Gate C real runner has not been released")
    if args.dry_run:
        payload = dry_run_payload(matrix)
    elif args.operator_smoke_all:
        try:
            seeds = [int(value) for value in args.operator_seeds.split(",") if value.strip()]
        except ValueError as exc:
            raise SystemExit("Gate C operator seeds must be comma-separated integers") from exc
        payload = run_operator_smoke_all(
            matrix_path=matrix_path,
            matrix=matrix,
            output=Path(args.output_dir).resolve(),
            seeds=seeds,
            steps=args.operator_steps,
        )
    else:
        payload = run_synthetic_smoke(
            matrix_path=matrix_path,
            matrix=matrix,
            candidate_id=args.synthetic_smoke,
            output=Path(args.output_dir).resolve(),
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
