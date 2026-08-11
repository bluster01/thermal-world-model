#!/usr/bin/env python3
"""Run the frozen Phase 3.5-MS5 full free+response validation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.multistep_mismatch import (  # noqa: E402
    _assert_clean_source_tree,
    _canonical,
    _sha256,
)
from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.full_training import (  # noqa: E402
    FullCouplingTrainingConfig,
    train_full_synthetic_run,
)
from src.phase35.multistep.synthetic import SyntheticSpec  # noqa: E402
from src.phase35.multistep.training import _json_dump  # noqa: E402


PROTOCOL_VERSION = "phase3.5-ms5-v1"
DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms5_full_coupling_matrix.json"
FROZEN_EXECUTION_PATHS = (
    "configs/phase3_5/ms5_full_coupling_matrix.json",
    "experiments/phase3_5/ms5_full_coupling.py",
    "experiments/phase3_5/summarize_ms5_full_coupling.py",
    "experiments/phase3_5/multistep_mismatch.py",
    "src/phase35/multistep/contracts.py",
    "src/phase35/multistep/model.py",
    "src/phase35/multistep/operators.py",
    "src/phase35/multistep/staging.py",
    "src/phase35/multistep/synthetic.py",
    "src/phase35/multistep/training.py",
    "src/phase35/multistep/full_training.py",
)
FORBIDDEN_TEST_ARTIFACTS = {
    "summary_test.json",
    "metrics_test.json",
    "episode_metrics_test.json",
    "synthetic_test_access_ledger.json",
    "synthetic_test_matrix_access_ledger.json",
}


def _expected_operator() -> dict:
    return {
        "route": "graybox",
        "horizon": 60,
        "context_dim": 4,
        "dt_seconds": 10.0,
        "opening_map": "monotone",
        "poles": 3,
        "latent_dim": 4,
        "hidden_dim": 32,
        "tau_min_seconds": 20.0,
        "tau_max_seconds": 900.0,
        "ode_substeps": 2,
        "closure_scale": 0.02,
        "context_scheduled": True,
        "schedule_log_scale": 0.5,
        "delay_mode": "none",
        "fixed_delay_steps": 0,
        "max_delay_steps": 0,
    }


def _expected_synthetic() -> dict:
    return {
        "train_samples": 1024,
        "validation_samples": 256,
        "test_samples": 256,
        "horizon": 60,
        "context_dim": 4,
        "dt_seconds": 10.0,
        "seed": 20260817,
        "noise_std": 0.02,
        "gain_c_per_pct": -0.10,
        "tau_seconds": [40.0, 70.0, 210.0],
        "truth_regime": "full_coupled_context_scheduled",
        "truth_opening_map": "equal_percentage_r50",
        "context_gain_log_scale": 0.35,
        "context_tau_log_scale": 0.30,
        "input_delay_steps": 0,
        "disturbance_std": 0.0,
        "disturbance_tau_seconds": 0.0,
        "free_trajectory_scale": 1.0,
        "action_context_coupling_pct": 4.0,
    }


def _expected_training() -> dict:
    return asdict(FullCouplingTrainingConfig())


def _expected_gates() -> dict:
    return {
        "oracle_total_clean_nmae_max": 0.10,
        "oracle_free_clean_nmae_max": 0.10,
        "oracle_response_clean_nmae_max": 0.10,
        "eligible_total_clean_nmae_max": 0.10,
        "eligible_free_clean_nmae_max": 0.10,
        "eligible_response_clean_nmae_max": 0.15,
        "response_amplitude_ratio_min": 0.80,
        "response_amplitude_ratio_max": 1.20,
        "staged_total_mae_ratio_max": 1.10,
    }


def _expected_candidates() -> list[dict]:
    return [
        {
            "candidate_id": "ms5_free_only",
            "mode": "free_only",
            "role": "prediction_only_negative_control",
        },
        {
            "candidate_id": "ms5_joint_total",
            "mode": "joint_total",
            "role": "primary_simple_strategy",
        },
        {
            "candidate_id": "ms5_staged_total",
            "mode": "staged_total",
            "role": "fallback_staged_strategy",
        },
        {
            "candidate_id": "ms5_component_oracle",
            "mode": "component_oracle",
            "role": "decomposition_positive_control",
        },
    ]


def load_matrix(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    required = {
        "protocol_version",
        "evidence_scope",
        "seeds",
        "operator",
        "synthetic",
        "training",
        "gates",
        "d3_reference",
        "candidates",
    }
    if set(matrix) != required:
        raise ValueError("MS5 matrix keys differ from the frozen protocol")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_scope": "synthetic_full_free_response_coupling_validation_not_field_causality",
        "seeds": [0, 1, 2],
        "operator": _expected_operator(),
        "synthetic": _expected_synthetic(),
        "training": _expected_training(),
        "gates": _expected_gates(),
        "d3_reference": {
            "path": "results/phase3_5/ms2d_disturbance/summary_validation.json",
            "sha256": "3e9db6df8d9f8e152eba838c3c1af7f851b4b42d6b99e2799be8d6fb505ea668",
            "decision": "validation_stress_pass_no_test_by_budget_decision",
        },
        "candidates": _expected_candidates(),
    }
    for key, value in expected.items():
        if _canonical(matrix.get(key)) != _canonical(value):
            raise ValueError(f"MS5 {key} differs from the frozen protocol")
    reference = ROOT / matrix["d3_reference"]["path"]
    if not reference.is_file() or _sha256(reference) != matrix["d3_reference"]["sha256"]:
        raise ValueError("MS5 frozen D3 reference is missing or changed")
    OperatorConfig.from_mapping(matrix["operator"])
    FullCouplingTrainingConfig(**matrix["training"]).validate()
    SyntheticSpec(
        samples=matrix["synthetic"]["train_samples"],
        **{
            key: value
            for key, value in matrix["synthetic"].items()
            if key not in {"train_samples", "validation_samples", "test_samples"}
        },
    ).validate()
    return matrix


def expand_runs(matrix: dict) -> list[dict]:
    return [
        {**candidate, "seed": int(seed)}
        for candidate in matrix["candidates"]
        for seed in matrix["seeds"]
    ]


def _configs(matrix: dict, smoke: bool):
    operator = OperatorConfig.from_mapping(matrix["operator"])
    full = FullCouplingTrainingConfig(**matrix["training"])
    synthetic_raw = matrix["synthetic"]
    synthetic = SyntheticSpec(
        samples=synthetic_raw["train_samples"],
        **{
            key: value
            for key, value in synthetic_raw.items()
            if key not in {"train_samples", "validation_samples", "test_samples"}
        },
    )
    validation_samples = int(synthetic_raw["validation_samples"])
    if smoke:
        full = replace(
            full,
            batch_size=16,
            epochs=6,
            patience=2,
            stage_a_epochs=2,
            stage_b_epochs=2,
            stage_c_epochs=2,
            stage_patience=2,
        )
        synthetic = replace(synthetic, samples=40, horizon=12)
        operator = replace(operator, horizon=12)
        validation_samples = 40
    return operator, full, synthetic, validation_samples


def _assert_no_test_artifacts(output_root: Path) -> None:
    if not output_root.exists():
        return
    found = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_TEST_ARTIFACTS
    )
    if found:
        raise RuntimeError(f"MS5 validation refuses test artifacts: {found}")


def _existing_compatible(
    output_dir: Path,
    matrix: dict,
    matrix_path: Path,
    run: dict,
    operator: OperatorConfig,
    full: FullCouplingTrainingConfig,
    synthetic: SyntheticSpec,
) -> bool:
    if not output_dir.exists():
        return False
    required = (
        "manifest.json",
        "history.json",
        "metrics_validation.json",
        "episode_metrics_validation.json",
        "checkpoint_best_val.pt",
    )
    if not all((output_dir / name).is_file() for name in required):
        raise RuntimeError(f"partial MS5 output cannot be skipped: {output_dir}")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_spec = replace(synthetic, seed=synthetic.seed + run["seed"] * 1_000_003)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "route_id": run["candidate_id"],
        "seed": run["seed"],
        "training_mode": run["mode"],
        "operator_config": operator.to_dict(),
        "full_training_config": asdict(full),
        "synthetic_spec": asdict(expected_spec),
        "matrix_sha256": _sha256(matrix_path),
        "d3_reference_sha256": matrix["d3_reference"]["sha256"],
        "test_accessed": False,
        "test_authorized": False,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if _canonical(manifest.get(key)) != _canonical(value)
    ]
    if mismatches:
        raise RuntimeError(f"existing MS5 run is incompatible: {mismatches}")
    expected_stages = (
        {
            "stage_a_free_hold",
            "stage_b_response_frozen_free",
            "stage_c_low_lr_joint",
        }
        if run["mode"] == "staged_total"
        else set()
    )
    stage_records = manifest.get("stage_checkpoints")
    if not isinstance(stage_records, list) or {
        item.get("stage") for item in stage_records if isinstance(item, dict)
    } != expected_stages:
        raise RuntimeError("existing MS5 stage checkpoint manifest is incompatible")
    for item in stage_records:
        stage_path = output_dir / item.get("path", "")
        if not stage_path.is_file() or item.get("sha256") != _sha256(stage_path):
            raise RuntimeError("existing MS5 stage checkpoint is missing or changed")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/ms5_full_coupling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-matrix", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix = load_matrix(matrix_path)
    runs = expand_runs(matrix)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_version": matrix["protocol_version"],
                    "evidence_scope": matrix["evidence_scope"],
                    "run_count": len(runs),
                    "runs": runs,
                    "test_authorized": False,
                },
                indent=2,
            )
        )
        return
    if args.execute == args.execute_matrix:
        raise SystemExit("choose exactly one of --execute or --execute-matrix")
    if args.overwrite and args.skip_existing:
        raise SystemExit("choose either --overwrite or --skip-existing")
    if not args.smoke and matrix_path != DEFAULT_MATRIX.resolve():
        raise SystemExit("formal MS5 execution requires the frozen repository matrix")
    if args.execute:
        if args.candidate_id is None or args.seed is None:
            raise SystemExit("--execute requires --candidate-id and --seed")
        selected = [
            run
            for run in runs
            if run["candidate_id"] == args.candidate_id and run["seed"] == args.seed
        ]
        if len(selected) != 1:
            raise SystemExit("requested MS5 candidate/seed is not frozen")
    else:
        if args.candidate_id is not None or args.seed is not None or args.smoke:
            raise SystemExit("matrix execution does not accept candidate/seed/smoke")
        selected = runs
    output_root = Path(args.output_root).resolve()
    _assert_no_test_artifacts(output_root)
    if not args.smoke:
        _assert_clean_source_tree(output_root, allow_generated_outputs=args.skip_existing)
    operator, full, synthetic, validation_samples = _configs(matrix, args.smoke)
    completed = []
    for index, run in enumerate(selected, start=1):
        output_dir = output_root / f"{run['candidate_id']}_s{run['seed']}"
        if args.skip_existing and _existing_compatible(
            output_dir,
            matrix,
            matrix_path,
            run,
            operator,
            full,
            synthetic,
        ):
            completed.append({**run, "status": "skipped_existing"})
            continue
        print(
            f"[{index}/{len(selected)}] MS5 candidate={run['candidate_id']} seed={run['seed']}",
            file=sys.stderr,
            flush=True,
        )
        result = train_full_synthetic_run(
            operator_config=operator,
            full_config=full,
            synthetic_spec=synthetic,
            validation_samples=validation_samples,
            seed=run["seed"],
            mode=run["mode"],
            route_id=run["candidate_id"],
            output_dir=output_dir,
            device=args.device,
            repo_root=ROOT,
            overwrite=args.overwrite,
            protocol_version=PROTOCOL_VERSION,
        )
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "candidate_role": run["role"],
                "matrix_sha256": _sha256(matrix_path),
                "d3_reference_sha256": matrix["d3_reference"]["sha256"],
                "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
            }
        )
        _json_dump(manifest_path, manifest)
        completed.append(
            {
                **run,
                "status": "completed",
                "output_dir": str(result.output_dir),
                "best_epoch": result.best_epoch,
                "validation_total_clean_nmae": result.validation_metrics[
                    "total_clean_nmae"
                ],
                "validation_response_clean_nmae": result.validation_metrics[
                    "response_clean_nmae"
                ],
                "test_accessed": False,
            }
        )
    print(json.dumps({"status": "matrix_completed", "run_count": len(completed), "runs": completed}, indent=2))


if __name__ == "__main__":
    main()
