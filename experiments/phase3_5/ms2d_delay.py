#!/usr/bin/env python3
"""Run the frozen Phase 3.5-MS2-D1 pure-delay validation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.multistep_mismatch import (  # noqa: E402
    _assert_clean_source_tree,
    _canonical,
    _current_git_sha,
    _sha256,
)
from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.synthetic import SyntheticSpec  # noqa: E402
from src.phase35.multistep.training import (  # noqa: E402
    TrainingConfig,
    _json_dump,
    train_synthetic_run,
)


PROTOCOL_VERSION = "phase3.5-ms2d-d1-v1"
DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms2d_delay_matrix.json"
FROZEN_EXECUTION_PATHS = (
    "configs/phase3_5/ms2d_delay_matrix.json",
    "experiments/phase3_5/ms2d_delay.py",
    "src/phase35/multistep/contracts.py",
    "src/phase35/multistep/operators.py",
    "src/phase35/multistep/synthetic.py",
    "src/phase35/multistep/training.py",
)


def load_matrix(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    required = {
        "protocol_version",
        "evidence_scope",
        "seeds",
        "operator_defaults",
        "synthetic_defaults",
        "training",
        "gates",
        "regimes",
    }
    missing = sorted(required - set(matrix))
    if missing:
        raise ValueError(f"MS2-D1 matrix missing keys: {missing}")
    if matrix["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"MS2-D1 runner accepts only {PROTOCOL_VERSION}")
    if matrix["evidence_scope"] != "synthetic_delay_pressure_validation":
        raise ValueError("MS2-D1 evidence scope is not frozen")
    candidates = [
        candidate
        for regime in matrix["regimes"]
        for candidate in regime["candidates"]
    ]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("MS2-D1 candidate IDs must be globally unique")
    if len(matrix["regimes"]) != 1 or len(candidates) != 6:
        raise ValueError("MS2-D1 freezes one regime with six candidates")
    if [int(seed) for seed in matrix["seeds"]] != [0, 1, 2]:
        raise ValueError("MS2-D1 freezes seeds [0,1,2]")
    if sum(candidate.get("role") == "primary_model" for candidate in candidates) != 1:
        raise ValueError("MS2-D1 requires exactly one primary model")
    expected_candidates = {
        "d1_g2_no_delay": {
            "role": "primary_ablation",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "delay_mode": "none",
        },
        "d1_g2_learned_delay": {
            "role": "primary_model",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "delay_mode": "learned",
            "max_delay_steps": 4,
        },
        "d1_g2_oracle_delay": {
            "role": "positive_control",
            "route": "graybox",
            "opening_map": "equal_percentage_r50",
            "context_scheduled": True,
            "delay_mode": "fixed",
            "fixed_delay_steps": 2,
            "max_delay_steps": 4,
        },
        "d1_k4_monotone": {
            "role": "secondary_representation",
            "route": "koopman",
            "opening_map": "monotone",
        },
        "d1_pi_monotone": {
            "role": "secondary_representation",
            "route": "pi_ode",
            "opening_map": "monotone",
        },
        "d1_deeponet": {
            "role": "secondary_representation",
            "route": "deeponet",
            "opening_map": "identity",
        },
    }
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if set(by_id) != set(expected_candidates):
        raise ValueError("MS2-D1 candidate IDs differ from the frozen protocol")
    for candidate_id, expected in expected_candidates.items():
        candidate = by_id[candidate_id]
        for key, expected_value in expected.items():
            observed = candidate.get(key, matrix["operator_defaults"].get(key))
            if observed != expected_value:
                raise ValueError(
                    f"MS2-D1 {candidate_id}.{key} differs from the frozen protocol"
                )
    truth = matrix["synthetic_defaults"]
    expected_truth = {
        "truth_regime": "delayed_context_scheduled",
        "truth_opening_map": "equal_percentage_r50",
        "input_delay_steps": 2,
        "dt_seconds": 10.0,
    }
    if any(truth.get(key) != value for key, value in expected_truth.items()):
        raise ValueError("MS2-D1 truth differs from the frozen 20 s delay protocol")
    expected_gates = {
        "oracle_clean_nmae_max": 0.05,
        "learned_delay_relative_improvement_min": 0.20,
        "delay_identification_error_steps_max": 1.0,
        "delay_truth_neighborhood_mass_min": 0.80,
    }
    if matrix["gates"] != expected_gates:
        raise ValueError("MS2-D1 gates differ from the frozen protocol")
    return matrix


def expand_runs(matrix: dict) -> list[dict]:
    return [
        {
            "regime_id": regime["regime_id"],
            "candidate_id": candidate["candidate_id"],
            "role": candidate["role"],
            "route": candidate["route"],
            "seed": int(seed),
        }
        for regime in matrix["regimes"]
        for candidate in regime["candidates"]
        for seed in matrix["seeds"]
    ]


def _select(matrix: dict, candidate_id: str) -> tuple[dict, dict]:
    matches = [
        (regime, candidate)
        for regime in matrix["regimes"]
        for candidate in regime["candidates"]
        if candidate["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate_id={candidate_id!r} is not uniquely defined")
    return matches[0]


def _build_configs(matrix: dict, regime: dict, candidate: dict, smoke: bool):
    operator_values = dict(matrix["operator_defaults"])
    operator_values.update(
        {
            key: value
            for key, value in candidate.items()
            if key not in {"candidate_id", "role"}
        }
    )
    operator = OperatorConfig.from_mapping(operator_values)
    training_values = dict(matrix["training"])
    synthetic_values = dict(matrix["synthetic_defaults"])
    synthetic_values.update(regime["synthetic"])
    validation_samples = int(synthetic_values.pop("validation_samples"))
    synthetic_values.pop("test_samples")
    synthetic_values["samples"] = int(synthetic_values.pop("train_samples"))
    if smoke:
        training_values.update(batch_size=16, epochs=2, patience=2)
        synthetic_values["samples"] = 64
        validation_samples = 32
    return (
        operator,
        TrainingConfig(**training_values),
        SyntheticSpec(**synthetic_values),
        validation_samples,
    )


def _augment_manifest(
    output_dir: Path,
    matrix: dict,
    matrix_path: Path,
    regime: dict,
    candidate: dict,
) -> None:
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        evidence_scope=matrix["evidence_scope"],
        regime_id=regime["regime_id"],
        candidate_role=candidate["role"],
        matrix_sha256=_sha256(matrix_path),
        frozen_execution_paths=list(FROZEN_EXECUTION_PATHS),
        test_authorized=False,
    )
    _json_dump(path, manifest)


def _existing_run_is_compatible(
    output_dir: Path,
    matrix: dict,
    matrix_path: Path,
    regime: dict,
    candidate: dict,
    seed: int,
    operator: OperatorConfig,
    training: TrainingConfig,
    synthetic: SyntheticSpec,
) -> bool:
    required = [
        output_dir / "manifest.json",
        output_dir / "checkpoint_best_val.pt",
        output_dir / "metrics_validation.json",
        output_dir / "history.json",
    ]
    present = [path.is_file() for path in required]
    if not any(present):
        return False
    if not all(present):
        missing = [path.name for path, exists in zip(required, present) if not exists]
        raise RuntimeError(f"incomplete existing MS2-D1 run {output_dir}; missing={missing}")
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    expected_spec = replace(synthetic, seed=synthetic.seed + seed * 1_000_003)
    expected = {
        "protocol_version": matrix["protocol_version"],
        "evidence_scope": matrix["evidence_scope"],
        "regime_id": regime["regime_id"],
        "candidate_role": candidate["role"],
        "route_id": candidate["candidate_id"],
        "seed": seed,
        "operator_config": operator.to_dict(),
        "training_config": asdict(training),
        "synthetic_spec": asdict(expected_spec),
        "git_sha": _current_git_sha(),
        "matrix_sha256": _sha256(matrix_path),
        "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
        "test_accessed": False,
        "test_authorized": False,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if _canonical(manifest.get(key)) != _canonical(value)
    ]
    if manifest.get("checkpoint_sha256") != _sha256(required[1]):
        mismatches.append("checkpoint_sha256")
    history = json.loads(required[3].read_text(encoding="utf-8"))
    best_epoch = manifest.get("best_epoch")
    if not isinstance(history, list) or not history:
        mismatches.append("history")
    elif not isinstance(best_epoch, int) or not 1 <= best_epoch <= len(history):
        mismatches.append("best_epoch")
    if mismatches:
        raise RuntimeError(
            f"existing MS2-D1 run mismatch {output_dir}: {sorted(set(mismatches))}"
        )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/ms2d_delay")
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
    for regime in matrix["regimes"]:
        for candidate in regime["candidates"]:
            _build_configs(matrix, regime, candidate, False)
    runs = expand_runs(matrix)
    if args.dry_run or not (args.execute or args.execute_matrix):
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
        raise SystemExit("formal MS2-D1 execution requires the frozen repository matrix")
    if args.execute:
        if args.candidate_id is None or args.seed is None:
            raise SystemExit("--execute requires --candidate-id and --seed")
        selected = [{"candidate_id": args.candidate_id, "seed": args.seed}]
    else:
        if args.candidate_id is not None or args.seed is not None or args.smoke:
            raise SystemExit("matrix execution does not accept candidate/seed/smoke")
        selected = runs
    output_root = Path(args.output_root).resolve()
    if not args.smoke:
        _assert_clean_source_tree(output_root, allow_generated_outputs=args.skip_existing)
    allowed_seeds = {int(seed) for seed in matrix["seeds"]}
    completed = []
    for index, run in enumerate(selected, start=1):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        if seed not in allowed_seeds:
            raise SystemExit(f"seed={seed} is not frozen")
        regime, candidate = _select(matrix, candidate_id)
        operator, training, synthetic, validation_samples = _build_configs(
            matrix, regime, candidate, args.smoke
        )
        output_dir = output_root / f"ms2d_{candidate_id}_s{seed}"
        if args.skip_existing and _existing_run_is_compatible(
            output_dir,
            matrix,
            matrix_path,
            regime,
            candidate,
            seed,
            operator,
            training,
            synthetic,
        ):
            completed.append(
                {"candidate_id": candidate_id, "seed": seed, "status": "skipped_existing"}
            )
            continue
        print(
            f"[{index}/{len(selected)}] candidate={candidate_id} seed={seed}",
            file=sys.stderr,
            flush=True,
        )
        result = train_synthetic_run(
            operator_config=operator,
            training_config=training,
            synthetic_spec=synthetic,
            validation_samples=validation_samples,
            seed=seed,
            route_id=candidate_id,
            output_dir=output_dir,
            device=args.device,
            repo_root=ROOT,
            overwrite=args.overwrite,
            protocol_version=matrix["protocol_version"],
        )
        _augment_manifest(output_dir, matrix, matrix_path, regime, candidate)
        completed.append(
            {
                "regime_id": regime["regime_id"],
                "candidate_id": candidate_id,
                "seed": seed,
                "status": "completed",
                "output_dir": str(result.output_dir),
                "checkpoint_sha256": _sha256(result.checkpoint),
                "best_epoch": result.best_epoch,
                "validation_clean_effect_nmae": result.validation_metrics[
                    "clean_effect_nmae"
                ],
                "test_accessed": False,
            }
        )
    payload = completed[0] if len(completed) == 1 else {
        "status": "matrix_completed",
        "run_count": len(completed),
        "runs": completed,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
