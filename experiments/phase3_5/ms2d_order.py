#!/usr/bin/env python3
"""Run the frozen Phase 3.5-MS2-D2 inertial-order validation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch


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
from src.phase35.multistep.staging import environment_payload  # noqa: E402
from src.phase35.multistep.synthetic import SyntheticSpec  # noqa: E402
from src.phase35.multistep.training import (  # noqa: E402
    TrainingConfig,
    _json_dump,
    train_synthetic_run,
)


PROTOCOL_VERSION = "phase3.5-ms2d-d2-v1"
DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms2d_order_matrix.json"
FROZEN_EXECUTION_PATHS = (
    "configs/phase3_5/ms2d_order_matrix.json",
    "experiments/phase3_5/ms2d_order.py",
    "experiments/phase3_5/summarize_ms2d_order.py",
    "src/phase35/multistep/contracts.py",
    "src/phase35/multistep/operators.py",
    "src/phase35/multistep/staging.py",
    "src/phase35/multistep/synthetic.py",
    "src/phase35/multistep/training.py",
)
FORBIDDEN_TEST_ARTIFACT_NAMES = {
    "summary_test.json",
    "metrics_test.json",
    "episode_metrics_test.json",
    "synthetic_test_access_ledger.json",
    "synthetic_test_matrix_access_ledger.json",
}


def _assert_no_test_artifacts(output_root: Path) -> None:
    if not output_root.exists():
        return
    found = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_TEST_ARTIFACT_NAMES
    )
    if found:
        raise RuntimeError(f"MS2-D2 validation refuses test artifacts: {found}")


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
        raise ValueError(f"MS2-D2 matrix missing keys: {missing}")
    if matrix["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"MS2-D2 runner accepts only frozen {PROTOCOL_VERSION}")
    if matrix["evidence_scope"] != (
        "synthetic_order_pressure_validation_not_field_causality"
    ):
        raise ValueError("MS2-D2 evidence scope differs from the frozen protocol")
    if [int(seed) for seed in matrix["seeds"]] != [0, 1, 2]:
        raise ValueError("MS2-D2 freezes seeds [0,1,2]")

    candidates = [
        candidate
        for regime in matrix["regimes"]
        for candidate in regime["candidates"]
    ]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(matrix["regimes"]) != 1 or len(candidates) != 7:
        raise ValueError("MS2-D2 freezes one regime with seven candidates")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("MS2-D2 candidate IDs must be globally unique")
    if sum(candidate.get("role") == "primary_model" for candidate in candidates) != 1:
        raise ValueError("MS2-D2 requires exactly one primary model")

    expected_operator_defaults = {
        "horizon": 60,
        "context_dim": 4,
        "dt_seconds": 10.0,
        "opening_map": "identity",
        "poles": 2,
        "latent_dim": 4,
        "hidden_dim": 32,
        "tau_min_seconds": 20.0,
        "tau_max_seconds": 900.0,
        "ode_substeps": 2,
        "closure_scale": 0.02,
        "context_scheduled": False,
        "schedule_log_scale": 0.5,
        "delay_mode": "none",
        "fixed_delay_steps": 0,
        "max_delay_steps": 0,
    }
    expected_synthetic_defaults = {
        "train_samples": 1024,
        "validation_samples": 256,
        "test_samples": 256,
        "horizon": 60,
        "context_dim": 4,
        "dt_seconds": 10.0,
        "seed": 20260813,
        "noise_std": 0.02,
        "gain_c_per_pct": -0.10,
        "tau_seconds": [40.0, 70.0, 210.0],
        "truth_regime": "context_scheduled",
        "truth_opening_map": "equal_percentage_r50",
        "context_gain_log_scale": 0.35,
        "context_tau_log_scale": 0.30,
        "input_delay_steps": 0,
    }
    expected_training = {
        "batch_size": 64,
        "epochs": 300,
        "patience": 30,
        "learning_rate": 0.002,
        "weight_decay": 0.000001,
        "physics_weight": 0.01,
        "gradient_clip": 1.0,
    }
    if matrix["operator_defaults"] != expected_operator_defaults:
        raise ValueError("MS2-D2 operator defaults differ from the frozen protocol")
    if matrix["synthetic_defaults"] != expected_synthetic_defaults:
        raise ValueError("MS2-D2 synthetic defaults differ from the frozen protocol")
    if matrix["training"] != expected_training:
        raise ValueError("MS2-D2 training budget differs from the frozen protocol")
    if (
        matrix["regimes"][0].get("regime_id")
        != "third_order_r50_context_scheduled"
        or matrix["regimes"][0].get("synthetic") != {}
    ):
        raise ValueError("MS2-D2 regime differs from the frozen protocol")

    expected_candidates = {
        "d2_g2_two_pole": {
            "role": "primary_ablation",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 2,
            "delay_mode": "none",
        },
        "d2_g3_three_pole": {
            "role": "primary_model",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 3,
            "delay_mode": "none",
        },
        "d2_g3_oracle_structure": {
            "role": "positive_control",
            "route": "graybox",
            "opening_map": "equal_percentage_r50",
            "context_scheduled": True,
            "poles": 3,
            "delay_mode": "none",
        },
        "d2_g2_delay_compensation": {
            "role": "alternative_mechanism_diagnostic",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 2,
            "delay_mode": "learned",
            "max_delay_steps": 4,
        },
        "d2_k4_monotone": {
            "role": "secondary_representation",
            "route": "koopman",
            "opening_map": "monotone",
            "latent_dim": 4,
        },
        "d2_pi_monotone": {
            "role": "secondary_representation",
            "route": "pi_ode",
            "opening_map": "monotone",
        },
        "d2_deeponet": {
            "role": "secondary_representation",
            "route": "deeponet",
            "opening_map": "identity",
            "latent_dim": 8,
            "hidden_dim": 32,
        },
    }
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if set(by_id) != set(expected_candidates):
        raise ValueError("MS2-D2 candidate IDs differ from the frozen protocol")
    defaults = matrix["operator_defaults"]
    for candidate_id, expected in expected_candidates.items():
        candidate = by_id[candidate_id]
        for key, expected_value in expected.items():
            observed = candidate.get(key, defaults.get(key))
            if observed != expected_value:
                raise ValueError(
                    f"MS2-D2 {candidate_id}.{key} differs from the frozen protocol"
                )

    expected_candidate_payloads = {
        "d2_g2_two_pole": {
            "candidate_id": "d2_g2_two_pole",
            "role": "primary_ablation",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 2,
        },
        "d2_g3_three_pole": {
            "candidate_id": "d2_g3_three_pole",
            "role": "primary_model",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 3,
        },
        "d2_g3_oracle_structure": {
            "candidate_id": "d2_g3_oracle_structure",
            "role": "positive_control",
            "route": "graybox",
            "opening_map": "equal_percentage_r50",
            "context_scheduled": True,
            "poles": 3,
        },
        "d2_g2_delay_compensation": {
            "candidate_id": "d2_g2_delay_compensation",
            "role": "alternative_mechanism_diagnostic",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 2,
            "delay_mode": "learned",
            "max_delay_steps": 4,
        },
        "d2_k4_monotone": {
            "candidate_id": "d2_k4_monotone",
            "role": "secondary_representation",
            "route": "koopman",
            "opening_map": "monotone",
            "latent_dim": 4,
        },
        "d2_pi_monotone": {
            "candidate_id": "d2_pi_monotone",
            "role": "secondary_representation",
            "route": "pi_ode",
            "opening_map": "monotone",
        },
        "d2_deeponet": {
            "candidate_id": "d2_deeponet",
            "role": "secondary_representation",
            "route": "deeponet",
            "opening_map": "identity",
            "latent_dim": 8,
            "hidden_dim": 32,
        },
    }
    for candidate_id, expected_payload in expected_candidate_payloads.items():
        if by_id[candidate_id] != expected_payload:
            raise ValueError(
                f"MS2-D2 {candidate_id} contains non-frozen candidate fields"
            )

    truth = matrix["synthetic_defaults"]
    expected_truth = {
        "truth_regime": "context_scheduled",
        "truth_opening_map": "equal_percentage_r50",
        "tau_seconds": [40.0, 70.0, 210.0],
        "input_delay_steps": 0,
        "dt_seconds": 10.0,
        "context_gain_log_scale": 0.35,
        "context_tau_log_scale": 0.30,
    }
    if any(truth.get(key) != value for key, value in expected_truth.items()):
        raise ValueError("MS2-D2 truth differs from the frozen third-order protocol")
    expected_gates = {
        "oracle_clean_nmae_max": 0.05,
        "order_aware_clean_nmae_max": 0.10,
        "order_aware_relative_improvement_min": 0.10,
        "tau_set_log_mae_max": 0.35,
        "no_true_delay_expected_steps_max": 0.50,
        "no_true_delay_zero_step_mass_min": 0.80,
    }
    if matrix["gates"] != expected_gates:
        raise ValueError("MS2-D2 gates differ from the frozen protocol")
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
    device: str,
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
        environment=environment_payload(torch.device(device)),
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
        raise RuntimeError(f"incomplete existing MS2-D2 run {output_dir}; missing={missing}")
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
    environment = manifest.get("environment")
    required_environment = {
        "python",
        "torch",
        "cuda_runtime",
        "cuda_available",
        "device",
        "platform",
    }
    if not isinstance(environment, dict) or not required_environment <= set(environment):
        mismatches.append("environment")
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
            f"existing MS2-D2 run mismatch {output_dir}: {sorted(set(mismatches))}"
        )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/ms2d_order")
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
        raise SystemExit("formal MS2-D2 execution requires the frozen repository matrix")
    if args.execute:
        if args.candidate_id is None or args.seed is None:
            raise SystemExit("--execute requires --candidate-id and --seed")
        selected = [{"candidate_id": args.candidate_id, "seed": args.seed}]
    else:
        if args.candidate_id is not None or args.seed is not None or args.smoke:
            raise SystemExit("matrix execution does not accept candidate/seed/smoke")
        selected = runs
    output_root = Path(args.output_root).resolve()
    _assert_no_test_artifacts(output_root)
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
        output_dir = output_root / f"ms2o_{candidate_id}_s{seed}"
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
        _augment_manifest(
            output_dir,
            matrix,
            matrix_path,
            regime,
            candidate,
            args.device,
        )
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
