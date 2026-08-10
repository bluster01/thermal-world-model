#!/usr/bin/env python3
"""Run the frozen Phase 3.5-MS2-D3 colored-disturbance validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.ms2d_order import (  # noqa: E402
    _assert_no_test_artifacts,
    _build_configs,
)
from experiments.phase3_5.multistep_mismatch import (  # noqa: E402
    _assert_clean_source_tree,
    _canonical,
    _current_git_sha,
    _sha256,
)
from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.operators import build_response_operator  # noqa: E402
from src.phase35.multistep.staging import environment_payload  # noqa: E402
from src.phase35.multistep.synthetic import (  # noqa: E402
    SyntheticSpec,
    generate_synthetic_split,
)
from src.phase35.multistep.training import (  # noqa: E402
    TrainingConfig,
    _json_dump,
    evaluate_operator,
    structural_diagnostics,
    train_synthetic_run,
)


PROTOCOL_VERSION = "phase3.5-ms2d-d3-v1"
DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms2d_disturbance_matrix.json"
FROZEN_EXECUTION_PATHS = (
    "configs/phase3_5/ms2d_disturbance_matrix.json",
    "experiments/phase3_5/ms2d_delay.py",
    "experiments/phase3_5/ms2d_delay_test.py",
    "experiments/phase3_5/ms2d_order.py",
    "experiments/phase3_5/ms2d_order_test.py",
    "experiments/phase3_5/summarize_ms2d_order_test.py",
    "experiments/phase3_5/multistep_mismatch.py",
    "experiments/phase3_5/ms2d_disturbance.py",
    "experiments/phase3_5/summarize_ms2d_disturbance.py",
    "src/phase35/multistep/contracts.py",
    "src/phase35/multistep/operators.py",
    "src/phase35/multistep/staging.py",
    "src/phase35/multistep/synthetic.py",
    "src/phase35/multistep/training.py",
)
FROZEN_QUESTION = (
    "Does the confirmed three-pole response advantage survive an unobserved "
    "action-independent stationary AR(1) output disturbance?"
)
VALIDATION_EPISODES_NAME = "episode_metrics_validation.json"


def _expected_operator_defaults() -> dict:
    return {
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


def _expected_synthetic_defaults() -> dict:
    return {
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
        "truth_regime": "disturbed_context_scheduled",
        "truth_opening_map": "equal_percentage_r50",
        "context_gain_log_scale": 0.35,
        "context_tau_log_scale": 0.30,
        "input_delay_steps": 0,
        "disturbance_std": 0.03,
        "disturbance_tau_seconds": 120.0,
    }


def _expected_training() -> dict:
    return {
        "batch_size": 64,
        "epochs": 300,
        "patience": 30,
        "learning_rate": 0.002,
        "weight_decay": 0.000001,
        "physics_weight": 0.01,
        "gradient_clip": 1.0,
    }


def _expected_gates() -> dict:
    return {
        "oracle_clean_nmae_max": 0.05,
        "disturbance_robust_clean_nmae_max": 0.10,
        "disturbance_robust_ci_lower_min": 0.10,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260814,
        "tau_set_log_mae_max": 0.35,
        "no_true_delay_expected_steps_max": 0.50,
        "no_true_delay_zero_step_mass_min": 0.80,
    }


def _expected_candidates() -> list[dict]:
    return [
        {
            "candidate_id": "d3_g2_two_pole",
            "role": "primary_ablation",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 2,
        },
        {
            "candidate_id": "d3_g3_three_pole",
            "role": "primary_model",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 3,
        },
        {
            "candidate_id": "d3_g3_oracle_structure",
            "role": "positive_control",
            "route": "graybox",
            "opening_map": "equal_percentage_r50",
            "context_scheduled": True,
            "poles": 3,
        },
        {
            "candidate_id": "d3_g2_delay_compensation",
            "role": "alternative_mechanism_diagnostic",
            "route": "graybox",
            "opening_map": "monotone",
            "context_scheduled": True,
            "poles": 2,
            "delay_mode": "learned",
            "max_delay_steps": 4,
        },
        {
            "candidate_id": "d3_k4_monotone",
            "role": "secondary_representation",
            "route": "koopman",
            "opening_map": "monotone",
            "latent_dim": 4,
        },
        {
            "candidate_id": "d3_pi_monotone",
            "role": "secondary_representation",
            "route": "pi_ode",
            "opening_map": "monotone",
        },
        {
            "candidate_id": "d3_deeponet",
            "role": "secondary_representation",
            "route": "deeponet",
            "opening_map": "identity",
            "latent_dim": 8,
            "hidden_dim": 32,
        },
    ]


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
        "d2_reference",
        "regimes",
    }
    if set(matrix) != required:
        raise ValueError("MS2-D3 matrix keys differ from the frozen protocol")
    if matrix["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"MS2-D3 accepts only frozen {PROTOCOL_VERSION}")
    if matrix["evidence_scope"] != (
        "synthetic_colored_disturbance_pressure_validation_not_field_causality"
    ):
        raise ValueError("MS2-D3 evidence scope differs from the frozen protocol")
    if matrix["seeds"] != [0, 1, 2]:
        raise ValueError("MS2-D3 seeds differ from the frozen protocol")
    exact_sections = {
        "operator_defaults": _expected_operator_defaults(),
        "synthetic_defaults": _expected_synthetic_defaults(),
        "training": _expected_training(),
        "gates": _expected_gates(),
        "d2_reference": {
            "path": "results/phase3_5/ms2d_order/summary_test.json",
            "sha256": "9c4e9300314379436d9b3e3a3bd004114fb76aeaf406164def86fce95cb71ea0",
        },
    }
    for name, expected in exact_sections.items():
        if _canonical(matrix[name]) != _canonical(expected):
            raise ValueError(f"MS2-D3 {name} differs from the frozen protocol")
    if len(matrix["regimes"]) != 1:
        raise ValueError("MS2-D3 freezes exactly one regime")
    regime = matrix["regimes"][0]
    if set(regime) != {"regime_id", "question", "synthetic", "candidates"}:
        raise ValueError("MS2-D3 regime keys differ from the frozen protocol")
    if regime["question"] != FROZEN_QUESTION:
        raise ValueError("MS2-D3 scientific question differs from the frozen protocol")
    if (
        regime["regime_id"]
        != "third_order_r50_context_scheduled_colored_disturbance"
        or regime["synthetic"] != {}
        or regime["candidates"] != _expected_candidates()
    ):
        raise ValueError("MS2-D3 regime or candidates differ from the frozen protocol")
    reference = ROOT / matrix["d2_reference"]["path"]
    if not reference.is_file() or _sha256(reference) != matrix["d2_reference"]["sha256"]:
        raise ValueError("MS2-D3 frozen D2 reference is missing or changed")
    SyntheticSpec(
        **{
            "samples": matrix["synthetic_defaults"]["train_samples"],
            **{
                key: value
                for key, value in matrix["synthetic_defaults"].items()
                if key not in {"train_samples", "validation_samples", "test_samples"}
            },
        }
    ).validate()
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


def _trajectory_digest(batch) -> str:
    digest = hashlib.sha256()
    for tensor in (
        batch.context,
        batch.action,
        batch.reference,
        batch.clean_effect,
        batch.colored_disturbance,
        batch.profile_ids,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _episode_metrics(batch, prediction: torch.Tensor) -> dict:
    prediction = prediction.detach().cpu()
    target = batch.target_effect.detach().cpu()
    clean = batch.clean_effect.detach().cpu()
    disturbance = batch.colored_disturbance.detach().cpu()
    observed_error = (prediction - target).abs()
    clean_error = (prediction - clean).abs()
    points = (1, 6, 18, 60)
    return {
        "episode_ids": list(range(clean.shape[0])),
        "profile_ids": batch.profile_ids.detach().cpu().tolist(),
        "profile_names": list(batch.profile_names),
        "trajectory_design_sha256": _trajectory_digest(batch),
        "observed_effect_mae": observed_error.mean(dim=1).tolist(),
        "clean_effect_mae": clean_error.mean(dim=1).tolist(),
        "clean_effect_scale": clean.abs().mean(dim=1).tolist(),
        "colored_disturbance_mae": disturbance.abs().mean(dim=1).tolist(),
        "colored_disturbance_mean": disturbance.mean(dim=1).tolist(),
        "clean_horizon_absolute_error": {
            f"H{point}": clean_error[:, point - 1].tolist() for point in points
        },
    }


def _evaluate_validation(
    checkpoint: Path, validation_samples: int, device: str
) -> tuple[dict, dict]:
    dev = torch.device(device)
    payload = torch.load(checkpoint, map_location=dev, weights_only=False)
    operator = build_response_operator(
        OperatorConfig.from_mapping(payload["operator_config"])
    ).to(dev)
    operator.load_state_dict(payload["model_state_dict"])
    operator.eval()
    spec = SyntheticSpec(**payload["synthetic_spec"])
    batch = generate_synthetic_split(
        replace(spec, samples=validation_samples), "validation"
    )
    metrics, prediction = evaluate_operator(operator, batch, dev)
    metrics["structural_diagnostics"] = structural_diagnostics(operator, batch, dev)
    metrics["truth"] = batch.truth
    return metrics, _episode_metrics(batch, prediction)


def _assert_reproduced_metrics(stored: dict, replayed: dict) -> None:
    for key in (
        "effect_mae",
        "clean_effect_mae",
        "clean_effect_scale",
        "clean_effect_nmae",
    ):
        if not math.isclose(
            float(stored[key]), float(replayed[key]), rel_tol=1e-6, abs_tol=1e-8
        ):
            raise RuntimeError(f"validation checkpoint replay differs for {key}")
    if _canonical(stored.get("truth")) != _canonical(replayed.get("truth")):
        raise RuntimeError("validation checkpoint replay truth differs")


def _augment_manifest(
    output_dir: Path,
    matrix: dict,
    matrix_path: Path,
    regime: dict,
    candidate: dict,
    episodes: dict,
    device: str,
) -> None:
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        evidence_scope=matrix["evidence_scope"],
        regime_id=regime["regime_id"],
        candidate_role=candidate["role"],
        matrix_sha256=_sha256(matrix_path),
        d2_reference_sha256=matrix["d2_reference"]["sha256"],
        frozen_execution_paths=list(FROZEN_EXECUTION_PATHS),
        validation_episode_metrics=VALIDATION_EPISODES_NAME,
        validation_trajectory_design_sha256=episodes[
            "trajectory_design_sha256"
        ],
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
        output_dir / VALIDATION_EPISODES_NAME,
    ]
    present = [path.is_file() for path in required]
    if not any(present):
        return False
    if not all(present):
        missing = [path.name for path, exists in zip(required, present) if not exists]
        raise RuntimeError(f"incomplete existing MS2-D3 run {output_dir}; missing={missing}")
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    episodes = json.loads(required[4].read_text(encoding="utf-8"))
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
        "d2_reference_sha256": matrix["d2_reference"]["sha256"],
        "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
        "validation_episode_metrics": VALIDATION_EPISODES_NAME,
        "validation_trajectory_design_sha256": episodes.get(
            "trajectory_design_sha256"
        ),
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
            f"existing MS2-D3 run mismatch {output_dir}: {sorted(set(mismatches))}"
        )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/ms2d_disturbance")
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
        raise SystemExit("formal MS2-D3 execution requires the frozen repository matrix")
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
        output_dir = output_root / f"ms2d3_{candidate_id}_s{seed}"
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
            f"[{index}/{len(selected)}] D3 candidate={candidate_id} seed={seed}",
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
        replayed, episodes = _evaluate_validation(
            result.checkpoint, validation_samples, args.device
        )
        _assert_reproduced_metrics(result.validation_metrics, replayed)
        _json_dump(output_dir / VALIDATION_EPISODES_NAME, episodes)
        _augment_manifest(
            output_dir,
            matrix,
            matrix_path,
            regime,
            candidate,
            episodes,
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
                "validation_trajectory_design_sha256": episodes[
                    "trajectory_design_sha256"
                ],
                "test_accessed": False,
            }
        )
    payload = (
        completed[0]
        if len(completed) == 1
        else {"status": "matrix_completed", "run_count": len(completed), "runs": completed}
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
