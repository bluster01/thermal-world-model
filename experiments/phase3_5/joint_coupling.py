#!/usr/bin/env python3
"""Run the frozen Phase 3.5-MS2-J joint-coupling validation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
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
    _sha256,
)
from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.staging import (  # noqa: E402
    StagedTrainingConfig,
    environment_payload,
    train_staged_synthetic_run,
    trajectory_design_sha256,
)
from src.phase35.multistep.synthetic import (  # noqa: E402
    SyntheticSpec,
    generate_synthetic_split,
)
from src.phase35.multistep.training import (  # noqa: E402
    TrainingConfig,
    _json_dump,
    train_synthetic_run,
)


PROTOCOL_VERSION = "phase3.5-ms2j-v1"
FROZEN_EXECUTION_PATHS = (
    "configs/phase3_5/joint_coupling_matrix.json",
    "experiments/phase3_5/joint_coupling.py",
    "src/phase35/multistep/staging.py",
    "src/phase35/multistep/contracts.py",
    "src/phase35/multistep/operators.py",
    "src/phase35/multistep/synthetic.py",
    "src/phase35/multistep/training.py",
)


def _current_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


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
        "staged_training",
        "regimes",
    }
    missing = sorted(required - set(matrix))
    if missing:
        raise ValueError(f"MS2-J matrix missing keys: {missing}")
    if matrix["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"MS2-J runner accepts only {PROTOCOL_VERSION}")
    candidates = [
        candidate
        for regime in matrix["regimes"]
        for candidate in regime["candidates"]
    ]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("MS2-J candidate IDs must be globally unique")
    if set(candidate["training_mode"] for candidate in candidates) - {
        "joint",
        "staged",
    }:
        raise ValueError("MS2-J training_mode must be joint or staged")
    if sum(candidate["training_mode"] == "staged" for candidate in candidates) != 1:
        raise ValueError("MS2-J freezes exactly one staged candidate")
    if len(candidates) != 9 or len(matrix["seeds"]) != 3:
        raise ValueError("MS2-J formal matrix must contain 9 candidates and 3 seeds")
    return matrix


def expand_runs(matrix: dict) -> list[dict]:
    return [
        {
            "regime_id": regime["regime_id"],
            "candidate_id": candidate["candidate_id"],
            "route": candidate["route"],
            "training_mode": candidate["training_mode"],
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
            if key not in {"candidate_id", "training_mode"}
        }
    )
    operator = OperatorConfig.from_mapping(operator_values)
    training_values = dict(matrix["training"])
    staged_values = dict(matrix["staged_training"])
    synthetic_values = dict(matrix["synthetic_defaults"])
    synthetic_values.update(regime["synthetic"])
    validation_samples = int(synthetic_values.pop("validation_samples"))
    synthetic_values.pop("test_samples")
    synthetic_values["samples"] = int(synthetic_values.pop("train_samples"))
    if smoke:
        synthetic_values["samples"] = 64
        validation_samples = 32
        if candidate["training_mode"] == "staged":
            training_values.update(batch_size=16, epochs=6, patience=2)
            staged_values.update(
                stage_a_epochs=2,
                stage_b_epochs=2,
                stage_c_epochs=2,
                stage_patience=2,
            )
        else:
            training_values.update(batch_size=16, epochs=2, patience=2)
    return (
        operator,
        TrainingConfig(**training_values),
        StagedTrainingConfig(**staged_values),
        SyntheticSpec(**synthetic_values),
        validation_samples,
    )


def _augment_joint_manifest(
    output_dir: Path,
    synthetic: SyntheticSpec,
    validation_samples: int,
    seed: int,
    device: str,
) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeded_spec = replace(synthetic, seed=synthetic.seed + seed * 1_000_003)
    validation_batch = generate_synthetic_split(
        replace(seeded_spec, samples=validation_samples), "validation"
    )
    manifest["evidence_scope"] = (
        "synthetic_joint_coupling_validation_not_field_causality"
    )
    manifest["training_mode"] = "joint"
    manifest["environment"] = environment_payload(torch.device(device))
    manifest["validation_trajectory_design_sha256"] = trajectory_design_sha256(
        validation_batch
    )
    _json_dump(manifest_path, manifest)


def _existing_run_is_compatible(
    output_dir: Path,
    matrix: dict,
    candidate: dict,
    seed: int,
    operator: OperatorConfig,
    training: TrainingConfig,
    staged: StagedTrainingConfig,
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
        raise RuntimeError(f"incomplete existing MS2-J run {output_dir}; missing={missing}")
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    expected_spec = replace(synthetic, seed=synthetic.seed + seed * 1_000_003)
    expected = {
        "protocol_version": matrix["protocol_version"],
        "route_id": candidate["candidate_id"],
        "seed": seed,
        "training_mode": candidate["training_mode"],
        "operator_config": operator.to_dict(),
        "training_config": asdict(training),
        "synthetic_spec": asdict(expected_spec),
        "git_sha": _current_git_sha(),
        "test_accessed": False,
    }
    if candidate["training_mode"] == "staged":
        expected["staged_training_config"] = asdict(staged)
    mismatches = [
        key
        for key, value in expected.items()
        if _canonical(manifest.get(key)) != _canonical(value)
    ]
    if manifest.get("checkpoint_sha256") != _sha256(required[1]):
        mismatches.append("checkpoint_sha256")
    if candidate["training_mode"] == "staged":
        for stage in ("stage_a", "stage_b", "stage_c"):
            info = manifest.get("stage_checkpoints", {}).get(stage, {})
            path = output_dir / info.get("file", "missing")
            if not path.is_file() or info.get("sha256") != _sha256(path):
                mismatches.append(f"checkpoint_{stage}")
    if mismatches:
        raise RuntimeError(
            f"existing MS2-J run mismatch {output_dir}: {sorted(set(mismatches))}"
        )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/joint_coupling_matrix.json"),
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/joint_coupling")
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
    canonical_matrix = (
        ROOT / "configs/phase3_5/joint_coupling_matrix.json"
    ).resolve()
    if (args.execute or args.execute_matrix) and not args.smoke:
        if matrix_path != canonical_matrix:
            raise SystemExit("formal MS2-J execution requires the repository-frozen matrix")
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
    if args.execute:
        if args.candidate_id is None or args.seed is None:
            raise SystemExit("--execute requires --candidate-id and --seed")
        selected = [{"candidate_id": args.candidate_id, "seed": args.seed}]
    else:
        if args.candidate_id is not None or args.seed is not None or args.smoke:
            raise SystemExit("matrix execution does not accept candidate/seed/smoke")
        selected = runs
    allowed_seeds = {int(seed) for seed in matrix["seeds"]}
    output_root = Path(args.output_root).resolve()
    if not args.smoke:
        _assert_clean_source_tree(output_root, allow_generated_outputs=args.skip_existing)
    completed = []
    for index, run in enumerate(selected, start=1):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        if seed not in allowed_seeds:
            raise SystemExit(f"seed={seed} is not frozen")
        regime, candidate = _select(matrix, candidate_id)
        operator, training, staged, synthetic, validation_samples = _build_configs(
            matrix, regime, candidate, args.smoke
        )
        output_dir = output_root / f"ms2j_{candidate_id}_s{seed}"
        if args.skip_existing and _existing_run_is_compatible(
            output_dir,
            matrix,
            candidate,
            seed,
            operator,
            training,
            staged,
            synthetic,
        ):
            completed.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "status": "skipped_existing",
                }
            )
            continue
        print(
            f"[{index}/{len(selected)}] candidate={candidate_id} seed={seed} "
            f"mode={candidate['training_mode']}",
            file=sys.stderr,
            flush=True,
        )
        common = {
            "operator_config": operator,
            "training_config": training,
            "synthetic_spec": synthetic,
            "validation_samples": validation_samples,
            "seed": seed,
            "route_id": candidate_id,
            "output_dir": output_dir,
            "device": args.device,
            "repo_root": ROOT,
            "overwrite": args.overwrite,
            "protocol_version": matrix["protocol_version"],
        }
        if candidate["training_mode"] == "staged":
            result = train_staged_synthetic_run(
                staged_config=staged,
                **common,
            )
        else:
            result = train_synthetic_run(**common)
            _augment_joint_manifest(
                output_dir, synthetic, validation_samples, seed, args.device
            )
        completed.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "training_mode": candidate["training_mode"],
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
