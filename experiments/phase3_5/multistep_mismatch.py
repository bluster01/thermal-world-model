#!/usr/bin/env python3
"""Run the frozen Phase 3.5-MS2 synthetic mismatch validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.synthetic import SyntheticSpec  # noqa: E402
from src.phase35.multistep.training import TrainingConfig, train_synthetic_run  # noqa: E402


def load_matrix(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    required = {
        "protocol_version",
        "seeds",
        "operator_defaults",
        "synthetic_defaults",
        "training",
        "regimes",
    }
    missing = sorted(required - set(matrix))
    if missing:
        raise ValueError(f"matrix missing keys: {missing}")
    candidate_ids = [
        candidate["candidate_id"]
        for regime in matrix["regimes"]
        for candidate in regime["candidates"]
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("MS2 candidate_id values must be globally unique")
    if matrix["protocol_version"] != "phase3.5-ms2-v1":
        raise ValueError("MS2 runner accepts only protocol_version=phase3.5-ms2-v1")
    return matrix


def expand_runs(matrix: dict) -> list[dict]:
    return [
        {
            "regime_id": regime["regime_id"],
            "candidate_id": candidate["candidate_id"],
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
        {key: value for key, value in candidate.items() if key != "candidate_id"}
    )
    operator = OperatorConfig.from_mapping(operator_values)

    training_values = dict(matrix["training"])
    synthetic_values = dict(matrix["synthetic_defaults"])
    synthetic_values.update(regime["synthetic"])
    validation_samples = int(synthetic_values.pop("validation_samples"))
    synthetic_values.pop("test_samples")
    train_samples = int(synthetic_values.pop("train_samples"))
    synthetic_values["samples"] = train_samples
    if smoke:
        training_values.update(batch_size=16, epochs=2, patience=2)
        synthetic_values["samples"] = 64
        validation_samples = 32
    training = TrainingConfig(**training_values)
    synthetic = SyntheticSpec(**synthetic_values)
    return operator, training, synthetic, validation_samples


def _current_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_clean_source_tree(output_root: Path, allow_generated_outputs: bool) -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    )
    dirty = []
    for line in status.splitlines():
        relative = line[3:].strip().strip('"')
        candidate = (ROOT / relative).resolve()
        inside_output = candidate == output_root or output_root in candidate.parents
        if not (allow_generated_outputs and inside_output):
            dirty.append(line)
    if dirty:
        raise SystemExit(
            "formal execution requires clean source/config state; dirty entries:\n"
            + "\n".join(dirty)
        )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _existing_run_is_compatible(
    output_dir: Path,
    protocol_version: str,
    candidate_id: str,
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
        raise RuntimeError(f"incomplete existing run {output_dir}; missing={missing}")
    with (output_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_spec = replace(synthetic, seed=synthetic.seed + seed * 1_000_003)
    expected = {
        "protocol_version": protocol_version,
        "route_id": candidate_id,
        "seed": seed,
        "operator_config": operator.to_dict(),
        "training_config": asdict(training),
        "synthetic_spec": asdict(expected_spec),
        "git_sha": _current_git_sha(),
        "test_accessed": False,
    }
    mismatches = [
        key for key, value in expected.items()
        if _canonical(manifest.get(key)) != _canonical(value)
    ]
    checkpoint_hash = manifest.get("checkpoint_sha256")
    if not checkpoint_hash or _sha256(required[1]) != checkpoint_hash:
        mismatches.append("checkpoint_sha256")
    if mismatches:
        raise RuntimeError(
            f"existing run {output_dir} does not match frozen MS2 run: "
            f"{sorted(set(mismatches))}"
        )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"),
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/multistep_mismatch")
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
    matrix = load_matrix(args.matrix)
    runs = expand_runs(matrix)
    if args.dry_run or not (args.execute or args.execute_matrix):
        print(json.dumps({
            "protocol_version": matrix["protocol_version"],
            "run_count": len(runs),
            "runs": runs,
        }, indent=2))
        return
    if args.execute and args.execute_matrix:
        raise SystemExit("choose either --execute or --execute-matrix")
    if args.overwrite and args.skip_existing:
        raise SystemExit("choose either --overwrite or --skip-existing")
    allowed_seeds = {int(seed) for seed in matrix["seeds"]}
    if args.execute:
        if args.candidate_id is None or args.seed is None:
            raise SystemExit("--execute requires --candidate-id and --seed")
        selected = [{"candidate_id": args.candidate_id, "seed": args.seed}]
    else:
        if args.candidate_id is not None or args.seed is not None:
            raise SystemExit("--execute-matrix does not accept candidate or seed filters")
        selected = runs

    output_root = Path(args.output_root).resolve()
    if not args.smoke:
        _assert_clean_source_tree(output_root, allow_generated_outputs=args.skip_existing)
    completed = []
    for index, run in enumerate(selected, start=1):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        if seed not in allowed_seeds:
            raise SystemExit(f"seed={seed} is not frozen in seeds={sorted(allowed_seeds)}")
        regime, candidate = _select(matrix, candidate_id)
        operator, training, synthetic, validation_samples = _build_configs(
            matrix, regime, candidate, args.smoke
        )
        output_dir = output_root / f"ms2_{candidate_id}_s{seed}"
        if args.skip_existing and _existing_run_is_compatible(
            output_dir,
            matrix["protocol_version"],
            candidate_id,
            seed,
            operator,
            training,
            synthetic,
        ):
            completed.append({
                "regime_id": regime["regime_id"],
                "candidate_id": candidate_id,
                "seed": seed,
                "status": "skipped_existing",
                "output_dir": str(output_dir),
            })
            continue
        print(
            f"[{index}/{len(selected)}] regime={regime['regime_id']} "
            f"candidate={candidate_id} seed={seed}",
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
        completed.append({
            "regime_id": regime["regime_id"],
            "candidate_id": candidate_id,
            "seed": seed,
            "status": "completed",
            "output_dir": str(result.output_dir),
            "checkpoint": str(result.checkpoint),
            "checkpoint_sha256": _sha256(result.checkpoint),
            "best_epoch": result.best_epoch,
            "validation_effect_mae": result.validation_metrics["effect_mae"],
            "validation_clean_effect_mae": result.validation_metrics["clean_effect_mae"],
            "test_accessed": False,
        })
    payload = completed[0] if len(completed) == 1 else {
        "status": "matrix_completed",
        "run_count": len(completed),
        "runs": completed,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
