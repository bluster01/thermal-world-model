#!/usr/bin/env python3
"""Dry-run or execute the frozen Phase 3.5 multi-step synthetic benchmark."""

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

from src.phase35.multistep.contracts import OperatorConfig  # noqa: E402
from src.phase35.multistep.synthetic import SyntheticSpec  # noqa: E402
from src.phase35.multistep.training import (  # noqa: E402
    TrainingConfig,
    evaluate_synthetic_test_checkpoint,
    train_synthetic_run,
)


def load_matrix(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    required = {"protocol_version", "seeds", "operator_defaults", "synthetic", "training", "routes"}
    missing = sorted(required - set(matrix))
    if missing:
        raise ValueError(f"matrix missing keys: {missing}")
    route_ids = [route["route_id"] for route in matrix["routes"]]
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("matrix route_id values must be unique")
    return matrix


def expand_runs(matrix: dict) -> list[dict]:
    return [
        {"route_id": route["route_id"], "route": route["route"], "seed": int(seed)}
        for route in matrix["routes"]
        for seed in matrix["seeds"]
    ]


def _select_route(matrix: dict, route_id: str) -> dict:
    matches = [route for route in matrix["routes"] if route["route_id"] == route_id]
    if len(matches) != 1:
        raise ValueError(f"route_id={route_id!r} is not uniquely defined in matrix")
    return matches[0]


def _build_configs(matrix: dict, route: dict, seed: int, smoke: bool):
    operator_values = dict(matrix["operator_defaults"])
    operator_values.update({key: value for key, value in route.items() if key != "route_id"})
    operator = OperatorConfig.from_mapping(operator_values)
    training_values = dict(matrix["training"])
    synthetic_values = dict(matrix["synthetic"])
    validation_samples = int(synthetic_values.pop("validation_samples"))
    test_samples = int(synthetic_values.pop("test_samples"))
    train_samples = int(synthetic_values.pop("train_samples"))
    synthetic_values["samples"] = train_samples
    synthetic_values["seed"] = int(synthetic_values.get("seed", 20260809))
    if smoke:
        training_values.update(batch_size=16, epochs=2, patience=2)
        synthetic_values["samples"] = 64
        validation_samples = 32
        test_samples = 32
    training = TrainingConfig(**training_values)
    synthetic = SyntheticSpec(**synthetic_values)
    return operator, training, synthetic, validation_samples, test_samples


def _current_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


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
            "formal execution requires clean source/config state; dirty entries:\n" + "\n".join(dirty)
        )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _existing_run_is_compatible(
    output_dir: Path,
    route_id: str,
    seed: int,
    operator: OperatorConfig,
    training: TrainingConfig,
    synthetic: SyntheticSpec,
    required_test_access: bool | None,
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
        "protocol_version": "phase3.5-ms-v1",
        "route_id": route_id,
        "seed": seed,
        "operator_config": operator.to_dict(),
        "training_config": asdict(training),
        "synthetic_spec": asdict(expected_spec),
        "git_sha": _current_git_sha(),
    }
    mismatches = [key for key, value in expected.items() if _canonical(manifest.get(key)) != _canonical(value)]
    if required_test_access is not None and manifest.get("test_accessed") is not required_test_access:
        mismatches.append("test_accessed")
    if mismatches:
        raise RuntimeError(
            f"existing run {output_dir} does not match current frozen run: {sorted(set(mismatches))}"
        )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default=str(ROOT / "configs/phase3_5/multistep_operator_matrix.json")
    )
    parser.add_argument("--route-id", help="one frozen route id; omit only for dry-run")
    parser.add_argument("--seed", type=int, help="one frozen seed; omit only for dry-run")
    parser.add_argument("--output-root", default="results/phase3_5/multistep_synthetic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-matrix", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--evaluate-synthetic-test", action="store_true")
    parser.add_argument("--allow-synthetic-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    runs = expand_runs(matrix)
    allowed_seeds = {int(seed) for seed in matrix["seeds"]}
    if args.evaluate_synthetic_test:
        if not args.allow_synthetic_test:
            raise SystemExit("--evaluate-synthetic-test requires explicit --allow-synthetic-test")
        if args.execute or args.execute_matrix or args.smoke or args.route_id is None or args.seed is None:
            raise SystemExit(
                "synthetic test evaluation requires only --route-id, --seed, "
                "--evaluate-synthetic-test, and --allow-synthetic-test"
            )
        if args.seed not in allowed_seeds:
            raise SystemExit(f"seed={args.seed} is not frozen in matrix seeds={sorted(allowed_seeds)}")
        route = _select_route(matrix, args.route_id)
        operator, training, synthetic, _, test_samples = _build_configs(matrix, route, args.seed, False)
        output_root = Path(args.output_root).resolve()
        _assert_clean_source_tree(output_root, allow_generated_outputs=True)
        output_dir = output_root / f"synthetic_{args.route_id}_s{args.seed}"
        _existing_run_is_compatible(
            output_dir, args.route_id, args.seed, operator, training, synthetic, required_test_access=None
        )
        metrics = evaluate_synthetic_test_checkpoint(
            output_dir=output_dir,
            test_samples=test_samples,
            device=args.device,
            repo_root=ROOT,
        )
        print(json.dumps({
            "status": "synthetic_test_completed",
            "output_dir": str(output_dir),
            "effect_mae": metrics["effect_mae"],
            "test_accessed": True,
        }, indent=2))
        return
    if args.allow_synthetic_test:
        raise SystemExit("--allow-synthetic-test is valid only with --evaluate-synthetic-test")
    if args.overwrite and args.skip_existing:
        raise SystemExit("choose either --overwrite or --skip-existing")
    if args.dry_run or not (args.execute or args.execute_matrix):
        print(json.dumps({"protocol_version": matrix["protocol_version"], "run_count": len(runs), "runs": runs}, indent=2))
        return
    if args.execute and args.execute_matrix:
        raise SystemExit("choose either --execute for one run or --execute-matrix for the frozen matrix")
    if args.execute:
        if args.route_id is None or args.seed is None:
            raise SystemExit("--execute requires one --route-id and one --seed")
        selected = [{"route_id": args.route_id, "seed": args.seed}]
    else:
        if args.route_id is not None or args.seed is not None:
            raise SystemExit("--execute-matrix does not accept --route-id or --seed filters")
        selected = runs
    output_root = Path(args.output_root).resolve()
    if not args.smoke:
        _assert_clean_source_tree(output_root, allow_generated_outputs=args.skip_existing)
    completed = []
    for index, run in enumerate(selected, start=1):
        route_id, seed = run["route_id"], int(run["seed"])
        if seed not in allowed_seeds:
            raise SystemExit(f"seed={seed} is not frozen in matrix seeds={sorted(allowed_seeds)}")
        route = _select_route(matrix, route_id)
        operator, training, synthetic, validation_samples, test_samples = _build_configs(
            matrix, route, seed, args.smoke
        )
        output_dir = output_root / f"synthetic_{route_id}_s{seed}"
        if args.skip_existing and _existing_run_is_compatible(
            output_dir, route_id, seed, operator, training, synthetic, required_test_access=False
        ):
            completed.append({"route_id": route_id, "seed": seed, "status": "skipped_existing", "output_dir": str(output_dir)})
            continue
        print(f"[{index}/{len(selected)}] route={route_id} seed={seed}", file=sys.stderr, flush=True)
        result = train_synthetic_run(
            operator_config=operator,
            training_config=training,
            synthetic_spec=synthetic,
            validation_samples=validation_samples,
            seed=seed,
            route_id=route_id,
            output_dir=output_dir,
            device=args.device,
            repo_root=ROOT,
            overwrite=args.overwrite,
        )
        completed.append({
            "route_id": route_id,
            "seed": seed,
            "status": "completed",
            "output_dir": str(result.output_dir),
            "checkpoint": str(result.checkpoint),
            "best_epoch": result.best_epoch,
            "validation_effect_mae": result.validation_metrics["effect_mae"],
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
