#!/usr/bin/env python3
"""Run the frozen Phase 3.5-MS3 A/B observational validation matrix."""

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

from experiments.phase3_5.multistep_mismatch import _assert_clean_source_tree, _sha256
from src.phase35.data import load_cache
from src.phase35.multistep.real_training import (
    RealModelConfig,
    RealTrainingConfig,
    train_real_run,
)
from src.phase35.multistep.training import _json_dump
from src.phase35.schema import MS3_HISTORY_FEATURES


PROTOCOL_VERSION = "phase3.5-ms3-v1"
DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms3_real_adaptation_matrix.json"
FROZEN_MATRIX_SHA256 = "09dd01d02b4d94cec88b6bec4fbcbc0dc9eb4d3406d68f961de4796886b7b3d2"
FROZEN_EXECUTION_PATHS = (
    "configs/phase3_5/ms3_real_adaptation_matrix.json",
    "experiments/phase3_5/prepare_ms3_cross_data.py",
    "experiments/phase3_5/ms3_real_adaptation.py",
    "experiments/phase3_5/summarize_ms3_real_adaptation.py",
    "experiments/phase3_5/multistep_mismatch.py",
    "src/phase35/data.py",
    "src/phase35/model.py",
    "src/phase35/schema.py",
    "src/phase35/multistep/contracts.py",
    "src/phase35/multistep/full_training.py",
    "src/phase35/multistep/operators.py",
    "src/phase35/multistep/real_training.py",
    "src/phase35/multistep/training.py",
)
FORBIDDEN_TEST_ARTIFACTS = {
    "summary_test.json",
    "metrics_test.json",
    "episode_metrics_test.json",
    "test_access_ledger.json",
}


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_matrix(path: str | Path) -> dict:
    matrix_path = Path(path)
    if _sha256(matrix_path) != FROZEN_MATRIX_SHA256:
        raise ValueError("MS3 matrix differs from the frozen protocol")
    with matrix_path.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    required = {
        "protocol_version",
        "evidence_scope",
        "seeds",
        "data_contract",
        "model",
        "training",
        "gates",
        "ms5_reference",
        "candidates",
    }
    if set(matrix) != required or matrix["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("MS3 matrix keys/version differ from the frozen protocol")
    if matrix["evidence_scope"] != "real_ab_observational_validation_not_causal":
        raise ValueError("MS3 evidence scope changed")
    if matrix["seeds"] != [0, 1, 2]:
        raise ValueError("MS3 seeds changed")
    if tuple(matrix["data_contract"]["history_features"]) != tuple(
        MS3_HISTORY_FEATURES
    ):
        raise ValueError("MS3 history features changed")
    if set(matrix["data_contract"]["side_mappings"]) != {"A", "B"}:
        raise ValueError("MS3 cross-side mappings are incomplete")
    if [item["mode"] for item in matrix["candidates"]] != [
        "joint_total",
        "free_only",
    ]:
        raise ValueError("MS3 candidates changed")
    RealModelConfig(**matrix["model"]).validate()
    RealTrainingConfig(**matrix["training"]).validate()
    reference = ROOT / matrix["ms5_reference"]["path"]
    if not reference.is_file() or _sha256(reference) != matrix["ms5_reference"]["sha256"]:
        raise ValueError("MS3 frozen MS5 reference is missing or changed")
    with reference.open("r", encoding="utf-8") as handle:
        ms5 = json.load(handle)
    if (
        ms5.get("strategy_decision", {}).get("selected_strategy")
        != matrix["ms5_reference"]["selected_strategy"]
    ):
        raise ValueError("MS3 MS5 strategy pin is inconsistent")
    return matrix


def expand_runs(matrix: dict) -> list[dict]:
    return [
        {**candidate, "side": side, "seed": int(seed)}
        for candidate in matrix["candidates"]
        for side in ("A", "B")
        for seed in matrix["seeds"]
    ]


def _configs(matrix: dict, smoke: bool) -> tuple[RealModelConfig, RealTrainingConfig]:
    model = RealModelConfig(**matrix["model"])
    training = RealTrainingConfig(**matrix["training"])
    if smoke:
        model = replace(model, window=12, horizon=12, d_model=8, n_heads=2, dropout=0.0)
        training = replace(
            training,
            batch_size=8,
            epochs=2,
            patience=2,
            steps_per_epoch=2,
            max_train_anchors=64,
            max_selector_anchors=16,
            max_validation_anchors=32,
        )
    return model, training


def _assert_no_test_artifacts(output_root: Path) -> None:
    if not output_root.exists():
        return
    found = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_TEST_ARTIFACTS
    )
    if found:
        raise RuntimeError(f"MS3 validation refuses test artifacts: {found}")


def _validate_cache(cache, side: str, matrix: dict, matrix_sha: str) -> None:
    metadata = cache.metadata
    expected_contract = matrix["data_contract"]["side_mappings"][side]
    if metadata.get("side") != side or metadata.get("cross_pairing_frozen") is not True:
        raise ValueError(f"MS3 {side} cache lacks frozen cross-pairing metadata")
    if metadata.get("control_loop") != expected_contract["control_loop"]:
        raise ValueError(f"MS3 {side} cache control-loop mapping changed")
    if metadata.get("column_map") != expected_contract["column_map"]:
        raise ValueError(f"MS3 {side} cache column mapping changed")
    if tuple(cache.columns) != tuple(matrix["data_contract"]["history_features"]):
        raise ValueError(f"MS3 {side} cache feature order changed")
    if metadata.get("source", {}).get("sha256") != matrix["data_contract"]["source_sha256"]:
        raise ValueError(f"MS3 {side} cache source SHA changed")
    if metadata.get("matrix_sha256") != matrix_sha:
        raise ValueError(f"MS3 {side} cache matrix pin changed")


def _existing_compatible(output_dir: Path, run: dict, matrix_sha: str, git_sha: str) -> bool:
    if not output_dir.exists():
        return False
    required = (
        "manifest.json",
        "history.json",
        "checkpoint_best_val.pt",
        "metrics_validation.json",
        "episode_metrics_validation.json",
    )
    if not all((output_dir / name).is_file() for name in required):
        raise RuntimeError(f"partial MS3 output cannot be skipped: {output_dir}")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": f"{run['side']}_{run['candidate_id']}_s{run['seed']}",
        "side": run["side"],
        "seed": run["seed"],
        "mode": run["mode"],
        "matrix_sha256": matrix_sha,
        "git_sha": git_sha,
        "test_accessed": False,
        "test_authorized": False,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(f"existing MS3 run is incompatible: {mismatches}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--candidate-id")
    parser.add_argument("--side", choices=["A", "B"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="results/phase3_5/ms3_real_adaptation")
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
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.execute == args.execute_matrix:
        raise SystemExit("choose exactly one of --execute or --execute-matrix")
    if args.overwrite and args.skip_existing:
        raise SystemExit("choose either --overwrite or --skip-existing")
    if not args.cache_a or not args.cache_b:
        raise SystemExit("MS3 execution requires --cache-a and --cache-b")
    if not args.smoke and matrix_path != DEFAULT_MATRIX.resolve():
        raise SystemExit("formal MS3 execution requires the frozen repository matrix")
    if args.execute:
        if args.candidate_id is None or args.side is None or args.seed is None:
            raise SystemExit("--execute requires candidate-id, side, and seed")
        selected = [
            run
            for run in runs
            if run["candidate_id"] == args.candidate_id
            and run["side"] == args.side
            and run["seed"] == args.seed
        ]
        if len(selected) != 1:
            raise SystemExit("requested MS3 run is not frozen")
    else:
        if args.candidate_id is not None or args.side is not None or args.seed is not None or args.smoke:
            raise SystemExit("matrix execution does not accept single-run/smoke arguments")
        selected = runs

    output_root = Path(args.output_root).resolve()
    _assert_no_test_artifacts(output_root)
    if not args.smoke:
        _assert_clean_source_tree(output_root, allow_generated_outputs=args.skip_existing)
    matrix_sha = _sha256(matrix_path)
    caches = {"A": load_cache(args.cache_a), "B": load_cache(args.cache_b)}
    cache_paths = {"A": Path(args.cache_a), "B": Path(args.cache_b)}
    for side in ("A", "B"):
        _validate_cache(caches[side], side, matrix, matrix_sha)
    model_config, training_config = _configs(matrix, args.smoke)
    git_sha = _git_sha()
    completed = []
    for index, run in enumerate(selected, start=1):
        run_id = f"{run['side']}_{run['candidate_id']}_s{run['seed']}"
        output_dir = output_root / run_id
        if args.skip_existing and _existing_compatible(
            output_dir, run, matrix_sha, git_sha
        ):
            completed.append({**run, "status": "skipped_existing"})
            continue
        print(f"[{index}/{len(selected)}] MS3 {run_id}", file=sys.stderr, flush=True)
        result = train_real_run(
            cache=caches[run["side"]],
            cache_path=cache_paths[run["side"]],
            feature_columns=matrix["data_contract"]["history_features"],
            model_config=model_config,
            training_config=training_config,
            side=run["side"],
            seed=run["seed"],
            mode=run["mode"],
            run_id=run_id,
            output_dir=output_dir,
            device=args.device,
            protocol_version=PROTOCOL_VERSION,
            matrix_sha256=matrix_sha,
            repo_git_sha=git_sha,
            overwrite=args.overwrite,
        )
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "candidate_role": run["role"],
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
                "validation_logged_mae_c": result.validation_metrics["logged_mae_c"],
                "test_accessed": False,
            }
        )
    print(
        json.dumps(
            {"status": "matrix_completed", "run_count": len(completed), "runs": completed},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
