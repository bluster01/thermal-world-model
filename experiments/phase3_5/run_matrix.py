#!/usr/bin/env python3
"""Expand and optionally execute the frozen Phase 3.5 development matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.matrix import expand_matrix, load_matrix


def _csv_set(value: str | None, cast=str):
    return None if not value else {cast(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/phase3_5/experiment_matrix.json")
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--output-root", default="results/phase3_5/runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--configs", help="optional comma-separated config ids")
    parser.add_argument("--sides", help="optional comma-separated A,B")
    parser.add_argument("--seeds", help="optional comma-separated seeds")
    parser.add_argument("--execute", action="store_true", help="otherwise print deterministic dry-run commands")
    parser.add_argument("--evaluate-validation", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    configs, sides, seeds = _csv_set(args.configs), _csv_set(args.sides), _csv_set(args.seeds, int)
    caches = {"A": args.cache_a, "B": args.cache_b}
    runs = [
        run for run in expand_matrix(matrix, sorted(seeds) if seeds is not None else None)
        if (configs is None or run.config.config_id in configs)
        and (sides is None or run.side in sides)
    ]
    if args.execute and any(not caches[run.side] for run in runs):
        parser.error("--cache-a/--cache-b are required for every executed side")
    print(f"[phase3.5] protocol={matrix['protocol_version']} runs={len(runs)} execute={args.execute}")
    for run in runs:
        cache = caches[run.side] or f"<CACHE_{run.side}>"
        run_dir = Path(args.output_root) / run.run_id
        checkpoint = run_dir / "checkpoint_best_val.pt"
        train_cmd = [
            sys.executable,
            str(ROOT / "experiments" / "phase3_5" / "train.py"),
            "--matrix", str(matrix_path),
            "--config-id", run.config.config_id,
            "--cache", cache,
            "--side", run.side,
            "--seed", str(run.seed),
            "--output-root", args.output_root,
            "--device", args.device,
        ]
        if args.skip_existing and checkpoint.exists():
            print(f"SKIP {run.run_id}: checkpoint exists")
        elif args.execute:
            print(f"RUN  {run.run_id}", flush=True)
            subprocess.run(train_cmd, cwd=ROOT, check=True)
        else:
            print(subprocess.list2cmdline(train_cmd))
        if args.evaluate_validation:
            eval_cmd = [
                sys.executable,
                str(ROOT / "experiments" / "phase3_5" / "evaluate.py"),
                "--checkpoint", str(checkpoint),
                "--cache", cache,
                "--split", "validation",
                "--device", args.device,
            ]
            if args.execute and checkpoint.exists():
                subprocess.run(eval_cmd, cwd=ROOT, check=True)
            elif not args.execute:
                print(subprocess.list2cmdline(eval_cmd))


if __name__ == "__main__":
    main()
