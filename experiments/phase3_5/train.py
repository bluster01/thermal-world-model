#!/usr/bin/env python3
"""Train exactly one Phase 3.5 run; checkpoint selection is validation-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.phase35.data import load_cache
from src.phase35.matrix import get_experiment_config, load_matrix
from src.phase35.training import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--side", required=True, choices=["A", "B"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default="results/phase3_5/runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    config = get_experiment_config(matrix, args.config_id)
    cache = load_cache(args.cache)
    run_id = f"{args.side}_{config.config_id}_s{args.seed}"
    result = train_one(
        cache=cache,
        config=config,
        side=args.side,
        seed=args.seed,
        output_dir=Path(args.output_root) / run_id,
        device=args.device,
        overwrite=args.overwrite,
        repo_root=ROOT,
    )
    print(json.dumps({
        "run_id": run_id,
        "checkpoint": str(result.checkpoint),
        "best_epoch": result.best_epoch,
        "validation": result.validation_metrics,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
