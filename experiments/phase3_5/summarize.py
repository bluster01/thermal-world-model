#!/usr/bin/env python3
"""Aggregate Phase 3.5 runs and emit conservative, preregistered conclusions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.reporting import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="results/phase3_5/runs")
    parser.add_argument("--matrix", default="configs/phase3_5/experiment_matrix.json")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output-dir", default="results/phase3_5")
    args = parser.parse_args()
    result = summarize(args.run_root, args.matrix, args.split)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"summary_{args.split}.json"
    markdown_path = output / f"summary_{args.split}.md"
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_md = markdown_path.with_suffix(".md.tmp")
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "markdown"}, f, ensure_ascii=False, indent=2)
    with tmp_md.open("w", encoding="utf-8") as f:
        f.write(result["markdown"])
    os.replace(tmp_json, json_path)
    os.replace(tmp_md, markdown_path)
    print(result["markdown"])


if __name__ == "__main__":
    main()
