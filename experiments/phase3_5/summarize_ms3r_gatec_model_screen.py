#!/usr/bin/env python3
"""Summarize local Gate C synthetic diagnostics without making a decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def summarize(input_dir: Path) -> dict[str, Any]:
    manifests = sorted(input_dir.glob("run_manifest*.json"))
    if not manifests:
        raise RuntimeError("Gate C summary found no run manifests")
    runs = [_read_json(path) for path in manifests]
    if any(run.get("scope") != "local_synthetic_smoke_only" for run in runs):
        raise RuntimeError("Gate C local summarizer only accepts synthetic smoke runs")
    return {
        "scope": "local_synthetic_diagnostic_only",
        "run_count": len(runs),
        "candidate_ids": [run["candidate_id"] for run in runs],
        "selector_eligible_count": sum(bool(run["selector_eligible"]) for run in runs),
        "test_accessed": any(bool(run["test_accessed"]) for run in runs),
        "real_training_executed": any(bool(run["real_training_executed"]) for run in runs),
        "automatic_scientific_pass": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    payload = summarize(input_dir)
    output = Path(args.output).resolve() if args.output else input_dir / "summary_local.json"
    _atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
