#!/usr/bin/env python3
"""Zero-training RM3-AV0 replay of returned RM3/RM3-A validation artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.multistep.rm3av_replay import build_av0_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rm3-root", default=str(ROOT / "results/phase3_5/ms3r_rm3"))
    parser.add_argument("--rm3a-root", default=str(ROOT / "results/phase3_5/ms3r_rm3a"))
    parser.add_argument("--cache-a")
    parser.add_argument("--cache-b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchor-count", type=int, default=64)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results/phase3_5/ms3r_rm3av0/supervisor_replay_validation.json"),
    )
    args = parser.parse_args()
    if bool(args.cache_a) != bool(args.cache_b):
        parser.error("functional replay requires both --cache-a and --cache-b")
    payload = build_av0_replay(
        Path(args.rm3_root).resolve(),
        Path(args.rm3a_root).resolve(),
        cache_a=Path(args.cache_a).resolve() if args.cache_a else None,
        cache_b=Path(args.cache_b).resolve() if args.cache_b else None,
        device=args.device,
        anchor_count=args.anchor_count,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
