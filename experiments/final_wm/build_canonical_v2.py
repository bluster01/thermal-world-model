"""Execution entry: build canonical v2 for one side (v0.6 Phase 1).

    python experiments/final_wm/build_canonical_v2.py --side A \
        --data-root /home/bluster/Desktop/AI --v1 artifacts/final_wm/canonical_sideA.npz \
        --out artifacts/final_wm/canonical_sideA_v2.npz

Execution must not edit the mapping or gates; gate breaches are reported as-is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.final_wm.data_v2 import build_canonical_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["A", "B"], required=True)
    parser.add_argument("--data-root", required=True,
                        help="root containing merged_all_data/all_merged_10s.csv")
    parser.add_argument("--v1", required=True, help="path to the side's v1 canonical npz")
    parser.add_argument("--out", required=True, help="output v2 npz path")
    parser.add_argument("--mapping", default="configs/final_wm/channel_mapping_v2.json")
    args = parser.parse_args()

    meta = build_canonical_v2(args.v1, args.data_root, args.mapping, args.out, side=args.side)
    print(f"[v2 side {args.side}] built {args.out}")
    print(f"  n={meta['n_samples']} alignment checks: "
          + ", ".join(f"{r['channel']}->{r['column']} corr={r['corr']:.4f}"
                      for r in meta["alignment"]))
    print(json.dumps({"side": args.side, "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
