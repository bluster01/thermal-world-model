#!/usr/bin/env python3
"""Cache-free RM3-AV2 integrity replay and Supervisor evidence assembly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase35.multistep.rm3av_audit import build_av2_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", default=str(ROOT / "configs/phase3_5/ms3r_rm3av_matrix.json")
    )
    parser.add_argument(
        "--input-root", default=str(ROOT / "results/phase3_5/ms3r_rm3av")
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results/phase3_5/ms3r_rm3av2/supervisor_evidence_validation.json"
        ),
    )
    args = parser.parse_args()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    payload = build_av2_audit(Path(args.input_root).resolve(), matrix, repo_root=ROOT)
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
