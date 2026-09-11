"""Materialize the frozen spec and source identities without opening data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experiments.fmts_mainsteam_20260911.spec import serialized_spec

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "frozen_spec.json"
HASHED_SOURCES = (
    "src/final_wm/contracts.py",
    "src/final_wm/observer.py",
    "src/final_wm/training.py",
    "src/final_wm/model.py",
    "src/final_wm/transition.py",
    "src/final_wm/closure.py",
    "src/final_wm/boundary.py",
    "experiments/fmts_mainsteam_20260911/spec.py",
    "experiments/fmts_mainsteam_20260911/data.py",
    "experiments/fmts_mainsteam_20260911/models.py",
    "experiments/fmts_mainsteam_20260911/run.py",
    "experiments/fmts_mainsteam_20260911/audit.py",
    "experiments/fmts_mainsteam_20260911/smoke_pipeline.py",
    "configs/final_wm/channel_mapping_v2.json",
    "experiments/fmts_mainsteam_20260911/manifests.py",
    "experiments/fmts_mainsteam_20260911/smoke.py",
    "experiments/fmts_mainsteam_20260911/DESIGN_DECISION_ZH.md",
    "docs/fmts2026/PREREG_THREE_MODEL_MAIN_STEAM_20260911.md",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    payload = serialized_spec()
    payload["preparation"] = {
        "git_head_before_new_commit": git_head(),
        "canonical_data_opened": False,
        "locked_test_opened": False,
        "source_sha256": {name: sha256(ROOT / name) for name in HASHED_SOURCES},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
