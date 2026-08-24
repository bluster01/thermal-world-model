"""Design-side T1 verdict replay for sideB (read-only).

Mirrors the sideA reissue (results/final_wm/t1_verdict_reissue_20260823.md):
loads archived metrics blobs, reconstructs WindowMetrics, and applies the
runner's frozen functions (_seed_passes with THRESH_T1_NLL, _verdict with
MIN_SEED_PASSES).  No retraining, no runner edits, no summary writes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.final_wm.evaluation import WindowMetrics
from experiments.final_wm import matrix_spec as ms
from experiments.final_wm.run_matrix import _seed_passes, _verdict

OUT = Path("artifacts/final_wm_sideB")


def load(run_id: str) -> WindowMetrics:
    blob = torch.load(OUT / "metrics" / f"{run_id}.pt", map_location="cpu",
                      weights_only=False)
    if "metrics" not in blob:
        raise SystemExit(f"{run_id}: legacy flat metrics blob (no fingerprint) -- refuse")
    return WindowMetrics(**blob["metrics"])


def main() -> None:
    seeds = [0, 1, 2]
    pairs_list = {
        "closure_cons_vs_physics_only": ("physics_only", "closure_cons"),
        "closure_cons_norew_vs_physics_only": ("physics_only", "closure_cons_norew"),
        "closure_cons_norew_vs_closure_cons": ("closure_cons", "closure_cons_norew"),
    }
    report = {"side": "B", "threshold": ms.THRESH_T1_NLL,
              "min_seed_passes": ms.MIN_SEED_PASSES, "comparisons": {}}
    for name, (base_arm, arm) in pairs_list.items():
        pairs = [(load(f"t1_{base_arm}_seed{s}"), load(f"t1_{arm}_seed{s}"))
                 for s in seeds]
        passes, details = _seed_passes(pairs, ms.THRESH_T1_NLL)
        report["comparisons"][name] = {
            "verdict": _verdict(passes, len(seeds)),
            "passes": passes,
            "per_seed": details,
        }
    print(json.dumps(report, indent=2))
    Path("results/final_wm/t1_verdict_sideB_replay_20260824.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
