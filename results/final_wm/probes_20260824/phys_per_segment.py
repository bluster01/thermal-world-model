"""Per-load-segment accuracy: production arm vs (later) black-box baselines.

Same protocol as v05 black-box pack and auditpack: sideA record, val split,
256 windows (sample_windows seed 50_000), oracle boundary, H=18.
Bins = load quintiles of the window-start steam flow (identical to
binning_stats). Outputs per-quintile per-channel MAE (H1/H18) for the
norew production arm, 3 seeds, plus per-quintile window counts (data
sparsity = extrapolation exposure).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import binning_stats, window_abs_errors
from src.final_wm.contracts import OBSERVATION_ELEMENTS
from src.final_wm.data import CanonicalRecord, SPLIT_VAL
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
OUT = Path("/tmp/grid_out")
SEEDS = (0, 1, 2)
N_WIN = 256

props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")

per_seed = {}
for seed in SEEDS:
    spec = ms._base(
        "t1", "closure_cons_norew", seed, boundary_mode="oracle",
        initial_state_mode="hybrid", closure_mode="conservative_norew",
        epochs=60, patience=10,
    )
    model = build_world_model(spec, props).to(DEVICE)
    ckpt = torch.load(
        ROOT / f"artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed{seed}.pt",
        map_location=DEVICE, weights_only=False,
    )["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    errs = window_abs_errors(
        model, record, SPLIT_VAL, n_windows=N_WIN, batch_size=32,
        history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
        boundary_mode="oracle", seed=50_000, device=DEVICE,
    )
    bins = binning_stats(errs)
    # window counts per quintile (same edges as binning_stats)
    load = errs.load.numpy()
    edges = np.quantile(load, np.linspace(0, 1, 6)[1:-1])
    bin_id = np.clip(np.digitize(load, edges), 0, 4)
    counts = [int((bin_id == b).sum()) for b in range(5)]
    per_seed[f"seed{seed}"] = {"bins": bins, "quintile_window_counts": counts,
                               "load_edges": edges.tolist()}
    print(f"seed{seed} done; counts={counts}", flush=True)

# aggregate across seeds: mean + std of bin_means per channel per horizon
agg = {"H1": {}, "H18": {}}
for h in ("H1", "H18"):
    for ch in OBSERVATION_ELEMENTS:
        rows = np.array([per_seed[f"seed{s}"]["bins"][h][ch]["bin_means"] for s in SEEDS])
        agg[h][ch] = {
            "mean": rows.mean(0).tolist(),
            "std": rows.std(0).tolist(),
        }
agg["quintile_window_counts"] = per_seed["seed0"]["quintile_window_counts"]
agg["load_edges"] = per_seed["seed0"]["load_edges"]

(OUT / "phys_per_segment.json").write_text(
    json.dumps({"protocol": "sideA val, 256 win seed 50k, oracle bnd, H18",
                "arm": "closure_cons_norew", "seeds": SEEDS, "agg": agg,
                "per_seed": per_seed}, indent=2))
print("saved phys_per_segment.json", flush=True)

# compact print: H18 final_outlet + sh1_outlet per quintile
print("\n=== H18 per-quintile MAE [degC] (mean over seeds) ===")
hdr = "channel      Q1(low)  Q2      Q3      Q4      Q5(high)  counts"
for ch in ("final_outlet_temp", "sh1_outlet_temp", "sh1_inlet_temp"):
    m = agg["H18"][ch]["mean"]
    print(f"{ch:14s} " + " ".join(f"{v:7.2f}" for v in m) + f"   {agg['quintile_window_counts']}")
print("\n=== H1 per-quintile MAE [degC] (mean over seeds) ===")
for ch in ("final_outlet_temp", "sh1_outlet_temp", "sh1_inlet_temp"):
    m = agg["H1"][ch]["mean"]
    print(f"{ch:14s} " + " ".join(f"{v:7.2f}" for v in m))
print("DONE", flush=True)
