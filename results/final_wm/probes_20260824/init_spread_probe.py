"""Cross-seed spread decomposition: init luck vs training noise (2026-08-25).

(1) val NLL at INITIALIZATION (untrained) for seeds 0..5 — if these differ
    a lot, the init point itself carries signal; if near-identical, the
    spread is generated during training (SGD data-stream noise + landscape).
(2) pairwise L2 distance between FINAL checkpoints (prod s0/s1/s2, armA
    s0/s1) per module group — how far apart are the reached optima.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.evaluation import evaluate_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)

def spec_for(seed):
    return ms._base("t1", "closure_cons_norew", seed, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative_norew",
                    epochs=60, patience=10)

print("[1] val NLL at initialization (untrained), 64 windows seed 50k:")
for seed in range(6):
    torch.manual_seed(seed)
    m = build_world_model(spec_for(seed), props).to(DEVICE)
    met = evaluate_windows(m, record, SPLIT_VAL, n_windows=64, batch_size=32,
                           history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
                           boundary_mode="oracle", seed=50_000, device=DEVICE)
    print(f"  seed{seed}: init val_nll={met.nll.mean().item():.4f} "
          f"mae={met.mae.mean().item():.4f}")
    del m
    torch.cuda.empty_cache()

print("\n[2] final-checkpoint pairwise L2 distances (per module group):")
def load_ckpt(path):
    torch.manual_seed(0)
    m = build_world_model(spec_for(0), props).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE, weights_only=False)["state_dict"]
    m.load_state_dict(sd)
    return m

def param_vecs(m):
    return {k: v.detach().flatten().float() for k, v in m.state_dict().items()}

def group_dist(a, b, prefix):
    keys = [k for k in a if k.startswith(prefix)]
    d = sum((a[k] - b[k]).pow(2).sum().item() for k in keys) ** 0.5
    n = sum(a[k].numel() for k in keys)
    return d, d / (n ** 0.5)

ckpts = {
    "prod_s0": ROOT / "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed0.pt",
    "prod_s1": ROOT / "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed1.pt",
    "prod_s2": ROOT / "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed2.pt",
    "armA_s0": ROOT / "results/final_wm/probes_20260824/retrain_probe/armA_budget/checkpoints/t1_closure_cons_norew_seed0.pt",
    "armA_s1": ROOT / "results/final_wm/probes_20260824/retrain_probe/armA_budget_seed1/checkpoints/t1_closure_cons_norew_seed1.pt",
}
models = {k: param_vecs(load_ckpt(p)) for k, p in ckpts.items()}
groups = ["transition", "observation", "observer", "boundary_model", "closure"]
for a, b in (("prod_s0", "prod_s1"), ("prod_s0", "prod_s2"), ("prod_s1", "prod_s2"),
             ("armA_s0", "armA_s1"), ("prod_s0", "armA_s0")):
    print(f"  {a} vs {b}:")
    for g in groups:
        d, dn = group_dist(models[a], models[b], g)
        print(f"    {g:16s} L2={d:9.1f}  per-param RMS={dn:8.4f}")
