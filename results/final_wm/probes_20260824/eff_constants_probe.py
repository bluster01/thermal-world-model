"""Effective transition-constant divergence across seeds (2026-08-25).

raw params are prior*softplus(raw) / prior*tanh(raw). Compare EFFECTIVE
constants (physics space) across prod s0/1/2 and armA s0/1/2, relative
to the priors. Also inspect which groups moved furthest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.transition import (
    TRANSITION_PARAM_PRIORS,
    _SIGNED_PARAMS,
)

CKPTS = {
    "prod_s0": "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed0.pt",
    "prod_s1": "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed1.pt",
    "prod_s2": "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed2.pt",
    "armA_s0": "results/final_wm/probes_20260824/retrain_probe/armA_budget/checkpoints/t1_closure_cons_norew_seed0.pt",
    "armA_s1": "results/final_wm/probes_20260824/retrain_probe/armA_budget_seed1/checkpoints/t1_closure_cons_norew_seed1.pt",
    "armA_s2": "results/final_wm/probes_20260824/retrain_probe/armA_budget_seed2/checkpoints/t1_closure_cons_norew_seed2.pt",
}


def eff_constants(raw: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    for name, r in raw.items():
        prior = TRANSITION_PARAM_PRIORS[name]
        v = float(r.item()) if r.numel() == 1 else None
        if name in _SIGNED_PARAMS:
            out[name] = prior * np.tanh(v)
        else:
            out[name] = prior * float(np.log1p(np.exp(v)))
    return out


data = {}
for tag, p in CKPTS.items():
    sd = torch.load(ROOT / p, map_location="cpu", weights_only=False)["state_dict"]
    raw = {k.split("transition.raw.")[1]: v for k, v in sd.items()
           if k.startswith("transition.raw.")}
    data[tag] = eff_constants(raw)

names = list(TRANSITION_PARAM_PRIORS)
print(f"{'param':10s} {'prior':>9s} | " + " | ".join(
    f"{t.replace('armA_','A').replace('prod_','P')}" for t in CKPTS))
for n in names:
    pr = TRANSITION_PARAM_PRIORS[n]
    row = f"{n:10s} {pr:9.1f} |"
    for t in CKPTS:
        row += f" {data[t][n]:9.2f}"
    print(row)
print()
# ratio spread per param: max/min of effective across armA seeds
print("eff spread (max/min across armA s0/s1/s2):")
spreads = []
for n in names:
    vs = [data[f"armA_s{s}"][n] for s in (0, 1, 2)]
    r = max(vs) / max(min(vs), 1e-9)
    spreads.append((r, n))
for r, n in sorted(spreads, reverse=True)[:10]:
    print(f"  {n:10s} x{r:.2f}   values={[round(data[f'armA_s{s}'][n], 2) for s in (0,1,2)]} pri={TRANSITION_PARAM_PRIORS[n]}")
# and prod spread
print("eff spread (max/min across prod s0/s1/s2):")
for r, n in sorted([(max(data[f'prod_s{s}'][n] for s in (0,1,2)) / max(min(data[f'prod_s{s}'][n] for s in (0,1,2)), 1e-9), n)
                    for n in names], reverse=True)[:8]:
    print(f"  {n:10s} x{r:.2f}   values={[round(data[f'prod_s{s}'][n], 2) for s in (0,1,2)]}")
