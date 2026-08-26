"""Anchor bisection iteration sweep: iters vs accuracy vs speed (probe-side).

Monkeypatches _invert_spray_anchor's iters default; measures temps_mu drift
vs the protocol value (24) and per-forecast wall time. No src changes.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.data import SPLIT_TRAIN, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from src.final_wm.transition import Fan2020UDETransition
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative_norew",
                epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
ckpt_path = (ROOT / "results/final_wm/probes_20260824/retrain_probe"
             "/armC_anchor_s1const_seed0/checkpoints/t1_closure_cons_norew_seed0.pt")

gen = torch.Generator().manual_seed(7)
batch = sample_windows(record, SPLIT_TRAIN, 32, 96, 18, gen)
history = batch.history.__class__(obs=batch.history.obs.to(DEVICE),
                                  actions=batch.history.actions.to(DEVICE),
                                  boundary=batch.history.boundary.to(DEVICE))
fa = batch.future_actions.to(DEVICE)
fbnd = batch.future_boundary.to(DEVICE)

orig = Fan2020UDETransition._invert_spray_anchor

def set_iters(k):
    if k is None:
        Fan2020UDETransition._invert_spray_anchor = orig
    else:
        Fan2020UDETransition._invert_spray_anchor = functools.partialmethod(orig, iters=k)

def build():
    m = build_world_model(spec, props).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False)["state_dict"])
    m.eval()
    return m

def fwd(m):
    with torch.no_grad():
        return m.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)

def bench(fn, warmup=3, reps=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.mean(ts))

set_iters(None)
m24 = build()
r24 = fwd(m24)
t24 = bench(lambda: fwd(m24))

print(f"iters=24 (protocol): forecast {t24:.1f} ms", flush=True)
for k in (16, 12, 8, 6, 4):
    set_iters(k)
    m = build()
    r = fwd(m)
    d = float((r.temps_mu - r24.temps_mu).abs().max())
    t = bench(lambda: fwd(m))
    print(f"iters={k:2d}: forecast {t:6.1f} ms ({t24/t:5.2f}x) | max|d temps_mu| vs 24 = {d:.3e} degC",
          flush=True)
    del m
    torch.cuda.empty_cache()
print("DONE")
