"""Minimal phase-2: fwd+bwd eager vs full-compile (train mode)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.data import SPLIT_TRAIN, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative_norew",
                epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
model = build_world_model(spec, props).to(DEVICE)
model.load_state_dict(torch.load(
    ROOT / "results/final_wm/probes_20260824/retrain_probe"
    "/armC_anchor_s1const_seed0/checkpoints/t1_closure_cons_norew_seed0.pt",
    map_location=DEVICE, weights_only=False)["state_dict"])
model.train()

gen = torch.Generator().manual_seed(7)
batch = sample_windows(record, SPLIT_TRAIN, 32, 96, 18, gen)
history = batch.history.__class__(obs=batch.history.obs.to(DEVICE),
                                  actions=batch.history.actions.to(DEVICE),
                                  boundary=batch.history.boundary.to(DEVICE))
fa = batch.future_actions.to(DEVICE)
fobs = batch.future_obs.to(DEVICE)
fbnd = batch.future_boundary.to(DEVICE)

import copy
m_full = copy.deepcopy(model)
m_full.forecast = torch.compile(m_full.forecast)


def fwd_bwd(m):
    r = m.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
    loss = m.observation_nll(r.temps_mu, r.temps_sigma, fobs)
    loss.backward()
    return float(loss)


def bench(fn, warmup=3, reps=6):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.mean(ts))


res = {}
for rnd in range(2):
    for name, m in (("eager", model), ("full", m_full)):
        m.zero_grad(set_to_none=True)
        ms = bench(lambda: fwd_bwd(m), warmup=3, reps=6)
        res.setdefault(name, []).append(ms)
        print(f"r{rnd} {name:6s} fwd+bwd: {ms:.1f} ms", flush=True)
print("MEAN", " ".join(f"{k}={np.mean(v):.1f}ms ({np.mean(res['eager'])/np.mean(v):.2f}x)"
                       for k, v in res.items()), flush=True)
print("DONE")
