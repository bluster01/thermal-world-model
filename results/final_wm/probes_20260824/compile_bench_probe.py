"""torch.compile A/B benchmark (2026-08-25).

Measures the anchored seed0 model (real weights) in three configs:
  eager / compile(transition) / compile(full forecast)
Metrics: forward ms, forward+backward+step ms, numerical drift vs eager.
Interleaved rounds to cancel GPU contention from the running v1fix probe.
"""
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
ckpt = torch.load(ROOT / "results/final_wm/probes_20260824/retrain_probe"
                  "/armC_anchor_s1const_seed0/checkpoints/t1_closure_cons_norew_seed0.pt",
                  map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

gen = torch.Generator().manual_seed(7)
batch = sample_windows(record, SPLIT_TRAIN, 32, 96, 18, gen)
history = batch.history.__class__(obs=batch.history.obs.to(DEVICE),
                                  actions=batch.history.actions.to(DEVICE),
                                  boundary=batch.history.boundary.to(DEVICE))
fa = batch.future_actions.to(DEVICE)
fobs = batch.future_obs.to(DEVICE)
fbnd = batch.future_boundary.to(DEVICE)


def fwd(m):
    return m.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)


def fwd_bwd(m):
    r = m.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
    loss = m.observation_nll(r.temps_mu, r.temps_sigma, fobs)
    loss.backward()
    return loss


eager_fwd = fwd(model)
print(f"[sanity] eager temps_mu shape={tuple(eager_fwd.temps_mu.shape)} "
      f"range=[{float(eager_fwd.temps_mu.min()):.1f},{float(eager_fwd.temps_mu.max()):.1f}]", flush=True)

# --- compiled variants ---
import copy

m_trans = copy.deepcopy(model)
m_trans.transition = torch.compile(m_trans.transition)

m_full = copy.deepcopy(model)
m_full.forecast = torch.compile(m_full.forecast)

m_full_ro = copy.deepcopy(model)
m_full_ro.forecast = torch.compile(m_full_ro.forecast, mode="reduce-overhead")


def bench(fn, warmup=3, reps=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.mean(ts))


print("[phase 1] interleaved forward timing (ms per batch of 32)", flush=True)
res = {}
for rnd in range(3):
    for name, fn in (("eager", lambda: fwd(model)),
                     ("trans", lambda: fwd(m_trans)),
                     ("full", lambda: fwd(m_full)),
                     ("full_ro", lambda: fwd(m_full_ro))):
        ms = bench(fn, warmup=3, reps=8)
        res.setdefault(name, []).append(ms)
        print(f"  r{rnd} {name:8s}: {ms:8.1f} ms", flush=True)
print("\n== forward mean ms: " +
      " ".join(f"{k}={np.mean(v):.1f} (speedup {np.mean(res['eager'])/np.mean(v):.2f}x)" for k, v in res.items()))

# numerical drift (compiled vs eager, no grad, same batch)
with torch.no_grad():
    r_t = m_trans.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
    r_f = m_full.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
    r_ro = m_full_ro.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
d_trans = float((r_t.temps_mu - eager_fwd.temps_mu).abs().max())
d_full = float((r_f.temps_mu - eager_fwd.temps_mu).abs().max())
d_ro = float((r_ro.temps_mu - eager_fwd.temps_mu).abs().max())
print(f"\n[numerics] max|d temps_mu|: transition={d_trans:.2e}  full={d_full:.2e}  full_ro={d_ro:.2e}")

model.train(); m_trans.train(); m_full.train(); m_full_ro.train()
print("\n[phase 2] forward+backward (no opt step), train mode, ms", flush=True)
for name, m in (("eager", model), ("trans", m_trans), ("full", m_full), ("full_ro", m_full_ro)):
    m.zero_grad(set_to_none=True)
    ms = bench(lambda: fwd_bwd(m), warmup=3, reps=8)
    print(f"  {name:8s}: {ms:8.1f} ms", flush=True)
