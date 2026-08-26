"""IAPWS interp optimization probe (2026-08-25).

GridThermoProperties hot path = searchsorted per element per substep.
If grids are regular, replace with pure arithmetic index math (vectorized,
dynamo-friendly). Probe-side subclass only -- no src changes.
Numerical consistency gate: max|d| vs original interp + full-forecast drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.properties import (GridThermoProperties, interp1d, interp2d,
                                     load_grid_properties, ste_clamp)

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
npz = ROOT / "artifacts/final_wm/iapws_surrogate.npz"
raw = dict(np.load(npz))
props = GridThermoProperties(raw, device=DEVICE)

# ---- grid regularity ----
print("== grid shapes & regularity")
for key in ("P", "H", "Tg", "Psub", "t_liq"):
    g = torch.tensor(np.asarray(raw[key]), dtype=torch.float64)
    d = (g[1:] - g[:-1])
    reg = float((d.max() / d.min()).item())
    print(f"  {key}: n={g.numel()} range=[{float(g[0]):.4g},{float(g[-1]):.4g}] "
          f"spacing_max/min={reg:.6f}")


def _idx_regular(g0: float, g1: float, n: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Regular-grid index math: idx = upper-node (insertion) index in [1, n-1],
    weight = fraction within cell [idx-1, idx], in [0,1]."""
    d = (g1 - g0) / (n - 1)
    raw_i = (x - g0) / d
    idx = raw_i.ceil().clamp(1, n - 1).long()
    weight = (raw_i - idx.float() + 1.0).clamp(0.0, 1.0)
    return idx, weight


class FastGridThermoProperties(GridThermoProperties):
    """Same grids, arithmetic-index interpolation instead of searchsorted."""

    def __init__(self, arrays, *, device="cpu"):
        super().__init__(arrays, device=device)
        for k, t in (("P", self._p), ("H", self._h), ("Tg", self._tg),
                     ("Psub", self._psub), ("t_liq", self._tliq)):
            d = (t[1:] - t[:-1])
            rel = float(((d.max() / d.min()) - 1.0).item())
            if rel > 1e-4:
                raise RuntimeError(f"grid {k} not regular enough (rel={rel:.2e})")
        self._d = {k: (float(t[0]), float(t[-1]), t.numel())
                   for k, t in (("P", self._p), ("H", self._h), ("Tg", self._tg),
                                ("Psub", self._psub), ("t_liq", self._tliq))}

    def _i2d(self, grid_k, grid_v, table, r, c):
        g0, g1, n = self._d[grid_k]
        ir, wr = _idx_regular(g0, g1, n, r)
        h0, h1, m = self._d[grid_v]
        ic, wc = _idx_regular(h0, h1, m, c)
        v00 = table[ir - 1, ic - 1]
        v01 = table[ir - 1, ic]
        v10 = table[ir, ic - 1]
        v11 = table[ir, ic]
        top = v00 + wc * (v01 - v00)
        bot = v10 + wc * (v11 - v10)
        return top + wr * (bot - top)

    def _i1d(self, grid_k, values, x):
        g0, g1, n = self._d[grid_k]
        idx, w = _idx_regular(g0, g1, n, x)
        return values[idx - 1] + w * (values[idx] - values[idx - 1])

    def temperature_of_ph(self, p, h):
        b = self.bounds
        p = p.clamp(b.p_lo, b.p_hi)
        h = ste_clamp(h, b.h_lo, b.h_hi)
        return self._i2d("P", "H", self._tph, p, h)

    def enthalpy_of_pt(self, p, temperature):
        b = self.bounds
        p = p.clamp(b.p_lo, b.p_hi)
        temperature = temperature.clamp(b.t_lo, b.t_hi)
        return self._i2d("P", "Tg", self._hpt, p, temperature)

    def saturated_vapor_enthalpy(self, p):
        return self._i1d("Psub", self._hsatv, p)

    def liquid_enthalpy(self, temperature):
        return self._i1d("t_liq", self._hliq, temperature)


fast = FastGridThermoProperties(raw, device=DEVICE)

# ---- numerical consistency ----
print("\n== numerics: fast vs original (random queries, GPU)")
torch.manual_seed(0)
for name, fn_orig, fn_fast, n in (
        ("temperature_of_ph", props.temperature_of_ph, fast.temperature_of_ph, 100_000),
        ("enthalpy_of_pt", props.enthalpy_of_pt, fast.enthalpy_of_pt, 100_000),
        ("sat_vapor_enthalpy", props.saturated_vapor_enthalpy, fast.saturated_vapor_enthalpy, 100_000),
        ("liquid_enthalpy", props.liquid_enthalpy, fast.liquid_enthalpy, 100_000)):
    b = props.bounds
    if name == "temperature_of_ph":
        p = torch.rand(n, device=DEVICE) * (b.p_hi - b.p_lo) + b.p_lo
        q = torch.rand(n, device=DEVICE) * (b.h_hi - b.h_lo) + b.h_lo
    elif name == "enthalpy_of_pt":
        p = torch.rand(n, device=DEVICE) * (b.p_hi - b.p_lo) + b.p_lo
        q = torch.rand(n, device=DEVICE) * (b.t_hi - b.t_lo) + b.t_lo
    else:
        p = torch.rand(n, device=DEVICE) * (b.p_hi - b.p_lo) + b.p_lo
        q = torch.rand(n, device=DEVICE) * (b.t_hi - b.t_lo) + b.t_lo
    a = fn_orig(p, q) if name in ("temperature_of_ph", "enthalpy_of_pt") else fn_orig(p)
    c = fn_fast(p, q) if name in ("temperature_of_ph", "enthalpy_of_pt") else fn_fast(p)
    print(f"  {name}: max|d|={float((a - c).abs().max()):.3e}")

# ---- micro bench ----
print("\n== micro bench (ms, 100k queries)", flush=True)
def t(fn, args, reps=10):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(*args); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.mean(ts))

n = 100_000
b = props.bounds
p1 = torch.rand(n, device=DEVICE) * (b.p_hi - b.p_lo) + b.p_lo
h1 = torch.rand(n, device=DEVICE) * (b.h_hi - b.h_lo) + b.h_lo
t1 = torch.rand(n, device=DEVICE) * (b.t_hi - b.t_lo) + b.t_lo
p2 = torch.rand(n, device=DEVICE) * (b.p_hi - b.p_lo) + b.p_lo

pairs = [
    ("temp_of_ph  orig", props.temperature_of_ph, (p1, h1)),
    ("temp_of_ph  fast", fast.temperature_of_ph, (p1, h1)),
    ("enth_of_pt  orig", props.enthalpy_of_pt, (p1, t1)),
    ("enth_of_pt  fast", fast.enthalpy_of_pt, (p1, t1)),
    ("sat_vap     orig", props.saturated_vapor_enthalpy, (p2,)),
    ("sat_vap     fast", fast.saturated_vapor_enthalpy, (p2,)),
    ("liq_enth    orig", props.liquid_enthalpy, (t1,)),
    ("liq_enth    fast", fast.liquid_enthalpy, (t1,)),
]
res = {}
for name, fn, args in pairs:
    ms = t(fn, args)
    res[name] = ms
    print(f"  {name}: {ms:8.3f} ms", flush=True)
print("\n== speedup summary")
for a, b_ in (("temp_of_ph", "temp_of_ph"), ("enth_of_pt", "enth_of_pt"),
              ("sat_vap", "sat_vap"), ("liq_enth", "liq_enth")):
    print(f"  {a}: {res[f'{b_}  orig']/res[f'{b_}  fast']:.2f}x")

# ---- end-to-end forecast check ----
print("\n== end-to-end: forecast with fast props vs orig props")
from src.final_wm.data import SPLIT_TRAIN, CanonicalRecord, sample_windows
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative_norew",
                epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
ckpt_path = (ROOT / "results/final_wm/probes_20260824/retrain_probe"
             "/armC_anchor_s1const_seed0/checkpoints/t1_closure_cons_norew_seed0.pt")
mo = build_world_model(spec, props).to(DEVICE)
mo.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False)["state_dict"])
mf = build_world_model(spec, fast).to(DEVICE)
mf.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False)["state_dict"])
mo.eval(); mf.eval()
gen = torch.Generator().manual_seed(7)
batch = sample_windows(record, SPLIT_TRAIN, 32, 96, 18, gen)
history = batch.history.__class__(obs=batch.history.obs.to(DEVICE),
                                  actions=batch.history.actions.to(DEVICE),
                                  boundary=batch.history.boundary.to(DEVICE))
fa = batch.future_actions.to(DEVICE)
fbnd = batch.future_boundary.to(DEVICE)
with torch.no_grad():
    ro = mo.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
    rf = mf.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
d = float((ro.temps_mu - rf.temps_mu).abs().max())
print(f"  forecast temps_mu max|d| (orig vs fast props) = {d:.3e}", flush=True)
for name, m in (("orig", mo), ("fast", mf)):
    ms_t = t(lambda: m.forecast(history, fa, boundary_mode="oracle",
                                true_future_boundary=fbnd), (), reps=8)
    print(f"  forecast {name}: {ms_t:8.1f} ms", flush=True)
print("DONE")
