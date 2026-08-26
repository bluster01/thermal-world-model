"""Transport-delay diagnostic (2026-08-26): how long is the real valve->temp delay,
and can a fixed first-order lag represent it?

Data-only. For each load bin, cross-correlate differenced valve position against
differenced downstream temperature over lags 0..60 steps (0..600 s) and report
the peak lag. Plug-flow transport delay should scale ~1/flow, so a LOAD-DEPENDENT
peak lag falsifies any fixed time constant -- which is what the learned constants
(tau_mix1 365-757 s vs prior 80 s; tauB 423-804 s vs prior 120 s) are straining to
emulate.

Pairs (own-side wiring: 1A->left, 2B->left):
  stage-1: 一级A valve -> left sh1_outlet_temp  (and left sh2_inlet as check)
  stage-2: 二级B valve -> left final_outlet_temp
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.contracts import BOUNDARY_ELEMENTS, OBSERVATION_ELEMENTS

A = ROOT / "artifacts/final_wm"
IDX_FLOW = BOUNDARY_ELEMENTS.index("steam_flow")
MAX_LAG = 60           # 600 s
SMOOTH = 3             # 30 s differencing to suppress sensor noise

a_new = np.load(A / "canonical_sideA_v2.npz")
b_new = np.load(A / "canonical_sideB_v2.npz")
obs = a_new["obs"]
flow = a_new["boundary"][:, IDX_FLOW]
print("observation channels:", list(OBSERVATION_ELEMENTS))

pairs = [
    ("stage1 一级A", a_new["actions"][:, 0], "sh1_outlet_temp"),
    ("stage1 一级A", a_new["actions"][:, 0], "sh2_inlet_temp"),
    ("stage2 二级B", a_new["actions"][:, 1], "sh2_outlet_temp"),
    ("stage2 二级B", a_new["actions"][:, 1], "final_outlet_temp"),
    ("stage1 一级B(cross)", b_new["actions"][:, 0], "sh1_outlet_temp"),
]
pairs = [(n, v, t) for n, v, t in pairs if t in OBSERVATION_ELEMENTS]


def diff(x, k=SMOOTH):
    return x[k:] - x[:-k]


def lag_profile(valve, temp, mask):
    """corr(dvalve[t], dtemp[t+lag]) restricted to mask; returns array over lags."""
    dv_full, dt_full = diff(valve), diff(temp)
    out = np.full(MAX_LAG + 1, np.nan)
    m_full = mask[SMOOTH:]
    for lag in range(MAX_LAG + 1):
        dv = dv_full[: len(dv_full) - lag]
        dt = dt_full[lag:]
        m = m_full[: len(dv)]
        if m.sum() < 2000:
            continue
        x, y = dv[m], dt[m]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        out[lag] = np.corrcoef(x, y)[0, 1]
    return out


edges = np.quantile(flow, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
print(f"\nspan n={len(flow)}  flow {flow.min():.0f}-{flow.max():.0f} kg/s")
print(f"lag sweep 0-{MAX_LAG*10} s, {SMOOTH*10} s differencing\n")

for name, valve, tname in pairs:
    temp = obs[:, OBSERVATION_ELEMENTS.index(tname)]
    print(f"--- {name} -> {tname} ---")
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        m = (flow >= lo) & (flow <= hi if i == 4 else flow < hi)
        prof = lag_profile(valve, temp, m)
        if np.all(np.isnan(prof)):
            print(f"  Q{i+1} {lo:5.0f}-{hi:5.0f}: no data")
            continue
        # cooling response is negative: take the most negative correlation
        k = int(np.nanargmin(prof))
        k0 = float(prof[0])
        # half-peak crossing = rough delay onset
        peak = prof[k]
        onset = next((j for j in range(k + 1) if not np.isnan(prof[j])
                      and prof[j] <= 0.5 * peak), k)
        print(f"  Q{i+1} {lo:5.0f}-{hi:5.0f}: peak lag={k*10:4d}s (corr {peak:+.3f})  "
              f"onset(50%)={onset*10:4d}s  corr@0s={k0:+.3f}  "
              f"gain0/peak={abs(k0/peak) if peak else float('nan'):.2f}")
    print()

print("model's learned first-order constants for comparison:")
print("  tau_mix1 prior 80s -> learned 365 / 757 / 658 s   (2.1-9.5x)")
print("  tauB     prior 120s -> learned 423 / 804 / 615 s  (3.5-6.7x)")
print("  UA1      prior 600 -> learned 246 / 402 / 494")
print("done")
