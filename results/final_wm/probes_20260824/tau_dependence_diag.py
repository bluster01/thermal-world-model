"""Is the transport lag only load-dependent, or does it need other boundary vars?
(2026-08-26, user question)

The probe scheduled tau by flow alone: tau = tau0 * (mdot/mdot_ref)^-alpha, and
training learned alpha_tau = 1.35 (1.42 in the earlier void run) instead of the
plug-flow value 1.0. But the physical residence time is

    tau = V * rho / mdot            (mass in the volume / mass flow)

so tau depends on DENSITY too, and rho = rho(p, T) swings hugely in a sliding-
pressure supercritical unit (12 -> 26 MPa here). A flow-only schedule folds the
density variation into the flow exponent, which would explain alpha > 1.

Questions answered here, data-side:
  Q1 how collinear are mdot and the other boundary channels? (if p is ~fully
     predictable from mdot, a flow-only schedule is observationally equivalent
     and alpha absorbs the rest)
  Q2 is there independent variation at FIXED load (which would make a 2-D
     schedule identifiable at all)?
  Q3 what does the physically correct group rho/mdot predict for the Q1->Q5
     lag ratio, versus pure 1/mdot, versus the MEASURED lag ratio 2.33?
  Q4 which boundary channels carry information about the residence time beyond
     flow (partial correlations)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from iapws import IAPWS97

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.contracts import BOUNDARY_ELEMENTS, OBSERVATION_ELEMENTS

P = ROOT / "results/final_wm/probes_20260824"
raw = np.load(P / "v1fix_probe/canonical_sideA_v1fixed.npz")
bnd, obs = raw["boundary"], raw["obs"]
B = {n: bnd[:, i] for i, n in enumerate(BOUNDARY_ELEMENTS)}
O = {n: obs[:, i] for i, n in enumerate(OBSERVATION_ELEMENTS)}
flow = B["steam_flow"]
edges = np.quantile(flow, [0, .2, .4, .6, .8, 1.0])
bins = [(flow >= edges[i]) & (flow <= edges[i+1] if i == 4 else flow < edges[i+1])
        for i in range(5)]

print("=" * 78)
print("Q1  collinearity of each boundary/obs channel with steam flow")
print("=" * 78)
print(f"{'channel':24s} {'corr':>8s} {'R2 on flow':>11s} {'resid std':>10s} {'std':>9s}")
for name, v in list(B.items()) + [("sh1_outlet_temp", O["sh1_outlet_temp"]),
                                  ("final_outlet_temp", O["final_outlet_temp"])]:
    if name == "steam_flow":
        continue
    c = np.corrcoef(flow, v)[0, 1]
    # linear + quadratic fit on flow (sliding-pressure curves are nonlinear)
    A = np.vstack([np.ones_like(flow), flow, flow ** 2]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ coef
    resid = v - pred
    r2 = 1 - resid.var() / v.var()
    print(f"{name:24s} {c:+8.3f} {r2:11.3f} {resid.std():10.3f} {v.std():9.3f}")

print("\n" + "=" * 78)
print("Q2  independent variation at FIXED load (pressure spread inside a load bin)")
print("=" * 78)
for i, m in enumerate(bins):
    p = B["separator_pressure"][m]
    print(f"  Q{i+1} flow {edges[i]:5.0f}-{edges[i+1]:5.0f} (n={m.sum():6d}): "
          f"p_sep mean={p.mean():6.2f} std={p.std():5.2f} MPa "
          f"range={p.min():5.2f}-{p.max():5.2f} "
          f"(spread/mean = {p.std()/p.mean()*100:4.1f}%)")
# narrow-load slice: is there enough pressure variation to identify a 2-D schedule?
mid = (flow > np.quantile(flow, .48)) & (flow < np.quantile(flow, .52))
print(f"\n  narrow load slice (48-52 pct, n={mid.sum()}): "
      f"flow {flow[mid].min():.0f}-{flow[mid].max():.0f} kg/s, "
      f"p_sep {B['separator_pressure'][mid].min():.2f}-"
      f"{B['separator_pressure'][mid].max():.2f} MPa "
      f"(std {B['separator_pressure'][mid].std():.2f})")

print("\n" + "=" * 78)
print("Q3  physical residence time rho/mdot vs pure 1/mdot vs MEASURED lag")
print("=" * 78)
print("  density from IAPWS97 at (p_out, final_outlet_temp) -- the SH2->final path")


def rho_of(p_mpa, t_c, n=4000):
    idx = np.linspace(0, len(p_mpa) - 1, min(n, len(p_mpa))).astype(int)
    out = np.full(len(idx), np.nan)
    for k, j in enumerate(idx):
        try:
            st = IAPWS97(P=float(p_mpa[j]), T=float(t_c[j]) + 273.15)
            out[k] = st.rho
        except Exception:
            pass
    return idx, out


rows = []
for i, m in enumerate(bins):
    p = B["outlet_pressure"][m]
    t = O["final_outlet_temp"][m]
    idx, r = rho_of(p, t, n=1500)
    r = r[np.isfinite(r)]
    rows.append((flow[m].mean(), np.nanmean(r), p.mean(), t.mean()))
    print(f"  Q{i+1}: flow={rows[-1][0]:6.1f} kg/s  p_out={p.mean():6.2f} MPa  "
          f"T={t.mean():6.1f} C  ->  rho={np.nanmean(r):7.2f} kg/m3  "
          f"(residence proxy rho/mdot = {np.nanmean(r)/flow[m].mean():.4f})")

f1, r1, _, _ = rows[0]
f5, r5, _, _ = rows[-1]
meas = 560 / 240          # measured stage2->final peak lag ratio Q1/Q5
pure = f5 / f1            # 1/mdot prediction
phys = (r1 / f1) / (r5 / f5)
print(f"\n  Q1/Q5 lag ratio:")
print(f"    MEASURED (lag sweep)      = {meas:.2f}")
print(f"    pure 1/mdot               = {pure:.2f}   (error {abs(pure-meas)/meas*100:5.1f}%)")
print(f"    physical rho/mdot         = {phys:.2f}   (error {abs(phys-meas)/meas*100:5.1f}%)")
print(f"    flow-only with alpha=1.35 = {pure**1.35:.2f}   "
      f"(error {abs(pure**1.35-meas)/meas*100:5.1f}%)")
print(f"    density ratio rho_Q1/rho_Q5 = {r1/r5:.3f}")

print("\n" + "=" * 78)
print("Q4  partial correlation with the residence proxy, controlling for flow")
print("=" * 78)
idx, rho_all = rho_of(B["outlet_pressure"], O["final_outlet_temp"], n=6000)
ok = np.isfinite(rho_all)
idx, rho_all = idx[ok], rho_all[ok]
res_proxy = rho_all / flow[idx]


def partial(x, y, z):
    """corr(x, y) after removing the linear+quadratic part of z from both."""
    A = np.vstack([np.ones_like(z), z, z ** 2]).T
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


print(f"  corr(residence proxy, flow)             = "
      f"{np.corrcoef(res_proxy, flow[idx])[0,1]:+.3f}")
for name in ("separator_pressure", "outlet_pressure", "separator_temperature",
             "feedwater_temperature", "coal_command", "spray_flow_total"):
    pc = partial(res_proxy, B[name][idx], flow[idx])
    print(f"  partial corr(residence, {name:22s} | flow) = {pc:+.3f}")
print("\n  -> a large partial correlation means that channel carries residence-time")
print("     information the flow-only schedule cannot represent.")
print("\ndone")
