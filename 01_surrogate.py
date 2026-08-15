#!/usr/bin/env python3
"""ad hoc2 Step 1 前置：IAPWS-IF97 可微代理构建（离线，一次性）
产物 out/iapws_surrogate.npz：
  - T_of_ph: 网格 (Np, Nh) = T(p,h) °C（亚临界两相区=Tsat 平台；超临界区连续）
  - h_of_pT: 网格 (Np, NT) = h(p,T) kJ/kg（亚临界两相区为液态焓——仅限过热/超临界查询）
  - tsat_coef / hsatv_coef: Tsat(p)、h_satV(Tsat) 多项式（仅亚临界 p<=22.064MPa 拟合）
  - p_crit=22.064；边界、分辨率、分段最大拟合误差（对 iapws 参考值校验）
训练/rollout 中所有物性查询走 torch 双线性 grid_sample（可微、快）。
机组为超临界/超超临界（分离器出口压力 max≈27.6MPa）——网格覆盖 [8,30]MPa。
"""
import os
import json
import time
import numpy as np
from iapws import IAPWS97

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)

P_CRIT = 22.064  # MPa
P_LO, P_HI, DP = 8.0, 30.0, 0.05
H_LO, H_HI, DH = 1100.0, 3700.0, 10.0
T_LO, T_HI, DT = 300.0, 650.0, 2.0

P = np.arange(P_LO, P_HI + 1e-9, DP)
H = np.arange(H_LO, H_HI + 1e-9, DH)
Tg = np.arange(T_LO, T_HI + 1e-9, DT)

t0 = time.time()
Tph = np.empty((len(P), len(H)), dtype=np.float64)
for i, p in enumerate(P):
    for j, h in enumerate(H):
        Tph[i, j] = IAPWS97(P=p, h=h).T - 273.15
print(f"T(p,h) grid {Tph.shape} in {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
hpT = np.empty((len(P), len(Tg)), dtype=np.float64)
for i, p in enumerate(P):
    for j, T in enumerate(Tg):
        hpT[i, j] = IAPWS97(P=p, T=T + 273.15).h
print(f"h(p,T) grid {hpT.shape} in {time.time()-t0:.0f}s", flush=True)

# 饱和曲线：仅亚临界
Psub = P[P <= P_CRIT - 1e-6]
Tsat = np.array([IAPWS97(P=p, x=0).T - 273.15 for p in Psub])
hsatV = np.array([IAPWS97(P=p, x=1).h for p in Psub])
tsat_coef = np.polyfit(Psub, Tsat, 8)
tsat_fit = np.polyval(tsat_coef, Psub)
err_tsat = float(np.max(np.abs(tsat_fit - Tsat)))
# hsatV：多项式在临界点病态，改用 1D 精确网格（查询时线性插值，网格点处精确）
hsatv_linerr = float(np.max(np.abs(0.5 * (hsatV[:-2] + hsatV[2:]) - hsatV[1:-1])))  # 相邻中点线性误差上界
print(f"Tsat(p) poly maxerr={err_tsat:.4f}°C, hsatV 1D grid midpoint-linerr<={hsatv_linerr:.3f} kJ/kg", flush=True)

# ---- 校验：双线性 vs iapws 真值 ----
def bilinear2d(grid, p_grid, h_grid, p, h):
    fp = (p - p_grid[0]) / (p_grid[-1] - p_grid[0]) * (len(p_grid) - 1)
    fh = (h - h_grid[0]) / (h_grid[-1] - h_grid[0]) * (len(h_grid) - 1)
    i0 = np.clip(np.floor(fp).astype(int), 0, len(p_grid) - 2)
    j0 = np.clip(np.floor(fh).astype(int), 0, len(h_grid) - 2)
    wp = fp - i0
    wh = fh - j0
    v00 = grid[i0, j0]
    v01 = grid[i0, j0 + 1]
    v10 = grid[i0 + 1, j0]
    v11 = grid[i0 + 1, j0 + 1]
    return ((1 - wp) * ((1 - wh) * v00 + wh * v01) +
            wp * ((1 - wh) * v10 + wh * v11))

rng = np.random.default_rng(0)
errs = {}
# T(p,h)：亚临界 + 超临界 分段
for name, plo, phi in [("Tph_sub", P_LO + 0.3, P_CRIT - 0.5), ("Tph_sup", P_CRIT + 0.5, P_HI - 0.3)]:
    pv = rng.uniform(plo, phi, 300)
    hv = rng.uniform(H_LO + 30, H_HI - 30, 300)
    ref = np.array([IAPWS97(P=float(p), h=float(h)).T - 273.15 for p, h in zip(pv, hv)])
    pred = bilinear2d(Tph, P, H, pv, hv)
    errs[name] = float(np.max(np.abs(pred - ref)))
    print(f"{name}: bilinear maxerr = {errs[name]:.4f}°C", flush=True)
# h(p,T)：仅过热/超临界（T > Tsat(p)+5，p>p_crit 无约束）
pv = rng.uniform(P_LO + 0.3, P_HI - 0.3, 300)
ts_of_p = np.array([np.polyval(tsat_coef, min(float(p), P_CRIT)) for p in pv])
Tv = np.array([rng.uniform(t + 5.0, T_HI - 10.0) for t in ts_of_p])
ref2 = np.array([IAPWS97(P=float(p), T=float(T + 273.15)).h for p, T in zip(pv, Tv)])
pred2 = bilinear2d(hpT, P, Tg, pv, Tv)
errs["hpT_sup"] = float(np.max(np.abs(pred2 - ref2)))
print(f"hpT (过热/超临界): bilinear maxerr = {errs['hpT_sup']:.4f} kJ/kg", flush=True)

np.savez(os.path.join(OUT, "iapws_surrogate.npz"),
         P=P, H=H, Tg=Tg, Tph=Tph, hpT=hpT,
         Psub=Psub, Tsat=Tsat, hsatV=hsatV, tsat_coef=tsat_coef,
         p_crit=np.float32(P_CRIT))
meta = {"p": [P_LO, P_HI, DP], "h": [H_LO, H_HI, DH], "T": [T_LO, T_HI, DT],
        "p_crit": P_CRIT, "errs": errs,
        "err_tsat_C": err_tsat, "hsatv_linerr_bound_kJkg": hsatv_linerr,
        "built": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(os.path.join(OUT, "iapws_surrogate_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("saved out/iapws_surrogate.npz", flush=True)
