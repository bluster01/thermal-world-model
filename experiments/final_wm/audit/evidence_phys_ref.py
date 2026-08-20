"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""物理参考独立求解 + 真实阀位事件研究 (canonical_sideA, val 段)
1) dW/dv: 实测 W(t/h) 对阀位 v 的回归斜率 (数据锚定的阀门灵敏度)
2) 混合能量平衡: 每 kg/s 喷水的稳态温降 ΔT/Δdsw = (h_steam - h_water)/(D*cp)
3) 事件研究: 真实数据里 |Δv|>=0.04 持续阶跃, 量 H1/H6/H18/H60 温度响应
"""
import sys
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[3]))
import numpy as np

from src.final_wm.data import CanonicalRecord
from src.final_wm.properties import load_grid_properties

GRID = "/home/bluster/.hermes/workspace/adhoc2_lumped_enthalpy/out/iapws_surrogate.npz"  # env-specific
record = CanonicalRecord("artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(GRID)

obs = record.obs.numpy()
bnd = record.boundary.numpy()
act = record.actions.numpy()
sp = record.split.numpy()
m = sp == 1
O, B, A = obs[m], bnd[m], act[m]
D, pm, Tfw, p_out, W = B[:, 0], B[:, 2], B[:, 4], B[:, 5], B[:, 6]
v1, v2 = A[:, 0], A[:, 1]
KAPPA = 0.2777778

print("=== 1) dW/dv 回归 (val, W in t/h) ===")
for name, v in [("v1", v1), ("v2", v2)]:
    ok = v > 0.01
    if ok.sum() > 100:
        sxx = np.sum((v[ok] - v[ok].mean()) ** 2)
        sxy = np.sum((v[ok] - v[ok].mean()) * (W[ok] - W[ok].mean()))
        slope = sxy / sxx
        print(f"  {name}: dW/dv = {slope:.3f} t/h per full opening (n={ok.sum()})")
        # 每 2% 阀位 → 喷水量
        dsw2p = slope * 0.02 * KAPPA
        print(f"    -> +2% {name} = {dsw2p:.4f} kg/s")

print("\n=== 2) 混合能量平衡: 每 kg/s 喷水的温降 ===")
# h_steam at 屏过入口 (T~450-520C, p1) vs h_water(Tfw)
import torch
with torch.no_grad():
    T_steam = torch.tensor(O[:, 1], dtype=torch.float32)[::100]
    pm_t = torch.tensor(pm, dtype=torch.float32)[::100]
    p_out_t = torch.tensor(p_out, dtype=torch.float32)[::100]
    p1 = pm_t + 2 * (p_out_t - pm_t) / 3
    h_steam = props.enthalpy_of_pt(p1, T_steam).numpy()
    h_water = props.liquid_enthalpy(torch.tensor(Tfw, dtype=torch.float32)[::100]).numpy()
dh = h_steam - h_water
print(f"  h_steam-h_water: mean={dh.mean():.0f} kJ/kg")
for Dq in [200, 400, 560]:
    dt_per_kgps = dh.mean() / (Dq * 2.2)  # cp ~ 2.2 kJ/kgK
    print(f"  D={Dq} kg/s: ΔT per 1 kg/s spray = {dt_per_kgps:.3f} °C")

print("\n=== 3) 事件研究: 真实阀位阶跃的温度响应 ===")
CH = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "final"]
STEP = 0.04
for vi, vname in [(0, "v1"), (1, "v2")]:
    dv = np.diff(v1 if vi == 0 else v2)
    idx = np.where(np.abs(dv) >= STEP)[0]
    ups, downs = idx[dv[idx] > 0], idx[dv[idx] < 0]
    print(f"\n{vname}: |Δv|>=0.04 事件: up={len(ups)}, down={len(downs)}")
    for tag, ii in [("up(+阀开大)", ups), ("down(阀关小)", downs)]:
        for hk, h in [("H1", 1), ("H6", 6), ("H18", 18), ("H60", 60)]:
            resps = []
            for i in ii:
                if i + h + 5 >= len(O) or i < 5:
                    continue
                base = O[i + 4:i + 5].mean(axis=0)   # 阶跃前 1 步
                fut = O[i + h + 4:i + h + 5].mean(axis=0)  # 阶跃后 h 步
                # 排除阶跃后 h 步内又发生大阶跃的事件
                seg = dv[i + 1:i + h]
                if seg.size and np.abs(seg).max() >= STEP:
                    continue
                resps.append(fut - base)
            if len(resps) >= 5:
                r = np.array(resps)
                # 阀开大应该降温: 用 -r 看"冷却方向"占比
                cool_frac = (r[:, 4] < 0).mean() if tag.startswith("up") else (r[:, 4] > 0).mean()
                print(f"  {tag} {hk}: final ΔT mean={r[:, 4].mean():+.3f}  "
                      f"(n={len(r)}, 物理方向占比={cool_frac:.2f})  sh1_out={r[:, 1].mean():+.3f}")
