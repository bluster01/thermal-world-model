#!/usr/bin/env python3
"""11b_step5_e0check.py: 排查 — 湿态v2阶跃符号翻转是残差污染还是灰盒本体缺陷?

对照: qh(带残差) vs e0(冻结灰盒无残差) 在同一湿态工况点注 v2+5%, 比较 main 响应方向。
另加 qh 在 v2+5% 时同步抬高 W (保持喷水硬守恒下总喷水随阀开增加——更接近真实阀门→总喷水链)。
"""
import importlib.util
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t02 = _imp("02_train.py", "t02")
r09 = _imp("09_residual.py", "r09")
import numpy as np
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
P_CRIT = t02.P_CRIT
N = 600
V2 = 6


def init_state(model0, row, obs):
    p0 = row[2] + (row[7] - row[2]) / 3.0
    p1 = row[2] + 2.0 * (row[7] - row[2]) / 3.0
    h0 = t02.h_of_pT(torch.tensor(p0, device=DEVICE), torch.tensor(float(obs[0]), device=DEVICE))
    h1 = t02.h_of_pT(torch.tensor(p1, device=DEVICE), torch.tensor(float(obs[2]), device=DEVICE))
    h2 = t02.h_of_pT(torch.tensor(row[7], device=DEVICE), torch.tensor(float(obs[4]), device=DEVICE))
    h = torch.stack([h0, h1, h2])[:, None]
    pst = torch.stack([torch.tensor(p0, device=DEVICE), torch.tensor(p1, device=DEVICE),
                       torch.tensor(row[7], device=DEVICE)])[:, None]
    ts = t02.T_of_ph(pst, h)
    rB = torch.tensor([row[1]], device=DEVICE)
    pm0 = torch.tensor([row[2]], device=DEVICE)
    Tm = ts + model0.k_of(pm0) * rB / 3600.0 / model0.tri("UA")[:, None] + model0.tri("dTm")[:, None]
    return h, Tm, rB


def run(model0, res, mode, row, h, Tm, rB, steps, d_v2=0.0, d_W=0.0):
    exo = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, steps, 1)
    exo[:, :, V2] += d_v2
    exo[:, :, 8] += d_W
    with torch.no_grad():
        out, *_ = r09.integrate_res(model0, res, exo, h, Tm, rB, steps, None, mode)
    return out[0].cpu().numpy()


df = r09.load_e0_df()
model0 = r09.load_e0(0)
res_qh = r09.ResMLP(13, r09.Q_SCALE).to(DEVICE)
res_qh.load_state_dict(torch.load(os.path.join(OUT, "model_res_qh_seed0.pt"),
                                  map_location=DEVICE, weights_only=True))
res_qh.eval()
res0 = r09.ResMLP(13, r09.Q_SCALE).to(DEVICE)  # mode none 占位

E = df[r09.E0_COLS].copy()
E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
Ea = E.to_numpy(np.float32)
T_all = df[t02.OUTPUTS].to_numpy(np.float32)

row_idx = 40161  # 湿态代表点 (与 11_step5 相同)
row = Ea[row_idx]
obs = T_all[row_idx]
print(f"wet op row={row_idx} pm={row[2]:.2f} v1={row[5]:.3f} v2={row[6]:.3f} W={row[8]:.2f}")

for name, res, mode in (("e0(无残差)", res0, "none"), ("qh(带残差)", res_qh, "qh")):
    h, Tm, rB = init_state(model0, row, obs)
    base = run(model0, res, mode, row, h, Tm, rB, N)
    h, Tm, rB = init_state(model0, row, obs)
    tr = run(model0, res, mode, row, h, Tm, rB, N, d_v2=0.05)
    d = tr - base
    print(f"{name}: v2+5% → main Δ={d[-1,4]:+.3f}°C (末值), 60s={d[5,4]:+.3f}, "
          f"sh1_in Δ={d[-1,0]:+.3f}, sh2_in Δ={d[-1,2]:+.3f}, sh2_out Δ={d[-1,3]:+.3f}")

# qh + W 联动 (v2+5% 同时 W+5%, 总喷水随阀开上升 — 更接近真实阀门链)
h, Tm, rB = init_state(model0, row, obs)
base2 = run(model0, res_qh, "qh", row, h, Tm, rB, N)
h, Tm, rB = init_state(model0, row, obs)
tr2 = run(model0, res_qh, "qh", row, h, Tm, rB, N, d_v2=0.05, d_W=0.05 * row[8])
d2 = tr2 - base2
print(f"qh v2+5%&W+5%: main Δ={d2[-1,4]:+.3f}°C, sh1_in={d2[-1,0]:+.3f}, sh2_in={d2[-1,2]:+.3f}")
