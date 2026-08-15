#!/usr/bin/env python3
"""trace: 训练后模型 (seed0) 首步内部状态 + 稳态平衡量 诊断"""
import importlib.util, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t02", os.path.join(os.getcwd(), "02_train.py"))
t02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t02)
import numpy as np, pandas as pd, torch

df = pd.read_csv(t02.CSV,
                 usecols=t02.EXO + t02.EXO_EXTRA + t02.OUTPUTS + [
                     "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
                 dtype=np.float32).iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)

model = t02.E0Model().to(t02.DEVICE)
model.load_state_dict(torch.load(os.path.join(t02.OUT, "model_e0_seed0.pt"),
                                 map_location=t02.DEVICE, weights_only=True))
model.eval()
print("=== 学得参数 ===")
for k in ["M0","M1","M2","UA0","UA1","UA2","Cm0","Cm1","Cm2","k0","k1","k2","tauB","th1","th2","dTm0","dTm1","dTm2"]:
    print(f"  {k}: {model.val(k).item():.4g}", end="  ")
print()

START = t02.TRAIN_N + t02.VAL_N
Xte, Yte, Ite, Ite_T = t02.e0_build_windows(df, START, len(df) - 1, 10)

# 取 20 个窗口做首步 trace
idx = np.random.RandomState(0).choice(len(Xte), 20, replace=False)
exo = torch.from_numpy(Xte[idx]).to(t02.DEVICE)
obs = torch.from_numpy(Ite_T[idx]).to(t02.DEVICE)
init = torch.from_numpy(Ite[idx]).to(t02.DEVICE)
truth1 = Yte[idx, 0]  # t+1 真值 (20,5)
truth0 = Ite_T[idx]   # t 真值

D = init[:, 0]; pm = init[:, 2]; p_out = init[:, 7]
p0 = pm + (p_out - pm) / 3.0
p1 = pm + 2.0 * (p_out - pm) / 3.0
h0 = t02.h_of_pT(p0, obs[:, 0]); h1 = t02.h_of_pT(p1, obs[:, 2]); h2 = t02.h_of_pT(p_out, obs[:, 4])
h = torch.stack([h0, h1, h2])
ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
Tm = ts + model.tri("dTm")[:, None]
rB = init[:, 1].clone()

k = model.tri("k")[:, None]; UA = model.tri("UA")[:, None]
Cm = model.tri("Cm")[:, None]; M = model.tri("M")[:, None]
Q_init = UA * (Tm - ts)
Q_eq = k * rB[None, :] / 3600.0
print("\n=== 首步前：Q 初值 vs 稳态平衡 Q（kW，20窗×3段均值）===")
print("  Q_init:", (Q_init / 1000).mean(dim=1).tolist())
print("  Q_eq(k*rB/3600):", (Q_eq / 1000).mean(dim=1).tolist())
print("  Tm-ts 初值:", (Tm - ts).mean(dim=1).tolist(), "  Tm-ts 平衡:", ((k * rB[None,:] / 3600.0) / UA).mean(dim=1).tolist())
print("  dTm learned:", [model.val(f"dTm{i}").item() for i in range(3)])

with torch.no_grad():
    out, h_a, Tm_a, rB_a, hm1a, hm2a = model.integrate(exo, h, Tm, rB, 1, return_states=True)
dh = (h_a - h) * M / 10.0   # 每步等效加热功率 kW
print("\n=== 首步后 ===")
print("  Δh per step (kJ/kg):", (h_a - h).mean(dim=1).tolist())
print("  等效加热功率 (kW):", (dh / 1000).mean(dim=1).tolist())
print("  ΔTm per step (K):", (Tm_a - Tm).mean(dim=1).tolist())
print("  ΔrB per step:", (rB_a - rB).mean().item())
print("  h clamp 激活占比:", ((h_a <= t02.H_LO + 1e-3) | (h_a >= t02.H_HI - 1e-3)).float().mean().item())
print("  pred t+1 vs truth t+1 (每列 mean±std, 20窗):")
for j, nm in enumerate(["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]):
    e = (out[:, 0, j] - torch.from_numpy(truth1[:, j]).to(t02.DEVICE)).cpu().numpy()
    print(f"    {nm}: err {e.mean():+.2f} ± {e.std():.2f} | truth Δ={np.mean(truth1[:, j] - truth0[:, j]):+.2f} | pred Δ={np.mean(out[:,0,j].cpu().numpy() - truth0[:, j]):+.2f}")

# 物性 sanity：h(p,T) 圆整
print("\n=== 物性圆整 sanity ===")
for nm, p_, T_ in [("sh1_in", p0[0].item(), obs[0, 0].item()), ("main", p_out[0].item(), obs[0, 4].item())]:
    hh = t02.h_of_pT(torch.tensor(p_, device=t02.DEVICE), torch.tensor(T_, device=t02.DEVICE))
    tt = t02.T_of_ph(torch.tensor(p_, device=t02.DEVICE), hh)
    print(f"  {nm}: p={p_:.2f} T={T_:.2f} -> h={hh.item():.2f} -> T_roundtrip={tt.item():.2f} (Δ={tt.item()-T_:+.3f})")
print(f"  hsep: pm={pm[0].item():.2f} Tm_sep={init[0,3].item():.2f} -> h={t02.h_sep_of(pm[0:1], init[0:1,3]).item():.2f}")
print(f"  h(p0,T_sh1_in)[0]={h[0,0].item():.2f} vs hsep={t02.h_sep_of(pm[0:1], init[0:1,3]).item():.2f}")
print(f"  喷水 h_sw=CP_W*Tfw[0]={t02.CP_W*init[0,4].item():.2f}")
