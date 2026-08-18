#!/usr/bin/env python3
"""08_diag_dry.py: 干态修复后变差诊断（方向E，2026-08-16）

三个问题，三段输出：
  A. 干窗吸引子偏差 + b 斜率消融（b×0 / b×0.5 / b×1.0，恒定外生 120 步模拟 30 窗样本）
  B. 喷水校正需求 k 的湿/干分模态线性斜率 vs 模型 b（滚动 test 段逐步反算）
  C. 喷水/阀位/混合权重统计（干窗起点）

结论（写入 NOTES.md §2.4）：
  干态退化不是代码 bug——是迭代4"共享 pm 斜率 b"在干态的外推错误：
  干态需求 k0 斜率 −0.046e6/MPa，模型 b0=+0.011（符号反）；消融 b→0 时
  干态 main 吸引子偏差 +12.1°C → −2.7°C。sigmoid 混合带非主因（干步 a>0.1 仅 5.5%）。
"""
import importlib.util, json, os, sys
import numpy as np
import pandas as pd
import torch

PROJ = "/home/bluster/.hermes/workspace/adhoc2_lumped_enthalpy"
os.chdir(PROJ)
spec = importlib.util.spec_from_file_location("t02", os.path.join(PROJ, "02_train.py"))
t02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t02)

START = t02.TRAIN_N + t02.VAL_N
P_CRIT = t02.P_CRIT
OUTPUTS = t02.OUTPUTS
EXO_COLS = ["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
            "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
            "末级过热器出口压力", "减温水总流量"]

df = pd.read_csv(t02.CSV, usecols=EXO_COLS + OUTPUTS, dtype=np.float32)\
    .iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)
raw = df[EXO_COLS].copy()
raw["主蒸汽流量"] = raw["主蒸汽流量"] / 3.6
raw["一级减温调节门阀位"] = raw["一级减温调节门阀位"].clip(lower=0) / 100.0
raw["二级减温调节门阀位"] = raw["二级减温调节门阀位"].clip(lower=0) / 100.0
E = raw.to_numpy(np.float32)
T = df[OUTPUTS].to_numpy(np.float32)

model = t02.E0Model().to(t02.DEVICE)
model.load_state_dict(torch.load(os.path.join(t02.OUT, "model_e0_seed0.pt"),
                                 map_location=t02.DEVICE, weights_only=True))
model.eval()

i0 = np.arange(START, len(df) - 1 - t02.SEQ, 10)
pm0 = E[i0, 2]
dry = pm0 > P_CRIT
wet = ~dry
Wr = df["减温水总流量"].to_numpy()

# ---------------- A. 吸引子偏差 + b 消融 ----------------
def attractor_bias_ablated(exo_row, obs_T, model, bscale):
    saved = {k: model.raw[k].detach().clone() for k in ("b0", "b1", "b2")}
    try:
        with torch.no_grad():
            for k in ("b0", "b1", "b2"):
                cur = model.val(k)
                scaled = cur * bscale
                model.raw[k].copy_(torch.atanh((scaled / t02.E0_PRIORS[k]).clamp(-0.999, 0.999)))
        exo = torch.from_numpy(np.tile(exo_row[None], (1, 120, 1))).to(t02.DEVICE)
        pm_v = exo[:, 0, 2]
        p_out = exo[:, 0, 7]
        p0 = pm_v + (p_out - pm_v) / 3.0
        p1 = pm_v + 2.0 * (p_out - pm_v) / 3.0
        obs = torch.from_numpy(obs_T[None]).to(t02.DEVICE)
        h0 = t02.h_of_pT(p0, obs[:, 0])
        h1 = t02.h_of_pT(p1, obs[:, 2])
        h2 = t02.h_of_pT(p_out, obs[:, 4])
        h = torch.stack([h0, h1, h2])
        ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
        rB = torch.tensor([exo_row[1]], device=t02.DEVICE)
        Tm = (ts + model.k_of(pm_v) * rB[None, :] / 3600.0 / model.tri("UA")[:, None]
              + model.tri("dTm")[:, None])
        with torch.no_grad():
            pred = model.integrate(exo, h, Tm, rB, 120)
        return (pred[0, -1] - obs[0]).cpu().numpy()
    finally:
        with torch.no_grad():
            for k, v in saved.items():
                model.raw[k].copy_(v)

print("=== A. 干窗吸引子偏差 (恒定外生120步终态, 30窗) + b消融 ===")
rng = np.random.RandomState(0)
samp = rng.choice(np.where(dry)[0], min(30, dry.sum()), replace=False)
for bscale in (0.0, 0.5, 1.0):
    bias_all = np.array([attractor_bias_ablated(E[i0[k]], T[i0[k]], model, bscale) for k in samp])
    print(f"b×{bscale}: mean={bias_all.mean(0).round(2)}  rmse={np.sqrt((bias_all**2).mean(0)).round(2)} "
          f"(sh1i,sh1o,sh2i,sh2o,main)")

# ---------------- B. 喷水校正需求 k 斜率 ----------------
n = t02.ROLL_STEPS
pm = E[START: START + n, 2]
D = E[START: START + n, 0]
rB = E[START: START + n, 1]
p_out = E[START: START + n, 7]
p0 = pm + (p_out - pm) / 3.0
p1 = pm + 2.0 * (p_out - pm) / 3.0
Tsep = E[START: START + n, 3]
Tfw = E[START: START + n, 4]
v1 = E[START: START + n, 5]
v2 = E[START: START + n, 6]
W = E[START: START + n, 8]
Tsh1i, Tsh1o, Tsh2i, Tsh2o, Tmain = [T[START: START + n, j] for j in range(5)]

with torch.no_grad():
    pm_t = torch.from_numpy(pm).to(t02.DEVICE)
    hsep = t02.h_sep_of(pm_t, torch.from_numpy(Tsep).to(t02.DEVICE)).cpu().numpy()
    h0 = t02.h_of_pT(torch.from_numpy(p0).to(t02.DEVICE),
                     torch.from_numpy(Tsh1i).to(t02.DEVICE)).cpu().numpy()
    h1 = t02.h_of_pT(torch.from_numpy(p1).to(t02.DEVICE),
                     torch.from_numpy(Tsh2i).to(t02.DEVICE)).cpu().numpy()
    h2 = t02.h_of_pT(torch.from_numpy(p_out).to(t02.DEVICE),
                     torch.from_numpy(Tmain).to(t02.DEVICE)).cpu().numpy()
    h_sw = t02.hliq_of_T(torch.from_numpy(Tfw).to(t02.DEVICE)).cpu().numpy()
    th1_eff, th2_eff = model.th_of(pm_t)
    th1_eff, th2_eff = th1_eff.cpu().numpy(), th2_eff.cpu().numpy()
    k_model = model.k_of(pm_t).cpu().numpy()

s_den = th1_eff * v1 + th2_eff * v2 + 1e-6
Dsw1 = W / 3.6 * (th1_eff * v1) / s_den
Dsw2 = W / 3.6 * (th2_eff * v2) / s_den
hm1 = (D * h0 + Dsw1 * h_sw) / (D + Dsw1 + 1e-6)
hm2 = (D * h1 + Dsw2 * h_sw) / (D + Dsw2 + 1e-6)
k0_req = 3600.0 * D * (h0 - hsep) / np.maximum(rB, 1e-3)
k1_req = 3600.0 * D * (h1 - hm1) / np.maximum(rB, 1e-3)
k2_req = 3600.0 * D * (h2 - hm2) / np.maximum(rB, 1e-3)

print("\n=== B. 喷水校正需求 k vs 模型 k (湿/干分模态, 逐步反算) ===")
print(f"{'mode':>5} {'n':>5} | {'k0req':>7} {'k0mod':>7} {'slope_r':>8} {'slope_m':>8} | "
      f"{'k1req':>7} {'k1mod':>7} {'slope_r':>8} {'slope_m':>8} | "
      f"{'k2req':>7} {'k2mod':>7} {'slope_r':>8} {'slope_m':>8}")
for mode, mask in (("wet", pm <= P_CRIT), ("dry", pm > P_CRIT)):
    rows = []
    for req, mod in ((k0_req, k_model[0]), (k1_req, k_model[1]), (k2_req, k_model[2])):
        r, mo = req[mask], mod[mask]
        pmc = pm[mask]
        sr = np.polyfit(pmc, r / 1e6, 1)[0]
        sm = np.polyfit(pmc, mo / 1e6, 1)[0]
        rows.append((r.mean() / 1e6, mo.mean() / 1e6, sr, sm))
    print(f"{mode:>5} {mask.sum():>5} | {rows[0][0]:>7.2f} {rows[0][1]:>7.2f} {rows[0][2]:>+8.3f} {rows[0][3]:>+8.3f} | "
          f"{rows[1][0]:>7.2f} {rows[1][1]:>7.2f} {rows[1][2]:>+8.3f} {rows[1][3]:>+8.3f} | "
          f"{rows[2][0]:>7.2f} {rows[2][1]:>7.2f} {rows[2][2]:>+8.3f} {rows[2][3]:>+8.3f}")

# ---------------- C. 统计 ----------------
print("\n=== C. 统计 ===")
print(f"窗口: dry={dry.sum()}, wet={wet.sum()}")
print(f"减温水总流量 (raw, t/h): 全段 mean={Wr.mean():.1f}; dry窗起点 mean={Wr[i0[dry]].mean():.1f}, "
      f"wet窗起点 mean={Wr[i0[wet]].mean():.1f}")
print(f"干窗起点 pm: mean={pm0[dry].mean():.2f}, quantiles={np.percentile(pm0[dry],[5,25,50,75,95]).round(2)}")
a_roll = 1.0 / (1.0 + np.exp(-(P_CRIT - pm) / t02.K_BLEND))
dry_roll = pm > P_CRIT
print(f"rollout 干步混合权重 a: mean={a_roll[dry_roll].mean():.4f}, a>0.1 占比={(a_roll[dry_roll]>0.1).mean():.1%}, "
      f"a>0.01 占比={(a_roll[dry_roll]>0.01).mean():.1%}")
print(f"干窗 v1 mean={E[i0[dry],5].mean():.3f}, v2 mean={E[i0[dry],6].mean():.3f}")
