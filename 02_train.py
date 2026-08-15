#!/usr/bin/env python3
"""ad hoc2 Step 1: e0 主赛 + 基线 v0/v2/v2o（设计稿 §5/§6）

进程纪律（设计稿 §3）：每 seed 一个进程，同进程内跑完全部变体 + 基线。
用法:
  python 02_train.py --seed 0                     # 完整跑 e0,v0,v2,v2o
  python 02_train.py --seed 0 --variants e0       # 只跑 e0
  python 02_train.py --seed 0 --fast              # 冒烟（2 epoch / 短 rollout，产物带 .fast 后缀）

产物（out/）:
  results_{v}_seed{s}.json   rollout_{v}_seed{s}.npz（preds/truths 1800x5）
  params_e0_seed{s}.json     model_{v}_seed{s}.pt
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据_cleaned_10s.csv"
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WIN_START, WIN, SEQ = 70686, 50000, 60
TRAIN_N, VAL_N = 30000, 10000
ROLL_STEPS = 1800
T_BAND = [557.75, 572.13]
OUTPUTS = ["一级减温器入口温度", "一级减温器出口温度", "二级减温器入口温度",
           "二级减温器出口温度", "末级过热器出口汽温"]

# ---------------- 数据统计（供 sanity / 单位确认） ----------------
def data_stats(df):
    for c in ["主蒸汽流量", "减温水总流量", "分离器出口压力", "末级过热器出口压力",
              "分离器出口温度", "省煤器出口给水温度"]:
        s = df[c]
        print(f"  {c}: mean={s.mean():.2f} min={s.min():.2f} max={s.max():.2f}", flush=True)

# ---------------- IAPWS 代理（torch，可微） ----------------
S = np.load(os.path.join(OUT, "iapws_surrogate.npz"))
P_GRID = torch.tensor(S["P"], dtype=torch.float32, device=DEVICE)          # (Np,)
H_GRID = torch.tensor(S["H"], dtype=torch.float32, device=DEVICE)          # (Nh,)
TG_GRID = torch.tensor(S["Tg"], dtype=torch.float32, device=DEVICE)        # (NT,)
TPH = torch.tensor(S["Tph"], dtype=torch.float32, device=DEVICE)           # (Np, Nh)
HPT = torch.tensor(S["hpT"], dtype=torch.float32, device=DEVICE)           # (Np, NT)
Psub_t = torch.tensor(S["Psub"], dtype=torch.float32, device=DEVICE)       # 亚临界 p 网格
HSATV_t = torch.tensor(S["hsatV"], dtype=torch.float32, device=DEVICE)
TSAT_C = torch.tensor(S["tsat_coef"], dtype=torch.float32, device=DEVICE)  # deg 8, in p(亚临界)
P_CRIT = float(S["p_crit"])
P_LO, P_HI = float(P_GRID[0]), float(P_GRID[-1])
H_LO, H_HI = float(H_GRID[0]), float(H_GRID[-1])
T_LO, T_HI = float(TG_GRID[0]), float(TG_GRID[-1])


def _grid_sample(img, x_lo, x_hi, x, y_lo, y_hi, y):
    """双线性查询 img(Ny, Nx)：x=列（h 或 T），y=行（p）。返回与输入同 shape。"""
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    gx = 2.0 * (flat_x - x_lo) / (x_hi - x_lo) - 1.0
    gy = 2.0 * (flat_y - y_lo) / (y_hi - y_lo) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    out = F.grid_sample(img[None, None], grid, mode="bilinear",
                        align_corners=True, padding_mode="border")
    return out.view_as(x)


def _ste_clamp(x, lo, hi):
    """straight-through clamp：值硬限幅但梯度直通（饱和时不死梯度）。"""
    return x + (x.clamp(lo, hi) - x).detach()


def T_of_ph(p, h):
    p = p.clamp(P_LO, P_HI)
    h = _ste_clamp(h, H_LO, H_HI)
    return _grid_sample(TPH, H_LO, H_HI, h, P_LO, P_HI, p)


def h_of_pT(p, T):
    p = p.clamp(P_LO, P_HI)
    T = T.clamp(T_LO, T_HI)
    return _grid_sample(HPT, T_LO, T_HI, T, P_LO, P_HI, p)


def _polyval(coef, x):
    """Horner 多项式求值（torch 2.11 无 polyval）。coef: 降幂排列。"""
    y = torch.full_like(x, coef[0])
    for c in coef[1:]:
        y = y * x + c
    return y


def tsat_poly(p):
    """亚临界 Tsat(p)；p>p_crit 时返回临界温度 374.15°C（clamp 到 p_crit）。"""
    return _polyval(TSAT_C, p.clamp(Psub_t[0], P_CRIT))


def hsatv_of_p(p):
    """饱和汽焓 h_satV(p)，1D 网格线性插值（仅亚临界查询）。"""
    p = p.clamp(Psub_t[0], Psub_t[-1])
    flat = p.reshape(-1)
    idx = torch.searchsorted(Psub_t, flat).clamp(1, len(Psub_t) - 1)
    p0 = Psub_t[idx - 1]
    p1 = Psub_t[idx]
    w = (flat - p0) / (p1 - p0 + 1e-12)
    h = HSATV_t[idx - 1] + w * (HSATV_t[idx] - HSATV_t[idx - 1])
    return h.view_as(p)


def h_sep_of(pm, Tm):
    """分离器出口焓。超临界（p>p_crit）：直接 h(p,T)（无两相）。
    亚临界两相（Tm<Tsat）：取饱和汽边界（clamp 到 Tsat+0.5°C 微过热）。"""
    ts = tsat_poly(pm)
    supercrit = pm > P_CRIT
    T_eff = torch.where(supercrit, Tm, torch.maximum(Tm, ts + 0.5))
    return h_of_pT(pm, T_eff)


# ---------------- e0: Fan2020 集总焓模型 ----------------
def sp_init(target):
    """softplus 反函数：参数 raw=ln(e^target−1) → softplus(raw)≈target。大值直接取 target。"""
    if target > 30.0:
        return float(target)
    return float(np.log(np.expm1(target)))


E0_PRIORS = {  # 物理量级先验（660MW 超临界，稳态能量平衡自洽：Q_typ=1e5kW/段, ΔTm-ts≈167K）
    "M0": 500.0, "M1": 500.0, "M2": 500.0,          # kg（蒸汽驻留 τ=M/D≈1.4s）
    "UA0": 600.0, "UA1": 600.0, "UA2": 600.0,       # kW/K（Q=UA·ΔT）
    "Cm0": 60000.0, "Cm1": 60000.0, "Cm2": 60000.0,  # kJ/K（τ_metal=Cm/UA≈100s）
    "k0": 1.2e6, "k1": 1.2e6, "k2": 1.2e6,          # kJ/t（k·rB[t/h]/3600≈1e5 kW）
    "tauB": 120.0,                                   # s
    "th1": 10.0, "th2": 20.0,                        # kg/s per 0-1 阀位（Step0: 0.70t/h·raw→19.4kg/s）
    "dTm0": 20.0, "dTm1": 20.0, "dTm2": 20.0,        # K
}
E0_KEYS = list(E0_PRIORS.keys())
KAPPA = 1.0 / 3.6          # t/h → kg/s
LAM_CAL = 0.1
CP_W = 4.18                # kJ/kg·K，喷水焓
DT_SUB = 2.0               # 子步长 s（10s 步内 5 子步）
N_SUB = 5


class E0Model(nn.Module):
    """参数化：value = prior × softplus(raw)，raw 初始 ≈ ln(e−1)≈0.54（全参数 O(1)）。
    避免物理尺度（0.1~8e5）下 Adam 绝对步长失效。"""

    def __init__(self):
        super().__init__()
        self.raw = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(sp_init(1.0), dtype=torch.float32))
            for k in E0_PRIORS.keys()})
        self._g = {"M": ["M0", "M1", "M2"], "UA": ["UA0", "UA1", "UA2"],
                   "Cm": ["Cm0", "Cm1", "Cm2"], "k": ["k0", "k1", "k2"],
                   "dTm": ["dTm0", "dTm1", "dTm2"]}

    def val(self, k):
        return E0_PRIORS[k] * F.softplus(self.raw[k])

    def tri(self, grp):
        return torch.stack([E0_PRIORS[k] * F.softplus(self.raw[k]) for k in self._g[grp]], dim=0)  # (3,)

    def integrate(self, exo, h, Tm, rB, steps, return_states=False):
        """exo: (B, steps, 9)=[D,uB,pm,Tm_sep,Tfw,v1,v2,p_out,W]
        h,Tm: (3,B)；rB: (B,)。半隐式欧拉（Tm 隐式，h/rB 显式），5 子步。
        返回 (B, steps, 5) 输出温度；return_states=True 时追加 (h,Tm,rB,hm1,hm2)。
        训练与 rollout 共用本函数（同一代码路径）。"""
        Bsz = exo.shape[0]
        M = self.tri("M")[:, None]
        UA = self.tri("UA")[:, None]
        Cm = self.tri("Cm")[:, None]
        k = self.tri("k")[:, None]
        tauB = self.val("tauB")
        th1 = self.val("th1")
        th2 = self.val("th2")
        D, uB, pm, Tm_sep, Tfw, v1, v2, p_out, W = [exo[:, :, j] for j in range(9)]
        h_sw = CP_W * Tfw
        p0 = pm + (p_out - pm) / 3.0
        p1 = pm + 2.0 * (p_out - pm) / 3.0
        hsep = h_sep_of(pm, Tm_sep)
        out_list = []
        # 初始混合（用初始 h0/h1）
        Dsw1 = th1 * v1[:, 0]
        Dsw2 = th2 * v2[:, 0]
        hm1 = (D[:, 0] * h[0] + Dsw1 * h_sw[:, 0]) / (D[:, 0] + Dsw1 + 1e-6)
        hm2 = (D[:, 0] * h[1] + Dsw2 * h_sw[:, 0]) / (D[:, 0] + Dsw2 + 1e-6)
        for t in range(steps):
            for _ in range(N_SUB):
                ts = T_of_ph(torch.stack([p0[:, t], p1[:, t], p_out[:, t]]), h)  # (3,B)
                Q = UA * (Tm - ts)
                # Tm 半隐式（k·rB 为 kJ/h，/3600 → kJ/s）
                Tm = (Tm + DT_SUB * (k * rB[None, :] / 3600.0 + UA * ts) / Cm) / (1.0 + DT_SUB * UA / Cm)
                # h 半隐式（D·(hin−h) 项隐式——无条件稳定；Q 项显式，dt=2s 下衰减因子 0.84）
                hin = torch.stack([hsep[:, t], hm1, hm2])
                h = (h + DT_SUB * (D[:, t][None, :] * hin + Q) / M) / (1.0 + DT_SUB * D[:, t][None, :] / M)
                h = _ste_clamp(h, H_LO, H_HI)
                # 喷水混合
                Dsw1 = th1 * v1[:, t]
                Dsw2 = th2 * v2[:, t]
                hm1 = (D[:, t] * h[0] + Dsw1 * h_sw[:, t]) / (D[:, t] + Dsw1 + 1e-6)
                hm2 = (D[:, t] * h[1] + Dsw2 * h_sw[:, t]) / (D[:, t] + Dsw2 + 1e-6)
                # 制粉一阶滞后
                rB = rB + DT_SUB * (uB[:, t] - rB) / tauB
            p = torch.stack([p0[:, t], p0[:, t], p1[:, t], p1[:, t], p_out[:, t]])
            hh = torch.stack([h[0], hm1, h[1], hm2, h[2]])
            out_list.append(T_of_ph(p, hh))  # (5, B)
        out = torch.stack(out_list, dim=2).permute(1, 2, 0)  # (5,B,steps) -> (B, steps, 5)
        if return_states:
            return out, h, Tm, rB, hm1, hm2
        return out


def e0_build_windows(df, lo, hi, stride, need_init=True):
    """窗口 i0∈[lo, hi−60)：exo (N,60,9)，target (N,60,5)，init_exo (N,9)，init_T (N,5)。"""
    exo_cols = ["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
                "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
                "末级过热器出口压力", "减温水总流量"]
    raw = df[exo_cols].copy()
    raw["主蒸汽流量"] = raw["主蒸汽流量"] / 3.6
    raw["一级减温调节门阀位"] = raw["一级减温调节门阀位"].clip(lower=0) / 100.0
    raw["二级减温调节门阀位"] = raw["二级减温调节门阀位"].clip(lower=0) / 100.0
    E = raw.to_numpy(np.float32)
    T = df[OUTPUTS].to_numpy(np.float32)
    i0 = np.arange(lo, hi - SEQ, stride)
    exo = np.stack([E[s: s + SEQ] for s in i0])
    tgt = np.stack([T[s + 1: s + SEQ + 1] for s in i0])
    init_exo = E[i0] if need_init else None
    init_T = T[i0] if need_init else None
    return exo, tgt, init_exo, init_T


def e0_train(df, seed, fast, tag, max_epochs=60):
    torch.manual_seed(seed)
    np.random.seed(seed)
    exo_cols = ["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
                "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
                "末级过热器出口压力", "减温水总流量"]
    t0 = time.time()
    tr_s = 25 if fast else 5
    va_s = 100 if fast else 20
    Xtr, Ytr, Itr, Itr_T = e0_build_windows(df, 0, TRAIN_N, tr_s)
    Xva, Yva, Iva, Iva_T = e0_build_windows(df, TRAIN_N, TRAIN_N + VAL_N, va_s)
    print(f"[e0 s{seed}] train={len(Xtr)} val={len(Xva)} build {time.time()-t0:.0f}s", flush=True)

    mu_o = df[OUTPUTS].iloc[:TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[OUTPUTS].iloc[:TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    D_mean = float(df["主蒸汽流量"].to_numpy()[:TRAIN_N].mean() / 3.6)

    model = E0Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w5 = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=DEVICE)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    Itr_t = torch.from_numpy(Itr).to(DEVICE)
    ItrT_t = torch.from_numpy(Itr_T).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Yva_t = torch.from_numpy(Yva).to(DEVICE)
    Iva_t = torch.from_numpy(Iva).to(DEVICE)
    IvaT_t = torch.from_numpy(Iva_T).to(DEVICE)

    def init_states(init_rows, obs_T):
        """init_rows: (B,9) 窗口起点外生；obs_T: (B,5) 起点观测温度。"""
        D = init_rows[:, 0]
        pm = init_rows[:, 2]
        p_out = init_rows[:, 7]
        p0 = pm + (p_out - pm) / 3.0
        p1 = pm + 2.0 * (p_out - pm) / 3.0
        h0 = h_of_pT(p0, obs_T[:, 0])
        h1 = h_of_pT(p1, obs_T[:, 2])
        h2 = h_of_pT(p_out, obs_T[:, 4])
        ts0 = T_of_ph(p0, h0)
        ts1 = T_of_ph(p1, h1)
        ts2 = T_of_ph(p_out, h2)
        dTm = model.tri("dTm")[:, None]  # (3,1)
        Tm = torch.stack([ts0, ts1, ts2]) + dTm
        return torch.stack([h0, h1, h2]), Tm, init_rows[:, 1].clone()

    def fwd(exo, init_rows, obs_T):
        h, Tm, rB = init_states(init_rows, obs_T)
        return model.integrate(exo, h, Tm, rB, exo.shape[1])

    max_ep = 2 if fast else max_epochs
    best_va, best_state, patience = 1e9, None, 0
    n_batch = len(Xtr_t) // 256
    t_train = time.time()
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        for b in range(n_batch):
            i = perm[b * 256: (b + 1) * 256]
            pred = fwd(Xtr_t[i], Itr_t[i], ItrT_t[i])
            mse = (((pred - Ytr_t[i]) ** 2) * w5).mean()
            cal = (model.val("th1") * Xtr_t[i][:, :, 5] + model.val("th2") * Xtr_t[i][:, :, 6]
                   - KAPPA * Xtr_t[i][:, :, 8]) ** 2
            loss = mse + LAM_CAL * cal.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = fwd(Xva_t, Iva_t, IvaT_t)
            va = (((pred - Yva_t) ** 2) * w5).mean().item()
        if va < best_va:
            best_va, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, f"model_e0_seed{seed}{tag}.pt"))
    print(f"[e0 s{seed}] trained {ep+1}ep val_mse={best_va:.3f} ({time.time()-t_train:.0f}s)", flush=True)

    # ---- 学得参数快照 ----
    params = {k: float(model.val(k).item()) for k in E0_KEYS}
    with open(os.path.join(OUT, f"params_e0_seed{seed}{tag}.json"), "w") as f:
        json.dump({"seed": seed, "priors": E0_PRIORS, "learned": params,
                   "val_mse_C2": round(best_va, 4)}, f, ensure_ascii=False, indent=2)


def e0_rollout(model, df, start, n_steps, seed, tag, mu_o, sd_o):
    """1800 步递归 rollout，无状态重置，外生喂真实值。返回 metrics + (preds, truths)。
    每步调用 model.integrate(..., return_states=True)——与训练同一代码路径。"""
    exo_cols = ["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
                "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
                "末级过热器出口压力", "减温水总流量"]
    E_full = df[exo_cols].copy()
    E_full["主蒸汽流量"] = E_full["主蒸汽流量"] / 3.6
    E_full["一级减温调节门阀位"] = E_full["一级减温调节门阀位"].clip(lower=0) / 100.0
    E_full["二级减温调节门阀位"] = E_full["二级减温调节门阀位"].clip(lower=0) / 100.0
    E = E_full.to_numpy(np.float32)
    T_all = df[OUTPUTS].to_numpy(np.float32)
    th1 = model.val("th1")
    th2 = model.val("th2")
    preds = np.empty((n_steps, 5), dtype=np.float32)
    truths = T_all[start: start + n_steps]
    h, Tm, rB, hm1, hm2 = None, None, None, None, None
    with torch.no_grad():
        for t in range(n_steps):
            row = E[start + t]
            exo_t = torch.tensor(row, device=DEVICE)[None, None, :]  # (1,1,9)
            if t == 0:
                p0 = row[2] + (row[7] - row[2]) / 3.0
                p1 = row[2] + 2.0 * (row[7] - row[2]) / 3.0
                obs = T_all[start]
                h0 = h_of_pT(torch.tensor(p0, device=DEVICE), torch.tensor(float(obs[0]), device=DEVICE))
                h1 = h_of_pT(torch.tensor(p1, device=DEVICE), torch.tensor(float(obs[2]), device=DEVICE))
                h2 = h_of_pT(torch.tensor(row[7], device=DEVICE), torch.tensor(float(obs[4]), device=DEVICE))
                h = torch.stack([h0, h1, h2])[:, None]  # (3,1)
                pst = torch.stack([torch.tensor(p0, device=DEVICE),
                                   torch.tensor(p1, device=DEVICE),
                                   torch.tensor(row[7], device=DEVICE)])[:, None]
                ts = T_of_ph(pst, h)
                Tm = ts + model.tri("dTm")[:, None]
                rB = torch.tensor([row[1]], device=DEVICE)
            out, h, Tm, rB, hm1, hm2 = model.integrate(exo_t, h, Tm, rB, 1, return_states=True)
            preds[t] = out[0, 0].cpu().numpy()
    main_p, main_t = preds[:, 4], truths[:, 4]
    PAIRS = [(1, 2), (3, 4), (2, 3), (1, 0), (0, 2)]  # 同 ad-hoc1 04_mechanism 5 对
    viol = np.zeros(n_steps, dtype=bool)
    for lo, hi in PAIRS:
        viol |= (preds[:, lo] >= preds[:, hi])
    z_all = np.abs((preds - mu_o) / sd_o)
    r = {
        "rmse_main": float(np.sqrt(np.mean((main_p - main_t) ** 2))),
        "maxerr_main": float(np.max(np.abs(main_p - main_t))),
        "rmse_all": float(np.sqrt(np.mean((preds - truths) ** 2))),
        "band_viol_frac": float(np.mean((main_p > T_BAND[1]) | (main_p < T_BAND[0]))),
        "order_viol_any_frac": float(viol.mean()),
        "order_main_lt_sh2out_frac": float(np.mean(preds[:, 3] >= preds[:, 4])),
        "drift_main_mean_z": round(float(z_all[:, 4].mean()), 3),
        "drift_main_max_z": round(float(z_all[:, 4].max()), 3),
        "drift_all_mean_z": round(float(z_all.mean()), 3),
    }
    # 动作通道活跃判据（设计稿 §2 附加通道判据）
    Dsw_mean = float(th1.item() * E[start: start + n_steps, 5].mean()
                     + th2.item() * E[start: start + n_steps, 6].mean())
    r["dsw_mean_kgs"] = round(Dsw_mean, 4)
    r["dsw_thresh_kgs"] = round(0.005 * float(E[start: start + n_steps, 0].mean()), 4)
    r["channel_active"] = bool(Dsw_mean >= r["dsw_thresh_kgs"])
    return r, preds, truths


def e0_windowed_eval(model, df, mu_o, sd_o):
    """测试段窗口内评估（每窗重置，stride 10）：首步 10s 前向 RMSE + 60 步窗口 RMSE（主汽温）。"""
    Xte, Yte, Ite, Ite_T = e0_build_windows(df, TRAIN_N + VAL_N, len(df) - 1, 10)
    model.eval()
    first_errs, all_err_main = [], []
    with torch.no_grad():
        for b in range(0, len(Xte), 256):
            xb = torch.from_numpy(Xte[b: b + 256]).to(DEVICE)
            yb = torch.from_numpy(Yte[b: b + 256]).to(DEVICE)
            ib = Ite[b: b + 256]
            D = torch.from_numpy(ib[:, 0]).to(DEVICE)
            pm = torch.from_numpy(ib[:, 2]).to(DEVICE)
            p_out = torch.from_numpy(ib[:, 7]).to(DEVICE)
            p0 = pm + (p_out - pm) / 3.0
            p1 = pm + 2.0 * (p_out - pm) / 3.0
            obs = torch.from_numpy(Ite_T[b: b + 256]).to(DEVICE)
            h0 = h_of_pT(p0, obs[:, 0])
            h1 = h_of_pT(p1, obs[:, 2])
            h2 = h_of_pT(p_out, obs[:, 4])
            h = torch.stack([h0, h1, h2])
            ts = T_of_ph(torch.stack([p0, p1, p_out]), h)
            Tm = ts + model.tri("dTm")[:, None]
            rB = torch.from_numpy(ib[:, 1]).to(DEVICE).clone()
            pred = model.integrate(xb, h, Tm, rB, xb.shape[1])
            err = pred - yb
            first_errs.append(err[:, 0, 4].abs().cpu().numpy())
            all_err_main.append(err[:, :, 4].cpu().numpy())
    first = np.concatenate(first_errs)
    allm = np.concatenate(all_err_main)
    return {"single_step_rmse_main_C": float(np.sqrt(np.mean(first ** 2))),
            "window60_rmse_main_C": float(np.sqrt(np.mean(allm ** 2)))}


def run_e0(df, seed, fast, tag, max_epochs=60):
    e0_train(df, seed, fast, tag, max_epochs)
    # reload best model
    model = E0Model().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(OUT, f"model_e0_seed{seed}{tag}.pt"),
                                     map_location=DEVICE, weights_only=True))
    mu_o = df[OUTPUTS].iloc[:TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[OUTPUTS].iloc[:TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    n_steps = 120 if fast else ROLL_STEPS
    t0 = time.time()
    r, preds, truths = e0_rollout(model, df, TRAIN_N + VAL_N, n_steps, seed, tag, mu_o, sd_o)
    r["rollout_seconds"] = round(time.time() - t0, 1)
    if not fast:
        r.update(e0_windowed_eval(model, df, mu_o, sd_o))
    r["variant"] = "e0"
    r["seed"] = seed
    np.savez(os.path.join(OUT, f"rollout_e0_seed{seed}{tag}.npz"), preds=preds, truths=truths)
    with open(os.path.join(OUT, f"results_e0_seed{seed}{tag}.json"), "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"[e0 s{seed}] {json.dumps(r, ensure_ascii=False)}", flush=True)


# ---------------- 基线 v0/v2/v2o（ad-hoc1 协议原样，按 seed 复跑） ----------------
EXO = ["机组负荷", "主蒸汽压力", "主蒸汽流量", "主给水流量", "总风量指令", "总二次风量",
       "燃料主控输出", "未校正总煤量", "水煤比", "减温水总流量",
       "一级减温喷水调节门指令", "二级减温喷水调节门指令",
       "省煤器出口给水温度", "分离器出口温度", "分离器出口压力", "AGC指令", "机组负荷变化率"]
EXO_EXTRA = ["过热器出口温度升速率"]


def parse_variant(v):
    phys = v in ("v1", "v1x", "v2", "v2x", "v2xb", "v2b", "v2o")
    band = v in ("v2", "v2x", "v2xb", "v2b", "v0b")
    order = v in ("v2", "v2x", "v2o")
    return phys, band, order


class Net(nn.Module):
    def __init__(self, F_in):
        super().__init__()
        self.gru = nn.GRU(F_in, 32, batch_first=True)
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


def physics_features(df):
    w = df["减温水总流量"].to_numpy()
    feco = df["省煤器出口给水温度"].to_numpy()
    sh1i = df["一级减温器入口温度"].to_numpy()
    sh1o = df["一级减温器出口温度"].to_numpy()
    sh2i = df["二级减温器入口温度"].to_numpy()
    sh2o = df["二级减温器出口温度"].to_numpy()
    main = df["末级过热器出口汽温"].to_numpy()
    sep = df["分离器出口温度"].to_numpy()
    steam = df["主蒸汽流量"].to_numpy()
    coal = df["未校正总煤量"].to_numpy()
    f = pd.DataFrame({
        "spray_cool_1": w * (sh1i - feco),
        "spray_cool_2": w * (sh2i - feco),
        "adv_1": steam * (sh1i - sep),
        "adv_2": steam * (sh2i - sh1o),
        "adv_3": steam * (main - sh2o),
        "heat_intensity": coal / (steam + 1.0),
    })
    f["dspray_6"] = pd.Series(df["二级减温喷水调节门指令"]).diff(6).fillna(0.0).to_numpy()
    return f


def baseline_build_data(df, v, phys):
    exo_cols = EXO + (EXO_EXTRA if phys else [])
    feats = physics_features(df) if phys else None
    phy_cols = list(feats.columns) if feats is not None else []
    frames = [df[exo_cols], df[OUTPUTS]] + ([feats] if feats is not None else [])
    raw = pd.concat(frames, axis=1)
    mu = raw.iloc[:TRAIN_N].mean()
    sd = raw.iloc[:TRAIN_N].std().replace(0, 1.0)
    Z = ((raw - mu) / sd).to_numpy(np.float32)
    F_in = Z.shape[1]
    N = len(df)

    def windows(lo, hi, stride):
        idx = np.arange(lo + SEQ - 1, hi, stride)
        X = np.stack([Z[i - SEQ + 1: i + 1] for i in idx])
        out_start = F_in - len(phy_cols) - 5 if phys else F_in - 5
        Y = Z[idx + 1][:, out_start: out_start + 5]
        return X, Y

    Xtr, Ytr = windows(0, TRAIN_N, 5)
    Xva, Yva = windows(TRAIN_N, TRAIN_N + VAL_N, 1)
    Xte, Yte = windows(TRAIN_N + VAL_N, N - 1, 1)
    info = {"mu": mu.to_dict(), "sd": sd.to_dict(), "F": F_in,
            "exo_cols": exo_cols, "phy_cols": phy_cols, "n_train": len(Xtr)}
    return Xtr, Ytr, Xva, Yva, Xte, Yte, df, info


def model_out(yz, info):
    cols = OUTPUTS
    mu = torch.tensor([info["mu"][c] for c in cols], device=DEVICE, dtype=torch.float32)
    sd = torch.tensor([info["sd"][c] for c in cols], device=DEVICE, dtype=torch.float32)
    return yz * sd + mu


def order_loss(phys, t_sep, use_order):
    if not use_order:
        return torch.zeros((), device=DEVICE)
    s1i, s1o, s2i, s2o, main = phys[:, 0], phys[:, 1], phys[:, 2], phys[:, 3], phys[:, 4]
    L = torch.relu(t_sep - s1i + 0.5)
    L = L + torch.relu(s1o - s2i + 0.5)
    L = L + torch.relu(s2o - main + 0.5)
    L = L + 0.75 * torch.relu(s2i - s2o + 0.5)
    L = L + 0.3 * torch.relu(s1o - s1i + 0.5)
    return L.mean()


def band_loss(phys, use_band):
    if not use_band:
        return torch.zeros((), device=DEVICE)
    main = phys[:, 4]
    return (torch.relu(main - T_BAND[1]) + torch.relu(T_BAND[0] - main)).mean()


def baseline_train(Xtr, Ytr, Xva, Yva, info, use_band, use_order):
    F_in = info["F"]
    sep_idx = info["exo_cols"].index("分离器出口温度")
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Yva_t = torch.from_numpy(Yva).to(DEVICE)
    mu_sep = info["mu"]["分离器出口温度"]
    sd_sep = info["sd"]["分离器出口温度"]
    model = Net(F_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=DEVICE)
    best_va, best_state, patience = 1e9, None, 0
    n_batch = len(Xtr_t) // 256
    for ep in range(60):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        for b in range(n_batch):
            i = perm[b * 256: (b + 1) * 256]
            yz = model(Xtr_t[i])
            phys = model_out(yz, info)
            t_sep = Xtr_t[i][:, -1, sep_idx] * sd_sep + mu_sep
            mse = ((yz - Ytr_t[i]) ** 2 * w).mean()
            lo = order_loss(phys, t_sep, use_order)
            lb = band_loss(phys, use_band)
            loss = mse + 0.05 * lo + 0.05 * lb
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            yz = model(Xva_t)
            va = ((yz - Yva_t) ** 2 * w).mean().item()
        if va < best_va:
            best_va, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    return model, best_va


def baseline_rollout(model, df, info, phys, v, seed, tag, fast):
    use_phys = phys
    exo_cols = info["exo_cols"]
    phy_cols = info["phy_cols"]
    cols = exo_cols + OUTPUTS + (phy_cols if use_phys else [])
    feats = physics_features(df) if use_phys else None
    full = pd.concat([df[exo_cols], df[OUTPUTS]] + ([feats] if feats is not None else []), axis=1)
    mu = pd.Series(info["mu"])
    sd = pd.Series(info["sd"])
    start = TRAIN_N + VAL_N
    n_steps = 120 if fast else ROLL_STEPS
    window = full.iloc[start - SEQ: start].reset_index(drop=True).copy()
    preds, truths = [], []
    model.eval()
    with torch.no_grad():
        for t in range(n_steps):
            i = start + t
            true_row = df.iloc[i]
            z = np.ascontiguousarray(((window[cols] - mu[cols]) / sd[cols]).to_numpy(np.float32))
            yz = model(torch.from_numpy(z[None]).to(DEVICE))[0]
            pred_T = model_out(yz.unsqueeze(0), info)[0].cpu().numpy()
            preds.append(pred_T)
            truths.append([true_row[c] for c in OUTPUTS])
            new = pd.Series(0.0, index=cols)
            for c in exo_cols:
                new[c] = true_row[c]
            for j, c in enumerate(OUTPUTS):
                new[c] = pred_T[j]
            if use_phys:
                for k, c in enumerate(phy_cols):
                    if c == "dspray_6":
                        prev = df.iloc[i - 6]["二级减温喷水调节门指令"]
                        new[c] = true_row["二级减温喷水调节门指令"] - prev
                    else:
                        new[c] = 0.0
                new = recompute_phys(new)
            window = pd.concat([window.iloc[1:], new.to_frame().T], ignore_index=True)
    preds = np.array(preds)
    truths = np.array(truths)
    main_p, main_t = preds[:, 4], truths[:, 4]
    PAIRS = [(1, 2), (3, 4), (2, 3), (1, 0), (0, 2)]
    viol = np.zeros(n_steps, dtype=bool)
    for lo, hi in PAIRS:
        viol |= (preds[:, lo] >= preds[:, hi])
    mu_o = np.array([info["mu"][c] for c in OUTPUTS], dtype=np.float32)
    sd_o = np.array([info["sd"][c] for c in OUTPUTS], dtype=np.float32)
    z_all = np.abs((preds - mu_o) / sd_o)
    r = {
        "rmse_main": float(np.sqrt(np.mean((main_p - main_t) ** 2))),
        "maxerr_main": float(np.max(np.abs(main_p - main_t))),
        "rmse_all": float(np.sqrt(np.mean((preds - truths) ** 2))),
        "band_viol_frac": float(np.mean((main_p > T_BAND[1]) | (main_p < T_BAND[0]))),
        "order_viol_any_frac": float(viol.mean()),
        "order_main_lt_sh2out_frac": float(np.mean(preds[:, 3] >= preds[:, 4])),
        "drift_main_mean_z": round(float(z_all[:, 4].mean()), 3),
        "drift_main_max_z": round(float(z_all[:, 4].max()), 3),
        "drift_all_mean_z": round(float(z_all.mean()), 3),
    }
    return r, preds, truths


def recompute_phys(row):
    w = row["减温水总流量"]
    feco = row["省煤器出口给水温度"]
    sh1i = row["一级减温器入口温度"]
    sh1o = row["一级减温器出口温度"]
    sh2i = row["二级减温器入口温度"]
    sh2o = row["二级减温器出口温度"]
    main = row["末级过热器出口汽温"]
    sep = row["分离器出口温度"]
    steam = row["主蒸汽流量"]
    coal = row["未校正总煤量"]
    row["spray_cool_1"] = w * (sh1i - feco)
    row["spray_cool_2"] = w * (sh2i - feco)
    row["adv_1"] = steam * (sh1i - sep)
    row["adv_2"] = steam * (sh2i - sh1o)
    row["adv_3"] = steam * (main - sh2o)
    row["heat_intensity"] = coal / (steam + 1.0)
    return row


def run_baseline(df, v, seed, fast, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    phys, band, order = parse_variant(v)
    t0 = time.time()
    Xtr, Ytr, Xva, Yva, Xte, Yte, dff, info = baseline_build_data(df, v, phys)
    model, va = baseline_train(Xtr, Ytr, Xva, Yva, info, band, order)
    torch.save(model.state_dict(), os.path.join(OUT, f"model_{v}_seed{seed}{tag}.pt"))
    t0 = time.time()
    r, preds, truths = baseline_rollout(model, dff, info, phys, v, seed, tag, fast)
    r["variant"] = v
    r["seed"] = seed
    r["val_mse_z"] = round(va, 4)
    r["rollout_seconds"] = round(time.time() - t0, 1)
    np.savez(os.path.join(OUT, f"rollout_{v}_seed{seed}{tag}.npz"), preds=preds, truths=truths)
    with open(os.path.join(OUT, f"results_{v}_seed{seed}{tag}.json"), "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"[{v} s{seed}] {json.dumps(r, ensure_ascii=False)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="e0,v0,v2,v2o")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()
    tag = ".fast" if args.fast else ""
    if args.fast:
        tag = ".fast"
    elif args.max_epochs:
        tag = f".ep{args.max_epochs}"
    variants = [x.strip() for x in args.variants.split(",")]
    df = pd.read_csv(CSV, usecols=EXO + EXO_EXTRA + OUTPUTS + [
        "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
        dtype=np.float32).iloc[WIN_START: WIN_START + WIN].ffill().bfill().reset_index(drop=True)
    print(f"[seed {args.seed}] variants={variants} n={len(df)}", flush=True)
    data_stats(df)
    for v in variants:
        if v == "e0":
            run_e0(df, args.seed, args.fast, tag, args.max_epochs or 60)
        else:
            run_baseline(df, v, args.seed, args.fast, tag)
    print(f"[seed {args.seed}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
