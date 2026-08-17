#!/usr/bin/env python3
"""09_residual.py: ODEWorld 借探针 — 冻结灰盒 + 条件残差矢量场（2026-08-17）

背景: ODEWorld(清华AIR×Berkeley, arXiv:2607.27924) 三个可借设计的最小化验证:
  1. 条件矢量场 v(z,t;z0,c): 灰盒骨架不动, 加学到的残差 dT/dt 修正, 以窗口初始态 z0 作锚
  2. z0 锚定消融 (其 decoupling 消融声称 z0 条件化是关键)
  3. 残差注入点对比: Q 残差(气侧热流修正, 通式) vs Δk 残差(option D: k_i=f(工况) MLP 特例)

变体:
  r0  = residual 关闭 (sanity: 须复现 e0-post seed0 数字)
  ra  = Q 残差 + z0 锚 (主探针, seeds 0/1)
  rb  = Q 残差无锚 (z0 消融, seed 0)
  rc  = Δk 残差 + z0 锚 (option D 对照, seed 0)

训练协议: 与 e0 相同 — 60 步开窗 open-loop MSE (w5=[1,1,1,1,2]), 灰盒参数冻结,
  Adam lr=1e-3, batch 256, max 40 ep, val patience 8。训测同一代码路径 integrate_res。

预注册判定门槛 (2026-08-17, 不可改):
  S0 sanity: |r0.rollout.rmse_main - 12.657| < 0.02; dry 17.059/wet 6.966/win60dry 10.836 ±0.05
  G1weak:  ra dry rollout rmse_main <= 11.95 (>=30% 改进 vs 17.059)
  G1strong: ra dry rollout rmse_main <= 9.75 (向 v2 2.44 关闭 >=50% 缺口)
  G2:      ra wet rollout rmse_main <= 7.66 (退化 <= +10%)
  G3:      rollout 无 NaN 且 maxerr_main <= 100
  G4(info): ra win60 dry <= 8.67 (>=20% 改进 vs 10.836)
  裁决: STRONG = G1strong∧G2∧G3; WEAK = G1weak∧G2∧G3; 否则 FAIL。

用法:
  python 09_residual.py --fast                    # 冒烟 (2ep/rollout 120 步)
  python 09_residual.py --variants ra,rb,rc       # 全量
产物: out/residual_probe_summary.json, out/rollout_res_*.npz, out/model_res_*.pt,
      out/figs/fig9a_case_dry.png fig9b_case_wet.png fig9c_rollout.png fig9d_drift.png
"""
import argparse
import importlib.util
import json
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t02", os.path.join(os.getcwd(), "02_train.py"))
t02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t02)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = t02.DEVICE
OUT = t02.OUT
START = t02.TRAIN_N + t02.VAL_N
P_CRIT = t02.P_CRIT
SEQ = t02.SEQ

# ---------------- 预注册门槛（冻结） ----------------
GATES = {
    "S0": {"rmse_main": 12.657, "tol": 0.02,
           "dry": 17.059, "wet": 6.966, "win60_dry": 10.836, "win60_wet": 5.984,
           "first_dry": 1.998, "first_wet": 1.252, "tol_strat": 0.05},
    "G1weak": 11.95, "G1strong": 9.75, "G2": 7.66, "G4": 8.67,
}
Q_SCALE = 3.0e4     # kW: Q 残差幅值 (Q_typ=UA·ΔT≈1.0e5 kW, 30%)
K_SCALE = 0.3e6     # kJ/t: Δk 残差幅值 (k≈2e6)

# 特征归一化常数 (物理尺度, 探针级)
TS_M, TS_S = 555.0, 25.0
TM_M, TM_S = 720.0, 40.0
PM_M, PM_S = 21.0, 5.0
D_M, D_S = 360.0, 60.0
UB_M, UB_S = 250.0, 40.0
V_M, V_S = 0.5, 0.5
W_M, W_S = 3.0, 3.0

E0_COLS = ["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
           "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
           "末级过热器出口压力", "减温水总流量"]
PAIRS_PHYS = [(1, 0), (3, 2), (1, 2), (3, 4), (0, 2)]


def load_e0_df():
    return pd.read_csv(t02.CSV, usecols=E0_COLS + t02.OUTPUTS, dtype=np.float32) \
        .iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)


# ---------------- 残差网络 ----------------
class ResMLP(nn.Module):
    def __init__(self, F_in, scale, out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_in, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, out))
        self.scale = scale

    def forward(self, x):
        y = self.net(x)
        if y.shape[1] == 3:
            return self.scale * torch.tanh(y)  # (B,3)
        z = self.scale * torch.tanh(y[:, :3])  # Q 残差幅值饱和保护
        return torch.cat([z, y[:, 3:]], dim=1)  # (B,6): 后3=λ logit 不缩放(防sigmoid饱和)


def build_feats(ts, Tm, pm, D, uB, rB, v1, v2, W, anchor, no_act=False):
    """ts/Tm: (3,B) raw; 其余 (B,); anchor: (B,7) raw 或 None。→ (B,F)
    no_act=True: 去掉动作通道特征 v1/v2/W (Step⑤ 发现残差污染动作通道——训练数据中
    阀门动作与扰动并发, 残差学到伪相关, 湿态下翻转阀门因果符号)。"""
    f = [
        (ts[0] - TS_M) / TS_S, (ts[1] - TS_M) / TS_S, (ts[2] - TS_M) / TS_S,
        (Tm[0] - TM_M) / TM_S, (Tm[1] - TM_M) / TM_S, (Tm[2] - TM_M) / TM_S,
        (pm - PM_M) / PM_S, (D - D_M) / D_S,
        (uB - UB_M) / UB_S, (rB - UB_M) / UB_S,
    ]
    if not no_act:
        f += [(v1 - V_M) / V_S, (v2 - V_M) / V_S, (W - W_M) / W_S]
    if anchor is not None:
        f += [(anchor[:, 0] - TS_M) / TS_S, (anchor[:, 1] - TS_M) / TS_S,
              (anchor[:, 2] - TS_M) / TS_S,
              (anchor[:, 3] - TM_M) / TM_S, (anchor[:, 4] - TM_M) / TM_S,
              (anchor[:, 5] - TM_M) / TM_S, (anchor[:, 6] - PM_M) / PM_S]
    return torch.stack(f, dim=1)


# ---------------- 带残差的 integrate（逐行复制 E0Model.integrate, mode="none" 时逐位一致） ----------------
def integrate_res(model, res, exo, h, Tm, rB, steps, anchor=None, mode="none",
                  lam_list=None, no_act=False):
    """exo: (B,steps,9); h,Tm: (3,B); rB: (B,)。anchor: (B,7) 或 None。
    mode: none=冻结灰盒原路径; q=Q残差(同注Tm+h); dk=Δk残差;
          qtm/qh/qcon=注入点消融(Tm-only/h-only/守恒传递Tm失h得);
          q0=q残差仅段0; qspl=学到的分配系数λ(Tm得λz,h得(1-λ)z)。
    lam_list: 可选 list, qspl 模式收集每子步 λ(3,B), 仅诊断图用。"""
    Bsz = exo.shape[0]
    M = model.tri("M")[:, None]
    UA = model.tri("UA")[:, None]
    Cm = model.tri("Cm")[:, None]
    tauB = model.val("tauB")
    D, uB, pm, Tm_sep, Tfw, v1, v2, p_out, W = [exo[:, :, j] for j in range(9)]
    h_sw = t02.hliq_of_T(Tfw)
    p0 = pm + (p_out - pm) / 3.0
    p1 = pm + 2.0 * (p_out - pm) / 3.0
    hsep = t02.h_sep_of(pm, Tm_sep)
    out_list = []
    th1_0, th2_0 = model.th_of(pm[:, 0])
    s_den0 = th1_0 * v1[:, 0] + th2_0 * v2[:, 0] + 1e-6
    W0 = W[:, 0].clamp(min=0.0)
    Dsw1 = t02.KAPPA * W0 * (th1_0 * v1[:, 0]) / s_den0
    Dsw2 = t02.KAPPA * W0 * (th2_0 * v2[:, 0]) / s_den0
    hm1 = (D[:, 0] * h[0] + Dsw1 * h_sw[:, 0]) / (D[:, 0] + Dsw1 + 1e-6)
    hm2 = (D[:, 0] * h[1] + Dsw2 * h_sw[:, 0]) / (D[:, 0] + Dsw2 + 1e-6)
    for t in range(steps):
        k_t = model.k_of(pm[:, t])
        th1_t, th2_t = model.th_of(pm[:, t])
        s_den = th1_t * v1[:, t] + th2_t * v2[:, t] + 1e-6
        Wt = W[:, t].clamp(min=0.0)
        for _ in range(t02.N_SUB):
            ts = t02.T_of_ph(torch.stack([p0[:, t], p1[:, t], p_out[:, t]]), h)  # (3,B)
            Q = UA * (Tm - ts)
            if mode == "none":
                Tm = (Tm + t02.DT_SUB * (k_t * rB[None, :] / 3600.0 + UA * ts) / Cm) / (
                    1.0 + t02.DT_SUB * UA / Cm)
                hin = torch.stack([hsep[:, t], hm1, hm2])
                h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q) / M) / (
                    1.0 + t02.DT_SUB * D[:, t][None, :] / M)
            else:
                feats = build_feats(ts, Tm, pm[:, t], D[:, t], uB[:, t], rB,
                                    v1[:, t], v2[:, t], Wt, anchor, no_act=no_act)
                zraw = res(feats)  # (B,out)
                if mode == "q":
                    z = zraw.permute(1, 0)  # (3,B) 同注 Tm 与 h
                    Tm = (Tm + t02.DT_SUB * (k_t * rB[None, :] / 3600.0 + UA * ts + z) / Cm) / (
                        1.0 + t02.DT_SUB * UA / Cm)
                    hin = torch.stack([hsep[:, t], hm1, hm2])
                    h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q + z) / M) / (
                        1.0 + t02.DT_SUB * D[:, t][None, :] / M)
                elif mode in ("qtm", "qh", "qcon", "q0"):
                    # 注入点消融: Tm-only / h-only / 守恒传递(Tm失h得) / 仅段0
                    z = zraw.permute(1, 0).clone()  # (3,B)
                    if mode == "q0":
                        z[1:] = 0.0
                    zT = {"qtm": z, "qh": torch.zeros_like(z),
                          "qcon": -z, "q0": z}[mode]
                    zh = {"qtm": torch.zeros_like(z), "qh": z,
                          "qcon": z, "q0": z}[mode]
                    Tm = (Tm + t02.DT_SUB * (k_t * rB[None, :] / 3600.0 + UA * ts + zT) / Cm) / (
                        1.0 + t02.DT_SUB * UA / Cm)
                    hin = torch.stack([hsep[:, t], hm1, hm2])
                    h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q + zh) / M) / (
                        1.0 + t02.DT_SUB * D[:, t][None, :] / M)
                elif mode == "qspl":
                    # 学到的分配系数 λ: Tm 得 λz, h 得 (1-λ)z
                    zz = zraw[:, :3].permute(1, 0)  # (3,B)
                    lam = torch.sigmoid(zraw[:, 3:]).permute(1, 0)  # (3,B)
                    if lam_list is not None:
                        lam_list.append(lam.detach())
                    Tm = (Tm + t02.DT_SUB * (k_t * rB[None, :] / 3600.0 + UA * ts + lam * zz) / Cm) / (
                        1.0 + t02.DT_SUB * UA / Cm)
                    hin = torch.stack([hsep[:, t], hm1, hm2])
                    h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q + (1.0 - lam) * zz) / M) / (
                        1.0 + t02.DT_SUB * D[:, t][None, :] / M)
                else:  # dk
                    k_eff = k_t + zraw.permute(1, 0)
                    Tm = (Tm + t02.DT_SUB * (k_eff * rB[None, :] / 3600.0 + UA * ts) / Cm) / (
                        1.0 + t02.DT_SUB * UA / Cm)
                    hin = torch.stack([hsep[:, t], hm1, hm2])
                    h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q) / M) / (
                        1.0 + t02.DT_SUB * D[:, t][None, :] / M)
            h = t02._ste_clamp(h, t02.H_LO, t02.H_HI)
            Dsw1 = t02.KAPPA * Wt * (th1_t * v1[:, t]) / s_den
            Dsw2 = t02.KAPPA * Wt * (th2_t * v2[:, t]) / s_den
            hm1 = (D[:, t] * h[0] + Dsw1 * h_sw[:, t]) / (D[:, t] + Dsw1 + 1e-6)
            hm2 = (D[:, t] * h[1] + Dsw2 * h_sw[:, t]) / (D[:, t] + Dsw2 + 1e-6)
            rB = rB + t02.DT_SUB * (uB[:, t] - rB) / tauB
        p = torch.stack([p0[:, t], p0[:, t], p1[:, t], p1[:, t], p_out[:, t]])
        hh = torch.stack([h[0], hm1, h[1], hm2, h[2]])
        out_list.append(t02.T_of_ph(p, hh))
    out = torch.stack(out_list, dim=2).permute(1, 2, 0)
    return out, h, Tm, rB, hm1, hm2


# ---------------- 初始化（同 02_train 闭包 init_states, 加 anchor） ----------------
def init_states(model, init_rows, obs_T):
    D = init_rows[:, 0]
    pm = init_rows[:, 2]
    p_out = init_rows[:, 7]
    p0 = pm + (p_out - pm) / 3.0
    p1 = pm + 2.0 * (p_out - pm) / 3.0
    h0 = t02.h_of_pT(p0, obs_T[:, 0])
    h1 = t02.h_of_pT(p1, obs_T[:, 2])
    h2 = t02.h_of_pT(p_out, obs_T[:, 4])
    ts0 = t02.T_of_ph(p0, h0)
    ts1 = t02.T_of_ph(p1, h1)
    ts2 = t02.T_of_ph(p_out, h2)
    dTm = model.tri("dTm")[:, None]
    ts = torch.stack([ts0, ts1, ts2])
    rB0 = init_rows[:, 1].clone()
    Tm = ts + model.k_of(pm) * rB0[None, :] / 3600.0 / model.tri("UA")[:, None] + dTm
    anchor = torch.stack([ts0, ts1, ts2, Tm[0], Tm[1], Tm[2], pm], dim=1)
    return torch.stack([h0, h1, h2]), Tm, rB0, anchor


def load_e0(seed=0):
    model = t02.E0Model().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(OUT, f"model_e0_seed{seed}.pt"),
                                     map_location=DEVICE, weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


# ---------------- 训练 ----------------
def train_res(df, seed, variant, mode, use_anchor, fast, out=3, no_act=False):
    tag = ".fast" if fast else ""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = load_e0(0)
    F_in = (20 if use_anchor else 13) - (3 if no_act else 0)
    scale = Q_SCALE if mode != "dk" else K_SCALE
    res = ResMLP(F_in, scale, out).to(DEVICE)
    opt = torch.optim.Adam(res.parameters(), lr=1e-3)
    w5 = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=DEVICE)
    tr_s = 25 if fast else 5
    va_s = 100 if fast else 20
    Xtr, Ytr, Itr, Itr_T = t02.e0_build_windows(df, 0, t02.TRAIN_N, tr_s)
    Xva, Yva, Iva, Iva_T = t02.e0_build_windows(df, t02.TRAIN_N, t02.TRAIN_N + t02.VAL_N, va_s)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    Itr_t = torch.from_numpy(Itr).to(DEVICE)
    ItrT_t = torch.from_numpy(Itr_T).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Yva_t = torch.from_numpy(Yva).to(DEVICE)
    Iva_t = torch.from_numpy(Iva).to(DEVICE)
    IvaT_t = torch.from_numpy(Iva_T).to(DEVICE)
    print(f"[{variant} s{seed}] train={len(Xtr)} val={len(Xva)} F={F_in} mode={mode}", flush=True)

    def fwd(exo_t, init_t, obs_t):
        h, Tm, rB, anchor = init_states(model, init_t, obs_t)
        out, *_ = integrate_res(model, res, exo_t, h, Tm, rB, exo_t.shape[1],
                                anchor if use_anchor else None, mode, no_act=no_act)
        return out

    max_ep = 2 if fast else 40
    best_va, best_state, patience = 1e9, None, 0
    n_batch = len(Xtr_t) // 256
    t0 = time.time()
    for ep in range(max_ep):
        res.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        for b in range(n_batch):
            i = perm[b * 256: (b + 1) * 256]
            pred = fwd(Xtr_t[i], Itr_t[i], ItrT_t[i])
            mse = (((pred - Ytr_t[i]) ** 2) * w5).mean()
            opt.zero_grad()
            mse.backward()
            torch.nn.utils.clip_grad_norm_(res.parameters(), 10.0)
            opt.step()
        res.eval()
        with torch.no_grad():
            pred = fwd(Xva_t, Iva_t, IvaT_t)
            va = (((pred - Yva_t) ** 2) * w5).mean().item()
        if va < best_va:
            best_va, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in res.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    res.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, f"model_res_{variant}_seed{seed}{tag}.pt"))
    print(f"[{variant} s{seed}] trained {ep+1}ep val_mse={best_va:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return res, best_va, ep + 1


# ---------------- rollout（复制 e0_rollout, 残差路径） ----------------
def rollout_res(model0, res, df, start, n_steps, mode, use_anchor, no_act=False):
    E_full = df[E0_COLS].copy()
    E_full["主蒸汽流量"] = E_full["主蒸汽流量"] / 3.6
    E_full["一级减温调节门阀位"] = E_full["一级减温调节门阀位"].clip(lower=0) / 100.0
    E_full["二级减温调节门阀位"] = E_full["二级减温调节门阀位"].clip(lower=0) / 100.0
    E = E_full.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    preds = np.empty((n_steps, 5), dtype=np.float32)
    truths = T_all[start: start + n_steps]
    mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    anchor = None
    with torch.no_grad():
        for t in range(n_steps):
            row = E[start + t]
            exo_t = torch.tensor(row, device=DEVICE)[None, None, :]
            if t == 0:
                p0 = row[2] + (row[7] - row[2]) / 3.0
                p1 = row[2] + 2.0 * (row[7] - row[2]) / 3.0
                obs = T_all[start]
                h0 = t02.h_of_pT(torch.tensor(p0, device=DEVICE), torch.tensor(float(obs[0]), device=DEVICE))
                h1 = t02.h_of_pT(torch.tensor(p1, device=DEVICE), torch.tensor(float(obs[2]), device=DEVICE))
                h2 = t02.h_of_pT(torch.tensor(row[7], device=DEVICE), torch.tensor(float(obs[4]), device=DEVICE))
                h = torch.stack([h0, h1, h2])[:, None]
                pst = torch.stack([torch.tensor(p0, device=DEVICE),
                                   torch.tensor(p1, device=DEVICE),
                                   torch.tensor(row[7], device=DEVICE)])[:, None]
                ts = t02.T_of_ph(pst, h)
                rB = torch.tensor([row[1]], device=DEVICE)
                pm0 = torch.tensor([row[2]], device=DEVICE)
                Tm = (ts + model0.k_of(pm0) * rB / 3600.0 / model0.tri("UA")[:, None]
                      + model0.tri("dTm")[:, None])
                if use_anchor:
                    anchor = torch.stack([ts[0], ts[1], ts[2], Tm[0], Tm[1], Tm[2], pm0], dim=1)
            out, h, Tm, rB, hm1, hm2 = integrate_res(model0, res, exo_t, h, Tm, rB, 1, anchor, mode,
                                                     no_act=no_act)
            preds[t] = out[0, 0].cpu().numpy()
    main_p, main_t = preds[:, 4], truths[:, 4]
    viol = np.zeros(n_steps, dtype=bool)
    for lo, hi in PAIRS_PHYS:
        viol |= (preds[:, lo] >= preds[:, hi])
    z_all = np.abs((preds - mu_o) / sd_o)
    r = {
        "rmse_main": float(np.sqrt(np.mean((main_p - main_t) ** 2))),
        "maxerr_main": float(np.max(np.abs(main_p - main_t))),
        "rmse_all": float(np.sqrt(np.mean((preds - truths) ** 2))),
        "band_viol_frac": float(np.mean((main_p > t02.T_BAND[1]) | (main_p < t02.T_BAND[0]))),
        "order_viol_any_frac": float(viol.mean()),
        "drift_main_mean_z": round(float(z_all[:, 4].mean()), 3),
        "nan": bool(np.isnan(preds).any()),
    }
    return r, preds, truths


# ---------------- windowed 分层（复制 07 的 e0_windowed_arrays + layer_agg） ----------------
def windowed_arrays_res(df, model, res, mode, use_anchor, no_act=False):
    Xte, Yte, Ite, Ite_T = t02.e0_build_windows(df, START, len(df) - 1, 10)
    errs_main, errs_sh1, preds_main, pm0_list = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(Xte), 256):
            xb = torch.from_numpy(Xte[b: b + 256]).to(DEVICE)
            yb = torch.from_numpy(Yte[b: b + 256]).to(DEVICE)
            ib = Ite[b: b + 256]
            pm = torch.from_numpy(ib[:, 2]).to(DEVICE)
            p_out = torch.from_numpy(ib[:, 7]).to(DEVICE)
            p0 = pm + (p_out - pm) / 3.0
            p1 = pm + 2.0 * (p_out - pm) / 3.0
            obs = torch.from_numpy(Ite_T[b: b + 256]).to(DEVICE)
            h0 = t02.h_of_pT(p0, obs[:, 0])
            h1 = t02.h_of_pT(p1, obs[:, 2])
            h2 = t02.h_of_pT(p_out, obs[:, 4])
            h = torch.stack([h0, h1, h2])
            ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
            rB = torch.from_numpy(ib[:, 1]).to(DEVICE).clone()
            Tm = (ts + model.k_of(pm) * rB[None, :] / 3600.0 / model.tri("UA")[:, None]
                  + model.tri("dTm")[:, None])
            anchor = torch.stack([ts[0], ts[1], ts[2], Tm[0], Tm[1], Tm[2], pm], dim=1)
            pred, *_ = integrate_res(model, res, xb, h, Tm, rB, xb.shape[1],
                                     anchor if use_anchor else None, mode, no_act=no_act)
            err = pred - yb
            errs_main.append(err[:, :, 4].cpu().numpy())
            errs_sh1.append(err[:, :, 0].cpu().numpy())
            preds_main.append(pred[:, :, 4].cpu().numpy())
            pm0_list.append(ib[:, 2])
    return (np.concatenate(errs_main), np.concatenate(errs_sh1),
            np.concatenate(preds_main), np.concatenate(pm0_list))


def strat_rollout(preds, truths, pm, mu_o, sd_o):
    out = {}
    for mode_name, mask in (("wet", pm <= P_CRIT), ("dry", pm > P_CRIT)):
        p, t = preds[mask], truths[mask]
        viol = np.zeros(mask.sum(), dtype=bool)
        for lo, hi in PAIRS_PHYS:
            viol |= (p[:, lo] >= p[:, hi])
        z = np.abs((p - mu_o) / sd_o)
        out[mode_name] = {
            "n": int(mask.sum()),
            "rmse_main": round(float(np.sqrt(np.mean((p[:, 4] - t[:, 4]) ** 2))), 3),
            "rmse_all": round(float(np.sqrt(np.mean((p - t) ** 2))), 3),
            "bias_5": [round(float((p[:, j] - t[:, j]).mean()), 2) for j in range(5)],
            "band_viol_frac": round(float(np.mean((p[:, 4] > t02.T_BAND[1]) | (p[:, 4] < t02.T_BAND[0]))), 4),
            "viol_phys_frac": round(float(viol.mean()), 4),
            "drift_main_mean_z": round(float(z[:, 4].mean()), 3),
        }
    return out


def layer_agg(errs_main, errs_sh1, preds_main, pm0):
    out = {}
    for mode_name, mask in (("wet", pm0 <= P_CRIT), ("dry", pm0 > P_CRIT)):
        m = errs_main[mask]
        s1 = errs_sh1[mask]
        p = preds_main[mask]
        first = m[:, 0]
        out[mode_name] = {
            "n_win": int(mask.sum()),
            "first_rmse_main": round(float(np.sqrt(np.mean(first ** 2))), 3),
            "win60_rmse_main": round(float(np.sqrt(np.mean(m ** 2))), 3),
            "first_bias_sh1in": round(float(s1[:, 0].mean()), 2),
            "first_bias_main": round(float(first.mean()), 2),
            "win60_band_viol_frac": round(float(np.mean((p > t02.T_BAND[1]) | (p < t02.T_BAND[0]))), 4),
        }
    return out


# ---------------- 图 ----------------
def make_figs(df, arrays, summ, rollout_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = df[E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    E = E.to_numpy(np.float32)
    pm_roll = E[START: START + t02.ROLL_STEPS, 2]
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    steps = np.arange(SEQ)
    C = {"e0": "#1f4e79", "ra": "#c55a11", "rb": "#8b008b", "rc": "#2e8b57"}
    lab = {"e0": "e0-post (frozen)", "ra": "resQ+z0", "rb": "resQ no-z0", "rc": "resΔk+z0"}

    # fig9a/9b: case 窗口轨迹
    for mode_name, fname in (("dry", "fig9a_case_dry.png"), ("wet", "fig9b_case_wet.png")):
        em = arrays["e0"][0]
        pm0 = arrays["e0"][3]
        mask = pm0 > P_CRIT if mode_name == "dry" else pm0 <= P_CRIT
        win_err = np.abs(em).mean(1)
        win_err[~mask] = -1
        idx = np.argsort(win_err)[::-1][:2]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for k, wi in enumerate(idx):
            ax = axes[k]
            s = int(np.floor(wi * 10 + START))
            truth = T_all[s + 1: s + 1 + SEQ, 4]
            ax.plot(steps, truth, color="0.3", lw=1.2, label="truth")
            for v in arrays:
                p = arrays[v][2][wi]
                ax.plot(steps, p, color=C[v], lw=1.0, alpha=0.95,
                        label=f"{lab[v]} (rmse={np.sqrt(np.mean((p-truth)**2)):.1f})")
            ax.set_ylabel("main steam (°C)")
            ax.set_title(f"{mode_name} window @row {s} (pm0={pm0[wi]:.1f} MPa), worst by e0 error")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        axes[1].set_xlabel("steps since window start (×10 s)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figs", fname), dpi=150)
        plt.close(fig)
        print(f"[fig] {fname}", flush=True)

    # fig9c: rollout 主汽温
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    t_axis = np.arange(t02.ROLL_STEPS) / 6.0
    wet_mask = pm_roll <= P_CRIT
    d0 = rollout_data["e0"]
    truth = d0["truths"][:, 4]
    axes[0].fill_between(t_axis, truth.min() - 2, truth.max() + 2, where=wet_mask,
                         color="steelblue", alpha=0.12, label="wet")
    axes[0].plot(t_axis, truth, color="0.35", lw=1.0, label="truth")
    for v, d in rollout_data.items():
        pred = d["preds"][:, 4]
        rmse = np.sqrt(np.mean((pred - truth) ** 2))
        axes[0].plot(t_axis, pred, color=C[v], lw=1.1,
                     label=f"{lab[v]} (rmse={rmse:.1f}°C)")
        err = pred - truth
        axes[1].plot(t_axis, err, color=C[v], lw=0.9)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].axhline(5, color="crimson", ls=":", lw=0.8)
    axes[1].axhline(-5, color="crimson", ls=":", lw=0.8)
    axes[1].set_ylabel("err (°C)")
    axes[0].set_ylabel("main steam (°C)")
    axes[0].set_title("1800-step rollout (seed 0) — e0 vs residual variants")
    axes[0].legend(fontsize=8, ncol=5, loc="upper right")
    axes[1].set_xlabel("time (min)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig9c_rollout.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig9c_rollout.png", flush=True)

    # fig9d: 漂移曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for k, mode_name in enumerate(("wet", "dry")):
        ax = axes[k]
        for v, arrs in arrays.items():
            em, _, _, pm0 = arrs
            mask = pm0 > P_CRIT if mode_name == "dry" else pm0 <= P_CRIT
            m = em[mask].mean(0)
            s = em[mask].std(0)
            nw = mask.sum()
            ax.plot(steps, m, color=C[v], lw=1.5, label=f"{lab[v]} (n={nw})")
            ax.fill_between(steps, m - s / np.sqrt(nw), m + s / np.sqrt(nw),
                            color=C[v], alpha=0.12)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{mode_name} windows — main err vs step")
        ax.set_xlabel("steps (×10 s)")
        if k == 0:
            ax.set_ylabel("err (°C)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig9d_drift.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig9d_drift.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="ra,rb,rc")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()

    if args.figs_only:
        # 不重训: 从已存模型重跑评估 + 出图
        df = load_e0_df()
        model0 = load_e0(0)
        spec = {"e0": ("none", False, "ResMLP"), "ra": ("q", True, "res"),
                "rb": ("q", False, "res"), "rc": ("dk", True, "res")}
        arrays, rollout_data = {}, {}
        for v, (mode, use_anchor, kind) in spec.items():
            if v == "e0":
                res = ResMLP(13, Q_SCALE).to(DEVICE)
            else:
                p = os.path.join(OUT, f"model_res_{v}_seed0.pt")
                if not os.path.exists(p):
                    continue
                F_in = 20 if use_anchor else 13
                scale = Q_SCALE if mode == "q" else K_SCALE
                res = ResMLP(F_in, scale).to(DEVICE)
                res.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
            res.eval()
            r, preds, truths = rollout_res(model0, res, df, START, t02.ROLL_STEPS, mode, use_anchor)
            rollout_data[v] = {"preds": preds, "truths": truths}
            arrays[v] = windowed_arrays_res(df, model0, res, mode, use_anchor)
            print(f"[figs-only {v}] rollout rmse={r['rmse_main']:.3f}", flush=True)
        make_figs(df, arrays, None, rollout_data)
        print("[figs-only] done", flush=True)
        return
    tag = ".fast" if args.fast else ""
    variants = args.variants.split(",")

    df = load_e0_df()
    mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    pm_all = df["分离器出口压力"].to_numpy(np.float32)
    pm_roll = pm_all[START: START + t02.ROLL_STEPS]
    n_roll = t02.ROLL_STEPS  # 1800 步 rollout 仅 ~7s, 不截断 (fast 只减训练 epoch)

    summ = {"probe": "ODEWorld residual borrow", "date": time.strftime("%Y-%m-%d %H:%M"),
            "gates": GATES, "rollout": {}, "windowed": {}, "sanity": {}, "gates_check": {},
            "train": {}}
    rollout_data = {}
    arrays = {}

    # ============ r0 sanity ============
    model0 = load_e0(0)
    res0 = ResMLP(13, Q_SCALE).to(DEVICE)  # 占位, mode=none 不调用
    t0 = time.time()
    r, preds, truths = rollout_res(model0, res0, df, START, n_roll, "none", False)
    r0_secs = time.time() - t0
    s = GATES["S0"]
    d = {"rmse_main_diff": abs(r["rmse_main"] - s["rmse_main"]),
         "rmse_main": round(r["rmse_main"], 4)}
    if n_roll == t02.ROLL_STEPS:
        sr = strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
        summ["sanity"]["rollout_strat"] = sr
        d["dry_diff"] = abs(sr["dry"]["rmse_main"] - s["dry"])
        d["wet_diff"] = abs(sr["wet"]["rmse_main"] - s["wet"])
    arrs = windowed_arrays_res(df, model0, res0, "none", False)
    lr = layer_agg(*arrs)
    summ["sanity"]["windowed"] = lr
    d["win60_dry_diff"] = abs(lr["dry"]["win60_rmse_main"] - s["win60_dry"])
    d["win60_wet_diff"] = abs(lr["wet"]["win60_rmse_main"] - s["win60_wet"])
    d["first_dry_diff"] = abs(lr["dry"]["first_rmse_main"] - s["first_dry"])
    d["first_wet_diff"] = abs(lr["wet"]["first_rmse_main"] - s["first_wet"])
    ok = (d["rmse_main_diff"] < s["tol"] and d["win60_dry_diff"] < s["tol_strat"]
          and d["win60_wet_diff"] < s["tol_strat"])
    if n_roll == t02.ROLL_STEPS:
        ok = ok and d["dry_diff"] < s["tol_strat"] and d["wet_diff"] < s["tol_strat"]
    d["PASS"] = bool(ok)
    summ["sanity"]["check"] = d
    print(f"[r0 sanity] {json.dumps(d)}", flush=True)
    if not ok:
        raise SystemExit("[ABORT] sanity 失败: 残差路径与 e0-post 不一致, 停止回传")
    summ["rollout"]["e0"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
    summ["windowed"]["e0"] = lr
    rollout_data["e0"] = {"preds": preds, "truths": truths}
    arrays["e0"] = arrs
    np.savez(os.path.join(OUT, f"rollout_res_e0{tag}.npz"), preds=preds, truths=truths)
    print(f"[r0 sanity PASS] {r0_secs:.0f}s", flush=True)

    # ============ 变体训练 + 评估 ============
    VSPEC = {"ra": ("q", True), "rb": ("q", False), "rc": ("dk", True)}
    trained = {}
    for v in variants:
        mode, use_anchor = VSPEC[v]
        seeds = [0, 1] if (v == "ra" and not args.fast) else [0]
        for sd_ in seeds:
            res, best_va, ep = train_res(df, sd_, v, mode, use_anchor, args.fast)
            trained[f"{v}_s{sd_}"] = (res, mode, use_anchor)
            summ["train"].setdefault(v, {})[str(sd_)] = {"val_mse": round(best_va, 4), "ep": ep}
            r, preds, truths = rollout_res(model0, res, df, START, n_roll, mode, use_anchor)
            summ["rollout"].setdefault(v, {})[str(sd_)] = {
                k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}
            rollout_data.setdefault(f"{v}_s{sd_}", {"preds": preds, "truths": truths})
            np.savez(os.path.join(OUT, f"rollout_res_{v}_seed{sd_}{tag}.npz"),
                     preds=preds, truths=truths)
            arrs = windowed_arrays_res(df, model0, res, mode, use_anchor)
            summ["windowed"].setdefault(v, {})[str(sd_)] = layer_agg(*arrs)
            arrays.setdefault(f"{v}_s{sd_}", arrs)
            if n_roll == t02.ROLL_STEPS:
                sr = strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
                summ["rollout"].setdefault(v + "_strat", {})[str(sd_)] = sr
            print(f"[{v} s{sd_}] rollout rmse={r['rmse_main']:.3f} "
                  f"win60dry={summ['windowed'][v][str(sd_)]['dry']['win60_rmse_main']:.2f}", flush=True)

    # ============ 门槛裁决 ============
    ck = summ["gates_check"]
    if "ra_s0" in trained:
        ra_roll = summ["rollout"]["ra"]["0"]["rmse_main"]
        ra_max = summ["rollout"]["ra"]["0"]["maxerr_main"]
        ra_nan = summ["rollout"]["ra"]["0"]["nan"]
        ra_win60dry = summ["windowed"]["ra"]["0"]["dry"]["win60_rmse_main"]
        if n_roll == t02.ROLL_STEPS:
            ra_dry = summ["rollout"]["ra_strat"]["0"]["dry"]["rmse_main"]
            ra_wet = summ["rollout"]["ra_strat"]["0"]["wet"]["rmse_main"]
        else:
            ra_dry = ra_wet = None
        ck["G1weak"] = bool(ra_dry is not None and ra_dry <= GATES["G1weak"])
        ck["G1strong"] = bool(ra_dry is not None and ra_dry <= GATES["G1strong"])
        ck["G2"] = bool(ra_wet is not None and ra_wet <= GATES["G2"])
        ck["G3"] = bool((not ra_nan) and ra_max <= 100.0)
        ck["G4"] = bool(ra_win60dry <= GATES["G4"])
        ck["ra_dry"] = ra_dry
        ck["ra_wet"] = ra_wet
        ck["ra_rollout_overall"] = round(ra_roll, 3)
        if ck["G1strong"] and ck["G2"] and ck["G3"]:
            ck["verdict"] = "STRONG"
        elif ck["G1weak"] and ck["G2"] and ck["G3"]:
            ck["verdict"] = "WEAK"
        else:
            ck["verdict"] = "FAIL"

    with open(os.path.join(OUT, f"residual_probe_summary{tag}.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(f"[json] out/residual_probe_summary{tag}.json", flush=True)

    if not args.fast:
        fig_arrays = {"e0": arrays["e0"], "ra": arrays["ra_s0"]}
        if "rb_s0" in arrays:
            fig_arrays["rb"] = arrays["rb_s0"]
        if "rc_s0" in arrays:
            fig_arrays["rc"] = arrays["rc_s0"]
        fig_roll = {"e0": rollout_data["e0"], "ra": rollout_data["ra_s0"]}
        if "rb_s0" in rollout_data:
            fig_roll["rb"] = rollout_data["rb_s0"]
        if "rc_s0" in rollout_data:
            fig_roll["rc"] = rollout_data["rc_s0"]
        make_figs(df, fig_arrays, summ, fig_roll)

    print("\n=== 裁决 ===", flush=True)
    print(json.dumps(ck, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
