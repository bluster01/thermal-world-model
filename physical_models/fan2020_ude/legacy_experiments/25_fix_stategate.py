#!/usr/bin/env python3
"""25_fix_stategate.py: 修复C 相态门控残差 — g(pm)=σ((P_CRIT−pm)/s), s可学习(先验0.5MPa)

背景: 段门控(②)失败 — 残差在湿态是精度+慢动态来源, 干态是通道破坏者; 按相态切而非按段切。
设计: 残差存在的物理理由=两相区单相近似失效, 故残差权威应随 pm 过临界点而熄灭。
门控以 pm (外生) 为条件, 不破坏因果; 转换带宽 s 可学习 = 数据自己定义干湿转换带宽度。

预注册 (冻结 2026-08-17):
  S1: 干态开环耦合阶跃 K<0 且 τ63∈[60,300]s (纯灰盒参考: K=−3.14, τ63=140s; 修复Q1未查符号的缺陷)
  S2: 干态闭环 (真实PI+速率限制0.0137/步) 收敛 norm600∈[0.8,1.2] ∧ tail_std≤0.05
  S3: 湿态开环 τ63∈[240,900]s 保持 (qslow 的 320s 不丢)
  S4: 湿态闭环收敛 norm600∈[0.8,1.2] ∧ tail_std≤0.05
  S5 (审计): rollout mean 报告 / 学得 s 与转换带 g(pm) 曲线 / 湿态 G1 报告
裁决: S1∧S2∧S3∧S4 = PASS
"""
import importlib.util
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t02 = _imp("02_train.py", "t02")
r09 = _imp("09_residual.py", "r09")
r22 = _imp("22_fix_slowdyn.py", "r22")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
RATE = 0.0137
S0 = 0.5
RAW0 = float(np.log(np.e - 1.0))
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])


class QnaLagGate(nn.Module):
    """qnaw残差 + 可学习 τ_sw/τ_sens/τ_res + 可学习相态门控带宽 s (先验0.5MPa)."""

    def __init__(self):
        super().__init__()
        self.mlp = r09.ResMLP(11, r09.Q_SCALE)
        self.raw_sw = nn.Parameter(torch.tensor(RAW0, dtype=torch.float32))
        self.raw_sens = nn.Parameter(torch.tensor(RAW0, dtype=torch.float32))
        self.raw_res = nn.Parameter(torch.tensor(RAW0, dtype=torch.float32))
        self.raw_gate = nn.Parameter(torch.tensor(RAW0, dtype=torch.float32))

    def params_(self):
        return (60.0 * F.softplus(self.raw_sw), 15.0 * F.softplus(self.raw_sens),
                120.0 * F.softplus(self.raw_res), S0 * F.softplus(self.raw_gate))


def integrate_gate(model0, mod, exo, h, Tm, rB, steps, T_sens=None, z_lag=None):
    """残差×相态门控 g(pm)=σ((P_CRIT−pm)/s), 加可学习滞后。返回 (out,h,Tm,rB,hm1,hm2,T_sens,z_lag,g_last)。"""
    tau_sw, tau_sens, tau_res, s_gate = mod.params_()
    a_sw, a_sens, a_res = DT / tau_sw, DT / tau_sens, DT / tau_res
    M = model0.tri("M")[:, None]
    UA = model0.tri("UA")[:, None]
    Cm = model0.tri("Cm")[:, None]
    tauB = model0.val("tauB")
    D, uB, pm, Tm_sep, Tfw, v1, v2, p_out, W = [exo[:, :, j] for j in range(9)]
    h_sw = t02.hliq_of_T(Tfw)
    p0 = pm + (p_out - pm) / 3.0
    p1 = pm + 2.0 * (p_out - pm) / 3.0
    hsep = t02.h_sep_of(pm, Tm_sep)
    out_list = []
    th1_0, th2_0 = model0.th_of(pm[:, 0])
    s_den0 = th1_0 * v1[:, 0] + th2_0 * v2[:, 0] + 1e-6
    W0 = W[:, 0].clamp(min=0.0)
    Dsw1_lag = t02.KAPPA * W0 * (th1_0 * v1[:, 0]) / s_den0
    Dsw2_lag = t02.KAPPA * W0 * (th2_0 * v2[:, 0]) / s_den0
    hm1 = (D[:, 0] * h[0] + Dsw1_lag * h_sw[:, 0]) / (D[:, 0] + Dsw1_lag + 1e-6)
    hm2 = (D[:, 0] * h[1] + Dsw2_lag * h_sw[:, 0]) / (D[:, 0] + Dsw2_lag + 1e-6)
    if z_lag is None:
        z_lag = torch.zeros(3, exo.shape[0], device=DEVICE, dtype=torch.float32)
    g_last = None
    for t in range(steps):
        k_t = model0.k_of(pm[:, t])
        th1_t, th2_t = model0.th_of(pm[:, t])
        s_den = th1_t * v1[:, t] + th2_t * v2[:, t] + 1e-6
        Wt = W[:, t].clamp(min=0.0)
        Dsw1_new = t02.KAPPA * Wt * (th1_t * v1[:, t]) / s_den
        Dsw2_new = t02.KAPPA * Wt * (th2_t * v2[:, t]) / s_den
        Dsw1_lag = Dsw1_lag + a_sw * (Dsw1_new - Dsw1_lag)
        Dsw2_lag = Dsw2_lag + a_sw * (Dsw2_new - Dsw2_lag)
        g = torch.sigmoid((P_CRIT - pm[:, t]) / s_gate)  # (B,) 湿态→1, 干态→0
        g_last = g.detach().clone()
        for _ in range(t02.N_SUB):
            ts = t02.T_of_ph(torch.stack([p0[:, t], p1[:, t], p_out[:, t]]), h)
            Q = UA * (Tm - ts)
            feats = r09.build_feats(ts, Tm, pm[:, t], D[:, t], uB[:, t], rB,
                                    v1[:, t], v2[:, t], Wt, None, no_v12=True)
            z = mod.mlp(feats).permute(1, 0)
            z_eff = z * g[None, :]
            Tm = (Tm + t02.DT_SUB * (k_t * rB[None, :] / 3600.0 + UA * ts + z_lag) / Cm) / (
                1.0 + t02.DT_SUB * UA / Cm)
            hin = torch.stack([hsep[:, t], hm1, hm2])
            h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q + z_lag) / M) / (
                1.0 + t02.DT_SUB * D[:, t][None, :] / M)
            h = t02._ste_clamp(h, t02.H_LO, t02.H_HI)
            hm1 = (D[:, t] * h[0] + Dsw1_lag * h_sw[:, t]) / (D[:, t] + Dsw1_lag + 1e-6)
            hm2 = (D[:, t] * h[1] + Dsw2_lag * h_sw[:, t]) / (D[:, t] + Dsw2_lag + 1e-6)
            rB = rB + t02.DT_SUB * (uB[:, t] - rB) / tauB
        z_lag = z_lag + a_res * (z_eff - z_lag)
        p = torch.stack([p0[:, t], p0[:, t], p1[:, t], p1[:, t], p_out[:, t]])
        hh = torch.stack([h[0], hm1, h[1], hm2, h[2]])
        T_raw = t02.T_of_ph(p, hh)
        if T_sens is None:
            T_sens = T_raw
        else:
            T_sens = T_sens + a_sens * (T_raw - T_sens)
        out_list.append(T_sens)
    out = torch.stack(out_list, dim=2).permute(1, 2, 0)
    return out, h, Tm, rB, hm1, hm2, T_sens, z_lag, g_last


def train_gate2(df, seed, fast=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model0 = r09.load_e0(0)
    mod = QnaLagGate().to(DEVICE)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
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

    def fwd(exo_t, init_t, obs_t):
        h, Tm, rB, _anchor = r09.init_states(model0, init_t, obs_t)
        T_sens0 = obs_t.permute(1, 0)
        out, *_ = integrate_gate(model0, mod, exo_t, h, Tm, rB, exo_t.shape[1],
                                 T_sens=T_sens0)
        return out

    max_ep = 2 if fast else 40
    best_va, best_state, patience = 1e9, None, 0
    n_batch = len(Xtr_t) // 256
    n_ep_done = 0
    for ep in range(max_ep):
        n_ep_done = ep + 1
        mod.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        for b in range(n_batch):
            i = perm[b * 256: (b + 1) * 256]
            pred = fwd(Xtr_t[i], Itr_t[i], ItrT_t[i])
            mse = (((pred - Ytr_t[i]) ** 2) * w5).mean()
            opt.zero_grad()
            mse.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 10.0)
            opt.step()
        mod.eval()
        with torch.no_grad():
            pred = fwd(Xva_t, Iva_t, IvaT_t)
            va = (((pred - Yva_t) ** 2) * w5).mean().item()
        if va < best_va:
            best_va, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in mod.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    mod.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, f"model_res_qstate_seed{seed}.pt"))
    tsw, tse, tre, sg = mod.params_()
    print(f"[qstate s{seed}] {n_ep_done}ep val={best_va:.3f} τ_sw={tsw.item():.1f} "
          f"τ_sens={tse.item():.1f} τ_res={tre.item():.1f} s_gate={sg.item():.2f}MPa",
          flush=True)
    return mod, best_va


def main():
    df = r09.load_e0_df()
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]
    model0 = r09.load_e0(0)

    summ = {"train": {}, "rollout": {}, "loop": {}, "step": {}, "judge": {}}
    for sd in (0, 1):
        mod, va = train_gate2(df, sd)
        summ["train"][str(sd)] = {"val_mse": round(va, 4),
                                  "params": [round(float(x.item()), 2) for x in mod.params_()]}

    mod0 = QnaLagGate().to(DEVICE)
    mod0.load_state_dict(torch.load(os.path.join(OUT, "model_res_qstate_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    mod0.eval()
    for p in mod0.parameters():
        p.requires_grad_(False)
    _, _, _, sg = mod0.params_()
    print(f"[gate curve] s={sg.item():.2f}MPa: g(20.0)={torch.sigmoid((P_CRIT-20.0)/sg).item():.3f} "
          f"g(22.064)={torch.sigmoid((P_CRIT-22.064)/sg).item():.3f} "
          f"g(24.2)={torch.sigmoid((P_CRIT-24.2)/sg).item():.3f}", flush=True)

    # rollout (S5)
    rolls = []
    for sd in (0, 1):
        mod = QnaLagGate().to(DEVICE)
        mod.load_state_dict(torch.load(os.path.join(OUT, f"model_res_qstate_seed{sd}.pt"),
                                       map_location=DEVICE, weights_only=True))
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
        preds = np.empty((t02.ROLL_STEPS, 5), dtype=np.float32)
        T_sens, z_lag = None, None
        with torch.no_grad():
            for t in range(t02.ROLL_STEPS):
                row = Ea[START + t]
                exo_t = torch.tensor(row, device=DEVICE)[None, None, :]
                if t == 0:
                    obs = T_all[START]
                    h, Tm, rB = r22.init_state(model0, row, obs)
                    T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
                out, h, Tm, rB, hm1, hm2, T_sens, z_lag, g_l = integrate_gate(
                    model0, mod, exo_t, h, Tm, rB, 1, T_sens=T_sens, z_lag=z_lag)
                preds[t] = out[0, 0].cpu().numpy()
        truths = T_all[START: START + t02.ROLL_STEPS]
        rmse = float(np.sqrt(np.mean((preds[:, 4] - truths[:, 4]) ** 2)))
        rolls.append(rmse)
        print(f"[qstate s{sd}] rollout={rmse:.3f}", flush=True)

    # k_w
    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    def fwd_one(row, h, Tm, rB, T_sens, z_lag, v2_val, W_val):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = v2_val
        exo[0, 0, 8] = W_val
        out, h, Tm, rB, hm1, hm2, T_sens, z_lag, g_l = integrate_gate(
            model0, mod0, exo, h, Tm, rB, 1, T_sens=T_sens, z_lag=z_lag)
        return float(out[0, 0, 4]), h, Tm, rB, T_sens, z_lag

    def base_run(row_idx):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r22.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        z_lag = None
        u0, W0 = float(row[V2]), float(row[8])
        b = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                b[t], h, Tm, rB, T_sens, z_lag = fwd_one(row, h, Tm, rB, T_sens, z_lag, u0, W0)
        return b

    base_wet = base_run(OP_WET)
    base_dry = base_run(OP_DRY)

    # 开环阶跃 (S1/S3)
    for name, row_idx, base, state in (("wet", OP_WET, base_wet, "wet"),
                                       ("dry", OP_DRY, base_dry, "dry")):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r22.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        z_lag = None
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[state]
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens, z_lag = fwd_one(row, h, Tm, rB, T_sens, z_lag,
                                                          u0 + 0.05, W0 * (1 + kw * 0.05))
        d = dT - base
        K = float(np.mean(d[-60:]))
        idx = np.where(d <= 0.63 * K)[0] if K < 0 else np.where(d >= 0.63 * K)[0]
        tau63 = int(idx[0]) * DT if len(idx) else None
        summ["step"][name] = {"K": round(K, 3), "tau63_s": tau63}
        print(f"[step {name}] K={K:.3f} τ63={tau63}s", flush=True)

    # 闭环 (S2/S4/S5)
    for name, row_idx, base, state, power in (("wet", OP_WET, base_wet, "wet", 332.85),
                                              ("dry", OP_DRY, base_dry, "dry", 464.53)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r22.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        z_lag = None
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[state]
        SP = float(obs[4]) + 2.0
        u, integ, v = u0, 0.0, u0
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens, z_lag = fwd_one(row, h, Tm, rB, T_sens, z_lag, v,
                                                          W0 * (1 + kw * (v - u0)))
                e = mh[t] - SP
                kp, ti = r22.pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                v = float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        anch = [float(norm[int(i) - 1]) for i in ANCH_T]
        deltas = [round(abs(a - b), 3) for a, b in zip(anch, ANCH_Y)] if name == "wet" else None
        summ["loop"][name] = {"anchors": [round(x, 3) for x in anch],
                              "norm600": round(float(norm[599]), 3),
                              "tail_std": round(float(np.std(norm[-120:])), 4),
                              "G1_deltas": deltas}
        print(f"[loop {name}] anchors={summ['loop'][name]['anchors']} "
              f"norm600={summ['loop'][name]['norm600']} "
              f"tail_std={summ['loop'][name]['tail_std']} Δ={deltas}", flush=True)

    S1 = bool(summ["step"]["dry"]["K"] < 0 and summ["step"]["dry"]["tau63_s"] is not None
              and 60 <= summ["step"]["dry"]["tau63_s"] <= 300)
    S2 = bool(0.8 <= summ["loop"]["dry"]["norm600"] <= 1.2
              and summ["loop"]["dry"]["tail_std"] <= 0.05)
    S3 = bool(summ["step"]["wet"]["tau63_s"] is not None
              and 240 <= summ["step"]["wet"]["tau63_s"] <= 900)
    S4 = bool(0.8 <= summ["loop"]["wet"]["norm600"] <= 1.2
              and summ["loop"]["wet"]["tail_std"] <= 0.05)
    judge = {"S1": S1, "S2": S2, "S3": S3, "S4": S4,
             "rollout_mean": round(float(np.mean(rolls)), 3),
             "s_gate_MPa": round(float(sg.item()), 2),
             "verdict": "PASS" if (S1 and S2 and S3 and S4) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fix_stategate_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    print("=== 修复C 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    t_axis = np.arange(N) * DT / 60.0
    ax = axes[0]
    for name, color in (("wet", "#c55a11"), ("dry", "#8b008b")):
        row_idx = OP_WET if name == "wet" else OP_DRY
        base = base_wet if name == "wet" else base_dry
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r22.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        z_lag = None
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[name]
        SP = float(obs[4]) + 2.0
        power = 332.85 if name == "wet" else 464.53
        u, integ, v = u0, 0.0, u0
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens, z_lag = fwd_one(row, h, Tm, rB, T_sens, z_lag, v,
                                                          W0 * (1 + kw * (v - u0)))
                e = mh[t] - SP
                kp, ti = r22.pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                v = float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        ax.plot(t_axis, dC / dC_ss, lw=1.4, color=color, label=f"{name} closed loop")
    ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.3", ms=7, label="exp_099")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_title("closed loop: state-gated residual")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for name, color in (("wet", "#c55a11"), ("dry", "#8b008b")):
        row_idx = OP_WET if name == "wet" else OP_DRY
        base = base_wet if name == "wet" else base_dry
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r22.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        z_lag = None
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[name]
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens, z_lag = fwd_one(row, h, Tm, rB, T_sens, z_lag,
                                                          u0 + 0.05, W0 * (1 + kw * 0.05))
        d = dT - base
        K = float(np.mean(d[-60:]))
        ax.plot(t_axis, d / K, lw=1.4, color=color, label=f"{name} open-loop step (norm)")
    ax.axhline(0.63, color="0.5", ls=":", lw=0.8)
    ax.set_title("open-loop normalized step")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[2]
    pm_grid = np.linspace(15, 26, 100)
    g_curve = 1.0 / (1.0 + np.exp(-(P_CRIT - pm_grid) / float(sg.item())))
    ax.plot(pm_grid, g_curve, lw=2, color="#8b008b")
    ax.axvline(P_CRIT, color="crimson", ls=":", lw=1, label="P_crit 22.064")
    ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax.set_title(f"learned phase gate (s={float(sg.item()):.2f} MPa)")
    ax.set_xlabel("pm (MPa)")
    ax.set_ylabel("g(pm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"FixC state-gated residual — verdict={judge['verdict']} "
                 f"(S1={S1} S2={S2} S3={S3} S4={S4})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig26_stategate.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig26_stategate.png", flush=True)


if __name__ == "__main__":
    main()
