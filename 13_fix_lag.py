#!/usr/bin/env python3
"""13_fix_lag.py: 修复② 喷水输运滞后(τ_sw=60s) + 输出传感器滞后(τ_sens=15s)

设计稿 FIX2_DESIGN.md (预注册冻结):
  B1: 干态闭环SP+2°C 归一化 vs exp_099锚点, 各|Δ|≤0.15 且600s≤0.10
  B2: 开环v2+5%→main τ63∈[240,900]s 且 θ∈[0,300]s
  B3: rollout main rmse 2seeds mean ≤ 3.2
  B4: 湿态闭环 norm600∈[0.8,1.2] 且湿态开环末值<0
残差: qnaw配置(去v1/v2保留W, 11特征) × 2 seeds; e0灰盒参数冻结。
integrate_lag 返回 T_sens 状态, 调用方(闭环/rollout)跨步持久化 — 滞后在单步调用中也生效。
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
import numpy as np
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
TAU_SW, TAU_SENS = 60.0, 15.0
A_SW = DT / TAU_SW
A_SENS = DT / TAU_SENS
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])

FX44_X = np.array([-12, -10, -8, -5, -3, 3, 5, 8, 10, 12])
FX44_Y = np.array([0.6, 0.6, 0.8, 1.0, 1.2, 1.2, 1.0, 0.8, 0.6, 0.6])
FX45_Y = np.array([800, 650, 550, 450, 350, 350, 450, 550, 650, 800])
FX49_X = np.array([150, 200, 300, 400, 500, 600])
FX49_Y = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.5])
FX50_Y = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.6])


def pi_params(dev, power):
    kp = float(np.interp(abs(dev), FX44_X, FX44_Y)) * float(np.interp(power, FX49_X, FX49_Y))
    ti = float(np.interp(abs(dev), FX44_X, FX45_Y)) * float(np.interp(power, FX49_X, FX50_Y))
    return kp, ti


def integrate_lag(model0, res, exo, h, Tm, rB, steps, T_sens=None):
    """qh 残差(no_v12) + 喷水输运滞后(τ_sw=60s, 每10s步一阶) + 输出传感器滞后(τ_sens=15s)。
    T_sens: (5,B) 传感器状态, None=首步原始输出起步。返回 (out, h, Tm, rB, hm1, hm2, T_sens)。"""
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
    for t in range(steps):
        k_t = model0.k_of(pm[:, t])
        th1_t, th2_t = model0.th_of(pm[:, t])
        s_den = th1_t * v1[:, t] + th2_t * v2[:, t] + 1e-6
        Wt = W[:, t].clamp(min=0.0)
        Dsw1_new = t02.KAPPA * Wt * (th1_t * v1[:, t]) / s_den
        Dsw2_new = t02.KAPPA * Wt * (th2_t * v2[:, t]) / s_den
        Dsw1_lag = Dsw1_lag + A_SW * (Dsw1_new - Dsw1_lag)
        Dsw2_lag = Dsw2_lag + A_SW * (Dsw2_new - Dsw2_lag)
        for _ in range(t02.N_SUB):
            ts = t02.T_of_ph(torch.stack([p0[:, t], p1[:, t], p_out[:, t]]), h)
            Q = UA * (Tm - ts)
            feats = r09.build_feats(ts, Tm, pm[:, t], D[:, t], uB[:, t], rB,
                                    v1[:, t], v2[:, t], Wt, None, no_v12=True)
            z = res(feats).permute(1, 0)
            Tm = (Tm + t02.DT_SUB * (k_t * rB[None, :] / 3600.0 + UA * ts + z) / Cm) / (
                1.0 + t02.DT_SUB * UA / Cm)
            hin = torch.stack([hsep[:, t], hm1, hm2])
            h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q + z) / M) / (
                1.0 + t02.DT_SUB * D[:, t][None, :] / M)
            h = t02._ste_clamp(h, t02.H_LO, t02.H_HI)
            hm1 = (D[:, t] * h[0] + Dsw1_lag * h_sw[:, t]) / (D[:, t] + Dsw1_lag + 1e-6)
            hm2 = (D[:, t] * h[1] + Dsw2_lag * h_sw[:, t]) / (D[:, t] + Dsw2_lag + 1e-6)
            rB = rB + t02.DT_SUB * (uB[:, t] - rB) / tauB
        p = torch.stack([p0[:, t], p0[:, t], p1[:, t], p1[:, t], p_out[:, t]])
        hh = torch.stack([h[0], hm1, h[1], hm2, h[2]])
        T_raw = t02.T_of_ph(p, hh)  # (5,B)
        if T_sens is None:
            T_sens = T_raw
        else:
            T_sens = T_sens + A_SENS * (T_raw - T_sens)
        out_list.append(T_sens)
    out = torch.stack(out_list, dim=2).permute(1, 2, 0)
    return out, h, Tm, rB, hm1, hm2, T_sens


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


def train_lag(df, seed, fast=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model0 = r09.load_e0(0)
    res = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
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

    def fwd(exo_t, init_t, obs_t):
        h, Tm, rB = init_state(model0, init_t, obs_t)
        T_sens0 = obs_t.permute(1, 0)  # (5,B) 传感器初值=窗口起点真值
        out, *_ = integrate_lag(model0, res, exo_t, h, Tm, rB, exo_t.shape[1], T_sens=T_sens0)
        return out

    max_ep = 2 if fast else 40
    best_va, best_state, patience = 1e9, None, 0
    n_batch = len(Xtr_t) // 256
    n_ep_done = 0
    for ep in range(max_ep):
        n_ep_done = ep + 1
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
    torch.save(best_state, os.path.join(OUT, f"model_res_qlag_seed{seed}.pt"))
    print(f"[qlag s{seed}] {n_ep_done}ep val={best_va:.3f}", flush=True)
    return res, best_va


def load_res(v, seed, F_in=11):
    res = r09.ResMLP(F_in, r09.Q_SCALE).to(DEVICE)
    res.load_state_dict(torch.load(os.path.join(OUT, f"model_res_{v}_seed{seed}.pt"),
                                   map_location=DEVICE, weights_only=True))
    res.eval()
    return res


def main():
    df = r09.load_e0_df()
    mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    pm_roll = df["分离器出口压力"].to_numpy(np.float32)[START: START + t02.ROLL_STEPS]
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    model0 = r09.load_e0(0)

    summ = {"train": {}, "rollout": {}, "step": {}, "cl": {}, "judge": {}}
    rolls = []
    for sd in (0, 1):
        res, va = train_lag(df, sd)
        summ["train"][str(sd)] = {"val_mse": round(va, 4)}
    res_s0 = load_res("qlag", 0)

    # ---- rollout (传感器状态跨步持久化) ----
    for sd in (0, 1):
        res = load_res("qlag", sd)
        preds = np.empty((t02.ROLL_STEPS, 5), dtype=np.float32)
        T_sens = None
        with torch.no_grad():
            for t in range(t02.ROLL_STEPS):
                row = Ea[START + t]
                exo_t = torch.tensor(row, device=DEVICE)[None, None, :]
                if t == 0:
                    obs = T_all[START]
                    h, Tm, rB = init_state(model0, row, obs)
                    T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
                out, h, Tm, rB, hm1, hm2, T_sens = integrate_lag(
                    model0, res, exo_t, h, Tm, rB, 1, T_sens=T_sens)
                preds[t] = out[0, 0].cpu().numpy()
        truths = T_all[START: START + t02.ROLL_STEPS]
        rmse_main = float(np.sqrt(np.mean((preds[:, 4] - truths[:, 4]) ** 2)))
        sr = r09.strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
        rolls.append(rmse_main)
        summ["rollout"][str(sd)] = {"rmse_main": round(rmse_main, 4),
                                    "dry": sr["dry"]["rmse_main"],
                                    "wet": sr["wet"]["rmse_main"]}
        print(f"[qlag s{sd}] rollout={rmse_main:.3f} dry={sr['dry']['rmse_main']:.2f} "
              f"wet={sr['wet']['rmse_main']:.2f}", flush=True)

    # ---- 开环阶跃 (B2/B4 符号) ----
    step_res = {}
    for name, row_idx in (("wet", OP_WET), ("dry", OP_DRY)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens0 = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        exo_b = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, N, 1)
        base, *_ = integrate_lag(model0, res_s0, exo_b, h, Tm, rB, N, T_sens=T_sens0)
        h, Tm, rB = init_state(model0, row, obs)
        exo_s = exo_b.clone()
        exo_s[:, :, V2] += 0.05
        tr, *_ = integrate_lag(model0, res_s0, exo_s, h, Tm, rB, N, T_sens=T_sens0)
        d = tr[0].cpu().numpy() - base[0].cpu().numpy()
        dT = d[:, 4]
        K = float(np.mean(dT[-60:]))
        theta = int(np.argmax(np.abs(dT) >= 0.02 * abs(K)))
        thr = 0.63 * K
        idx = np.where(dT <= thr)[0] if K < 0 else np.where(dT >= thr)[0]
        tau63 = int(idx[0]) if len(idx) else None
        step_res[name] = {"K": round(K, 3), "tau63": tau63, "theta": theta,
                          "main_final": round(float(dT[-1]), 3)}
        summ["step"][name] = step_res[name]
        print(f"[step {name}] K={K:.3f} tau63={tau63} theta={theta}", flush=True)

    # ---- 闭环 (B1/B4) ----
    cl_res = {}
    for name, row_idx, power in (("wet", OP_WET, 332.85), ("dry", OP_DRY, 464.53)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens0 = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        exo_b = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, N, 1)
        base, *_ = integrate_lag(model0, res_s0, exo_b, h, Tm, rB, N, T_sens=T_sens0)
        base_main = base[0, :, 4].cpu().numpy()
        h, Tm, rB = init_state(model0, row, obs)
        T_sens = T_sens0
        u0 = float(row[V2])
        u, integ = u0, 0.0
        SP = float(obs[4]) + 2.0
        main_hist = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
                exo[0, 0, V2] = u
                out, h, Tm, rB, hm1, hm2, T_sens = integrate_lag(
                    model0, res_s0, exo, h, Tm, rB, 1, T_sens=T_sens)
                main = float(out[0, 0, 4])
                e = main - SP
                kp, ti = pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                main_hist[t] = main
        dC = main_hist - base_main
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        anch_model = np.array([norm[int(i) - 1] for i in ANCH_T])
        anch_delta = np.abs(anch_model - ANCH_Y)
        cl_res[name] = {"anchors": [round(x, 3) for x in anch_model],
                        "delta": [round(x, 3) for x in anch_delta],
                        "norm600": round(float(norm[599]), 3)}
        summ["cl"][name] = cl_res[name]
        print(f"[cl {name}] anchors={cl_res[name]['anchors']} Δ={cl_res[name]['delta']}", flush=True)

    # ---- 判定 ----
    dry_delta = cl_res["dry"]["delta"]
    B1 = bool(all(d <= 0.15 for d in dry_delta) and dry_delta[-1] <= 0.10)
    sd_ = step_res["dry"]
    B2 = bool(sd_["tau63"] is not None and 240 <= sd_["tau63"] * DT <= 900
              and sd_["theta"] * DT <= 300)
    B3 = bool(np.mean(rolls) <= 3.2)
    B4 = bool(0.8 <= cl_res["wet"]["norm600"] <= 1.2 and step_res["wet"]["main_final"] < 0)
    judge = {"B1": B1, "B2": B2, "B3": B3, "B4": B4,
             "rollout_mean": round(float(np.mean(rolls)), 3),
             "verdict": "PASS" if (B1 and B2 and B3 and B4) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fix2_lag_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== 修复② 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    t_axis = np.arange(N) * DT / 60.0
    ax = axes[0]
    for name, color in (("wet", "#8b008b"), ("dry", "#2e8b57")):
        row, obs = Ea[OP_WET if name == "wet" else OP_DRY], T_all[OP_WET if name == "wet" else OP_DRY]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens0 = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        exo_b = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, N, 1)
        base, *_ = integrate_lag(model0, res_s0, exo_b, h, Tm, rB, N, T_sens=T_sens0)
        h, Tm, rB = init_state(model0, row, obs)
        exo_s = exo_b.clone()
        exo_s[:, :, V2] += 0.05
        tr, *_ = integrate_lag(model0, res_s0, exo_s, h, Tm, rB, N, T_sens=T_sens0)
        d = tr[0].cpu().numpy() - base[0].cpu().numpy()
        ax.plot(t_axis, d[:, 4], lw=1.5, color=color, label=f"qlag {name} v2+5%→main")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("open-loop step (τ_sw=60s + τ_sens=15s)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("ΔT (°C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.3", ms=7, label="exp_099")
    for name, color in (("wet", "#c55a11"), ("dry", "#8b008b")):
        row, obs = Ea[OP_WET if name == "wet" else OP_DRY], T_all[OP_WET if name == "wet" else OP_DRY]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens0 = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        exo_b = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, N, 1)
        base, *_ = integrate_lag(model0, res_s0, exo_b, h, Tm, rB, N, T_sens=T_sens0)
        base_main = base[0, :, 4].cpu().numpy()
        h, Tm, rB = init_state(model0, row, obs)
        T_sens = T_sens0
        u0 = float(row[V2])
        u, integ = u0, 0.0
        SP = float(obs[4]) + 2.0
        power = 332.85 if name == "wet" else 464.53
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
                exo[0, 0, V2] = u
                out, h, Tm, rB, hm1, hm2, T_sens = integrate_lag(
                    model0, res_s0, exo, h, Tm, rB, 1, T_sens=T_sens)
                e = float(out[0, 0, 4]) - SP
                kp, ti = pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                mh[t] = float(out[0, 0, 4])
        dC = mh - base_main
        dC_ss = float(np.mean(dC[-60:]))
        ax.plot(t_axis, dC / dC_ss, lw=1.5, color=color, label=f"qlag {name} closed-loop")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.axhline(0.63, color="0.5", ls=":", lw=0.8)
    ax.set_title("closed-loop SP+2 normalized")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Fix② lag structure — verdict={judge['verdict']} "
                 f"(B1={B1} B2={B2} B3={B3} B4={B4})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig16_fix_lag.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig16_fix_lag.png", flush=True)


if __name__ == "__main__":
    main()
