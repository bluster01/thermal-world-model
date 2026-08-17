#!/usr/bin/env python3
"""15_fix_learnlag.py: FIX3 = A(残差全动作屏蔽qna) + 可学习滞后(τ_sw/τ_sens) + B(分状态W子模型)

设计稿 FIX3_DESIGN.md (预注册冻结):
  A1': 湿+干 耦合阶跃 main 末值 < 0 (动作通道终极免疫)
  A2': rollout 2seeds mean ≤ 3.2
  A3': 湿态耦合闭环 norm600∈[0.8,1.2] 且 tail_std≤0.05
  B1': 干态耦合闭环 norm600∈[0.8,1.2] 且 tail_std≤0.05
  B2'(审计): τ_sw∈[10,600]s, τ_sens∈[5,120]s
τ 参数化: τ = prior×softplus(raw), prior 60s/15s, 与残差联合训练。
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
TAU_SW0, TAU_SENS0 = 60.0, 15.0
RAW0 = float(np.log(np.e - 1.0))  # softplus⁻¹(1) ≈ 0.5413
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


class QnaLag(nn.Module):
    """残差 MLP (10特征 qna) + 可学习滞后 τ_sw/τ_sens."""

    def __init__(self):
        super().__init__()
        self.mlp = r09.ResMLP(10, r09.Q_SCALE)
        self.raw_sw = nn.Parameter(torch.tensor(RAW0, dtype=torch.float32))
        self.raw_sens = nn.Parameter(torch.tensor(RAW0, dtype=torch.float32))

    def tau(self):
        return TAU_SW0 * F.softplus(self.raw_sw), TAU_SENS0 * F.softplus(self.raw_sens)


def integrate_learn(model0, mod, exo, h, Tm, rB, steps, T_sens=None):
    """qna残差(no_act) + 可学习喷水/传感器滞后。返回 (out, h, Tm, rB, hm1, hm2, T_sens)。"""
    tau_sw, tau_sens = mod.tau()
    a_sw = DT / tau_sw
    a_sens = DT / tau_sens
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
        Dsw1_lag = Dsw1_lag + a_sw * (Dsw1_new - Dsw1_lag)
        Dsw2_lag = Dsw2_lag + a_sw * (Dsw2_new - Dsw2_lag)
        for _ in range(t02.N_SUB):
            ts = t02.T_of_ph(torch.stack([p0[:, t], p1[:, t], p_out[:, t]]), h)
            Q = UA * (Tm - ts)
            feats = r09.build_feats(ts, Tm, pm[:, t], D[:, t], uB[:, t], rB,
                                    v1[:, t], v2[:, t], Wt, None, no_act=True)
            z = mod.mlp(feats).permute(1, 0)
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
        T_raw = t02.T_of_ph(p, hh)
        if T_sens is None:
            T_sens = T_raw
        else:
            T_sens = T_sens + a_sens * (T_raw - T_sens)
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


def train_learn(df, seed, fast=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model0 = r09.load_e0(0)
    mod = QnaLag().to(DEVICE)
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
        out, *_ = integrate_learn(model0, mod, exo_t, h, Tm, rB, exo_t.shape[1], T_sens=T_sens0)
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
    torch.save(best_state, os.path.join(OUT, f"model_res_qnal_seed{seed}.pt"))
    tsw, tse = mod.tau()
    print(f"[qnal s{seed}] {n_ep_done}ep val={best_va:.3f} "
          f"τ_sw={tsw.item():.1f}s τ_sens={tse.item():.1f}s", flush=True)
    return mod, best_va


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

    summ = {"train": {}, "rollout": {}, "w_fit": {}, "step": {}, "cl": {}, "judge": {}}
    rolls = []
    for sd in (0, 1):
        mod, va = train_learn(df, sd)
        summ["train"][str(sd)] = {"val_mse": round(va, 4),
                                  "tau_sw": round(float(mod.tau()[0].item()), 1),
                                  "tau_sens": round(float(mod.tau()[1].item()), 1)}
    mod0 = QnaLag().to(DEVICE)
    mod0.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnal_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    mod0.eval()
    for p in mod0.parameters():
        p.requires_grad_(False)
    tsw0, tse0 = mod0.tau()
    print(f"[qnal s0 loaded] τ_sw={tsw0.item():.1f} τ_sens={tse0.item():.1f}", flush=True)

    # ---- rollout (2 seeds) ----
    for sd in (0, 1):
        mod = QnaLag().to(DEVICE)
        mod.load_state_dict(torch.load(os.path.join(OUT, f"model_res_qnal_seed{sd}.pt"),
                                       map_location=DEVICE, weights_only=True))
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
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
                out, h, Tm, rB, hm1, hm2, T_sens = integrate_learn(
                    model0, mod, exo_t, h, Tm, rB, 1, T_sens=T_sens)
                preds[t] = out[0, 0].cpu().numpy()
        truths = T_all[START: START + t02.ROLL_STEPS]
        rmse_main = float(np.sqrt(np.mean((preds[:, 4] - truths[:, 4]) ** 2)))
        sr = r09.strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
        rolls.append(rmse_main)
        summ["rollout"][str(sd)] = {"rmse_main": round(rmse_main, 4),
                                    "dry": sr["dry"]["rmse_main"],
                                    "wet": sr["wet"]["rmse_main"]}
        print(f"[qnal s{sd}] rollout={rmse_main:.3f} dry={sr['dry']['rmse_main']:.2f} "
              f"wet={sr['wet']['rmse_main']:.2f}", flush=True)

    # ---- W 分状态拟合 (B) ----
    pm_seg = Ea[START: START + t02.ROLL_STEPS, 2]
    k_w_state = {}
    for state, mask in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][mask]
        v1s, v2s, Ws = sub[:, 5], sub[:, 6], sub[:, 8]
        A = np.stack([v1s, v2s, np.ones_like(v1s)], 1)
        coef, res_, _, _ = np.linalg.lstsq(A, Ws, rcond=None)
        Wmean = float(np.mean(Ws))
        r2 = float(1 - res_.sum() / np.sum((Ws - Ws.mean()) ** 2))
        k_raw = coef[1] / Wmean
        k_clip = float(np.clip(k_raw, 0.5, 4.0))
        k_w_state[state] = {"slope": round(float(coef[1]), 3), "R2": round(r2, 3),
                            "n": int(mask.sum()), "k_w_raw": round(float(k_raw), 3),
                            "k_w": k_clip}
        summ["w_fit"][state] = k_w_state[state]
        print(f"[wfit {state}] slope={coef[1]:.3f} R²={r2:.3f} n={mask.sum()} "
              f"k_w={k_clip:.3f}", flush=True)

    def fwd_one(row, h, Tm, rB, T_sens, u, W_val):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = u
        exo[0, 0, 8] = W_val
        out, h, Tm, rB, hm1, hm2, T_sens = integrate_learn(
            model0, mod0, exo, h, Tm, rB, 1, T_sens=T_sens)
        return float(out[0, 0, 4]), h, Tm, rB, T_sens

    # 基线 (恒定 u0/W0)
    row, obs = Ea[OP_WET], T_all[OP_WET]
    h, Tm, rB = init_state(model0, row, obs)
    T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
    u0w, W0w = float(row[V2]), float(row[8])
    base_wet = np.zeros(N)
    with torch.no_grad():
        for t in range(N):
            base_wet[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, u0w, W0w)
    row, obs = Ea[OP_DRY], T_all[OP_DRY]
    h, Tm, rB = init_state(model0, row, obs)
    T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
    u0d, W0d = float(row[V2]), float(row[8])
    base_dry = np.zeros(N)
    with torch.no_grad():
        for t in range(N):
            base_dry[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, u0d, W0d)

    # ---- 耦合阶跃 (A1') ----
    for name, row_idx, base, kws in (("wet", OP_WET, base_wet, "wet"),
                                     ("dry", OP_DRY, base_dry, "dry")):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[kws]["k_w"]
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens,
                                                   u0 + 0.05, W0 * (1 + kw * 0.05))
        d = dT - base
        summ["step"][name] = {"K": round(float(np.mean(d[-60:])), 3),
                              "main_final": round(float(d[-1]), 3)}
        print(f"[step {name}] K={summ['step'][name]['K']} "
              f"final={summ['step'][name]['main_final']}", flush=True)

    # ---- 耦合闭环 (A3'/B1') ----
    for name, row_idx, base, kws, power in (("wet", OP_WET, base_wet, "wet", 332.85),
                                            ("dry", OP_DRY, base_dry, "dry", 464.53)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[kws]["k_w"]
        SP = float(obs[4]) + 2.0
        u, integ = u0, 0.0
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, u,
                                                   W0 * (1 + kw * (u - u0)))
                e = mh[t] - SP
                kp, ti = pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        anch = [float(norm[int(i) - 1]) for i in ANCH_T]
        summ["cl"][name] = {"anchors": [round(x, 3) for x in anch],
                            "delta": [round(abs(x - y), 3) for x, y in zip(anch, ANCH_Y)],
                            "norm600": round(float(norm[599]), 3),
                            "tail_std": round(float(np.std(norm[-120:])), 4)}
        print(f"[cl {name}] anchors={summ['cl'][name]['anchors']} "
              f"norm600={summ['cl'][name]['norm600']} tail_std={summ['cl'][name]['tail_std']}",
              flush=True)

    # ---- 判定 ----
    A1p = bool(summ["step"]["wet"]["main_final"] < 0 and summ["step"]["dry"]["main_final"] < 0)
    A2p = bool(np.mean(rolls) <= 3.2)
    A3p = bool(0.8 <= summ["cl"]["wet"]["norm600"] <= 1.2 and summ["cl"]["wet"]["tail_std"] <= 0.05)
    B1p = bool(0.8 <= summ["cl"]["dry"]["norm600"] <= 1.2 and summ["cl"]["dry"]["tail_std"] <= 0.05)
    tsw = float(tsw0.item())
    tse = float(tse0.item())
    B2_ok = bool(10 <= tsw <= 600 and 5 <= tse <= 120)
    judge = {"A1p": A1p, "A2p": A2p, "A3p": A3p, "B1p": B1p,
             "B2_audit": {"tau_sw": round(tsw, 1), "tau_sens": round(tse, 1),
                          "in_bounds": B2_ok},
             "rollout_mean": round(float(np.mean(rolls)), 3),
             "verdict": "PASS" if (A1p and A2p and A3p and B1p) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fix3_learnlag_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== FIX3 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    t_axis = np.arange(N) * DT / 60.0
    ax = axes[0]
    for name, color in (("wet", "#8b008b"), ("dry", "#2e8b57")):
        row_idx = OP_WET if name == "wet" else OP_DRY
        base = base_wet if name == "wet" else base_dry
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[name]["k_w"]
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens,
                                                   u0 + 0.05, W0 * (1 + kw * 0.05))
        ax.plot(t_axis, dT - base, lw=1.5, color=color,
                label=f"{name} coupled step (k_w={kw:.2f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"open-loop coupled step — learned tau: sw={tsw:.0f}s sens={tse:.0f}s")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("ΔT (°C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.3", ms=7, label="exp_099")
    for name, color in (("wet", "#c55a11"), ("dry", "#8b008b")):
        row_idx = OP_WET if name == "wet" else OP_DRY
        base = base_wet if name == "wet" else base_dry
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[name]["k_w"]
        SP = float(obs[4]) + 2.0
        power = 332.85 if name == "wet" else 464.53
        u, integ = u0, 0.0
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, u,
                                                   W0 * (1 + kw * (u - u0)))
                e = mh[t] - SP
                kp, ti = pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        ax.plot(t_axis, dC / dC_ss, lw=1.4, color=color,
                label=f"{name} closed-loop coupled")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.axhline(0.63, color="0.5", ls=":", lw=0.8)
    ax.set_title("coupled closed-loop SP+2")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"FIX3 learnable-lag — verdict={judge['verdict']} "
                 f"(A1={A1p} A2={A2p} A3={A3p} B1={B1p})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig18_learnlag.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig18_learnlag.png", flush=True)


if __name__ == "__main__":
    main()
