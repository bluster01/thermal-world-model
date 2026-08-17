#!/usr/bin/env python3
"""27_fix_evap_residual.py: qnav — qnaw残差(no_v12) 上 e0-evap 底座

背景: FIXB 物理侧过慢 (湿态 τ63=1030s), 残差的'抵抗'从破坏者变回调节器 —
物理慢+残差快=中间节奏, 这是物理+残差混合设计的本来意图。
结构: e0-evap 冻结 + ResMLP(11特征, no_v12) 残差注入 Tm/h 方程 (与 qnaw 同注入点),
蒸发输出映射保持 (B1 的两相修正不被残差覆盖 — 残差不进输出层)。

预注册 (冻结 2026-08-17):
  E1: 湿态开环耦合阶跃 τ63∈[240,900]s (从 1030s 回落)
  E2: rollout 2seeds mean ≤ 3.5 (qnaw 旧底座 2.57, 预算 3.5)
  E3: 湿态闭环 (真实PI + 0.0137/步速率限制) 收敛 norm600∈[0.8,1.2] ∧ tail_std≤0.05
  E4: 干态开环 K<0 (符号保持)
  E5 (审计): 湿态 sh1_out 首步偏差 ≤8°C (B1 的 3.07°C 增益必须扛住残差)
裁决: E1∧E2∧E3∧E4∧E5 = PASS
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
r26 = _imp("26_fix_evap.py", "r26")
import numpy as np
import torch
import torch.nn as nn

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
TRAIN_N = t02.TRAIN_N
VAL_N = t02.VAL_N
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
RATE = 0.0137
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])
T_of_ph = t02.T_of_ph
tsat_poly = t02.tsat_poly


def integrate_evap_res(model0, res, exo, h, Tm, rB, m1, m2, steps):
    """e0-evap 动力学 + qnaw 残差注入 (Tm/h 方程, 同 qnaw 注入点)。"""
    M = model0.tri("M")[:, None]
    UA = model0.tri("UA")[:, None]
    Cm = model0.tri("Cm")[:, None]
    tauB = model0.val("tauB")
    tau_evap = model0.val("tau_evap")
    aW1 = model0.val("aW1")
    aW2 = model0.val("aW2")
    m_dry0 = model0.val("m_dry0")
    D, uB, pm, Tm_sep, Tfw, v1, v2, p_out, W = [exo[:, :, j] for j in range(9)]
    h_sw = t02.hliq_of_T(Tfw)
    p0 = pm + (p_out - pm) / 3.0
    p1 = pm + 2.0 * (p_out - pm) / 3.0
    hsep = t02.h_sep_of(pm, Tm_sep)
    out_list = []
    th1_0, th2_0 = model0.th_of(pm[:, 0])
    s_den0 = th1_0 * v1[:, 0] + th2_0 * v2[:, 0] + 1e-6
    W0 = W[:, 0].clamp(min=0.0)
    Dsw1 = t02.KAPPA * W0 * (th1_0 * v1[:, 0]) / s_den0
    Dsw2 = t02.KAPPA * W0 * (th2_0 * v2[:, 0]) / s_den0
    hm1 = (D[:, 0] * h[0] + Dsw1 * h_sw[:, 0]) / (D[:, 0] + Dsw1 + 1e-6)
    hm2 = (D[:, 0] * h[1] + Dsw2 * h_sw[:, 0]) / (D[:, 0] + Dsw2 + 1e-6)
    for t in range(steps):
        k_t = model0.k_of(pm[:, t])
        th1_t, th2_t = model0.th_of(pm[:, t])
        s_den = th1_t * v1[:, t] + th2_t * v2[:, t] + 1e-6
        Wt = W[:, t].clamp(min=0.0)
        for _ in range(t02.N_SUB):
            ts = T_of_ph(torch.stack([p0[:, t], p1[:, t], p_out[:, t]]), h)
            dry1 = torch.sigmoid(3.0 * (m_dry0 - m1) / m_dry0)
            dry2 = torch.sigmoid(3.0 * (m_dry0 - m2) / m_dry0)
            tsat0 = tsat_poly(p0[:, t])
            tsat1 = tsat_poly(p1[:, t])
            q_w1 = aW1 * (Tm[0] - tsat0) * (1.0 - dry1)
            q_w2 = aW2 * (Tm[1] - tsat1) * (1.0 - dry2)
            Q = UA * (Tm - ts)
            feats = r09.build_feats(ts, Tm, pm[:, t], D[:, t], uB[:, t], rB,
                                    v1[:, t], v2[:, t], Wt, None, no_v12=True)
            z = res(feats).permute(1, 0)
            Tm0_in = (k_t[0] * rB / 3600.0 + UA[0] * ts[0] - q_w1 + z[0]) / Cm[0]
            Tm1_in = (k_t[1] * rB / 3600.0 + UA[1] * ts[1] - q_w2 + z[1]) / Cm[1]
            Tm2_in = (k_t[2] * rB / 3600.0 + UA[2] * ts[2] + z[2]) / Cm[2]
            Tm_in = torch.stack([Tm0_in, Tm1_in, Tm2_in])
            Tm = (Tm + t02.DT_SUB * Tm_in) / (1.0 + t02.DT_SUB * UA / Cm)
            h_in1 = hm1 + q_w1 / (D[:, t] + 1e-6)
            h_in2 = hm2 + q_w2 / (D[:, t] + 1e-6)
            hin = torch.stack([hsep[:, t], h_in1, h_in2])
            h = (h + t02.DT_SUB * (D[:, t][None, :] * hin + Q + z) / M) / (
                1.0 + t02.DT_SUB * D[:, t][None, :] / M)
            h = t02._ste_clamp(h, t02.H_LO, t02.H_HI)
            Dsw1 = t02.KAPPA * Wt * (th1_t * v1[:, t]) / s_den
            Dsw2 = t02.KAPPA * Wt * (th2_t * v2[:, t]) / s_den
            hm1 = (D[:, t] * h[0] + Dsw1 * h_sw[:, t]) / (D[:, t] + Dsw1 + 1e-6)
            hm2 = (D[:, t] * h[1] + Dsw2 * h_sw[:, t]) / (D[:, t] + Dsw2 + 1e-6)
            m1 = (m1 + t02.DT_SUB * (Dsw1 - m1 / tau_evap)).clamp(min=0.0)
            m2 = (m2 + t02.DT_SUB * (Dsw2 - m2 / tau_evap)).clamp(min=0.0)
            rB = rB + t02.DT_SUB * (uB[:, t] - rB) / tauB
        dry1 = torch.sigmoid(3.0 * (m_dry0 - m1) / m_dry0)
        dry2 = torch.sigmoid(3.0 * (m_dry0 - m2) / m_dry0)
        tsat0 = tsat_poly(p0[:, t])
        tsat1 = tsat_poly(p1[:, t])
        q_w1o = aW1 * (Tm[0] - tsat0) * (1.0 - dry1)
        q_w2o = aW2 * (Tm[1] - tsat1) * (1.0 - dry2)
        h_o1 = hm1 + q_w1o / (D[:, t] + 1e-6)
        h_o2 = hm2 + q_w2o / (D[:, t] + 1e-6)
        p = torch.stack([p0[:, t], p0[:, t], p1[:, t], p1[:, t], p_out[:, t]])
        hh = torch.stack([h[0], h_o1, h[1], h_o2, h[2]])
        T_all5 = T_of_ph(p, hh)
        T_out = torch.stack([T_all5[0],
                             tsat0 + dry1 * (T_all5[1] - tsat0),
                             T_all5[2],
                             tsat1 + dry2 * (T_all5[3] - tsat1),
                             T_all5[4]])
        out_list.append(T_out)
    out = torch.stack(out_list, dim=2).permute(1, 2, 0)
    return out, h, Tm, rB, m1, m2


def train_qnav(df, seed, fast=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    warm0 = torch.load(os.path.join(OUT, "model_e0_seed0.pt"), map_location=DEVICE,
                       weights_only=True)
    model0 = r26.E0Evap(warm0).to(DEVICE)
    model0.load_state_dict(torch.load(os.path.join(OUT, "model_e0_evap_seed0.pt"),
                                      map_location=DEVICE, weights_only=True))
    for p in model0.parameters():
        p.requires_grad_(False)
    model0.eval()
    res = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
    opt = torch.optim.Adam(res.parameters(), lr=1e-3)
    w5 = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=DEVICE)
    tr_s = 25 if fast else 5
    va_s = 100 if fast else 20
    Xtr, Ytr, Itr, Itr_T = t02.e0_build_windows(df, 0, TRAIN_N, tr_s)
    Xva, Yva, Iva, Iva_T = t02.e0_build_windows(df, TRAIN_N, TRAIN_N + VAL_N, va_s)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    Itr_t = torch.from_numpy(Itr).to(DEVICE)
    ItrT_t = torch.from_numpy(Itr_T).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    Yva_t = torch.from_numpy(Yva).to(DEVICE)
    Iva_t = torch.from_numpy(Iva).to(DEVICE)
    IvaT_t = torch.from_numpy(Iva_T).to(DEVICE)

    def fwd(exo_t, init_t, obs_t):
        h, Tm, rB, m1, m2 = r26.init_states_evap(model0, init_t, obs_t)
        out, *_ = integrate_evap_res(model0, res, exo_t, h, Tm, rB, m1, m2,
                                     exo_t.shape[1])
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
    torch.save(best_state, os.path.join(OUT, f"model_res_qnav_seed{seed}.pt"))
    print(f"[qnav s{seed}] {n_ep_done}ep val={best_va:.3f}", flush=True)
    return res, best_va


def main():
    df = r09.load_e0_df()
    warm0 = torch.load(os.path.join(OUT, "model_e0_seed0.pt"), map_location=DEVICE,
                       weights_only=True)
    model0 = r26.E0Evap(warm0).to(DEVICE)
    model0.load_state_dict(torch.load(os.path.join(OUT, "model_e0_evap_seed0.pt"),
                                      map_location=DEVICE, weights_only=True))
    for p in model0.parameters():
        p.requires_grad_(False)
    model0.eval()

    summ = {"train": {}, "judge": {}}
    ckpts = [os.path.join(OUT, f"model_res_qnav_seed{sd}.pt") for sd in (0, 1)]
    if all(os.path.exists(c) for c in ckpts):
        print("[skip] 2 checkpoints exist", flush=True)
    else:
        for sd in (0, 1):
            _, va = train_qnav(df, sd)
            summ["train"][str(sd)] = {"val_mse": round(va, 4)}

    res0 = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
    res0.load_state_dict(torch.load(ckpts[0], map_location=DEVICE, weights_only=True))
    res0.eval()
    for p in res0.parameters():
        p.requires_grad_(False)

    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]

    # E5: 湿态 sh1_out 首步偏差 (B1 保持检查)
    wet_idx = np.where((pm_all[START: START + t02.ROLL_STEPS] <= P_CRIT))[0] + START
    wet_idx = wet_idx[wet_idx + 1 < len(T_all)]
    errs = []
    with torch.no_grad():
        for r in wet_idx:
            row = Ea[r]
            obs = T_all[r]
            exo_t = torch.tensor(Ea[r: r + 1], device=DEVICE)[None, :, :]
            h, Tm, rB, m1, m2 = r26.init_states_evap(
                model0, torch.tensor(row, device=DEVICE)[None, :],
                torch.tensor(obs, device=DEVICE)[None, :])
            out, *_ = integrate_evap_res(model0, res0, exo_t, h, Tm, rB, m1, m2, 1)
            errs.append(float(out[0, 0, 1] - T_all[r + 1, 1]))
    bias = float(np.mean(np.array(errs)))
    print(f"[E5] wet sh1_out first-step bias={bias:.2f}°C (B1=3.07, gate 8)", flush=True)

    # E2: rollout (2 seeds)
    rolls = []
    for sd in (0, 1):
        res = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
        res.load_state_dict(torch.load(ckpts[sd], map_location=DEVICE, weights_only=True))
        res.eval()
        for p in res.parameters():
            p.requires_grad_(False)
        preds = np.empty((t02.ROLL_STEPS, 5), dtype=np.float32)
        with torch.no_grad():
            for t in range(t02.ROLL_STEPS):
                row = Ea[START + t]
                exo_t = torch.tensor(row, device=DEVICE)[None, None, :]
                if t == 0:
                    obs = T_all[START]
                    h, Tm, rB, m1, m2 = r26.init_states_evap(
                        model0, torch.tensor(row, device=DEVICE)[None, :],
                        torch.tensor(obs, device=DEVICE)[None, :])
                out, h, Tm, rB, m1, m2 = integrate_evap_res(
                    model0, res, exo_t, h, Tm, rB, m1, m2, 1)
                preds[t] = out[0, 0].cpu().numpy()
        truths = T_all[START: START + t02.ROLL_STEPS]
        rmse = float(np.sqrt(np.mean((preds[:, 4] - truths[:, 4]) ** 2)))
        rolls.append(rmse)
        print(f"[qnav s{sd}] rollout={rmse:.3f}", flush=True)

    # k_w
    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    def run_step(row_idx, d_v2, W_mul):
        row, obs = Ea[row_idx], T_all[row_idx]
        exo = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, N, 1).clone()
        exo[0, :, V2] = exo[0, :, V2] + d_v2
        exo[0, :, 8] = exo[0, :, 8] * W_mul
        h, Tm, rB, m1, m2 = r26.init_states_evap(
            model0, torch.tensor(row, device=DEVICE)[None, :],
            torch.tensor(obs, device=DEVICE)[None, :])
        out, *_ = integrate_evap_res(model0, res0, exo, h, Tm, rB, m1, m2, N)
        return out[0, :, 4].cpu().numpy()

    for name, row_idx, state in (("wet", OP_WET, "wet"), ("dry", OP_DRY, "dry")):
        kw = k_w_state[state]
        base = run_step(row_idx, 0.0, 1.0)
        step = run_step(row_idx, 0.05, 1.0 + kw * 0.05)
        d = step - base
        K = float(np.mean(d[-60:]))
        idx = np.where(d <= 0.63 * K)[0] if K < 0 else np.where(d >= 0.63 * K)[0]
        tau63 = int(idx[0]) * DT if len(idx) else None
        summ[f"step_{name}"] = {"K": round(K, 3), "tau63_s": tau63}
        print(f"[step {name}] K={K:.3f} τ63={tau63}s", flush=True)

    # E3: 湿态闭环
    for name, row_idx, state, power in (("wet", OP_WET, "wet", 332.85),
                                        ("dry", OP_DRY, "dry", 464.53)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB, m1, m2 = r26.init_states_evap(
            model0, torch.tensor(row, device=DEVICE)[None, :],
            torch.tensor(obs, device=DEVICE)[None, :])
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[state]
        SP = float(obs[4]) + 2.0
        u, integ, v = u0, 0.0, u0
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
                exo[0, 0, V2] = v
                exo[0, 0, 8] = W0 * (1 + kw * (v - u0))
                out, h, Tm, rB, m1, m2 = integrate_evap_res(
                    model0, res0, exo, h, Tm, rB, m1, m2, 1)
                mh[t] = float(out[0, 0, 4])
                e = mh[t] - SP
                kp, ti = r22_pi(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                v = float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
        base_row, base_obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB, m1, m2 = r26.init_states_evap(
            model0, torch.tensor(base_row, device=DEVICE)[None, :],
            torch.tensor(base_obs, device=DEVICE)[None, :])
        b = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(base_row, device=DEVICE)[None, None, :].clone()
                out, h, Tm, rB, m1, m2 = integrate_evap_res(
                    model0, res0, exo, h, Tm, rB, m1, m2, 1)
                b[t] = float(out[0, 0, 4])
        dC = mh - b
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        anch = [float(norm[int(i) - 1]) for i in ANCH_T]
        deltas = [round(abs(a - b2), 3) for a, b2 in zip(anch, ANCH_Y)] if name == "wet" else None
        summ[f"loop_{name}"] = {"anchors": [round(x, 3) for x in anch],
                                "norm600": round(float(norm[599]), 3),
                                "tail_std": round(float(np.std(norm[-120:])), 4),
                                "G1_deltas": deltas}
        print(f"[loop {name}] anchors={summ['loop_' + name]['anchors']} "
              f"norm600={summ['loop_' + name]['norm600']} "
              f"tail_std={summ['loop_' + name]['tail_std']} Δ={deltas}", flush=True)

    E1 = bool(summ["step_wet"]["tau63_s"] is not None
              and 240 <= summ["step_wet"]["tau63_s"] <= 900)
    E2 = bool(np.mean(rolls) <= 3.5)
    E3 = bool(0.8 <= summ["loop_wet"]["norm600"] <= 1.2
              and summ["loop_wet"]["tail_std"] <= 0.05)
    E4 = bool(summ["step_dry"]["K"] < 0)
    E5 = bool(abs(bias) <= 8.0)
    judge = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5,
             "rollout_mean": round(float(np.mean(rolls)), 3),
             "verdict": "PASS" if (E1 and E2 and E3 and E4 and E5) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fixb_qnav_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    print("=== qnav 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)


def r22_pi(dev, power):
    FX44_X = np.array([-12, -10, -8, -5, -3, 3, 5, 8, 10, 12])
    FX44_Y = np.array([0.6, 0.6, 0.8, 1.0, 1.2, 1.2, 1.0, 0.8, 0.6, 0.6])
    FX45_Y = np.array([800, 650, 550, 450, 350, 350, 450, 550, 650, 800])
    FX49_X = np.array([150, 200, 300, 400, 500, 600])
    FX49_Y = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.5])
    FX50_Y = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.6])
    kp = float(np.interp(abs(dev), FX44_X, FX44_Y)) * float(np.interp(power, FX49_X, FX49_Y))
    ti = float(np.interp(abs(dev), FX44_X, FX45_Y)) * float(np.interp(power, FX49_X, FX50_Y))
    return kp, ti


if __name__ == "__main__":
    main()
