#!/usr/bin/env python3
"""24_fix_gate.py: 修复② 残差通道门控 — 残差只在段0(两相)工作, 喷水通道(段1-2)还给灰盒

背景: 干态探针证明残差经状态反馈抵抗开环扰动 (τ63: e0 140s / qnal 50s / qslow 10s)。
门控设计: z_mask=[1,0,0] — 段0是灰盒物理缺陷所在(两相欠热), 段1-2是喷水因果通道。

预注册 (冻结 2026-08-17):
  Q1: 干态开环耦合阶跃 τ63 ≥ 120s (e0 参考 140s 的 85%)
  Q2: 干态闭环 (真实PI + 阀位速率限制0.0137/步) 收敛 norm600∈[0.8,1.2] ∧ tail_std≤0.05
  Q3: 湿态开环 τ63 ∈ [240,900]s 保持 (FIXC 的 320s 不丢)
  Q4: rollout 2seeds mean ≤ 4.5 (段0-only 精度预算; 历史 q0 曾 4.8-5.3)
  Q5 (报告): 湿态闭环 G1 复测 + τ_res/τ_sw/τ_sens 学得值
裁决: Q1∧Q2∧Q3∧Q4 = PASS
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

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
RATE = 0.0137
Z_MASK = (1.0, 0.0, 0.0)
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])


def train_gate(df, seed, fast=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model0 = r09.load_e0(0)
    mod = r22.QnaLagSlow().to(DEVICE)
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
        out, *_ = r22.integrate_slow(model0, mod, exo_t, h, Tm, rB, exo_t.shape[1],
                                     T_sens=T_sens0, z_mask=Z_MASK)
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
    torch.save(best_state, os.path.join(OUT, f"model_res_qgate_seed{seed}.pt"))
    tsw, tse, tre = mod.taus()
    print(f"[qgate s{seed}] {n_ep_done}ep val={best_va:.3f} τ_sw={tsw.item():.1f} "
          f"τ_sens={tse.item():.1f} τ_res={tre.item():.1f}", flush=True)
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
        mod, va = train_gate(df, sd)
        summ["train"][str(sd)] = {"val_mse": round(va, 4),
                                  "taus": [round(float(x.item()), 1) for x in mod.taus()]}

    mod0 = r22.QnaLagSlow().to(DEVICE)
    mod0.load_state_dict(torch.load(os.path.join(OUT, "model_res_qgate_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    mod0.eval()
    for p in mod0.parameters():
        p.requires_grad_(False)

    # rollout (Q4)
    rolls = []
    for sd in (0, 1):
        mod = r22.QnaLagSlow().to(DEVICE)
        mod.load_state_dict(torch.load(os.path.join(OUT, f"model_res_qgate_seed{sd}.pt"),
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
                out, h, Tm, rB, hm1, hm2, T_sens, z_lag = r22.integrate_slow(
                    model0, mod, exo_t, h, Tm, rB, 1, T_sens=T_sens, z_lag=z_lag,
                    z_mask=Z_MASK)
                preds[t] = out[0, 0].cpu().numpy()
        truths = T_all[START: START + t02.ROLL_STEPS]
        rmse = float(np.sqrt(np.mean((preds[:, 4] - truths[:, 4]) ** 2)))
        rolls.append(rmse)
        print(f"[qgate s{sd}] rollout={rmse:.3f}", flush=True)

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
        out, h, Tm, rB, hm1, hm2, T_sens, z_lag = r22.integrate_slow(
            model0, mod0, exo, h, Tm, rB, 1, T_sens=T_sens, z_lag=z_lag, z_mask=Z_MASK)
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

    # 开环阶跃 (Q1/Q3)
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

    # 闭环 (Q2/Q5)
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

    Q1 = bool(summ["step"]["dry"]["tau63_s"] is not None
              and summ["step"]["dry"]["tau63_s"] >= 120)
    Q2 = bool(0.8 <= summ["loop"]["dry"]["norm600"] <= 1.2
              and summ["loop"]["dry"]["tail_std"] <= 0.05)
    Q3 = bool(summ["step"]["wet"]["tau63_s"] is not None
              and 240 <= summ["step"]["wet"]["tau63_s"] <= 900)
    Q4 = bool(np.mean(rolls) <= 4.5)
    judge = {"Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4,
             "rollout_mean": round(float(np.mean(rolls)), 3),
             "verdict": "PASS" if (Q1 and Q2 and Q3 and Q4) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fix_gate_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    print("=== 修复② 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
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
    ax.set_title("closed loop: gated residual (z=[1,0,0])")
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
    fig.suptitle(f"Fix② gated residual — verdict={judge['verdict']} "
                 f"(Q1={Q1} Q2={Q2} Q3={Q3} Q4={Q4})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig25_gate.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig25_gate.png", flush=True)


if __name__ == "__main__":
    main()
