#!/usr/bin/env python3
"""12_fix_action.py: 修复① 残差动作通道保护 (qna = qh 去掉 v1/v2/W 特征)

背景: Step⑤+11b 发现 qh 残差污染动作通道 (湿态 v2+5% → main +0.073, 符号翻转;
e0 无残差为 −0.047)。修复: 残差 MLP 输入去掉动作特征 v1/v2/W (13→10), 重训 3 seeds。
灰盒本体不动, 硬守恒/两相 clamp 保留。

预注册判定 (冻结 2026-08-17):
  A1: 湿态(row40161)与干态(row40437) v2+5% 开环 6000s 后 main ΔT 均 < 0 —— 符号修复
  A2: qna 3seeds rollout main mean ≤ 3.0 —— 精度不显著回退 (qh 2.816)
  A3: 闭环(真实POU107 PI)湿态 SP+2°C 的 norm(600s) ∈ [0.8,1.2] —— 无失控能收敛
产物: out/fix_action_summary.json, out/figs/fig15_fix_action.png
"""
import importlib.util
import json
import os
import time

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


def run_const(model0, res, row, h, Tm, rB, steps, d_v2=0.0, no_act=True):
    exo = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, steps, 1)
    exo[:, :, V2] += d_v2
    with torch.no_grad():
        out, *_ = r09.integrate_res(model0, res, exo, h, Tm, rB, steps, None, "qh",
                                    no_act=no_act)
    return out[0].cpu().numpy()


def run_closed_loop(model0, res, row, h, Tm, rB, steps, SP, power, no_act=True):
    u0 = float(row[V2])
    u, integ = u0, 0.0
    main_hist = np.zeros(steps)
    with torch.no_grad():
        for t in range(steps):
            exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
            exo[0, 0, V2] = u
            out, h, Tm, rB, hm1, hm2 = r09.integrate_res(
                model0, res, exo, h, Tm, rB, 1, None, "qh", no_act=no_act)
            main = float(out[0, 0, 4])
            e = main - SP
            kp, ti = pi_params(-e, power)
            integ += e * DT
            u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
            main_hist[t] = main
    return main_hist


def main():
    df = r09.load_e0_df()
    model0 = r09.load_e0(0)
    mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    pm_roll = df["分离器出口压力"].to_numpy(np.float32)[START: START + t02.ROLL_STEPS]

    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)

    summ = {"train": {}, "rollout": {}, "strat": {}, "windowed": {}, "step_test": {},
            "judge": {}}
    recs = []

    # ---- 训练 3 seeds ----
    for sd in (0, 1, 2):
        res, best_va, ep = r09.train_res(df, sd, "qna", "qh", False, False, out=3, no_act=True)
        res.eval()
        summ["train"][str(sd)] = {"val_mse": round(best_va, 4), "ep": ep}
        with torch.no_grad():
            r, preds, truths = r09.rollout_res(model0, res, df, START, t02.ROLL_STEPS,
                                               "qh", False, no_act=True)
        rec = {"seed": sd, "rollout": {k: (round(x, 4) if isinstance(x, float) else x)
                                       for k, x in r.items()}}
        rec["strat"] = r09.strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
        arrs = r09.windowed_arrays_res(df, model0, res, "qh", False, no_act=True)
        rec["windowed"] = r09.layer_agg(*arrs)
        recs.append(rec)
        summ["rollout"][str(sd)] = rec["rollout"]
        summ["strat"][str(sd)] = rec["strat"]
        summ["windowed"][str(sd)] = rec["windowed"]
        print(f"[qna s{sd}] rollout={r['rmse_main']:.3f} "
              f"dry={rec['strat']['dry']['rmse_main']:.2f} "
              f"wet={rec['strat']['wet']['rmse_main']:.2f} "
              f"win60dry={rec['windowed']['dry']['win60_rmse_main']:.2f}", flush=True)
        if sd == 0:
            res_s0 = res

    # ---- 符号测试 (A1) ----
    step_signs = {}
    for name, row_idx, power in (("wet", OP_WET, 332.85), ("dry", OP_DRY, 464.53)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        base = run_const(model0, res_s0, row, h, Tm, rB, N, no_act=True)
        h, Tm, rB = init_state(model0, row, obs)
        tr = run_const(model0, res_s0, row, h, Tm, rB, N, d_v2=0.05, no_act=True)
        d = tr - base
        step_signs[name] = {"main_final": round(float(d[-1, 4]), 3),
                            "main_60s": round(float(d[5, 4]), 3),
                            "sh1_in_final": round(float(d[-1, 0]), 3),
                            "sh2_in_final": round(float(d[-1, 2]), 3)}
        print(f"[step {name}] main_final={step_signs[name]['main_final']}", flush=True)
    summ["step_test"] = step_signs

    # ---- 闭环 (A3) ----
    row, obs = Ea[OP_WET], T_all[OP_WET]
    h, Tm, rB = init_state(model0, row, obs)
    base = run_const(model0, res_s0, row, h, Tm, rB, N, no_act=True)
    h, Tm, rB = init_state(model0, row, obs)
    cl = run_closed_loop(model0, res_s0, row, h, Tm, rB, N,
                         float(T_all[OP_WET, 4]) + 2.0, 332.85, no_act=True)
    dC = cl - base[:, 4]
    dC_ss = float(np.mean(dC[-60:]))
    norm600 = float(dC[599] / dC_ss)
    summ["cl_wet"] = {"dC_ss": round(dC_ss, 2), "norm_600s": round(norm600, 3)}
    print(f"[cl wet] ss={dC_ss:.2f} norm600={norm600:.3f}", flush=True)

    # ---- 判定 ----
    qna_roll = [rec["rollout"]["rmse_main"] for rec in recs]
    A1 = bool(step_signs["wet"]["main_final"] < 0 and step_signs["dry"]["main_final"] < 0)
    A2 = bool(np.mean(qna_roll) <= 3.0)
    A3 = bool(0.8 <= norm600 <= 1.2)
    judge = {"A1": A1, "A2": A2, "A3": A3,
             "rollout_mean": round(float(np.mean(qna_roll)), 3),
             "rollout_vals": [round(x, 3) for x in qna_roll],
             "verdict": "PASS" if (A1 and A2 and A3) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fix_action_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== 修复① 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t_axis = np.arange(N) * DT / 60.0
    # (a) 阶跃响应对比 (e0 历史数字 + qh + qna)
    ax = axes[0]
    for name, row_idx, color in (("wet", OP_WET, "#8b008b"), ("dry", OP_DRY, "#2e8b57")):
        row, obs = Ea[row_idx], T_all[row_idx]
        res_qh = r09.ResMLP(13, r09.Q_SCALE).to(DEVICE)
        res_qh.load_state_dict(torch.load(os.path.join(OUT, "model_res_qh_seed0.pt"),
                                          map_location=DEVICE, weights_only=True))
        res_qh.eval()
        h, Tm, rB = init_state(model0, row, obs)
        b0 = run_const(model0, res_qh, row, h, Tm, rB, N, no_act=False)
        h, Tm, rB = init_state(model0, row, obs)
        t0 = run_const(model0, res_qh, row, h, Tm, rB, N, d_v2=0.05, no_act=False)
        d_qh = t0[:, 4] - b0[:, 4]
        h, Tm, rB = init_state(model0, row, obs)
        b1 = run_const(model0, res_s0, row, h, Tm, rB, N, no_act=True)
        h, Tm, rB = init_state(model0, row, obs)
        t1 = run_const(model0, res_s0, row, h, Tm, rB, N, d_v2=0.05, no_act=True)
        d_qna = t1[:, 4] - b1[:, 4]
        ax.plot(t_axis, d_qh, ls="--", lw=1.2, color=color, label=f"qh {name}")
        ax.plot(t_axis, d_qna, lw=1.6, color=color, label=f"qna {name}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("v2+5% open-loop → main (differential)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("ΔT (°C)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # (b) 闭环湿态
    ax = axes[1]
    row, obs = Ea[OP_WET], T_all[OP_WET]
    res_qh = r09.ResMLP(13, r09.Q_SCALE).to(DEVICE)
    res_qh.load_state_dict(torch.load(os.path.join(OUT, "model_res_qh_seed0.pt"),
                                      map_location=DEVICE, weights_only=True))
    res_qh.eval()
    h, Tm, rB = init_state(model0, row, obs)
    b0 = run_const(model0, res_qh, row, h, Tm, rB, N, no_act=False)
    h, Tm, rB = init_state(model0, row, obs)
    cl0 = run_closed_loop(model0, res_qh, row, h, Tm, rB, N,
                          float(T_all[OP_WET, 4]) + 2.0, 332.85, no_act=False)
    d0 = cl0 - b0[:, 4]
    ax.plot(t_axis, d0 / np.mean(d0[-60:]), color="#c55a11", lw=1.2, label="qh closed-loop")
    ax.plot(t_axis, dC / dC_ss, color="#8b008b", lw=1.6, label="qna closed-loop")
    ax.plot([1, 2, 3, 5, 7, 10], [0.000, 0.10, 0.17, 0.49, 0.70, 0.97],
            "o", color="0.3", ms=6, label="exp_099")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_title(f"wet closed-loop SP+2 (norm600 qna={norm600:.2f})")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # (c) 精度对比 bars
    ax = axes[2]
    qh_vals = [2.854, 2.798, 2.968]
    x = np.arange(3)
    ax.bar(x - 0.2, qh_vals, width=0.38, color="#c55a11", label="qh (old)")
    ax.bar(x + 0.2, qna_roll, width=0.38, color="#8b008b", label="qna (no-act)")
    ax.axhline(2.93, color="0.4", ls="--", lw=1, label="v2 2.93")
    ax.set_xticks(x, ["s0", "s1", "s2"])
    ax.set_title("rollout rmse_main — accuracy check")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Fix① action-channel protection — verdict={judge['verdict']} "
                 f"(A1={A1} A2={A2} A3={A3})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig15_fix_action.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig15_fix_action.png", flush=True)


if __name__ == "__main__":
    main()
