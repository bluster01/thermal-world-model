#!/usr/bin/env python3
"""12b_qnaw_check.py: 修复①中间态 — qnaw = 残差只去 v1/v2 保留 W (11特征), 2 seeds

预注册 (冻结 2026-08-17):
  W1: qnaw 2seeds rollout main mean ≤ 3.0 (qna 为 3.04, qh 为 2.82)
  W2: 湿/干 v2+5% 开环 main 末值 ΔT < 0 (符号保持修复)
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
N = 600
V2 = 6
OP_WET, OP_DRY = 40161, 40437


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


def run_const(model0, res, row, h, Tm, rB, steps, d_v2=0.0):
    exo = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, steps, 1)
    exo[:, :, V2] += d_v2
    with torch.no_grad():
        out, *_ = r09.integrate_res(model0, res, exo, h, Tm, rB, steps, None, "qh",
                                    no_v12=True)
    return out[0].cpu().numpy()


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

    summ = {"train": {}, "rollout": {}, "strat": {}, "step_test": {}, "judge": {}}
    rolls = []
    res_s0 = None
    for sd in (0, 1):
        res, best_va, ep = r09.train_res(df, sd, "qnaw", "qh", False, False, out=3,
                                         no_v12=True)
        res.eval()
        summ["train"][str(sd)] = {"val_mse": round(best_va, 4), "ep": ep}
        with torch.no_grad():
            r, preds, truths = r09.rollout_res(model0, res, df, START, t02.ROLL_STEPS,
                                               "qh", False, no_v12=True)
        summ["rollout"][str(sd)] = {k: (round(x, 4) if isinstance(x, float) else x)
                                    for k, x in r.items()}
        sr = r09.strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
        summ["strat"][str(sd)] = sr
        rolls.append(r["rmse_main"])
        print(f"[qnaw s{sd}] rollout={r['rmse_main']:.3f} "
              f"dry={sr['dry']['rmse_main']:.2f} wet={sr['wet']['rmse_main']:.2f}", flush=True)
        if sd == 0:
            res_s0 = res

    signs = {}
    for name, row_idx in (("wet", OP_WET), ("dry", OP_DRY)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        base = run_const(model0, res_s0, row, h, Tm, rB, N)
        h, Tm, rB = init_state(model0, row, obs)
        tr = run_const(model0, res_s0, row, h, Tm, rB, N, d_v2=0.05)
        d = tr - base
        signs[name] = {"main_final": round(float(d[-1, 4]), 3),
                       "sh1_in_final": round(float(d[-1, 0]), 3)}
        print(f"[step {name}] main_final={signs[name]['main_final']}", flush=True)
    summ["step_test"] = signs

    W1 = bool(np.mean(rolls) <= 3.0)
    W2 = bool(signs["wet"]["main_final"] < 0 and signs["dry"]["main_final"] < 0)
    judge = {"W1": W1, "W2": W2,
             "rollout_mean": round(float(np.mean(rolls)), 3),
             "rollout_vals": [round(x, 3) for x in rolls],
             "verdict": "PASS" if (W1 and W2) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "qnaw_check_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== qnaw 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
