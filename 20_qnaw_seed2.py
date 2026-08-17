#!/usr/bin/env python3
"""20_qnaw_seed2.py: 审计行动项④ qnaw 补 seed2 → 3seeds 重评 W1/W2

审计发现: PAPER_MATERIALS 误写 3 seeds, 实际 qnaw 只有 2 seeds。补齐 s2 重评:
  W1: rollout 3seeds mean ≤ 3.0
  W2: 湿/干 op 点 v2+5% (W 恒定, 与 12b 协议一致) main 末值 < 0
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
    exo = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, steps, 1).clone()
    exo[0, :, V2] = exo[0, :, V2] + d_v2
    out, *_ = r09.integrate_res(model0, res, exo, h, Tm, rB, steps, None, "qh",
                                no_v12=True)
    return out[0, :, 4].cpu().numpy()


def main():
    df = r09.load_e0_df()
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    model0 = r09.load_e0(0)

    # 训练 seed 2
    res2, best_va, ep = r09.train_res(df, 2, "qnaw", "qh", False, False, out=3, no_v12=True)
    print(f"[qnaw s2] {ep}ep val={best_va:.3f}", flush=True)

    # 3 seeds rollout
    rolls = []
    for sd in (0, 1, 2):
        res = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
        res.load_state_dict(torch.load(os.path.join(OUT, f"model_res_qnaw_seed{sd}.pt"),
                                       map_location=DEVICE, weights_only=True))
        res.eval()
        for p in res.parameters():
            p.requires_grad_(False)
        r, preds, truths = r09.rollout_res(model0, res, df, START, t02.ROLL_STEPS,
                                           "qh", False, no_v12=True)
        rmse = float(r["rmse_main"])
        rolls.append(rmse)
        print(f"[qnaw s{sd}] rollout={rmse:.3f}", flush=True)

    # 符号测试 (s0/s1 原协议: W 恒定 v2+5%)
    signs = {}
    for name, row_idx in (("wet", OP_WET), ("dry", OP_DRY)):
        row, obs = Ea[row_idx], T_all[row_idx]
        res = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
        res.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnaw_seed2.pt"),
                                       map_location=DEVICE, weights_only=True))
        res.eval()
        for p in res.parameters():
            p.requires_grad_(False)
        h, Tm, rB = init_state(model0, row, obs)
        with torch.no_grad():
            base = run_const(model0, res, row, h.clone(), Tm.clone(), rB.clone(), N, 0.0)
            h2, Tm2, rB2 = init_state(model0, row, obs)
            step = run_const(model0, res, row, h2, Tm2, rB2, N, 0.05)
        signs[name] = round(float((step - base)[-1]), 3)
        print(f"[qnaw s2 sign {name}] final={signs[name]}", flush=True)

    # 汇总 (s0/s1 旧数字 + s2 新数字)
    old = json.load(open(os.path.join(OUT, "qnaw_check_summary.json")))
    mean3 = float(np.mean(rolls))
    W1 = bool(mean3 <= 3.0)
    W2 = bool(signs["wet"] < 0 and signs["dry"] < 0)
    judge = {"W1": W1, "W2": W2, "rollout_mean_3seeds": round(mean3, 3),
             "rollout_vals": [round(x, 3) for x in rolls],
             "signs_s2": signs,
             "verdict": "PASS" if (W1 and W2) else "FAIL"}
    summ = {"old_2seed": old["judge"], "new_3seed": judge}
    with open(os.path.join(OUT, "qnaw_3seed_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== qnaw 3seeds 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
