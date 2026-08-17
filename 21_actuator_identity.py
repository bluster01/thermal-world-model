#!/usr/bin/env python3
"""21_actuator_identity.py: 执行机构身份诊断 — 回应"883s不真实/干湿不同lag"质疑

假设 (用户): 883s 一阶拟合过高, 真实执行机构应快得多; 或干湿态执行机构行为不同,
单一混合拟合把湿态拉慢。
诊断四件:
  D1 指令身份: |Δcmd| 分布 (阶梯状=运行设定 / 连续=PI输出) + cmd 与温度偏差相关性
  D2 事件研究: 指令跳变事件 (|Δcmd|≥3%) 后阀位的真实响应 — 死区时间/初始速率/63%时间, 干湿分层
  D3 分层一阶拟合: τ_a 与 rate95 干/湿分别拟
  D4 闭环复测: 用事件测得的干/湿执行机构参数 (纯积分+速率限制+死区) 重跑干/湿闭环, 重算 G1
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
r15 = _imp("15_fix_learnlag.py", "r15")
import numpy as np
import pandas as pd
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])


def main():
    df = r09.load_e0_df()
    dfx = pd.read_csv(t02.CSV, usecols=["二级减温中间设定值", "二级减温调节阀设定",
                                        "二级减温调节门阀位", "一级减温调节门阀位"]) \
        .iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)
    dfx = dfx.astype(np.float32)
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]
    seg_s = slice(START, START + t02.ROLL_STEPS)
    cmd = dfx["二级减温调节阀设定"].to_numpy(np.float32)[seg_s] / 100.0
    pos = dfx["二级减温调节门阀位"].to_numpy(np.float32)[seg_s] / 100.0
    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    main_seg = T_all[START: START + t02.ROLL_STEPS, 4]

    summ = {"cmd_identity": {}, "events": {}, "stratified_fit": {}, "loop": {}}

    # ---- D1 指令身份 ----
    dcmd = np.abs(np.diff(cmd, prepend=cmd[0]))
    summ["cmd_identity"] = {
        "frac_gt_0.1pct": round(float((dcmd > 0.001).mean()), 4),
        "frac_gt_0.5pct": round(float((dcmd > 0.005).mean()), 4),
        "frac_gt_1pct": round(float((dcmd > 0.01).mean()), 4),
        "frac_gt_3pct": round(float((dcmd > 0.03).mean()), 4),
        "median_abs_dcmd": round(float(np.median(dcmd[dcmd > 0])), 4),
        "cmd_dev_corr": round(float(np.corrcoef(cmd, main_seg)[0, 1]), 3),
        "cmd_autocorr_lag1": round(float(np.corrcoef(cmd[:-1], cmd[1:])[0, 1]), 3)}
    print("[D1 cmd]", summ["cmd_identity"], flush=True)

    # ---- D2 事件研究 (干湿分层) ----
    events = {"wet": {"theta_steps": [], "init_rate": [], "t63_steps": [], "n": 0},
              "dry": {"theta_steps": [], "init_rate": [], "t63_steps": [], "n": 0}}
    ev_idx = np.where(dcmd >= 0.005)[0]
    groups = []
    for i in ev_idx:
        if groups and i - groups[-1][-1] <= 30:
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        i0 = g[0]
        if i0 < 5 or i0 + 120 >= len(cmd):
            continue
        d = cmd[i0] - cmd[i0 - 1]
        if abs(d) < 0.005:
            continue
        state = "wet" if pm_seg[i0] <= P_CRIT else "dry"
        # 事件后 120 步内状态一致
        if not np.all((pm_seg[i0: i0 + 121] <= P_CRIT) == (state == "wet")):
            continue
        dpos = np.diff(pos[i0 - 1: i0 + 120])
        # 死区: 首个 |dpos| ≥ 0.002 的步 (阀位开始动)
        onset = np.where(np.abs(dpos) >= 0.002)[0]
        if len(onset) == 0:
            continue
        theta = int(onset[0])
        # 初始速率: onset 后 10 步平均 |dpos|/步
        rate = float(np.mean(np.abs(dpos[theta: theta + 10])))
        # 63% 时间: pos 达到 cmd[i0] + 0.63·d (朝目标移动方向)
        target63 = cmd[i0] + 0.63 * d
        reached = None
        for t in range(theta, 120):
            if (d > 0 and pos[i0 - 1 + t + 1] >= target63) or \
               (d < 0 and pos[i0 - 1 + t + 1] <= target63):
                reached = t
                break
        if reached is not None:
            events[state]["theta_steps"].append(theta)
            events[state]["init_rate"].append(rate)
            events[state]["t63_steps"].append(reached)
            events[state]["n"] += 1
    for state in ("wet", "dry"):
        ev = events[state]
        summ["events"][state] = {
            "n": ev["n"],
            "theta_steps_med": int(np.median(ev["theta_steps"])) if ev["n"] else None,
            "init_rate_med_pct_s": round(float(np.median(ev["init_rate"])) / DT * 100, 3)
            if ev["n"] else None,
            "t63_med_s": round(float(np.median(ev["t63_steps"])) * DT, 1) if ev["n"] else None,
            "theta_steps": [int(x) for x in ev["theta_steps"]],
            "init_rates": [round(float(x), 4) for x in ev["init_rate"]],
            "t63_steps": [int(x) for x in ev["t63_steps"]]}
        print(f"[D2 events {state}] {summ['events'][state]}", flush=True)

    # ---- D3 分层一阶拟合 ----
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        c, p = cmd[msk], pos[msk]
        A = np.stack([p[:-1], c[1:]], 1)
        coef, res_, _, _ = np.linalg.lstsq(A, p[1:], rcond=None)
        a_hat = float(coef[0])
        tau = -DT / np.log(a_hat) if 0 < a_hat < 1 else float("nan")
        r2 = float(1 - res_.sum() / np.sum((p[1:] - p[1:].mean()) ** 2))
        rate95 = float(np.percentile(np.abs(np.diff(p)), 95))
        summ["stratified_fit"][state] = {
            "tau_s": round(tau, 1) if np.isfinite(tau) else None,
            "a_hat": round(a_hat, 4), "r2": round(r2, 3),
            "rate95_pct_s": round(rate95 / DT * 100, 3), "n": int(msk.sum())}
        print(f"[D3 fit {state}] {summ['stratified_fit'][state]}", flush=True)

    # ---- D4 闭环复测 (事件参数: 死区+速率限制纯积分) ----
    model0 = r09.load_e0(0)
    mod0 = r15.QnaLag().to(DEVICE)
    mod0.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnal_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    mod0.eval()
    for p in mod0.parameters():
        p.requires_grad_(False)

    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    act_params = {}
    for state in ("wet", "dry"):
        ev = events[state]
        if ev["n"] >= 3:
            act_params[state] = {"theta": int(np.median(ev["theta_steps"])),
                                 "rate": float(np.median(ev["init_rate"]))}
        else:
            # 样本不足回退: 用实测 rate95 (事件样本不足如实记录)
            act_params[state] = {"theta": 0,
                                 "rate": float(np.percentile(np.abs(np.diff(pos)), 95))}
            act_params[state]["fallback"] = True
    summ["act_params"] = act_params
    print("[D4 params]", act_params, flush=True)

    def fwd_one(row, h, Tm, rB, T_sens, v2_val, W_val):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = v2_val
        exo[0, 0, 8] = W_val
        out, h, Tm, rB, hm1, hm2, T_sens = r15.integrate_learn(
            model0, mod0, exo, h, Tm, rB, 1, T_sens=T_sens)
        return float(out[0, 0, 4]), h, Tm, rB, T_sens

    def base_run(row_idx):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        b = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                b[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, u0, W0)
        return b

    base_wet = base_run(OP_WET)
    base_dry = base_run(OP_DRY)

    for name, row_idx, base, state, power in (("wet", OP_WET, base_wet, "wet", 332.85),
                                              ("dry", OP_DRY, base_dry, "dry", 464.53)):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[state]
        ap = act_params[state]
        SP = float(obs[4]) + 2.0
        u, integ, v = u0, 0.0, u0
        ubuf = [u0] * (ap["theta"] + 1)
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, v,
                                                   W0 * (1 + kw * (v - u0)))
                e = mh[t] - SP
                kp, ti = r15.pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                ubuf = [u] + ubuf[:-1]
                u_del = ubuf[-1]
                v = float(np.clip(v + np.clip(u_del - v, -ap["rate"], ap["rate"]), 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        anch = [float(norm[int(i) - 1]) for i in ANCH_T]
        deltas = [round(abs(a - b), 3) for a, b in zip(anch, ANCH_Y)] if name == "wet" else None
        rec = {"anchors": [round(x, 3) for x in anch],
               "norm600": round(float(norm[599]), 3),
               "tail_std": round(float(np.std(norm[-120:])), 4),
               "converged": bool(0.8 <= norm[599] <= 1.2 and np.std(norm[-120:]) <= 0.05),
               "G1_deltas": deltas}
        summ["loop"][name] = rec
        print(f"[D4 loop {name}] anchors={rec['anchors']} conv={rec['converged']} "
              f"G1Δ={deltas}", flush=True)

    with open(os.path.join(OUT, "actuator_identity_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    print("=== 诊断汇总 ===", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    # 事件叠加: 对齐到事件点, 归一化 dpos
    for state, color in (("wet", "#8b008b"), ("dry", "#2e8b57")):
        ev = events[state]
        if ev["n"]:
            # 重跑事件提取 (保存轨迹) — 简化: 从 summary 读 theta/rate 画示意
            pass
    ax.set_title("cmd step events: position response (aggregate)")
    ax.set_xlabel("steps after command step")
    ax.set_ylabel("Δpos")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for name, color in (("wet", "#c55a11"), ("dry", "#8b008b")):
        row_idx = OP_WET if name == "wet" else OP_DRY
        base = base_wet if name == "wet" else base_dry
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[name]
        ap = act_params[name]
        SP = float(obs[4]) + 2.0
        power = 332.85 if name == "wet" else 464.53
        u, integ, v = u0, 0.0, u0
        ubuf = [u0] * (ap["theta"] + 1)
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, v,
                                                   W0 * (1 + kw * (v - u0)))
                e = mh[t] - SP
                kp, ti = r15.pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                ubuf = [u] + ubuf[:-1]
                v = float(np.clip(v + np.clip(ubuf[-1] - v, -ap["rate"], ap["rate"]), 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        ax.plot(np.arange(N) * DT / 60.0, dC / dC_ss, lw=1.4, color=color,
                label=f"{name} (event-based actuator)")
    ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.3", ms=7, label="exp_099")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_title("closed loop with event-measured actuator")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("FIX4 audit: actuator identity diagnosis", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig23_actuator_identity.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig23_actuator_identity.png", flush=True)


if __name__ == "__main__":
    main()
