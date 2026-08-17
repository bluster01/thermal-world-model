#!/usr/bin/env python3
"""16_fix_actuator.py: FIX4 干态闭环诊断 + 实测执行机构动态

D4诊断: ①实测执行机构(指令→阀位 τ_a/速率限制) ②真实干态SP事件闭环曲线 ③真实干态阀位事件开环形态
修复试验: qnal + 分状态k_w + 实测执行机构 → 干/湿耦合闭环
预注册 (FIX4_DESIGN.md): D1 干闭环收敛 / D2 湿闭环不退化 / D3 符号保持 / D4 诊断报告
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
    # 全列 df (指令/阀位/SP 在 CSV 里但不在 E0_COLS)
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
    model0 = r09.load_e0(0)
    mod0 = r15.QnaLag().to(DEVICE)
    mod0.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnal_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    mod0.eval()
    for p in mod0.parameters():
        p.requires_grad_(False)

    summ = {"diagnosis": {}, "sim": {}, "judge": {}}

    # ============ D4 诊断 ============
    # ① 执行机构: 指令→阀位
    cmd2 = dfx["二级减温调节阀设定"].to_numpy(np.float32) / 100.0
    pos2 = dfx["二级减温调节门阀位"].to_numpy(np.float32) / 100.0
    seg_s = slice(START, START + t02.ROLL_STEPS)
    c, p = cmd2[seg_s], pos2[seg_s]
    mask = np.isfinite(c) & np.isfinite(p)
    c, p = c[mask], p[mask]
    # 一阶: p_t = a·p_{t-1} + (1-a)·c_t
    x1, x2, y = p[:-1], c[1:], p[1:]
    A = np.stack([x1, x2], 1)
    coef, res_, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a_hat = float(coef[0])
    tau_a = -DT / np.log(a_hat) if 0 < a_hat < 1 else float("nan")
    r2 = float(1 - res_.sum() / np.sum((y - y.mean()) ** 2))
    dpos = np.abs(np.diff(p))
    rate95 = float(np.percentile(dpos, 95))  # per 10s step
    summ["diagnosis"]["actuator"] = {
        "tau_a_s": round(tau_a, 1), "a_hat": round(a_hat, 4),
        "r2": round(r2, 3), "rate95_per_step": round(rate95, 4),
        "rate95_pct_s": round(rate95 / DT * 100, 2)}
    print(f"[diag actuator] tau_a={tau_a:.1f}s rate95={rate95*100/DT:.2f}%/s R²={r2:.3f}", flush=True)

    # ② 真实干态 SP 事件闭环曲线
    SP = dfx["二级减温中间设定值"].to_numpy(np.float32)
    mainT = T_all[:, 4]
    dSP = np.diff(SP, prepend=SP[0])
    ev_idx = np.where((np.abs(dSP) >= 0.5))[0]
    # 合并 30 步内的事件
    groups = []
    for i in ev_idx:
        if groups and i - groups[-1][-1] <= 30:
            groups[-1].append(i)
        else:
            groups.append([i])
    dry_resp = []
    for g in groups:
        i0 = g[0]
        if i0 < START or i0 + 60 >= len(mainT):
            continue
        d = SP[i0] - SP[i0 - 1]
        if abs(d) < 0.5:
            continue
        # 事件后 60 步全部干态
        if not np.all(pm_all[i0: i0 + 61] > P_CRIT):
            continue
        resp = (mainT[i0 + 1: i0 + 61] - mainT[i0]) / d
        dry_resp.append(resp)
    dry_anch = None
    if dry_resp:
        R = np.stack(dry_resp)
        dry_anch = [float(np.mean(R[:, int(i) - 1])) for i in ANCH_T]
        summ["diagnosis"]["dry_sp_events"] = {
            "n": int(len(dry_resp)), "anchors": [round(x, 3) for x in dry_anch]}
        print(f"[diag dry SP] n={len(dry_resp)} anchors={dry_anch}", flush=True)
    else:
        summ["diagnosis"]["dry_sp_events"] = {"n": 0}
        print("[diag dry SP] 无事件", flush=True)

    # ③ 真实干态阀位事件 (开环形态)
    dv2 = np.abs(np.diff(pos2, prepend=pos2[0]))
    ev_v = np.where(dv2 >= 0.02)[0]
    gv = []
    for i in ev_v:
        if gv and i - gv[-1][-1] <= 30:
            gv[-1].append(i)
        else:
            gv.append([i])
    v_resp = []
    for g in gv:
        i0 = g[0]
        if i0 < START or i0 + 60 >= len(mainT):
            continue
        d = pos2[i0] - pos2[i0 - 1]
        if abs(d) < 0.02 or not np.all(pm_all[i0: i0 + 61] > P_CRIT):
            continue
        resp = (mainT[i0 + 1: i0 + 61] - mainT[i0]) / d
        v_resp.append(resp)
    if v_resp:
        R = np.stack(v_resp)
        v_anch = [float(np.mean(R[:, int(i) - 1])) for i in ANCH_T]
        summ["diagnosis"]["dry_valve_events"] = {
            "n": int(len(v_resp)), "anchors": [round(x, 3) for x in v_anch]}
        print(f"[diag dry valve] n={len(v_resp)} anchors={v_anch}", flush=True)
    else:
        summ["diagnosis"]["dry_valve_events"] = {"n": 0}
        print("[diag dry valve] 无事件", flush=True)

    # ============ 仿真: 实测执行机构进环 ============
    # 分状态 k_w (与 FIX3 相同)
    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))
    # 执行机构: 速率限制 (实测95分位) + 一阶 τ_a (若 tau_a 有效)
    rate = max(rate95, 0.01)
    use_tau = np.isfinite(tau_a) and 0.5 < tau_a < 120
    tau_a_eff = tau_a if use_tau else 5.0

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
        SP = float(obs[4]) + 2.0
        u, integ = u0, 0.0
        v2_act = u0  # 实际阀位 (执行机构)
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, v2_act,
                                                   W0 * (1 + kw * (v2_act - u0)))
                e = mh[t] - SP
                kp, ti = r15.pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                # 执行机构: 一阶+速率限制
                step = (u - v2_act) * (DT / tau_a_eff)
                step = float(np.clip(step, -rate, rate))
                v2_act = float(np.clip(v2_act + step, 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        anch = [float(norm[int(i) - 1]) for i in ANCH_T]
        summ["sim"][name] = {"anchors": [round(x, 3) for x in anch],
                             "norm600": round(float(norm[599]), 3),
                             "tail_std": round(float(np.std(norm[-120:])), 4)}
        print(f"[sim {name}] anchors={summ['sim'][name]['anchors']} "
              f"norm600={summ['sim'][name]['norm600']} "
              f"tail_std={summ['sim'][name]['tail_std']}", flush=True)

    # 阶跃符号 (D3)
    step_signs = {}
    for name, row_idx, base, state in (("wet", OP_WET, base_wet, "wet"),
                                       ("dry", OP_DRY, base_dry, "dry")):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[state]
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens,
                                                   u0 + 0.05, W0 * (1 + kw * 0.05))
        step_signs[name] = round(float(np.mean((dT - base)[-60:])), 3)
        print(f"[step {name}] K={step_signs[name]}", flush=True)
    summ["sim"]["step_K"] = step_signs

    D1 = bool(0.8 <= summ["sim"]["dry"]["norm600"] <= 1.2 and summ["sim"]["dry"]["tail_std"] <= 0.05)
    D2 = bool(0.8 <= summ["sim"]["wet"]["norm600"] <= 1.2 and summ["sim"]["wet"]["tail_std"] <= 0.05)
    D3 = bool(step_signs["wet"] < 0 and step_signs["dry"] < 0)
    judge = {"D1": D1, "D2": D2, "D3": D3,
             "verdict": "PASS" if (D1 and D2 and D3) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fix4_actuator_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== FIX4 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    t_axis = np.arange(N) * DT / 60.0
    ax = axes[0]
    if dry_anch is not None:
        ax.plot(ANCH_T / 6.0, dry_anch, "s-", color="0.3", ms=7,
                label=f"real dry SP events (n={len(dry_resp)})")
    ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.5", ms=7, label="exp_099 (mixed)")
    for name, color in (("wet", "#c55a11"), ("dry", "#8b008b")):
        row_idx = OP_WET if name == "wet" else OP_DRY
        base = base_wet if name == "wet" else base_dry
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[name]
        SP = float(obs[4]) + 2.0
        power = 332.85 if name == "wet" else 464.53
        u, integ = u0, 0.0
        v2_act = u0
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, v2_act,
                                                   W0 * (1 + kw * (v2_act - u0)))
                e = mh[t] - SP
                kp, ti = r15.pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                step = float(np.clip((u - v2_act) * (DT / tau_a_eff), -rate, rate))
                v2_act = float(np.clip(v2_act + step, 0.0, 1.0))
        dC = mh - base
        dC_ss = float(np.mean(dC[-60:]))
        ax.plot(t_axis, dC / dC_ss, lw=1.4, color=color,
                label=f"model {name} (actuator in loop)")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_title(f"closed-loop: model vs real (τ_a={tau_a_eff:.1f}s, rate={rate*100/DT:.2f}%/s)")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    if v_resp:
        R = np.stack(v_resp)
        m = R.mean(0)
        se = R.std(0) / np.sqrt(len(v_resp))
        ax.plot(t_axis[:60], m, color="0.3", lw=1.5,
                label=f"real dry valve events (n={len(v_resp)})")
        ax.fill_between(t_axis[:60], m - se, m + se, color="0.3", alpha=0.2)
    row, obs = Ea[OP_DRY], T_all[OP_DRY]
    h, Tm, rB = r15.init_state(model0, row, obs)
    T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
    u0, W0 = float(row[V2]), float(row[8])
    kw = k_w_state["dry"]
    dT = np.zeros(N)
    with torch.no_grad():
        for t in range(N):
            dT[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens,
                                               u0 + 0.05, W0 * (1 + kw * 0.05))
    d = dT - base_dry
    d_ss = float(np.mean(d[-60:]))
    ax.plot(t_axis[:60], d[:60] / d_ss, color="#8b008b", lw=1.5,
            label="model dry open-loop (normalized)")
    ax.set_title("dry open-loop: real valve events vs model step")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(f"FIX4 actuator dynamics — verdict={judge['verdict']} "
                 f"(D1={D1} D2={D2} D3={D3})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig19_actuator.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig19_actuator.png", flush=True)


if __name__ == "__main__":
    main()
