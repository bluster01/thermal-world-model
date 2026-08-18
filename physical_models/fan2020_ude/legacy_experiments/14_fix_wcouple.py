#!/usr/bin/env python3
"""14_fix_wcouple.py: 修复③ 阀门→W联动子模型 + 耦合通道重测滞后效果

背景: 修复② B1/B2 FAIL 的根因 = W(总喷水)恒定下阀位只做级间再分配, PI 在模型里无权威,
阶跃响应弱(K≈−0.07)且形态度量无意义。真实"阀门→总喷水→温度"链必须耦合 W。
方案: 数据拟合 dW/dv2 增量增益 → W_c = W0·(1+k_w·Δv2) → 闭环/阶跃用耦合通道。
对 qlag(带滞后) 与 qnaw(无滞后) 各跑一遍, 判定耦合后闭环收敛性与滞后是否改善锚点形态。

预注册判定 (冻结 2026-08-17):
  C1: qlag 闭环(耦合) 湿态与干态 norm600 ∈ [0.8,1.2] 且末120步相对std ≤ 0.05 (收敛不振荡)
  C2: 耦合开环阶跃 v2+5%+W联动: |K| ≥ 0.15 (湿或干) — 通道有权威
  C3: 湿/干 main 末值 < 0 (符号正确)
另报告: qlag vs qnaw 耦合闭环锚点 vs exp_099 (滞后是否改善形态, 不设门槛)
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
r13 = _imp("13_fix_lag.py", "r13")
import numpy as np
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])


def main():
    df = r09.load_e0_df()
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    model0 = r09.load_e0(0)

    # ---- 拟合 k_w = dW/W / dv2 (测试段) ----
    seg = Ea[START: START + t02.ROLL_STEPS]
    v2s = seg[:, V2]
    Ws = seg[:, 8]
    mask = v2s > 0.05
    slope, intercept = np.polyfit(v2s[mask], Ws[mask], 1)
    Wmean = float(np.mean(Ws[mask]))
    k_w = slope / Wmean
    r2 = float(np.corrcoef(v2s[mask], Ws[mask])[0, 1] ** 2)
    print(f"[fit] dW/dv2={slope:.3f} kg/s per unit, Wmean={Wmean:.2f}, k_w={k_w:.3f}, R²={r2:.3f}", flush=True)

    # 模型加载
    qlag = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
    qlag.load_state_dict(torch.load(os.path.join(OUT, "model_res_qlag_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    qlag.eval()
    for p in qlag.parameters():
        p.requires_grad_(False)
    qnaw = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
    qnaw.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnaw_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    qnaw.eval()
    for p in qnaw.parameters():
        p.requires_grad_(False)

    def fwd_one(res, row, h, Tm, rB, T_sens, u, W_val, lag):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = u
        exo[0, 0, 8] = W_val
        if lag:
            out, h, Tm, rB, hm1, hm2, T_sens = r13.integrate_lag(
                model0, res, exo, h, Tm, rB, 1, T_sens=T_sens)
        else:
            out, h, Tm, rB, hm1, hm2 = r09.integrate_res(
                model0, res, exo, h, Tm, rB, 1, None, "qh", no_v12=True)
            # 无滞后版: T_sens 不适用
        return float(out[0, 0, 4]), h, Tm, rB, T_sens

    def closed_loop(res, row_idx, lag):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0) if lag else None
        W0 = float(row[8])
        u0 = float(row[V2])
        u, integ = u0, 0.0
        SP = float(obs[4]) + 2.0
        power = 332.85 if row_idx == OP_WET else 464.53
        main_hist = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                W_val = W0 * (1.0 + k_w * (u - u0))
                main, h, Tm, rB, T_sens = fwd_one(res, row, h, Tm, rB, T_sens, u, W_val, lag)
                e = main - SP
                kp, ti = r13.pi_params(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                main_hist[t] = main
        # 基线: 恒定 u0/W0
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0) if lag else None
        base_main = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                m, h, Tm, rB, T_sens = fwd_one(res, row, h, Tm, rB, T_sens, u0, W0, lag)
                base_main[t] = m
        dC = main_hist - base_main
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        anch = [float(norm[int(i) - 1]) for i in ANCH_T]
        tail_std = float(np.std(norm[-120:]))
        return {"anchors": [round(x, 3) for x in anch],
                "delta": [round(abs(x - y), 3) for x, y in zip(anch, ANCH_Y)],
                "norm600": round(float(norm[599]), 3),
                "tail_std": round(tail_std, 4),
                "ss": round(dC_ss, 2)}

    def open_step(res, row_idx, lag):
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0) if lag else None
        u0, W0 = float(row[V2]), float(row[8])
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                W_val = W0 * (1.0 + k_w * 0.05)
                m, h, Tm, rB, T_sens = fwd_one(res, row, h, Tm, rB, T_sens,
                                                u0 + 0.05, W_val, lag)
                dT[t] = m
        # 基线
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0) if lag else None
        base = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                m, h, Tm, rB, T_sens = fwd_one(res, row, h, Tm, rB, T_sens, u0, W0, lag)
                base[t] = m
        d = dT - base
        K = float(np.mean(d[-60:]))
        return {"K": round(K, 3), "main_final": round(float(d[-1]), 3)}

    summ = {"fit": {"k_w": round(k_w, 3), "R2": round(r2, 3),
                    "slope": round(float(slope), 3), "Wmean": round(Wmean, 2)},
            "cl": {}, "step": {}, "judge": {}}
    for name, res, lag in (("qlag", qlag, True), ("qnaw", qnaw, False)):
        summ["cl"][name] = {}
        summ["step"][name] = {}
        for op in ("wet", "dry"):
            row_idx = OP_WET if op == "wet" else OP_DRY
            cl = closed_loop(res, row_idx, lag)
            st = open_step(res, row_idx, lag)
            summ["cl"][name][op] = cl
            summ["step"][name][op] = st
            print(f"[{name} {op}] cl anchors={cl['anchors']} Δ={cl['delta']} "
                  f"norm600={cl['norm600']} tail_std={cl['tail_std']} "
                  f"step K={st['K']}", flush=True)

    cl = summ["cl"]["qlag"]
    st = summ["step"]["qlag"]
    C1 = bool(all(0.8 <= cl[op]["norm600"] <= 1.2 for op in ("wet", "dry"))
              and all(cl[op]["tail_std"] <= 0.05 for op in ("wet", "dry")))
    C2 = bool(max(abs(st["wet"]["K"]), abs(st["dry"]["K"])) >= 0.15)
    C3 = bool(st["wet"]["main_final"] < 0 and st["dry"]["main_final"] < 0)
    judge = {"C1": C1, "C2": C2, "C3": C3,
             "verdict": "PASS" if (C1 and C2 and C3) else "FAIL"}
    summ["judge"] = judge
    with open(os.path.join(OUT, "fix_wcouple_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== 修复③ 判定 ===")
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
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        dT, base = np.zeros(N), np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                m, h, Tm, rB, T_sens = fwd_one(qlag, row, h, Tm, rB, T_sens,
                                               u0 + 0.05, W0 * (1 + k_w * 0.05), True)
                dT[t] = m
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        with torch.no_grad():
            for t in range(N):
                m, h, Tm, rB, T_sens = fwd_one(qlag, row, h, Tm, rB, T_sens, u0, W0, True)
                base[t] = m
        ax.plot(t_axis, dT - base, lw=1.5, color=color,
                label=f"qlag {name} coupled step→main")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("open-loop coupled step (v2+5% & W联动)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("ΔT (°C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.3", ms=7, label="exp_099")
    for name, res, lag, color in (("qlag", qlag, True, "#8b008b"), ("qnaw", qnaw, False, "#c55a11")):
        row_idx = OP_DRY
        row, obs = Ea[row_idx], T_all[row_idx]
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0) if lag else None
        W0, u0 = float(row[8]), float(row[V2])
        u, integ = u0, 0.0
        SP = float(obs[4]) + 2.0
        mh, bl = np.zeros(N), np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                m, h, Tm, rB, T_sens = fwd_one(res, row, h, Tm, rB, T_sens, u,
                                                W0 * (1 + k_w * (u - u0)), lag)
                e = m - SP
                kp, ti = r13.pi_params(-e, 464.53)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                mh[t] = m
        h, Tm, rB = r13.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0) if lag else None
        with torch.no_grad():
            for t in range(N):
                m, h, Tm, rB, T_sens = fwd_one(res, row, h, Tm, rB, T_sens, u0, W0, lag)
                bl[t] = m
        dC = mh - bl
        dC_ss = float(np.mean(dC[-60:]))
        ax.plot(t_axis, dC / dC_ss, lw=1.4, color=color,
                label=f"{name} dry closed-loop coupled")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.axhline(0.63, color="0.5", ls=":", lw=0.8)
    ax.set_title("dry closed-loop coupled (SP+2)")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Fix③ W-coupling — verdict={judge['verdict']} "
                 f"(C1={C1} C2={C2} C3={C3}, k_w={k_w:.3f} R²={r2:.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig17_fix_wcouple.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig17_fix_wcouple.png", flush=True)


if __name__ == "__main__":
    main()
