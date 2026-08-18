#!/usr/bin/env python3
"""19_actuator_ablation.py: 审计行动项③ 执行机构表征消融 — τ_a-only / rate-only / 双表征

审计发现: τ_a=883s 一阶拟合本身已吸收速率限制行为 (a_hat=0.989/步), 再叠 0.14%/s 速率限制
存在重复计慢嫌疑; 且无敏感性消融。本脚本三种配置闭环对比 (qnal + W联动 + 实测双参数):
  A tau_only:  v += (u−v)·dt/883 (无速率限制)
  B rate_only: v += clip(u−v, ±0.014) (纯速率限制积分器)
  C both:      v += clip((u−v)·dt/883, ±0.014) (FIX4 现状)
报告 (不设 PASS/FAIL, 敏感性探测): 每配置×湿/干 收敛性 + 湿态 G1 锚点差 + 阀位轨迹特征
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
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
TAU_A = 883.2
RATE = 0.014
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
    pm_all = Ea[:, 2]
    model0 = r09.load_e0(0)
    mod0 = r15.QnaLag().to(DEVICE)
    mod0.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnal_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    mod0.eval()
    for p in mod0.parameters():
        p.requires_grad_(False)

    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

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

    def actuator(u, v, cfg):
        if cfg == "tau_only":
            return float(np.clip(v + (u - v) * (DT / TAU_A), 0.0, 1.0))
        elif cfg == "rate_only":
            return float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
        else:  # both
            step = float(np.clip((u - v) * (DT / TAU_A), -RATE, RATE))
            return float(np.clip(v + step, 0.0, 1.0))

    summ = {"configs": {}}
    for cfg in ("tau_only", "rate_only", "both"):
        summ["configs"][cfg] = {}
        for name, row_idx, base, state, power in (("wet", OP_WET, base_wet, "wet", 332.85),
                                                  ("dry", OP_DRY, base_dry, "dry", 464.53)):
            row, obs = Ea[row_idx], T_all[row_idx]
            h, Tm, rB = r15.init_state(model0, row, obs)
            T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
            u0, W0 = float(row[V2]), float(row[8])
            kw = k_w_state[state]
            SP = float(obs[4]) + 2.0
            u, integ, v = u0, 0.0, u0
            mh = np.zeros(N)
            with torch.no_grad():
                for t in range(N):
                    mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, v,
                                                       W0 * (1 + kw * (v - u0)))
                    e = mh[t] - SP
                    kp, ti = r15.pi_params(-e, power)
                    integ += e * DT
                    u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                    v = actuator(u, v, cfg)
            dC = mh - base
            dC_ss = float(np.mean(dC[-60:]))
            norm = dC / dC_ss
            anch = [float(norm[int(i) - 1]) for i in ANCH_T]
            if name == "wet":
                deltas = [round(abs(a - b), 3) for a, b in zip(anch, ANCH_Y)]
            else:
                deltas = None
            rec = {"anchors": [round(x, 3) for x in anch],
                   "norm600": round(float(norm[599]), 3),
                   "tail_std": round(float(np.std(norm[-120:])), 4),
                   "converged": bool(0.8 <= norm[599] <= 1.2
                                     and np.std(norm[-120:]) <= 0.05),
                   "G1_deltas": deltas}
            summ["configs"][cfg][name] = rec
            print(f"[{cfg} {name}] anchors={rec['anchors']} norm600={rec['norm600']} "
                  f"tail_std={rec['tail_std']} conv={rec['converged']} "
                  f"G1Δ={deltas}", flush=True)

    with open(os.path.join(OUT, "actuator_ablation_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("=== 消融汇总 (G1 门槛: 各锚点|Δ|≤0.15 且 600s≤0.10) ===")
    for cfg in summ["configs"]:
        w = summ["configs"][cfg]["wet"]
        g1_ok = w["G1_deltas"] is not None and all(d <= 0.15 for d in w["G1_deltas"]) \
            and w["G1_deltas"][-1] <= 0.10
        print(f"{cfg}: wet_conv={w['converged']} dry_conv={summ['configs'][cfg]['dry']['converged']} "
              f"wet_G1={'PASS' if g1_ok else 'FAIL'} Δ={w['G1_deltas']}", flush=True)

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    t_axis = np.arange(N) * DT / 60.0
    ax = axes[0]
    ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.3", ms=7, label="exp_099")
    for cfg, color in (("tau_only", "#2e8b57"), ("rate_only", "#c55a11"), ("both", "#8b008b")):
        row, obs = Ea[OP_WET], T_all[OP_WET]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state["wet"]
        SP = float(obs[4]) + 2.0
        u, integ, v = u0, 0.0, u0
        mh = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                mh[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, v,
                                                   W0 * (1 + kw * (v - u0)))
                e = mh[t] - SP
                kp, ti = r15.pi_params(-e, 332.85)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                v = actuator(u, v, cfg)
        dC = mh - base_wet
        dC_ss = float(np.mean(dC[-60:]))
        ax.plot(t_axis, dC / dC_ss, lw=1.4, color=color, label=f"{cfg}")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_title("wet closed-loop: actuator representation ablation")
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar([0, 1, 2],
           [summ["configs"]["tau_only"]["wet"]["G1_deltas"][2],
            summ["configs"]["rate_only"]["wet"]["G1_deltas"][2],
            summ["configs"]["both"]["wet"]["G1_deltas"][2]],
           color=["#2e8b57", "#c55a11", "#8b008b"], alpha=0.8)
    ax.set_title("wet G1 delta at 180s anchor (gate ≤0.15)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["tau_only", "rate_only", "both"])
    ax.axhline(0.15, color="crimson", ls=":", lw=1)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("FIX4 audit: actuator representation sensitivity", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig22_actuator_ablation.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig22_actuator_ablation.png", flush=True)


if __name__ == "__main__":
    main()
