#!/usr/bin/env python3
"""29_final_fig.py: qnav 终态证据图 fig28_final_qnav.png
Panel A: 湿/干闭环响应 vs exp_099 (qnav, 真实PI+速率限制)
Panel B: τ63 全程 (各配置, 湿/干) vs G2 窗口 [240,900]s
Panel C: rollout 精度全程 (main RMSE)
Panel D: 湿态 sh1_out 首步偏差全程
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
r27 = _imp("27_fix_evap_residual.py", "r27")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET, OP_DRY = 40161, 40437
RATE = 0.0137
ANCH_T_MIN = np.array([1, 2, 3, 5, 7, 10])  # min
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])

QNAV_S0 = os.path.join(OUT, "model_res_qnav_seed0.pt")


def load_qnav():
    warm0 = torch.load(os.path.join(OUT, "model_e0_seed0.pt"), map_location=DEVICE,
                       weights_only=True)
    model0 = r26.E0Evap(warm0).to(DEVICE)
    model0.load_state_dict(torch.load(os.path.join(OUT, "model_e0_evap_seed0.pt"),
                                      map_location=DEVICE, weights_only=True))
    for p in model0.parameters():
        p.requires_grad_(False)
    model0.eval()
    res0 = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
    res0.load_state_dict(torch.load(QNAV_S0, map_location=DEVICE, weights_only=True))
    res0.eval()
    for p in res0.parameters():
        p.requires_grad_(False)
    return model0, res0


def main():
    df = r09.load_e0_df()
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]
    model0, res0 = load_qnav()

    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    def closed_loop(row_idx, state):
        row, obs = Ea[row_idx], T_all[row_idx]
        power = 332.85 if state == "wet" else 464.53
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
                out, h, Tm, rB, m1, m2 = r27.integrate_evap_res(
                    model0, res0, exo, h, Tm, rB, m1, m2, 1)
                mh[t] = float(out[0, 0, 4])
                e = mh[t] - SP
                kp, ti = r27.r22_pi(-e, power)
                integ += e * DT
                u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                v = float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
        h, Tm, rB, m1, m2 = r26.init_states_evap(
            model0, torch.tensor(row, device=DEVICE)[None, :],
            torch.tensor(obs, device=DEVICE)[None, :])
        b = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
                out, h, Tm, rB, m1, m2 = r27.integrate_evap_res(
                    model0, res0, exo, h, Tm, rB, m1, m2, 1)
                b[t] = float(out[0, 0, 4])
        dC = mh - b
        dC_ss = float(np.mean(dC[-60:]))
        return dC / dC_ss, (u, v)

    norm_wet, (uw, vw) = closed_loop(OP_WET, "wet")
    norm_dry, (ud, vd) = closed_loop(OP_DRY, "dry")
    t_axis = np.arange(N) * DT / 60.0

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    ax = axes[0, 0]
    ax.plot(t_axis, norm_wet, lw=1.8, color="#c55a11", label="qnav wet closed loop (SP+2C)")
    ax.plot(t_axis, norm_dry, lw=1.4, color="#8b008b", alpha=0.8,
            label="qnav dry closed loop (unstable, recorded)")
    ax.plot(ANCH_T_MIN, ANCH_Y, "o", color="0.15", ms=8, label="exp_099 event-study mean")
    ax.fill_between([1, 10], -0.3, 2.2, color="0.5", alpha=0.08)
    ax.axhline(1.0, color="0.5", ls=":", lw=0.9)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title("(a) Closed-loop response vs exp_099 anchors (qnav)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Normalized main-steam temp. response")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # B: τ63 journey
    ax = axes[0, 1]
    stages = ["e0", "qnal", "qslow", "FIXB evap", "qnav", "qnavlag"]
    tau_wet = [None, None, 320.0, 1030.0, 480.0, 540.0]
    tau_dry = [140.0, 50.0, 10.0, 1050.0, 670.0, 670.0]
    x = np.arange(len(stages))
    w = 0.38
    def plot_tau(vals, xoff, color, lab):
        yy = [v if v else np.nan for v in vals]
        ax.bar(x + xoff, yy, w, color=color, alpha=0.85, label=lab)
        for xi, v in zip(x + xoff, yy):
            if not np.isnan(v):
                ax.text(xi, min(v, 1080) + 18, f"{v:.0f}", ha="center", fontsize=7.5,
                        color=color)
    plot_tau(tau_wet, -w / 2, "#2e86c1", "wet")
    plot_tau(tau_dry, +w / 2, "#8b008b", "dry")
    ax.axhspan(240, 900, color="#2e8b57", alpha=0.10)
    ax.axhline(240, color="#2e8b57", ls="--", lw=1)
    ax.axhline(900, color="#2e8b57", ls="--", lw=1)
    ax.text(0.05, 910, "G2 window [240, 900] s", color="#2e8b57", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_title("(b) Open-loop τ63 journey (63% response time)")
    ax.set_ylabel("τ63 (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # C: rollout journey
    ax = axes[1, 0]
    stages_c = ["v2 black-box", "e0 grey-box", "qnaw", "qnal", "qslow", "FIXB evap", "qnav"]
    rolls = [2.93, 12.7, 2.570, 2.969, 2.91, 11.33, 2.463]
    colors_c = ["#7f8c8d", "#e74c3c", "#2e86c1", "#2e86c1", "#2e86c1", "#e74c3c", "#2e8b57"]
    bars = ax.bar(np.arange(len(stages_c)), rolls, 0.6, color=colors_c, alpha=0.85)
    for xi, v in zip(np.arange(len(stages_c)), rolls):
        ax.text(xi, v + 0.25, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.axhline(2.93, color="#7f8c8d", ls="--", lw=1)
    ax.text(6.15, 2.98, "v2 baseline", color="#7f8c8d", fontsize=8)
    ax.set_xticks(np.arange(len(stages_c)))
    ax.set_xticklabels(stages_c, fontsize=8.5)
    ax.set_title("(c) 5-h recursive rollout RMSE (main steam, °C)")
    ax.set_ylabel("RMSE (°C)")
    ax.grid(alpha=0.3, axis="y")

    # D: sh1_out bias journey
    ax = axes[1, 1]
    stages_d = ["e0 (instant mix)", "FIXB evap", "qnav"]
    bias = [21.4, 3.07, 2.79]
    colors_d = ["#e74c3c", "#f39c12", "#2e8b57"]
    bars = ax.bar(np.arange(len(stages_d)), bias, 0.55, color=colors_d, alpha=0.85)
    for xi, v in zip(np.arange(len(stages_d)), bias):
        ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(8.0, color="0.3", ls=":", lw=1.2)
    ax.text(0.02, 8.4, "B1 gate: 8°C", color="0.3", fontsize=8)
    ax.set_xticks(np.arange(len(stages_d)))
    ax.set_xticklabels(stages_d, fontsize=9)
    ax.set_title("(d) Wet sh1_out first-step bias (°C)")
    ax.set_ylabel("|bias| (°C)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("qnav final state: physics-first hybrid on the evaporation base "
                 "(rollout 2.46 < v2 2.93; wet τ63 480 s in window; two-phase bias 2.8 °C)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "figs", "fig28_final_qnav.png"), dpi=160)
    plt.close(fig)
    print("[fig] fig28_final_qnav.png")
    # 湿闭环锚点复核 (60/120/180/300/420/600s → 步 6/12/18/30/42/60)
    anch = [float(norm_wet[i - 1]) for i in (6, 12, 18, 30, 42, 60)]
    print("wet anchors:", [round(a, 3) for a in anch])


if __name__ == "__main__":
    main()
