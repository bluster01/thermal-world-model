#!/usr/bin/env python3
"""33_q32_hnow_fig.py: Q32 审计后 h_now (诚实配置) vs double_w 湿态闭环细节图

复用驱动 32 的冻结机器 (integrate/initialize/load_evap_model/pi_parameters),
在驱动自己选的湿态运行点 (F0, row 29771, pm 12.84 MPa) 上复现闭环轨迹,
输出 fig30_q32_hnow.png — 供论文A的干净配置叙事使用。
"""
import importlib.util
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(p, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(os.getcwd(), p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


q32 = _imp("32_qnav_first_principles.py", "q32")
t02 = _imp("02_train.py", "t02")
r09 = _imp("09_residual.py", "r09")
import json

import numpy as np
import torch

DEVICE = q32.DEVICE
OUT = t02.OUT
Q32_ROOT = os.path.join(OUT, "qnav_first_principles")
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])
ANCH_S = np.array([60, 120, 180, 300, 420, 600])


def main():
    matrix = q32.load_matrix(q32.BASE / "configs" / "qnav_first_principles_matrix.json")
    summary = json.load(open(os.path.join(Q32_ROOT, "summary_development.json")))
    # 从 summary 取驱动选定的运行点与 kw
    wet_row, kw_wet, power_mw = None, None, None
    for u in summary["units"]:
        if u["candidate"] == "evap_only" and u["fold"] == "F0":
            d = u["diagnostics"]["operating_points"]["wet"]
            wet_row, kw_wet = int(d["row"]), float(u["diagnostics"]["w_coupling"]["wet"])
            break

    # 按驱动方式读数据 (50000行窗口, 70686起)
    import pandas as pd

    use_columns = list(dict.fromkeys(q32.E0_COLS + t02.OUTPUTS + [q32.POWER_COLUMN]))
    frame = (
        pd.read_csv(t02.CSV, usecols=use_columns, dtype=np.float32)
        .iloc[
            int(matrix["data"]["window_start"]) : int(matrix["data"]["window_start"])
            + int(matrix["data"]["window_rows"])
        ]
        .ffill()
        .bfill()
        .reset_index(drop=True)
    )
    exo, targets = q32.prepare_arrays(frame)
    power_mw = float(frame[q32.POWER_COLUMN].to_numpy(np.float32)[wet_row])

    model = q32.load_evap_model(q32.BASE / "out" / "model_e0_evap_seed0.pt")
    row = exo[wet_row]
    observed = targets[wet_row]

    curves = {}
    for cand_id in ("h_now", "double_w"):
        cand = matrix["candidates"][cand_id]
        fc = 11 if cand["residual_reads_w"] else 10
        res = r09.ResMLP(fc, r09.Q_SCALE).to(DEVICE)
        res.load_state_dict(torch.load(
            os.path.join(Q32_ROOT, f"{cand_id}_F0_s0", "residual_best_validation.pt"),
            map_location=DEVICE, weights_only=True))
        res.eval()
        for p in res.parameters():
            p.requires_grad_(False)
        curves[cand_id] = run_loop(q32, model, res, cand, row, observed,
                                   kw_wet, power_mw, matrix)
        print(f"[{cand_id}] trk_err={curves[cand_id]['tracking_error_c']:.4f} "
              f"tail_std={curves[cand_id]['tail_std_delta_c']:.4f} "
              f"v_rev={curves[cand_id]['valve_reversals']}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    t_min = np.arange(600) / 6.0
    colors = {"h_now": "#2e8b57", "double_w": "#c0392b"}
    for cand_id, c in curves.items():
        delta = c["delta"]
        norm = delta / float(np.mean(delta[-60:]))
        axes[0].plot(t_min, norm, lw=1.4, color=colors[cand_id], label=cand_id)
        axes[1].plot(t_min, c["valve"], lw=1.1, color=colors[cand_id], label=cand_id)
        axes[2].plot(t_min[-120:], delta[-120:], lw=1.1, color=colors[cand_id], label=cand_id)
    axes[0].plot(ANCH_S / 60.0, ANCH_Y, "o", color="0.15", ms=7, label="exp_099 anchors")
    axes[0].axhline(1.0, color="0.5", ls=":", lw=0.8)
    axes[0].set_title("wet closed loop: normalized response (F0 fold, row 29771)")
    axes[0].set_xlabel("time (min)")
    axes[0].set_ylabel("normalized delta main (C)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("valve command history")
    axes[1].set_xlabel("time (min)")
    axes[1].set_ylabel("valve fraction")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[2].set_title("tail 20 min (delta main, C)")
    axes[2].set_xlabel("time (min)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.suptitle("Q32 audit: h_now (no-W, honest) vs double_w (qnav) — wet PI loop, rate limit only",
                 fontsize=11)
    fig.tight_layout()
    out_p = os.path.join(OUT, "figs", "fig30_q32_hnow.png")
    fig.savefig(out_p, dpi=150)
    plt.close(fig)
    print(f"[fig] {out_p}", flush=True)


def run_loop(q32, model, residual, candidate, row, observed, kw, power_mw, matrix):
    steps = int(matrix["interventions"]["steps"])
    rate = float(matrix["interventions"]["valve_rate_per_step"])
    sample_seconds = int(matrix["data"]["sample_seconds"])
    setpoint_delta = float(matrix["interventions"]["setpoint_step_c"])
    constant = torch.tensor(row, device=DEVICE)[None, None, :]
    state = q32.initialize(model, row, observed)
    with torch.no_grad():
        for _ in range(steps):
            output = q32.integrate(model, residual, constant, *state, 1,
                                   candidate["injection"],
                                   bool(candidate["residual_reads_w"]))
            state = output[1:]
    base_state = q32.clone_state(state)
    loop_state = q32.clone_state(state)
    base = np.empty(steps)
    controlled = np.empty(steps)
    valve_history = np.empty(steps)
    valve0 = float(row[6])
    w0 = float(row[8])
    valve = valve0
    command = valve0
    integral = 0.0
    baseline_initial = float(output[0][0, 0, 4])
    setpoint = baseline_initial + setpoint_delta
    with torch.no_grad():
        for index in range(steps):
            base_output = q32.integrate(model, residual, constant, *base_state, 1,
                                        candidate["injection"],
                                        bool(candidate["residual_reads_w"]))
            base[index] = float(base_output[0][0, 0, 4])
            base_state = base_output[1:]
            scenario = constant.clone()
            scenario[0, 0, 6] = valve
            scenario[0, 0, 8] = max(0.0, w0 * (1.0 + kw * (valve - valve0)))
            loop_output = q32.integrate(model, residual, scenario, *loop_state, 1,
                                        candidate["injection"],
                                        bool(candidate["residual_reads_w"]))
            controlled[index] = float(loop_output[0][0, 0, 4])
            loop_state = loop_output[1:]
            error = controlled[index] - setpoint
            kp, ti = q32.pi_parameters(-error, power_mw)
            integral += error * sample_seconds
            command = float(np.clip(valve0 + kp * error + kp / ti * integral, 0.0, 1.0))
            valve = float(np.clip(valve + np.clip(command - valve, -rate, rate), 0.0, 1.0))
            valve_history[index] = valve
    delta = controlled - base
    achieved = float(np.mean(delta[-60:]))
    valve_diff = np.diff(valve_history)
    nonzero = valve_diff[np.abs(valve_diff) > 1e-8]
    reversals = int(np.sum(nonzero[1:] * nonzero[:-1] < 0)) if len(nonzero) > 1 else 0
    return {
        "delta": delta, "valve": valve_history,
        "tracking_error_c": float(abs(achieved - setpoint_delta)),
        "tail_std_delta_c": float(np.std(delta[-120:])),
        "valve_reversals": reversals,
    }


if __name__ == "__main__":
    main()
