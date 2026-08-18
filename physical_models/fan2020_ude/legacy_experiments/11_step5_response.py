#!/usr/bin/env python3
"""11_step5_response.py: Step⑤ 动态响应域验证 (2026-08-17, 设计稿 STEP5_DESIGN.md 已确认)

对齐协议 + 差分阶跃响应 (step − baseline), qh seed0, 湿/干各1代表工况点。
通道: A=v2+5%开环 / B=v1+5%开环 / C=闭环SP+2°C (POU107真实PI: Kp=FX44(dev)×FX49(P), Ti=FX45(dev)×FX50(P), Td=0, 反作用)
预注册门槛 (冻结):
  G1: 闭环SP→main 归一化响应 vs exp_099锚点(60s 0.000/120s 0.10/180s 0.17/300s 0.49/420s 0.70/600s 0.97),
      各锚点|Δ|≤0.15 且 600s处≤0.10; 湿干两工况点都要过
  G2: 开环v2→main: K<0(阀开温降), 单调, τ63∈[240,900]s, θ∈[0,300]s
  G3: 湿/干开环增益同号且比值∈[0.3,3]
  G4(审计): v2→sh1_out 首步偏差记录(不设门槛)
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
import pandas as pd
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N_STEPS = 600          # 6000s
DT = 10.0              # s
V2, V1 = 6, 5          # exo 列索引
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0  # 步索引
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])  # exp_099

FX44_X = np.array([-12, -10, -8, -5, -3, 3, 5, 8, 10, 12])
FX44_Y = np.array([0.6, 0.6, 0.8, 1.0, 1.2, 1.2, 1.0, 0.8, 0.6, 0.6])
FX45_Y = np.array([800, 650, 550, 450, 350, 350, 450, 550, 650, 800])
FX49_X = np.array([150, 200, 300, 400, 500, 600])
FX49_Y = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.5])
FX50_Y = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.6])


def fx(x, xs, ys):
    return float(np.interp(x, xs, ys))


def pi_params(dev, power):
    kp = fx(abs(dev), FX44_X, FX44_Y) * fx(power, FX49_X, FX49_Y)
    ti = fx(abs(dev), FX44_X, FX45_Y) * fx(power, FX49_X, FX50_Y)
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


def run_const(model0, res, row, h, Tm, rB, steps, d_v1=0.0, d_v2=0.0):
    exo = torch.tensor(row, device=DEVICE)[None, None, :].repeat(1, steps, 1)
    exo[:, :, V1] += d_v1
    exo[:, :, V2] += d_v2
    with torch.no_grad():
        out, *_ = r09.integrate_res(model0, res, exo, h, Tm, rB, steps, None, "qh")
    return out[0].cpu().numpy()  # (steps,5)


def run_closed_loop(model0, res, row, h, Tm, rB, steps, SP, power):
    """反作用PI: e = main − SP; u = u0 + Kp·e + (Kp/Ti)·Σe·dt; v2=clip(u,0,1)。"""
    u0 = float(row[V2])
    u, integ = u0, 0.0
    main_hist = np.zeros(steps)
    u_hist = np.zeros(steps)
    with torch.no_grad():
        for t in range(steps):
            exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
            exo[0, 0, V2] = u
            out, h, Tm, rB, hm1, hm2 = r09.integrate_res(
                model0, res, exo, h, Tm, rB, 1, None, "qh")
            main = float(out[0, 0, 4])
            e = main - SP
            kp, ti = pi_params(-e, power)  # dev = SP − main = −e
            integ += e * DT
            u = np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0)
            main_hist[t] = main
            u_hist[t] = u
    return main_hist, u_hist


def fopdt(dT):
    K = float(np.mean(dT[-60:]))
    if abs(K) < 1e-6:
        return {"K": 0.0, "tau63": None, "theta": None, "monotonic": None}
    theta = int(np.argmax(np.abs(dT) >= 0.02 * abs(K)))
    if np.abs(dT).max() < 0.02 * abs(K):
        theta = None
    thr = 0.63 * K
    idx = np.where(dT <= thr)[0] if K < 0 else np.where(dT >= thr)[0]
    tau63 = int(idx[0]) if len(idx) else None
    post = dT[theta: ] if theta is not None else dT
    mono = bool(np.all(np.diff(post) <= 0)) if K < 0 else bool(np.all(np.diff(post) >= 0))
    return {"K": round(K, 3), "tau63": tau63, "theta": theta, "monotonic": mono}


def main():
    df = r09.load_e0_df()
    pow_df = pd.read_csv(t02.CSV, usecols=["机组负荷"], dtype=np.float32) \
        .iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)
    model0 = r09.load_e0(0)
    res = r09.ResMLP(13, r09.Q_SCALE).to(DEVICE)
    res.load_state_dict(torch.load(os.path.join(OUT, "model_res_qh_seed0.pt"),
                                   map_location=DEVICE, weights_only=True))
    res.eval()

    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]
    seg = pm_all[START: START + N_STEPS]
    seg_rows = np.arange(START, START + N_STEPS)

    ops = {}
    for mode_name, mask in (("wet", seg <= P_CRIT), ("dry", seg > P_CRIT)):
        cand = seg_rows[mask]
        med = np.median(seg[mask])
        sub = [r for r in cand if abs(seg[r - START] - med) < 0.5]
        if not sub:
            sub = cand
        # 选 main 未来 60 步波动最小的 (避开瞬态)
        vol = {r: float(np.std(np.diff(T_all[r: r + 61, 4]))) for r in sub[:400]}
        row = min(vol, key=vol.get)
        ops[mode_name] = {"row": int(row), "pm": float(pm_all[row]),
                          "power": float(pow_df["机组负荷"].iloc[row]),
                          "main0": float(T_all[row, 4])}
    print("op points:", ops, flush=True)

    summary = {"ops": ops, "anchors_exp099": ANCH_Y.tolist(), "channels": {}, "judge": {}}
    fig_data = {}
    for mode_name, op in ops.items():
        row_idx = op["row"]
        row = Ea[row_idx]
        obs = T_all[row_idx]
        h, Tm, rB = init_state(model0, row, obs)
        base = run_const(model0, res, row, h, Tm, rB, N_STEPS)
        # 注意: run_const 会推进状态 (原地修改 tensor), 每通道重新初始化
        h, Tm, rB = init_state(model0, row, obs)
        trA = run_const(model0, res, row, h, Tm, rB, N_STEPS, d_v2=0.05)
        h, Tm, rB = init_state(model0, row, obs)
        trB = run_const(model0, res, row, h, Tm, rB, N_STEPS, d_v1=0.05)
        dA = trA - base
        dB = trB - base
        # 闭环: SP = 真值 main0 + 2
        h, Tm, rB = init_state(model0, row, obs)
        cl_main, u_hist = run_closed_loop(model0, res, row, h, Tm, rB, N_STEPS,
                                          op["main0"] + 2.0, op["power"])
        dC = cl_main - base[:, 4]
        dC_ss = float(np.mean(dC[-60:]))
        normC = dC / dC_ss
        fA = fopdt(dA[:, 4])
        fB = fopdt(dB[:, 4])
        anch_model = np.array([normC[int(i) - 1] for i in ANCH_T])
        anch_delta = np.abs(anch_model - ANCH_Y)
        sh1o_off = float(base[0, 1] - T_all[row_idx + 1, 1])
        rec = {
            "fopdt_v2_main": fA, "fopdt_v1_main": fB,
            "cl_ss": round(dC_ss, 2), "cl_anchors": [round(x, 3) for x in anch_model],
            "anchor_delta": [round(x, 3) for x in anch_delta],
            "sh1_out_first_step_offset": round(sh1o_off, 2),
            "u_cl_min": round(float(u_hist.min()), 3), "u_cl_max": round(float(u_hist.max()), 3),
        }
        summary["channels"][mode_name] = rec
        fig_data[mode_name] = {"base": base, "dA": dA, "dB": dB,
                               "normC": normC, "u": u_hist, "dC": dC}
        print(f"[{mode_name}] v2->main FOPDT: {fA}", flush=True)
        print(f"[{mode_name}] cl anchors: {rec['cl_anchors']} Δ={rec['anchor_delta']}", flush=True)

    # ---- 判定 ----
    j = {}
    g1_ok = True
    for mode_name in ("wet", "dry"):
        rec = summary["channels"][mode_name]
        anchors_ok = all(d <= 0.15 for d in rec["anchor_delta"]) and rec["anchor_delta"][-1] <= 0.10
        j[f"G1_{mode_name}"] = bool(anchors_ok)
        g1_ok = g1_ok and anchors_ok
    j["G1"] = bool(g1_ok)
    g2_ok = True
    for mode_name in ("wet", "dry"):
        f = summary["channels"][mode_name]["fopdt_v2_main"]
        ok = (f["K"] < 0 and f["monotonic"] is True and f["tau63"] is not None
              and 240 <= f["tau63"] * DT <= 900 and f["theta"] is not None and f["theta"] * DT <= 300)
        j[f"G2_{mode_name}"] = bool(ok)
        g2_ok = g2_ok and ok
    j["G2"] = bool(g2_ok)
    Kw = summary["channels"]["wet"]["fopdt_v2_main"]["K"]
    Kd = summary["channels"]["dry"]["fopdt_v2_main"]["K"]
    ratio = (Kw / Kd) if Kd != 0 else float("inf")
    j["G3"] = bool(Kw * Kd > 0 and 0.3 <= ratio <= 3.0)
    j["G3_ratio"] = round(ratio, 2)
    j["G4_record"] = {m: summary["channels"][m]["sh1_out_first_step_offset"] for m in ("wet", "dry")}
    j["verdict"] = "PASS" if (j["G1"] and j["G2"] and j["G3"]) else "FAIL"
    summary["judge"] = j

    with open(os.path.join(OUT, "step5_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=== Step⑤ 判定 ===")
    print(json.dumps(j, ensure_ascii=False, indent=2), flush=True)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t_axis = np.arange(N_STEPS) * DT / 60.0  # min
    for k, mode_name in enumerate(("wet", "dry")):
        fd = fig_data[mode_name]
        ax = axes[0, k]
        ax.plot(t_axis, fd["dA"][:, 4], color="#8b008b", lw=1.5, label="v2+5% -> main")
        ax.plot(t_axis, fd["dB"][:, 4], color="#2e8b57", lw=1.5, label="v1+5% -> main")
        ax.plot(t_axis, fd["dA"][:, 0], color="#c55a11", lw=1.0, ls="--", label="v2+5% -> sh1_in")
        ax.plot(t_axis, fd["dA"][:, 1], color="#1f77b4", lw=1.0, ls="--", label="v2+5% -> sh1_out")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{mode_name} open-loop step responses (pm={summary['ops'][mode_name]['pm']:.1f})")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("ΔT (°C)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = axes[1, k]
        ax.plot(t_axis, fd["normC"], color="#d62728", lw=1.5, label="closed-loop SP+2 (model)")
        ax.plot(ANCH_T / 6.0, ANCH_Y, "o", color="0.3", ms=7, label="exp_099 measured")
        ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
        ax.axhline(0.63, color="0.5", ls=":", lw=0.8)
        ax.set_title(f"{mode_name} closed-loop normalized (POU107 PI, P={summary['ops'][mode_name]['power']:.0f}MW)")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("normalized ΔT")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Step⑤ dynamic-response validation — verdict={j['verdict']} "
                 f"(G1={j['G1']} G2={j['G2']} G3={j['G3']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig14_step_responses.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig14_step_responses.png", flush=True)


if __name__ == "__main__":
    main()
