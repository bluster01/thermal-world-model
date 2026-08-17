#!/usr/bin/env python3
"""30_fix_deadzone.py: 闭环协议补死区 — 回应"高频震荡"观察

证据: 真实指令 99.1% 步 |Δ|≤0.1% (基本冻结); 仿真 PI 每步都动阀 (0.0137/步蠕变)。
死区 (偏差 |e|<dz 时 u/integ 保持) 是真实 DCS 行为, 协议缺失导致高频狩猎。
基线量化 + dz∈{0.5, 1.0}°C 敏感性 (FX44 首个断点 ±3°C, 死区必然 ≤1°C)。

预注册 (冻结 2026-08-17):
  DZ1: 死区后指令更新占比 ≤5% (基线 100%, 真实 ~0.9%)
  DZ2: 尾段高频震荡指标下降: tail |Δmain| 中位数 ≤ 无死区基线
  DZ3: 前 3 锚点保持 ≤0.15 且 300s/420s 不恶化 (vs 无死区基线)
  DZ4: 收敛保持 norm600∈[0.8,1.2] ∧ tail_std≤0.05
裁决: DZ2∧DZ3∧DZ4 = PASS (任一 dz 通过即可, 取最优 dz 为最终协议)
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

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
OP_WET = 40161
RATE = 0.0137
ANCH_STEPS = (6, 12, 18, 30, 42, 60)
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

    warm0 = torch.load(os.path.join(OUT, "model_e0_seed0.pt"), map_location=DEVICE,
                       weights_only=True)
    model0 = r26.E0Evap(warm0).to(DEVICE)
    model0.load_state_dict(torch.load(os.path.join(OUT, "model_e0_evap_seed0.pt"),
                                      map_location=DEVICE, weights_only=True))
    for p in model0.parameters():
        p.requires_grad_(False)
    model0.eval()
    res0 = r09.ResMLP(11, r09.Q_SCALE).to(DEVICE)
    res0.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnav_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    res0.eval()
    for p in res0.parameters():
        p.requires_grad_(False)

    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    sub = Ea[START: START + t02.ROLL_STEPS][pm_seg <= P_CRIT]
    A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
    coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
    k_w = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    row, obs = Ea[OP_WET], T_all[OP_WET]
    u0, W0 = float(row[V2]), float(row[8])
    SP = float(obs[4]) + 2.0
    power = 332.85

    def run_loop(dz):
        h, Tm, rB, m1, m2 = r26.init_states_evap(
            model0, torch.tensor(row, device=DEVICE)[None, :],
            torch.tensor(obs, device=DEVICE)[None, :])
        u, integ, v = u0, 0.0, u0
        mh = np.zeros(N)
        u_hist = np.zeros(N)
        v_hist = np.zeros(N)
        u_moves = 0
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
                exo[0, 0, V2] = v
                exo[0, 0, 8] = W0 * (1 + k_w * (v - u0))
                out, h, Tm, rB, m1, m2 = r27.integrate_evap_res(
                    model0, res0, exo, h, Tm, rB, m1, m2, 1)
                mh[t] = float(out[0, 0, 4])
                e = mh[t] - SP
                if abs(e) > dz:
                    kp, ti = r27.r22_pi(-e, power)
                    integ += e * DT
                    u_new = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                    if abs(u_new - u) > 1e-4:
                        u_moves += 1
                    u = u_new
                v = float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
                u_hist[t] = u
                v_hist[t] = v
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
        norm = dC / dC_ss
        anch = [float(norm[i - 1]) for i in ANCH_STEPS]
        deltas = [round(abs(a - y), 3) for a, y in zip(anch, ANCH_Y)]
        dmain_tail = np.abs(np.diff(mh[-120:]))
        v_rev = int(np.sum(np.diff(np.sign(np.diff(v_hist))) != 0))
        return {
            "anchors": [round(a, 3) for a in anch], "deltas": deltas,
            "norm600": round(float(norm[599]), 3),
            "tail_std": round(float(np.std(norm[-120:])), 4),
            "u_move_frac": round(u_moves / N, 4),
            "dmain_tail_med": round(float(np.median(dmain_tail)), 4),
            "dmain_tail_p95": round(float(np.percentile(dmain_tail, 95)), 4),
            "valve_reversals": v_rev,
            "mh": mh, "norm": norm,
        }

    results = {}
    for dz in (0.0, 0.5, 1.0):
        r = run_loop(dz)
        r.pop("mh")
        r.pop("norm")
        results[str(dz)] = r
        print(f"[dz={dz}] anchors={r['anchors']} Δ={r['deltas']} norm600={r['norm600']} "
              f"tail_std={r['tail_std']} u_move={r['u_move_frac']*100:.1f}% "
              f"dmain_tail_med={r['dmain_tail_med']} p95={r['dmain_tail_p95']} "
              f"v_rev={r['valve_reversals']}", flush=True)

    base = results["0.0"]
    best = None
    for dz in ("0.5", "1.0"):
        r = results[dz]
        DZ2 = r["dmain_tail_med"] <= base["dmain_tail_med"]
        DZ3 = (all(d <= 0.15 for d in r["deltas"][:3])
               and r["deltas"][3] <= base["deltas"][3] + 0.05
               and r["deltas"][4] <= base["deltas"][4] + 0.05)
        DZ4 = 0.8 <= r["norm600"] <= 1.2 and r["tail_std"] <= 0.05
        DZ1 = r["u_move_frac"] <= 0.05
        ok = DZ2 and DZ3 and DZ4
        results[dz + "_judge"] = {"DZ1": DZ1, "DZ2": DZ2, "DZ3": DZ3, "DZ4": DZ4,
                                  "pass": ok}
        if ok and best is None:
            best = dz
        print(f"[dz={dz} judge] DZ1={DZ1} DZ2={DZ2} DZ3={DZ3} DZ4={DZ4} PASS={ok}", flush=True)
    results["best_dz"] = best
    results["verdict"] = "PASS" if best else "FAIL"
    with open(os.path.join(OUT, "deadzone_summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"=== 湿态死区判定: verdict={results['verdict']} best_dz={best} ===", flush=True)

    # ---- 干态: 死区的真正用武之地 (qnav 干态闭环震荡 tail_std=1.85) ----
    sub_d = Ea[START: START + t02.ROLL_STEPS][pm_seg > P_CRIT]
    A_d = np.stack([sub_d[:, 5], sub_d[:, 6], np.ones(len(sub_d))], 1)
    coef_d, _, _, _ = np.linalg.lstsq(A_d, sub_d[:, 8], rcond=None)
    k_wd = float(np.clip(coef_d[1] / np.mean(sub_d[:, 8]), 0.5, 4.0))
    row_d, obs_d = Ea[40437], T_all[40437]
    u0d, W0d = float(row_d[V2]), float(row_d[8])
    SPd = float(obs_d[4]) + 2.0
    power_d = 464.53

    def run_loop_dry(dz):
        h, Tm, rB, m1, m2 = r26.init_states_evap(
            model0, torch.tensor(row_d, device=DEVICE)[None, :],
            torch.tensor(obs_d, device=DEVICE)[None, :])
        u, integ, v = u0d, 0.0, u0d
        mh = np.zeros(N)
        v_hist = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row_d, device=DEVICE)[None, None, :].clone()
                exo[0, 0, V2] = v
                exo[0, 0, 8] = W0d * (1 + k_wd * (v - u0d))
                out, h, Tm, rB, m1, m2 = r27.integrate_evap_res(
                    model0, res0, exo, h, Tm, rB, m1, m2, 1)
                mh[t] = float(out[0, 0, 4])
                e = mh[t] - SPd
                if abs(e) > dz:
                    kp, ti = r27.r22_pi(-e, power_d)
                    integ += e * DT
                    u = float(np.clip(u0d + kp * e + (kp / ti) * integ, 0.0, 1.0))
                v = float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
                v_hist[t] = v
        h, Tm, rB, m1, m2 = r26.init_states_evap(
            model0, torch.tensor(row_d, device=DEVICE)[None, :],
            torch.tensor(obs_d, device=DEVICE)[None, :])
        b = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row_d, device=DEVICE)[None, None, :].clone()
                out, h, Tm, rB, m1, m2 = r27.integrate_evap_res(
                    model0, res0, exo, h, Tm, rB, m1, m2, 1)
                b[t] = float(out[0, 0, 4])
        dC = mh - b
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        v_rev = int(np.sum(np.diff(np.sign(np.diff(v_hist))) != 0))
        dmain_tail = np.abs(np.diff(mh[-120:]))
        return {"norm": norm, "norm600": round(float(norm[599]), 3),
                "tail_std": round(float(np.std(norm[-120:])), 4),
                "v_rev": v_rev,
                "dmain_tail_med": round(float(np.median(dmain_tail)), 4),
                "dmain_tail_p95": round(float(np.percentile(dmain_tail, 95)), 4)}

    dry_res = {}
    for dz in (0.0, 0.5, 1.0):
        r = run_loop_dry(dz)
        r.pop("norm")
        dry_res[str(dz)] = r
        print(f"[dry dz={dz}] norm600={r['norm600']} tail_std={r['tail_std']} "
              f"v_rev={r['v_rev']} dmain_tail_med={r['dmain_tail_med']} "
              f"p95={r['dmain_tail_p95']}", flush=True)
    results["dry"] = dry_res
    dry_base = dry_res["0.0"]
    dry_pass = None
    for dz in ("0.5", "1.0"):
        r = dry_res[dz]
        DD1 = 0.8 <= r["norm600"] <= 1.2 and r["tail_std"] <= 0.05
        DD2 = r["v_rev"] <= dry_base["v_rev"]
        DD3 = r["dmain_tail_med"] <= dry_base["dmain_tail_med"]
        ok = DD1 and DD2 and DD3
        dry_res[dz + "_judge"] = {"DD1": DD1, "DD2": DD2, "DD3": DD3, "pass": ok}
        if ok and dry_pass is None:
            dry_pass = dz
        print(f"[dry dz={dz} judge] DD1={DD1} DD2={DD2} DD3={DD3} PASS={ok}", flush=True)
    results["dry_best_dz"] = dry_pass
    with open(os.path.join(OUT, "deadzone_summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"=== 干态死区: best_dz={dry_pass} ===", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    t_axis = np.arange(N) * DT / 60.0
    ax = axes[0]
    for dz, color in (("0.0", "crimson"), ("0.5", "#2e86c1"), ("1.0", "#2e8b57")):
        r = run_loop(float(dz))
        ax.plot(t_axis, r["norm"], lw=1.3, color=color, label=f"dz={dz}°C")
    ax.plot(np.array(ANCH_STEPS) / 6.0, ANCH_Y, "o", color="0.15", ms=7, label="exp_099")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_xlim(0, 100)
    ax.set_title("wet closed loop with/without dead zone")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("normalized response")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for dz, color in (("0.0", "crimson"), ("0.5", "#2e86c1"), ("1.0", "#2e8b57")):
        r = run_loop(float(dz))
        mh = r["mh"]
        ax.plot(t_axis[-120:], mh[-120:], lw=1.1, color=color, label=f"dz={dz}°C")
    ax.set_title("tail 20 min: high-frequency zoom")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("main temp (°C)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"dead-zone protocol fix — verdict={results['verdict']} "
                 f"best dz={best}°C", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig29_deadzone.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig29_deadzone.png", flush=True)


if __name__ == "__main__":
    main()
