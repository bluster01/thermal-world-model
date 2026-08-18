#!/usr/bin/env python3
"""31_deadzone_retro.py: 死区回溯测试 — 在 fig14-22 的小幅震荡配置上验证死区效果

用户观察: fig28 干态=发散(失稳, 非死区范畴); fig14-22 的小幅尾段震荡才是死区该管的事。
选两个代表性配置 (checkpoint 均在):
  (a) FIX4 时代 qnal 湿态闭环 (rate-only 执行机构, tail_std≈0.02) — fig19
  (b) qgate 干态闭环 (tail_std≈0.0058) — fig25
每配置测 dz∈{0, 1.0}°C, 对比尾段震荡指标与指令行为。

预注册 (冻结 2026-08-17):
  W1: 尾段震荡下降: dmain_tail_med 与 p95 均 ≤ 无死区基线 (严格下降)
  W2: 收敛保持: norm600∈[0.8,1.2] ∧ tail_std 不恶化
  W3: 指令更新占比下降且 ≤5% (向真实 0.9% 靠拢)
裁决: 每配置 W1∧W2∧W3 = 该配置死区有效
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
r22 = _imp("22_fix_slowdyn.py", "r22")
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
RATE = 0.0137


def load_base_models():
    model0 = r09.load_e0(0)
    # (a) qnal (FIX3 可学习滞后, 无残差滞后)
    qnal = r15.QnaLag().to(DEVICE)
    qnal.load_state_dict(torch.load(os.path.join(OUT, "model_res_qnal_seed0.pt"),
                                    map_location=DEVICE, weights_only=True))
    qnal.eval()
    for p in qnal.parameters():
        p.requires_grad_(False)
    # (b) qgate (段门控 z=[1,0,0])
    qgate = r22.QnaLagSlow().to(DEVICE)
    qgate.load_state_dict(torch.load(os.path.join(OUT, "model_res_qgate_seed0.pt"),
                                     map_location=DEVICE, weights_only=True))
    qgate.eval()
    for p in qgate.parameters():
        p.requires_grad_(False)
    return model0, qnal, qgate


def main():
    df = r09.load_e0_df()
    E = df[r09.E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    Ea = E.to_numpy(np.float32)
    T_all = df[t02.OUTPUTS].to_numpy(np.float32)
    pm_all = Ea[:, 2]
    model0, qnal, qgate = load_base_models()

    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    def run_loop(fwd, init_fn, row, obs, u0, W0, kw, SP, power, dz):
        h, Tm, rB = init_fn(model0, row, obs)
        u, integ, v = u0, 0.0, u0
        mh = np.zeros(N)
        u_moves = 0
        with torch.no_grad():
            for t in range(N):
                mh[t] = fwd(row, h, Tm, rB, v, W0 * (1 + kw * (v - u0)))
                e = mh[t] - SP
                if abs(e) > dz:
                    kp, ti = r22.pi_params(-e, power)
                    integ += e * DT
                    u_new = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
                    if abs(u_new - u) > 1e-4:
                        u_moves += 1
                    u = u_new
                v = float(np.clip(v + np.clip(u - v, -RATE, RATE), 0.0, 1.0))
        h, Tm, rB = init_fn(model0, row, obs)
        b = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                b[t] = fwd(row, h, Tm, rB, u0, W0)
        dC = mh - b
        dC_ss = float(np.mean(dC[-60:]))
        norm = dC / dC_ss
        dmain_tail = np.abs(np.diff(mh[-120:]))
        return {"norm600": round(float(norm[599]), 3),
                "tail_std": round(float(np.std(norm[-120:])), 4),
                "u_move_frac": round(u_moves / N, 4),
                "dmain_tail_med": round(float(np.median(dmain_tail)), 4),
                "dmain_tail_p95": round(float(np.percentile(dmain_tail, 95)), 4)}

    # ---- (a) qnal 湿态闭环 (fig19 配置) ----
    row_w, obs_w = Ea[OP_WET], T_all[OP_WET]
    u0w, W0w = float(row_w[V2]), float(row_w[8])
    SPw = float(obs_w[4]) + 2.0

    def init_qnal(model0, row, obs):
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        return h, Tm, rB

    def fwd_qnal(row, h, Tm, rB, v2v, Wv):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = v2v
        exo[0, 0, 8] = Wv
        out, h2, Tm2, rB2, hm1, hm2, T_sens = r15.integrate_learn(
            model0, qnal, exo, h, Tm, rB, 1, T_sens=None)
        return float(out[0, 0, 4])

    results = {}
    for dz in (0.0, 1.0):
        r = run_loop(fwd_qnal, init_qnal, row_w, obs_w, u0w, W0w,
                     k_w_state["wet"], SPw, 332.85, dz)
        results[f"qnal_wet_dz{dz}"] = r
        print(f"[qnal wet dz={dz}] {r}", flush=True)
    b0 = results["qnal_wet_dz0.0"]
    b1 = results["qnal_wet_dz1.0"]
    W1 = b1["dmain_tail_med"] <= b0["dmain_tail_med"] and b1["dmain_tail_p95"] <= b0["dmain_tail_p95"]
    W2 = 0.8 <= b1["norm600"] <= 1.2 and b1["tail_std"] <= b0["tail_std"]
    W3 = b1["u_move_frac"] <= 0.05 and b1["u_move_frac"] <= b0["u_move_frac"]
    results["qnal_wet_judge"] = {"W1": W1, "W2": W2, "W3": W3, "pass": W1 and W2 and W3}
    print(f"[qnal wet judge] W1={W1} W2={W2} W3={W3} PASS={W1 and W2 and W3}", flush=True)

    # ---- (b) qgate 干态闭环 (fig25 配置) ----
    row_d, obs_d = Ea[OP_DRY], T_all[OP_DRY]
    u0d, W0d = float(row_d[V2]), float(row_d[8])
    SPd = float(obs_d[4]) + 2.0

    def init_qgate(model0, row, obs):
        h, Tm, rB = r22.init_state(model0, row, obs)
        return h, Tm, rB

    def fwd_qgate(row, h, Tm, rB, v2v, Wv):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = v2v
        exo[0, 0, 8] = Wv
        out, h2, Tm2, rB2, hm1, hm2, T_sens, z_lag = r22.integrate_slow(
            model0, qgate, exo, h, Tm, rB, 1, T_sens=None, z_lag=None,
            z_mask=(1.0, 0.0, 0.0))
        return float(out[0, 0, 4])

    for dz in (0.0, 0.3, 1.0):
        r = run_loop(fwd_qgate, init_qgate, row_d, obs_d, u0d, W0d,
                     k_w_state["dry"], SPd, 464.53, dz)
        results[f"qgate_dry_dz{dz}"] = r
        print(f"[qgate dry dz={dz}] {r}", flush=True)
    b0 = results["qgate_dry_dz0.0"]
    for dz in (0.3, 1.0):
        b1 = results[f"qgate_dry_dz{dz}"]
        W1 = b1["dmain_tail_med"] <= b0["dmain_tail_med"] and b1["dmain_tail_p95"] <= b0["dmain_tail_p95"]
        W2 = (np.isfinite(b1["norm600"]) and 0.8 <= b1["norm600"] <= 1.2
              and b1["tail_std"] <= b0["tail_std"])
        W3 = b1["u_move_frac"] <= 0.05 and b1["u_move_frac"] <= b0["u_move_frac"]
        results[f"qgate_dry_dz{dz}_judge"] = {"W1": W1, "W2": W2, "W3": W3,
                                              "pass": W1 and W2 and W3}
        print(f"[qgate dry dz={dz} judge] W1={W1} W2={W2} W3={W3} PASS={W1 and W2 and W3}",
              flush=True)

    with open(os.path.join(OUT, "deadzone_retro_summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("=== deadzone retro done ===", flush=True)


if __name__ == "__main__":
    main()
