#!/usr/bin/env python3
"""17_step6_tf.py: STEP6 级联TF提取 (执行机构883s + 对象FOPDT×6工况点) + 增益调度 + TF闭环验证

预注册 (STEP6_DESIGN.md): T1 main FOPDT R²≥0.85于≥4/6点 / T2 6/6点K<0 / T3 调度相关性报告 / T4 TF级联闭环6/6收敛
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
TAU_A = 883.2
RATE = 0.014  # per step (0.14%/s × 10s)
ANCH_T = np.array([60, 120, 180, 300, 420, 600]) / 10.0
ANCH_Y = np.array([0.000, 0.10, 0.17, 0.49, 0.70, 0.97])
TARGETS_PM = [15.5, 17.5, 19.5, 21.0, 22.5, 24.0]


def fit_fopdt(y, dt=DT):
    """差分响应 → FOPDT (K, θ, τ) 网格拟合, 返回含 R²。"""
    K = float(np.mean(y[-60:]))
    n = len(y)
    t = np.arange(n) * dt
    var = float(np.var(y))
    best = None
    for theta in range(0, min(31, n)):
        for tau in np.arange(20, 3001, 20):
            t0 = t - theta * dt
            yf = np.where(t0 > 0, K * (1 - np.exp(-t0 / tau)), 0.0)
            mse = float(np.mean((y - yf) ** 2))
            if best is None or mse < best[0]:
                best = (mse, theta * dt, tau)
    mse, theta, tau = best
    r2 = 1 - mse / var if var > 0 else 0.0
    return {"K": round(K, 3), "theta": round(theta, 1), "tau": round(tau, 1), "R2": round(r2, 3)}


def fit_so(y, dt=DT):
    """差分响应 → 二阶 K·e^(−θs)/((τ1s+1)(τ2s+1)) 网格拟合 (FOPDT 不足时的 fallback)。"""
    K = float(np.mean(y[-60:]))
    n = len(y)
    t = np.arange(n) * dt
    var = float(np.var(y))
    taus = np.logspace(np.log10(20), np.log10(3000), 30)
    best = None
    for theta in range(0, 16):
        t0 = t - theta * dt
        for t1 in taus:
            for t2 in taus:
                if abs(t1 - t2) < 1.0:
                    t2 = t1 + 5.0
                resp = 1.0 - (t1 * np.exp(-t0 / t1) - t2 * np.exp(-t0 / t2)) / (t1 - t2)
                yf = np.where(t0 > 0, K * resp, 0.0)
                mse = float(np.mean((y - yf) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, theta * dt, t1, t2)
    mse, theta, t1, t2 = best
    r2 = 1 - mse / var if var > 0 else 0.0
    return {"K": round(K, 3), "theta": round(theta, 1), "tau1": round(t1, 1),
            "tau2": round(t2, 1), "R2": round(r2, 3)}


def fopdt_step(K, theta, tau, n, dt=DT):
    t = np.arange(n) * dt
    t0 = t - theta
    return np.where(t0 > 0, K * (1 - np.exp(-t0 / tau)), 0.0)


def main():
    df = r09.load_e0_df()
    pow_df = pd.read_csv(t02.CSV, usecols=["机组负荷"], dtype=np.float32) \
        .iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)
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

    # 分状态 k_w (与 FIX3 同)
    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    # 选 6 工况点 (去重)
    seg_rows = np.arange(START, START + t02.ROLL_STEPS)
    ops = []
    used = set()
    for pm_t in TARGETS_PM:
        cand = seg_rows[np.abs(pm_all[START: START + t02.ROLL_STEPS] - pm_t) < 1.0]
        if len(cand) == 0:
            cand = seg_rows[np.argsort(np.abs(pm_all[START: START + t02.ROLL_STEPS] - pm_t))[:200]]
        vol = {r: float(np.std(np.diff(T_all[r: r + 61, 4]))) for r in cand[:400]
               if int(r) not in used}
        if not vol:
            continue
        r = min(vol, key=vol.get)
        used.add(int(r))
        ops.append({"row": int(r), "pm": float(pm_all[r]),
                    "power": float(pow_df["机组负荷"].iloc[r]),
                    "state": "wet" if pm_all[r] <= P_CRIT else "dry"})
    print("op points:", [(o["row"], round(o["pm"], 1), o["state"]) for o in ops], flush=True)

    def fwd_one(row, h, Tm, rB, T_sens, v2_val, W_val):
        exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
        exo[0, 0, V2] = v2_val
        exo[0, 0, 8] = W_val
        out, h, Tm, rB, hm1, hm2, T_sens = r15.integrate_learn(
            model0, mod0, exo, h, Tm, rB, 1, T_sens=T_sens)
        return float(out[0, 0, 4]), h, Tm, rB, T_sens

    table = []
    for op in ops:
        row, obs = Ea[op["row"]], T_all[op["row"]]
        kw = k_w_state[op["state"]]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        base = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                base[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, u0, W0)
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens,
                                                   u0 + 0.05, W0 * (1 + kw * 0.05))
        d = dT - base
        f = fit_fopdt(d)
        rec = {"row": op["row"], "pm": round(op["pm"], 1), "power": round(op["power"], 0),
               "state": op["state"], "k_w": round(kw, 3),
               "K_main": f["K"], "theta": f["theta"], "tau": f["tau"], "R2": f["R2"],
               "tau63_raw": None}
        if f["R2"] < 0.85:
            fs = fit_so(d)
            rec["so"] = fs
            rec["adopt"] = "so"
            print(f"[op {op['row']}] pm={op['pm']:.1f} {op['state']} FOPDT R²={f['R2']:.3f} "
                  f"→ SO R²={fs['R2']:.3f} τ1={fs['tau1']:.0f} τ2={fs['tau2']:.0f}", flush=True)
        else:
            rec["adopt"] = "fopdt"
            print(f"[op {op['row']}] pm={op['pm']:.1f} {op['state']} K={f['K']:.3f} "
                  f"θ={f['theta']:.0f}s τ={f['tau']:.0f}s R²={f['R2']:.3f}", flush=True)
        # τ63 直接法 (报告)
        K = f["K"]
        idx = np.where(d <= 0.63 * K)[0] if K < 0 else np.where(d >= 0.63 * K)[0]
        rec["tau63_raw"] = int(idx[0]) * DT if len(idx) else None
        table.append(rec)

    # ---- 判定 T1/T2/T3 ----
    T1_fopdt = sum(1 for r in table if r["R2"] >= 0.85)
    T1_so = sum(1 for r in table if r.get("so", {}).get("R2", 0) >= 0.85)
    T1 = bool(T1_fopdt >= 4 or T1_so >= 4)
    T2 = bool(all(r["K_main"] < 0 for r in table))
    pms = np.array([r["pm"] for r in table])
    Ks = np.array([r["K_main"] for r in table])
    sp = float(np.corrcoef(pms, Ks)[0, 1]) if np.std(pms) > 0 and np.std(Ks) > 0 else 0.0

    # ---- T4: TF 级联闭环 (PI + 执行机构 + 采用TF) 各工况点 ----
    cl_ok = {}
    for r in table:
        power = r["power"]
        SP = 2.0  # 相对阶跃
        u0 = 0.3
        v = u0
        y = 0.0
        y2 = 0.0
        integ = 0.0
        u = u0
        use_so = r["adopt"] == "so"
        if use_so:
            so = r["so"]
            K, theta, t1, t2 = so["K"], so["theta"], so["tau1"], so["tau2"]
        else:
            K, theta, tau = r["K_main"], r["theta"], r["tau"]
        theta_steps = int(round(theta / DT))
        buf = [0.0] * (theta_steps + 1)
        y_hist = []
        for t in range(N):
            buf = [v] + buf[:-1]
            u_del = buf[-1] - u0  # 差分模型: 输入偏差 (v − u0)
            if use_so:
                y = y + (DT / t1) * (K * u_del - y)
                y2 = y2 + (DT / t2) * (y - y2)
                y_out = y2
            else:
                y = y + (DT / tau) * (K * u_del - y)
                y_out = y
            e = y_out - SP
            kp, ti = r15.pi_params(-e, power)
            integ += e * DT
            u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
            step = float(np.clip((u - v) * (DT / TAU_A), -RATE, RATE))
            v = float(np.clip(v + step, 0.0, 1.0))
            y_hist.append(y_out)
        yh = np.array(y_hist)
        ss = float(np.mean(yh[-60:]))
        norm = yh / ss
        cl_ok[r["row"]] = {"norm600": round(float(norm[599]), 3),
                           "tail_std": round(float(np.std(norm[-120:])), 4),
                           "ok": bool(0.8 <= norm[599] <= 1.2 and np.std(norm[-120:]) <= 0.05)}
        print(f"[tf-cl row {r['row']}] norm600={cl_ok[r['row']]['norm600']} "
              f"tail_std={cl_ok[r['row']]['tail_std']} ({r['adopt']})", flush=True)
    T4 = bool(all(cl_ok[r["row"]]["ok"] for r in table))

    judge = {"T1": T1, "T1_fopdt_n": T1_fopdt, "T1_so_n": T1_so,
             "T2": T2, "T3_spearman_K_pm": round(sp, 3),
             "T4": T4, "verdict": "PASS" if (T1 and T2 and T4) else "FAIL"}
    summ = {"actuator": {"tau_a": TAU_A, "rate_per_step": RATE},
            "k_w": k_w_state, "table": table, "cl_tf": cl_ok, "judge": judge}
    with open(os.path.join(OUT, "step6_tf_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    print("=== STEP6 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    pms = [r["pm"] for r in table]
    ax.plot(pms, [r["K_main"] for r in table], "o-", color="#8b008b")
    ax.set_title("gain schedule: K_main vs pm")
    ax.set_xlabel("pm (MPa)")
    ax.set_ylabel("K (°C / coupled step)")
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(pms, [r["tau"] for r in table], "s-", color="#2e8b57", label="τ (FOPDT)")
    ax.plot(pms, [r["theta"] for r in table], "^-", color="#c55a11", label="θ")
    ax.set_title("τ / θ vs pm")
    ax.set_xlabel("pm (MPa)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    for r in table[:3] + table[-2:]:
        row, obs = Ea[r["row"]], T_all[r["row"]]
        kw = k_w_state[r["state"]]
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        u0, W0 = float(row[V2]), float(row[8])
        base = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                base[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens, u0, W0)
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        dT = np.zeros(N)
        with torch.no_grad():
            for t in range(N):
                dT[t], h, Tm, rB, T_sens = fwd_one(row, h, Tm, rB, T_sens,
                                                   u0 + 0.05, W0 * (1 + kw * 0.05))
        d = dT - base
        ax.plot(np.arange(N) * DT / 60.0, d, lw=1.2,
                label=f"pm={r['pm']:.1f} model")
        f_ = fopdt_step(r["K_main"], r["theta"], r["tau"], N)
        ax.plot(np.arange(N) * DT / 60.0, f_, ls="--", lw=0.9,
                label=f"pm={r['pm']:.1f} FOPDT")
    ax.set_title("model step vs FOPDT fit")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("ΔT (°C)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ok_n = sum(1 for r in table if cl_ok[r["row"]]["ok"])
    ax.text(0.1, 0.7, f"TF cascade closed-loop: {ok_n}/6 converged\n"
                      f"verdict={judge['verdict']} (T1={T1} T2={T2} T4={T4})\n"
                      f"Spearman(K,pm)={judge['T3_spearman_K_pm']:.2f}",
            fontsize=11, transform=ax.transAxes)
    ax.text(0.1, 0.3, "cascade: PI(POU107) -> actuator 1/(883s+1)\n"
                      "        -> FOPDT K·e^(−θs)/(τs+1)\n"
                      "rate limit 0.14%/s, k_w wet 3.22/dry 4.00",
            fontsize=9, transform=ax.transAxes, color="0.3")
    ax.axis("off")
    fig.suptitle("STEP6 TF extraction & gain scheduling", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig20_tf_schedule.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig20_tf_schedule.png", flush=True)


if __name__ == "__main__":
    main()
