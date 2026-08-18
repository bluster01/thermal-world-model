#!/usr/bin/env python3
"""18_step6b_lin.py: STEP6b 精确线性化 — qnal 模型逐工况点 Jacobian → 局部状态空间 → TF

背景: FOPDT/二阶拟合全部 R²<0.85 (T1 FAIL) — 模型阶跃响应含过冲, 单调滞后族表达不了。
正路: 模型可微 → autograd Jacobian → 14维局部状态空间 (h×3, Tm×3, rB, Dsw_lag×2, T_sens×5),
输入 = v2 (W 耦合内嵌: W = W0(1+k_w·Δv2)), 输出 = T_sens 5 通道。

预注册 (冻结 2026-08-17):
  L1: LTI 阶跃 vs 非线性差分阶跃 R² ≥ 0.95 于 ≥4/6 工况点 (线性化保真)
  L2: 级联 (POU107 PI + 执行机构883s + LTI) 闭环 6/6 收敛 (norm600∈[0.8,1.2] ∧ tail_std≤0.05)
  L3: LTI 特征值 |λ|<1 于全部工况点 (局部稳定, 内模可用)
产物: out/step6b_lin_summary.json, out/figs/fig21_linearized.png
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
import torch.autograd.functional as AF

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT
N = 600
DT = 10.0
V2 = 6
TAU_A = 883.2
RATE = 0.014
TARGETS_PM = [15.5, 17.5, 19.5, 21.0, 22.5, 24.0]
N_STATE = 14


def make_step_fn(model0, mod, row, W0, u0, kw):
    """返回 f(x, dv2) → x⁺ 的 14 维一步映射 (W 耦合内嵌)。x 顺序: h(3), Tm(3), rB, Dsw1l, Dsw2l, T_sens(5)。"""
    M = model0.tri("M")[:, None]
    UA = model0.tri("UA")[:, None]
    Cm = model0.tri("Cm")[:, None]
    tauB = model0.val("tauB")
    tau_sw, tau_sens = mod.tau()
    a_sw = DT / tau_sw
    a_sens = DT / tau_sens
    D0 = torch.tensor([row[0]], dtype=torch.float32, device=DEVICE)
    uB0 = torch.tensor([row[1]], dtype=torch.float32, device=DEVICE)
    pm0 = torch.tensor([row[2]], dtype=torch.float32, device=DEVICE)
    Tmsep = torch.tensor([row[3]], dtype=torch.float32, device=DEVICE)
    Tfw = torch.tensor([row[4]], dtype=torch.float32, device=DEVICE)
    v1_0 = torch.tensor([row[5]], dtype=torch.float32, device=DEVICE)
    pout = torch.tensor([row[7]], dtype=torch.float32, device=DEVICE)
    h_sw = t02.hliq_of_T(Tfw)
    p0 = pm0 + (pout - pm0) / 3.0
    p1 = pm0 + 2.0 * (pout - pm0) / 3.0
    hsep = t02.h_sep_of(pm0, Tmsep)
    k_t = model0.k_of(pm0)
    th1, th2 = model0.th_of(pm0)
    s_den = th1 * v1_0 + th2 * torch.tensor([u0], dtype=torch.float32, device=DEVICE) + 1e-6

    def f(x, dv2):
        # x: (14,) tensor (requires_grad); dv2: scalar tensor
        h = x[0:3][:, None]
        Tm = x[3:6][:, None]
        rB = x[6:7]
        Dsw1l = x[7:8]
        Dsw2l = x[8:9]
        T_sens = x[9:14][:, None]
        v2_t = u0 + dv2
        Wt = W0 * (1.0 + kw * dv2)
        Wt = torch.clamp(Wt, min=0.0)
        Dsw1_new = t02.KAPPA * Wt * (th1 * v1_0) / (th1 * v1_0 + th2 * v2_t + 1e-6)
        Dsw2_new = t02.KAPPA * Wt * (th2 * v2_t) / (th1 * v1_0 + th2 * v2_t + 1e-6)
        Dsw1l = Dsw1l + a_sw * (Dsw1_new - Dsw1l)
        Dsw2l = Dsw2l + a_sw * (Dsw2_new - Dsw2l)
        hm1 = (D0 * h[0:1] + Dsw1l * h_sw) / (D0 + Dsw1l + 1e-6)
        hm2 = (D0 * h[1:2] + Dsw2l * h_sw) / (D0 + Dsw2l + 1e-6)
        for _ in range(t02.N_SUB):
            ts = t02.T_of_ph(torch.stack([p0, p1, pout]), h)
            Q = UA * (Tm - ts)
            feats = r09.build_feats(ts, Tm, pm0, D0, uB0, rB, v1_0, v2_t, Wt, None,
                                    no_act=True)
            z = mod.mlp(feats).permute(1, 0)
            Tm = (Tm + t02.DT_SUB * (k_t * rB[None, :] / 3600.0 + UA * ts + z) / Cm) / (
                1.0 + t02.DT_SUB * UA / Cm)
            hin = torch.stack([hsep, hm1[:, 0], hm2[:, 0]])
            h = (h + t02.DT_SUB * (D0[None, :] * hin + Q + z) / M) / (
                1.0 + t02.DT_SUB * D0[None, :] / M)
            h = t02._ste_clamp(h, t02.H_LO, t02.H_HI)
            hm1 = (D0 * h[0:1] + Dsw1l * h_sw) / (D0 + Dsw1l + 1e-6)
            hm2 = (D0 * h[1:2] + Dsw2l * h_sw) / (D0 + Dsw2l + 1e-6)
            rB = rB + t02.DT_SUB * (uB0 - rB) / tauB
        p = torch.stack([p0, p0, p1, p1, pout])[:, 0]
        hh = torch.stack([h[0], hm1[0], h[1], hm2[0], h[2]])[:, 0]
        T_raw = t02.T_of_ph(p, hh)[:, None]
        T_sens = T_sens + a_sens * (T_raw - T_sens)
        x_next = torch.cat([h[:, 0], Tm[:, 0], rB, Dsw1l, Dsw2l, T_sens[:, 0]])
        return x_next

    return f


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

    pm_seg = pm_all[START: START + t02.ROLL_STEPS]
    k_w_state = {}
    for state, msk in (("wet", pm_seg <= P_CRIT), ("dry", pm_seg > P_CRIT)):
        sub = Ea[START: START + t02.ROLL_STEPS][msk]
        A = np.stack([sub[:, 5], sub[:, 6], np.ones(len(sub))], 1)
        coef, _, _, _ = np.linalg.lstsq(A, sub[:, 8], rcond=None)
        k_w_state[state] = float(np.clip(coef[1] / np.mean(sub[:, 8]), 0.5, 4.0))

    seg_rows = np.arange(START, START + t02.ROLL_STEPS)
    ops, used = [], set()
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

    summ = {"table": [], "judge": {}}
    results = []
    for op in ops:
        row, obs = Ea[op["row"]], T_all[op["row"]]
        u0, W0 = float(row[V2]), float(row[8])
        kw = k_w_state[op["state"]]
        # 非线性基准: 从 init 跑 600 步到平衡, 取终态 x̄ (差分测试与 LTI 同基准)
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
                out, h, Tm, rB, hm1, hm2, T_sens = r15.integrate_learn(
                    model0, mod0, exo, h, Tm, rB, 1, T_sens=T_sens)
        # 提取平衡态 (Dsw lag 未暴露 → 用初值近似: 从 x̄ 重算一步得到一致状态)
        x_bar = torch.cat([h[:, 0], Tm[:, 0], rB, torch.tensor([0.0], device=DEVICE),
                           torch.tensor([0.0], device=DEVICE), T_sens[:, 0]]).detach()
        # 用 f 从 x_bar 跑一步以修正 Dsw lag 值 (跑 30 步达到内部一致)
        f = make_step_fn(model0, mod0, row, W0, u0, kw)
        with torch.no_grad():
            xc = x_bar.clone()
            for _ in range(30):
                xc = f(xc, torch.tensor(0.0, device=DEVICE))
        x_bar = xc.clone()
        # Jacobian
        x0 = x_bar.clone().requires_grad_(True)
        dv0 = torch.tensor(0.0, device=DEVICE, dtype=torch.float32).requires_grad_(True)
        Jx, Ju = AF.jacobian(lambda x, dv: f(x, dv), (x0, dv0))
        Amat = Jx.detach().cpu().numpy()
        Bmat = Ju.detach().cpu().numpy()[:, None]
        Cmat = np.zeros((5, N_STATE))
        Cmat[:, 9:14] = np.eye(5)
        # 特征值
        ev = np.abs(np.linalg.eigvals(Amat))
        l3 = bool(ev.max() < 1.0)
        # LTI 阶跃响应 (从 0 状态)
        x_s = np.zeros(N_STATE)
        lti_resp = np.zeros((N, 5))
        for t in range(N):
            lti_resp[t] = Cmat @ x_s
            x_s = Amat @ x_s + Bmat[:, 0] * 0.05
        # 非线性差分 (从 x̄)
        with torch.no_grad():
            xb = x_bar.clone()
            base_nl = np.zeros((N, 5))
            for t in range(N):
                xb = f(xb, torch.tensor(0.0, device=DEVICE))
                base_nl[t] = xb[9:14].cpu().numpy()
            xs = x_bar.clone()
            step_nl = np.zeros((N, 5))
            for t in range(N):
                xs = f(xs, torch.tensor(0.05, device=DEVICE))
                step_nl[t] = xs[9:14].cpu().numpy()
        d_nl = step_nl - base_nl
        # R² per output (main = idx 4)
        r2s = []
        for j in range(5):
            v = float(np.var(d_nl[:, j]))
            mse = float(np.mean((lti_resp[:, j] - d_nl[:, j]) ** 2))
            r2s.append(1 - mse / v if v > 0 else 0.0)
        r2_main = r2s[4]
        rec = {"row": op["row"], "pm": round(op["pm"], 1), "state": op["state"],
               "power": round(op["power"], 0), "k_w": round(kw, 3),
               "R2_main": round(r2_main, 3), "R2_all": [round(x, 3) for x in r2s],
               "max_eig": round(float(ev.max()), 5), "stable": l3,
               "K_main_step": round(float(np.mean(d_nl[-60:, 4]) / 0.05), 3)}
        results.append(rec)
        print(f"[lin {op['row']}] pm={op['pm']:.1f} R²_main={r2_main:.3f} "
              f"max|λ|={ev.max():.4f} stable={l3}", flush=True)

    # L2: 级联闭环 (PI + actuator + LTI)
    cl_ok = {}
    for op, rec in zip(ops, results):
        row, obs = Ea[op["row"]], T_all[op["row"]]
        u0 = float(row[V2])
        # 重建 Amat/Bmat (重算一次, 简单起见重新跑线性化)
        h, Tm, rB = r15.init_state(model0, row, obs)
        T_sens = torch.tensor(obs, device=DEVICE)[None].permute(1, 0)
        with torch.no_grad():
            for t in range(N):
                exo = torch.tensor(row, device=DEVICE)[None, None, :].clone()
                out, h, Tm, rB, hm1, hm2, T_sens = r15.integrate_learn(
                    model0, mod0, exo, h, Tm, rB, 1, T_sens=T_sens)
        x_bar = torch.cat([h[:, 0], Tm[:, 0], rB, torch.tensor([0.0], device=DEVICE),
                           torch.tensor([0.0], device=DEVICE), T_sens[:, 0]]).detach()
        f = make_step_fn(model0, mod0, row, float(row[8]), u0, k_w_state[op["state"]])
        with torch.no_grad():
            xc = x_bar.clone()
            for _ in range(30):
                xc = f(xc, torch.tensor(0.0, device=DEVICE))
        x0 = xc.clone().requires_grad_(True)
        dv0 = torch.tensor(0.0, device=DEVICE).requires_grad_(True)
        Jx, Ju = AF.jacobian(lambda x, dv: f(x, dv), (x0, dv0))
        Amat = Jx.detach().cpu().numpy()
        Bmat = Ju.detach().cpu().numpy()[:, None]
        Cmat = np.zeros((5, N_STATE))
        Cmat[:, 9:14] = np.eye(5)
        # 闭环
        x_s = np.zeros(N_STATE)
        v, integ, u = u0, 0.0, u0
        SP = 2.0
        yh = np.zeros(N)
        for t in range(N):
            y = float((Cmat @ x_s)[4])
            e = y - SP
            kp, ti = r15.pi_params(-e, op["power"])
            integ += e * DT
            u = float(np.clip(u0 + kp * e + (kp / ti) * integ, 0.0, 1.0))
            step = float(np.clip((u - v) * (DT / TAU_A), -RATE, RATE))
            v = float(np.clip(v + step, 0.0, 1.0))
            x_s = Amat @ x_s + Bmat[:, 0] * (v - u0)
            yh[t] = y
        ss = float(np.mean(yh[-60:]))
        norm = yh / ss
        cl_ok[rec["row"]] = {"norm600": round(float(norm[599]), 3),
                             "tail_std": round(float(np.std(norm[-120:])), 4),
                             "ok": bool(0.8 <= norm[599] <= 1.2 and np.std(norm[-120:]) <= 0.05)}
        print(f"[lti-cl {rec['row']}] norm600={cl_ok[rec['row']]['norm600']} "
              f"tail_std={cl_ok[rec['row']]['tail_std']}", flush=True)

    L1 = bool(sum(1 for r in results if r["R2_main"] >= 0.95) >= 4)
    L2 = bool(all(cl_ok[r["row"]]["ok"] for r in results))
    L3 = bool(all(r["stable"] for r in results))
    judge = {"L1": L1, "L2": L2, "L3": L3,
             "verdict": "PASS" if (L1 and L2 and L3) else "FAIL"}
    summ = {"table": results, "cl_lti": cl_ok, "judge": judge}
    with open(os.path.join(OUT, "step6b_lin_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2, default=str)
    print("=== STEP6b 判定 ===")
    print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot([r["pm"] for r in results], [r["K_main_step"] for r in results], "o-",
            color="#8b008b")
    ax.set_title("linearized gain schedule: K vs pm")
    ax.set_xlabel("pm (MPa)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot([r["pm"] for r in results], [r["R2_main"] for r in results], "s-",
            color="#2e8b57", label="R² LTI vs nonlinear")
    ax.axhline(0.95, color="crimson", ls=":", lw=1, label="L1 gate 0.95")
    ax.set_title("linearization fidelity")
    ax.set_xlabel("pm (MPa)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"STEP6b exact linearization — verdict={judge['verdict']} "
                 f"(L1={L1} L2={L2} L3={L3})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig21_linearized.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig21_linearized.png", flush=True)


if __name__ == "__main__":
    main()
