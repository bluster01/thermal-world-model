#!/usr/bin/env python3
"""10f_windowed_per_output.py: 60步窗口评估 — 各输出 RMSE/MAE/bias 明细 (seed0)

对齐协议: 窗口起点状态@s + 外生[s..s+59] → 预测 T[s+1..s+60], 目标同 (无错位)。
灰盒家族: integrate_res 60步开环; v2/v0: 07 baseline_windowed_arrays 同反馈协议, 扩到全5输出。
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
OUTS = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]
SEQ = t02.SEQ


def grey_windowed_all(df, model0, res, mode, use_anchor):
    Xte, Yte, Ite, Ite_T = t02.e0_build_windows(df, START, len(df) - 1, 10)
    errs = np.zeros((len(Xte), SEQ, 5), np.float32)
    pm0_list = []
    with torch.no_grad():
        for b in range(0, len(Xte), 256):
            xb = torch.from_numpy(Xte[b: b + 256]).to(DEVICE)
            yb = torch.from_numpy(Yte[b: b + 256]).to(DEVICE)
            ib = Ite[b: b + 256]
            pm = torch.from_numpy(ib[:, 2]).to(DEVICE)
            p_out = torch.from_numpy(ib[:, 7]).to(DEVICE)
            p0 = pm + (p_out - pm) / 3.0
            p1 = pm + 2.0 * (p_out - pm) / 3.0
            obs = torch.from_numpy(Ite_T[b: b + 256]).to(DEVICE)
            h0 = t02.h_of_pT(p0, obs[:, 0])
            h1 = t02.h_of_pT(p1, obs[:, 2])
            h2 = t02.h_of_pT(p_out, obs[:, 4])
            h = torch.stack([h0, h1, h2])
            ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
            rB = torch.from_numpy(ib[:, 1]).to(DEVICE).clone()
            Tm = (ts + model0.k_of(pm) * rB[None, :] / 3600.0 / model0.tri("UA")[:, None]
                  + model0.tri("dTm")[:, None])
            anchor = torch.stack([ts[0], ts[1], ts[2], Tm[0], Tm[1], Tm[2], pm], dim=1)
            pred, *_ = r09.integrate_res(model0, res, xb, h, Tm, rB, xb.shape[1],
                                         anchor if use_anchor else None, mode)
            errs[b: b + 256] = (pred - yb).cpu().numpy()
            pm0_list.append(ib[:, 2])
    return errs, np.concatenate(pm0_list)


def baseline_windowed_all(df_b, model, info, phys):
    exo_cols = info["exo_cols"]
    phy_cols = info["phy_cols"]
    cols = exo_cols + t02.OUTPUTS + (phy_cols if phys else [])
    mu = pd.Series(info["mu"])
    sd = pd.Series(info["sd"])
    feats = t02.physics_features(df_b) if phys else None
    full = pd.concat([df_b[exo_cols], df_b[t02.OUTPUTS]] + ([feats] if feats is not None else []), axis=1)
    Z = ((full[cols] - mu[cols]) / sd[cols]).to_numpy(np.float32)
    out_start = len(cols) - len(phy_cols) - 5 if phys else len(cols) - 5
    mu_o = np.array([info["mu"][c] for c in t02.OUTPUTS], np.float32)
    sd_o = np.array([info["sd"][c] for c in t02.OUTPUTS], np.float32)
    i0 = np.arange(START, len(df_b) - 1 - SEQ, 10)
    hist = np.stack([Z[s - SEQ: s] for s in i0])  # (Nw, SEQ, F)
    errs = np.zeros((len(i0), SEQ, 5), np.float32)
    truth_all = df_b[t02.OUTPUTS].to_numpy()
    cmd = df_b["二级减温喷水调节门指令"].to_numpy()
    w_exo = df_b["减温水总流量"].to_numpy()
    feco = df_b["省煤器出口给水温度"].to_numpy()
    sep = df_b["分离器出口温度"].to_numpy()
    steam = df_b["主蒸汽流量"].to_numpy()
    coal = df_b["未校正总煤量"].to_numpy()
    pm_series = df_b["分离器出口压力"].to_numpy()
    cmd64, w64, feco64 = cmd.astype(np.float64), w_exo.astype(np.float64), feco.astype(np.float64)
    sep64, steam64, coal64 = sep.astype(np.float64), steam.astype(np.float64), coal.astype(np.float64)
    model.eval()
    with torch.no_grad():
        for t in range(SEQ):
            idx = i0 + t
            yz = model(torch.from_numpy(hist).to(DEVICE))
            pred = t02.model_out(yz, info).cpu().numpy()  # (Nw,5)
            truth = truth_all[idx]
            errs[:, t] = pred - truth
            new_z = Z[idx].copy()
            new_z[:, out_start: out_start + 5] = (
                (pred.astype(np.float64) - mu_o.astype(np.float64)) / sd_o.astype(np.float64)).astype(np.float32)
            if phys:
                sh1i, sh1o, sh2i, sh2o, main = [pred[:, j].astype(np.float64) for j in range(5)]
                feats_arr = np.stack([
                    w64[idx] * (sh1i - feco64[idx]),
                    w64[idx] * (sh2i - feco64[idx]),
                    steam64[idx] * (sh1i - sep64[idx]),
                    steam64[idx] * (sh2i - sh1o),
                    steam64[idx] * (main - sh2o),
                    coal64[idx] / (steam64[idx] + 1.0),
                    cmd64[idx] - cmd64[idx - 6],
                ], axis=1)
                for k, c in enumerate(phy_cols):
                    new_z[:, out_start + 5 + k] = ((feats_arr[:, k] - mu[c]) / sd[c]).astype(np.float32)
            hist = np.concatenate([hist[:, 1:], new_z[:, None]], axis=1)
    return errs, pm_series[i0]


def report(name, errs, pm0):
    n = errs.shape[0]
    wet = pm0 <= P_CRIT
    out = {"n_win": int(n)}
    print(f"\n{name} (n={n}, wet {wet.mean()*100:.0f}%):")
    print(f"  {'out':8s} | {'win60RMSE':>9s} | {'win60MAE':>9s} | {'firstRMSE':>9s} | {'bias':>7s} | {'wetRMSE':>8s} | {'dryRMSE':>8s}")
    for j, o in enumerate(OUTS):
        e = errs[:, :, j]
        r = float(np.sqrt(np.mean(e ** 2)))
        m = float(np.mean(np.abs(e)))
        f = float(np.sqrt(np.mean(e[:, 0] ** 2)))
        b = float(e.mean())
        rw = float(np.sqrt(np.mean(e[wet] ** 2)))
        rd = float(np.sqrt(np.mean(e[~wet] ** 2)))
        print(f"  {o:8s} | {r:9.2f} | {m:9.2f} | {f:9.2f} | {b:+7.2f} | {rw:8.2f} | {rd:8.2f}")
        out[o] = {"win60_rmse": round(r, 2), "win60_mae": round(m, 2),
                  "first_rmse": round(f, 2), "bias": round(b, 2),
                  "wet_rmse": round(rw, 2), "dry_rmse": round(rd, 2)}
    return out


def main():
    df = r09.load_e0_df()
    df_b = pd.read_csv(t02.CSV, usecols=t02.EXO + t02.EXO_EXTRA + t02.OUTPUTS + [
        "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
        dtype=np.float32).iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)
    model0 = r09.load_e0(0)
    summary = {}
    t0 = time.time()

    # ---- 灰盒家族 (seed0) ----
    for v, (mode, anchor) in [("e0", ("none", False)), ("rb", ("q", False)),
                              ("qh", ("qh", False)), ("qspl", ("qspl", False)),
                              ("q0", ("q0", False))]:
        if v == "e0":
            res = r09.ResMLP(13, r09.Q_SCALE).to(DEVICE)
        else:
            res = r09.ResMLP(13, r09.Q_SCALE, 6 if v == "qspl" else 3).to(DEVICE)
            res.load_state_dict(torch.load(os.path.join(OUT, f"model_res_{v}_seed0.pt"),
                                           map_location=DEVICE, weights_only=True))
        res.eval()
        errs, pm0 = grey_windowed_all(df, model0, res, mode, anchor)
        summary[v] = report(v, errs, pm0)
        print(f"[{v}] {time.time()-t0:.0f}s", flush=True)

    # ---- 黑盒基线 (seed0) ----
    for v, phys in [("v2", True), ("v0", False)]:
        info = t02.baseline_build_data(df_b, v, phys)[7]
        model = t02.Net(info["F"]).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(OUT, f"model_{v}_seed0.pt"),
                                         map_location=DEVICE, weights_only=True))
        errs, pm0 = baseline_windowed_all(df_b, model, info, phys)
        summary[v] = report(v, errs, pm0)
        print(f"[{v}] {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(OUT, "windowed_per_output_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[json] out/windowed_per_output_summary.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
