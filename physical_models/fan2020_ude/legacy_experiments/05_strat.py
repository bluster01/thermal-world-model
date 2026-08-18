#!/usr/bin/env python3
"""05_strat.py: 湿/干态分层评估（1800步 rollout + 测试段窗口内首步/60步）。
修复A（k0 分模态）前后通用：新模型走 model.k_of(pm)，旧模型回退 tri("k") 广播。

用法:
  python 05_strat.py --seeds 0,1,2 --out out/strat_pre_fixA.json --fig out/figs/fig3_strat_pre.png

产物 json 结构:
  {"mode_def": "wet=pm<=22.064", "wet_frac": {train/val/test, crossing_win_frac},
   "rollout": {wet/dry: {n, rmse_main, rmse_all, bias_5, band_viol_frac, viol_phys_frac, drift_main_mean_z, pm_mean}},
   "windowed": {wet/dry: {n_win, first_rmse_main, win60_rmse_main, first_bias_sh1in, first_bias_main}},
   "seeds": {...原始 seed 级...}}
"""
import argparse
import importlib.util
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t02", os.path.join(os.getcwd(), "02_train.py"))
t02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t02)
import numpy as np
import pandas as pd
import torch

PAIRS_PHYS = [(1, 0), (3, 2), (1, 2), (3, 4), (0, 2)]  # 与 step1_summary 物理对一致

OUTPUT_NAMES = t02.OUTPUTS


def load_df():
    return pd.read_csv(t02.CSV, usecols=t02.EXO + t02.EXO_EXTRA + t02.OUTPUTS + [
        "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
        dtype=np.float32).iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)


def k_eq(model, pm):
    """平衡初始化用 k：(3,B)。新模型走 k_of(pm)；旧模型 tri('k') 广播。"""
    if hasattr(model, "k_of"):
        return model.k_of(pm)
    return model.tri("k")[:, None].expand(3, pm.shape[0])


def load_model(seed):
    model = t02.E0Model().to(t02.DEVICE)
    model.load_state_dict(torch.load(os.path.join(t02.OUT, f"model_e0_seed{seed}.pt"),
                                     map_location=t02.DEVICE, weights_only=True))
    model.eval()
    return model


# ---------------- rollout 分层（复用落盘 npz，按 pm 分类） ----------------
def strat_rollout(df, seed):
    START = t02.TRAIN_N + t02.VAL_N
    n = t02.ROLL_STEPS
    d = np.load(os.path.join(t02.OUT, f"rollout_e0_seed{seed}.npz"))
    preds, truths = d["preds"], d["truths"]
    E = df[["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
            "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
            "末级过热器出口压力", "减温水总流量"]].to_numpy(np.float32)
    pm = E[START: START + n, 2]
    mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    out = {}
    for mode, mask in (("wet", pm <= t02.P_CRIT), ("dry", pm > t02.P_CRIT)):
        if mask.sum() == 0:
            out[mode] = None
            continue
        p, t = preds[mask], truths[mask]
        viol = np.zeros(mask.sum(), dtype=bool)
        for lo, hi in PAIRS_PHYS:
            viol |= (p[:, lo] >= p[:, hi])
        z = np.abs((p - mu_o) / sd_o)
        out[mode] = {
            "n": int(mask.sum()),
            "pm_mean": round(float(pm[mask].mean()), 2),
            "rmse_main": round(float(np.sqrt(np.mean((p[:, 4] - t[:, 4]) ** 2))), 3),
            "rmse_all": round(float(np.sqrt(np.mean((p - t) ** 2))), 3),
            "bias_5": [round(float((p[:, j] - t[:, j]).mean()), 2) for j in range(5)],
            "band_viol_frac": round(float(np.mean((p[:, 4] > t02.T_BAND[1]) | (p[:, 4] < t02.T_BAND[0]))), 4),
            "viol_phys_frac": round(float(viol.mean()), 4),
            "drift_main_mean_z": round(float(z[:, 4].mean()), 3),
        }
    return out


# ---------------- 窗口内分层（每窗重置，stride 10，与 e0_windowed_eval 同代码路径） ----------------
def strat_windowed(df, model, seed):
    START = t02.TRAIN_N + t02.VAL_N
    Xte, Yte, Ite, Ite_T = t02.e0_build_windows(df, START, len(df) - 1, 10)
    model.eval()
    first_errs, win_errs, modes = [], [], []
    with torch.no_grad():
        for b in range(0, len(Xte), 256):
            xb = torch.from_numpy(Xte[b: b + 256]).to(t02.DEVICE)
            yb = torch.from_numpy(Yte[b: b + 256]).to(t02.DEVICE)
            ib = Ite[b: b + 256]
            D = torch.from_numpy(ib[:, 0]).to(t02.DEVICE)
            pm = torch.from_numpy(ib[:, 2]).to(t02.DEVICE)
            p_out = torch.from_numpy(ib[:, 7]).to(t02.DEVICE)
            p0 = pm + (p_out - pm) / 3.0
            p1 = pm + 2.0 * (p_out - pm) / 3.0
            obs = torch.from_numpy(Ite_T[b: b + 256]).to(t02.DEVICE)
            h0 = t02.h_of_pT(p0, obs[:, 0])
            h1 = t02.h_of_pT(p1, obs[:, 2])
            h2 = t02.h_of_pT(p_out, obs[:, 4])
            h = torch.stack([h0, h1, h2])
            ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
            rB = torch.from_numpy(ib[:, 1]).to(t02.DEVICE).clone()
            Tm = (ts + k_eq(model, pm) * rB[None, :] / 3600.0 / model.tri("UA")[:, None]
                  + model.tri("dTm")[:, None])
            pred = model.integrate(xb, h, Tm, rB, xb.shape[1])
            err = pred - yb
            first_errs.append(err[:, 0].cpu().numpy())     # (B,5)
            win_errs.append(err[:, :, 4].cpu().numpy())    # (B,60)
            modes.append(ib[:, 2])                          # 窗口起点 pm
    first = np.concatenate(first_errs, axis=0)
    winm = np.concatenate(win_errs, axis=0)
    pm0 = np.concatenate(modes)
    # 窗口内跨临界占比（起点湿/干之外还报"混合窗"）
    E = df[["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
            "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
            "末级过热器出口压力", "减温水总流量"]].to_numpy(np.float32)
    i0 = np.arange(START, len(df) - 1 - t02.SEQ, 10)
    cross = np.array([
        bool(((E[s: s + t02.SEQ, 2] > t02.P_CRIT).any()) and ((E[s: s + t02.SEQ, 2] <= t02.P_CRIT).any()))
        for s in i0])
    out = {"n_win": int(len(pm0)), "crossing_win_frac": round(float(cross.mean()), 4)}
    for mode, mask in (("wet", pm0 <= t02.P_CRIT), ("dry", pm0 > t02.P_CRIT)):
        if mask.sum() == 0:
            out[mode] = None
            continue
        out[mode] = {
            "n_win": int(mask.sum()),
            "first_rmse_main": round(float(np.sqrt(np.mean(first[mask, 4] ** 2))), 3),
            "win60_rmse_main": round(float(np.sqrt(np.mean(winm[mask] ** 2))), 3),
            "first_bias_sh1in": round(float(first[mask, 0].mean()), 2),
            "first_bias_main": round(float(first[mask, 4].mean()), 2),
        }
    return out


def make_fig(df, seeds, strat, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    START = t02.TRAIN_N + t02.VAL_N
    n = t02.ROLL_STEPS
    E = df[["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
            "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
            "末级过热器出口压力", "减温水总流量"]].to_numpy(np.float32)
    pm = E[START: START + n, 2]
    t_axis = np.arange(n) / 6.0  # 10s → min
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    # panel 1: pm + wet shading
    ax = axes[0]
    ax.plot(t_axis, pm, color="#1f4e79", lw=1.2)
    ax.axhline(t02.P_CRIT, color="crimson", ls="--", lw=1, label=f"P_crit={t02.P_CRIT} MPa")
    wet_mask = pm <= t02.P_CRIT
    ax.fill_between(t_axis, pm.min(), pm.max(), where=wet_mask, color="steelblue", alpha=0.15, label="wet")
    ax.set_ylabel("Separator pressure (MPa)")
    ax.set_title("Test rollout: wet/dry stratification by separator pressure")
    ax.legend(loc="upper right", fontsize=8)
    # panel 2: per-mode bias bars (5 outputs), mean±std over seeds
    ax = axes[1]
    names = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]
    x = np.arange(5)
    for k, mode in enumerate(("wet", "dry")):
        rows = [strat["seeds"][str(s)]["rollout"][mode]["bias_5"] for s in seeds
                if strat["seeds"][str(s)]["rollout"][mode] is not None]
        if not rows:
            continue
        rows = np.array(rows)
        ax.bar(x + (k - 0.5) * 0.36, rows.mean(0), 0.32, yerr=rows.std(0),
               label=mode, color=["#2e75b6", "#c55a11"][k], alpha=0.9, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Rollout bias (°C)")
    ax.set_title("Per-mode rollout bias (mean±std over 3 seeds)")
    ax.legend(fontsize=9)
    # panel 3: windowed first-step & 60-step rmse per mode
    ax = axes[2]
    xw = np.arange(2)
    wets, drys = [], []
    for s in seeds:
        w = strat["seeds"][str(s)]["windowed"]
        wets.append(w["wet"]["win60_rmse_main"] if w["wet"] else np.nan)
        drys.append(w["dry"]["win60_rmse_main"] if w["dry"] else np.nan)
    vals = [np.nanmean(wets), np.nanmean(drys)]
    errs = [np.nanstd(wets), np.nanstd(drys)]
    bars = ax.bar(xw, vals, 0.5, yerr=errs, capsize=5, color=["#2e75b6", "#c55a11"], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(xw)
    ax.set_xticklabels(["wet windows", "dry windows"])
    ax.set_ylabel("Window-60 RMSE main (°C)")
    ax.set_title("Windowed eval (60-step, reset per window) — main steam temp RMSE by mode")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[fig] {fig_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    df = load_df()
    START = t02.TRAIN_N + t02.VAL_N
    E = df[["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
            "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
            "末级过热器出口压力", "减温水总流量"]].to_numpy(np.float32)
    pm_all = E[:, 2]
    wet_frac = {
        "train": round(float((pm_all[:t02.TRAIN_N] <= t02.P_CRIT).mean()), 4),
        "val": round(float((pm_all[t02.TRAIN_N:START] <= t02.P_CRIT).mean()), 4),
        "test": round(float((pm_all[START:] <= t02.P_CRIT).mean()), 4),
    }
    strat = {"mode_def": "wet = pm <= 22.064 MPa (P_CRIT)", "wet_frac": wet_frac,
             "seeds": {}, "rollout": {}, "windowed": {}}
    for s in seeds:
        model = load_model(s)
        r = strat_rollout(df, s)
        w = strat_windowed(df, model, s)
        strat["seeds"][str(s)] = {"rollout": r, "windowed": w}
        print(f"[seed {s}] rollout wet n={r['wet']['n']} dry n={r['dry']['n']} | "
              f"win wet n={w['wet']['n_win']} dry n={w['dry']['n_win']} cross={w['crossing_win_frac']}", flush=True)
    # 跨 seed 聚合
    for mode in ("wet", "dry"):
        rows = [strat["seeds"][str(s)]["rollout"][mode] for s in seeds]
        rows = [r for r in rows if r is not None]
        if rows:
            agg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in
                   ("rmse_main", "rmse_all", "band_viol_frac", "viol_phys_frac", "drift_main_mean_z")}
            agg["n"] = int(rows[0]["n"])
            agg["bias_5"] = [round(float(np.mean([r["bias_5"][j] for r in rows])), 2) for j in range(5)]
            strat["rollout"][mode] = agg
        wrows = [strat["seeds"][str(s)]["windowed"][mode] for s in seeds]
        wrows = [r for r in wrows if r is not None]
        if wrows:
            agg = {k: round(float(np.mean([r[k] for r in wrows])), 3) for k in
                   ("first_rmse_main", "win60_rmse_main", "first_bias_sh1in", "first_bias_main")}
            agg["n_win"] = int(wrows[0]["n_win"])
            strat["windowed"][mode] = agg
    with open(args.out, "w") as f:
        json.dump(strat, f, ensure_ascii=False, indent=2)
    print(f"[json] {args.out}", flush=True)
    if args.fig:
        make_fig(df, seeds, strat, args.fig)
    print("=== 聚合 ===", flush=True)
    print(json.dumps({"wet_frac": wet_frac, "rollout": strat["rollout"],
                      "windowed": strat["windowed"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
