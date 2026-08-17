#!/usr/bin/env python3
"""10_refine.py: adhoc2 残差探针细化验证三连 (2026-08-17)

Step① repro : ra/rb/rc × 3 seeds —— 3seeds 复现
Step② inject: qboth(=rb)/qtm/qh/qcon/qspl × seed0 —— 注入点消融
Step③ seg0  : q0(rb 残差只注段0) × 3 seeds —— 湿态段0专项
Step④ confirm: qspl/qh × 3 seeds —— 坐实"超 v2" (s0 复用 Step② ckpt, s1/s2 重训)

预注册判定 (冻结 2026-08-17, 不可改):
  R1: rb 3 个种子每个 dry rollout ≤ 3.0 且 wet ≤ 4.0 —— 方向跨种子复现
  R2: rb 3 种子 dry std ≤ 0.6 —— 种子稳健
  I1: ②胜者 = min 平均(dry, wet) rollout rmse; 胜者 dry ≤ rb_s0 dry + 0.5 —— 不劣于 rb
  I2: 若胜者 ∈ {qcon, qspl} 且优于 qboth 平均 ≥ 0.3 —— 守恒/分配版注入被实证
  S1: q0 wet rollout 3seed mean ≤ rb wet mean − 0.5 —— 段0确为湿态短板主因
  S2: q0 dry rollout 3seed mean ≤ rb dry mean + 0.5 —— 干态不回退
  C1: qspl 3seed mean rollout < 2.93 (v2 基线) —— 灰盒+残差超黑盒坐实
  C2: qh   3seed mean rollout < 2.93 —— 简单h-only版同样超 v2
  C3: qspl/qh 每个种子 rollout ≤ 3.2 —— 无种子显著劣于 v2

用法:
  python 10_refine.py repro [--fast]
  python 10_refine.py inject [--fast]
  python 10_refine.py seg0 [--fast]
  python 10_refine.py confirm [--fast]
  python 10_refine.py smoke     # 新模式冒烟 (2ep + rollout 120 步, 无 windowed, 仅崩溃检测)
产物:
  out/refine_repro_summary.json / refine_inject_summary.json / refine_seg0_summary.json
  out/figs/fig10_repro.png fig11_inject.png fig12_seg0.png
"""
import argparse
import importlib.util
import json
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _imp(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.getcwd(), path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t02 = _imp("02_train.py", "t02")
r09 = _imp("09_residual.py", "r09")
import numpy as np
import torch

DEVICE = t02.DEVICE
OUT = t02.OUT
START = r09.START
P_CRIT = t02.P_CRIT

VSPEC = {
    "ra":   ("q",    True,  3, "resQ+z0"),
    "rb":   ("q",    False, 3, "resQ no-z0"),
    "rc":   ("dk",   True,  3, "resΔk+z0"),
    "qtm":  ("qtm",  False, 3, "Q→Tm only"),
    "qh":   ("qh",   False, 3, "Q→h only"),
    "qcon": ("qcon", False, 3, "Q conserved (Tm−/h+)"),
    "qspl": ("qspl", False, 6, "Q split λ"),
    "q0":   ("q0",   False, 3, "Q seg0 only"),
}
C = {"e0": "#1f4e79", "ra": "#c55a11", "rb": "#8b008b", "rc": "#2e8b57",
     "qtm": "#1f77b4", "qh": "#ff7f0e", "qcon": "#17becf", "qspl": "#d62728",
     "q0": "#7f7f7f"}
# 基线 (预注册冻结): e0-post / v2 (GRU 黑盒 3seeds)
BASE = {"e0": {"dry": 17.059, "wet": 6.966, "all": 12.657},
        "v2": {"dry": 2.44, "wet": 3.29, "all": 2.93}}


def load_ctx(args):
    df = r09.load_e0_df()
    model0 = r09.load_e0(0)
    mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    pm_roll = df["分离器出口压力"].to_numpy(np.float32)[START: START + t02.ROLL_STEPS]
    n_roll = 120 if args.step == "smoke" else t02.ROLL_STEPS
    return df, model0, mu_o, sd_o, pm_roll, n_roll


def run_variant(df, model0, v, seeds, fast, n_roll, mu_o, sd_o, pm_roll, reuse=True):
    mode, use_anchor, out, _ = VSPEC[v]
    F_in = 20 if use_anchor else 13
    scale = r09.Q_SCALE if mode != "dk" else r09.K_SCALE
    recs = []
    for sd in seeds:
        p = os.path.join(OUT, f"model_res_{v}_seed{sd}.pt")
        if reuse and not fast and os.path.exists(p):
            res = r09.ResMLP(F_in, scale, out).to(DEVICE)
            res.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
            res.eval()
            tr = {"val_mse": None, "ep": None, "reused": True}
            print(f"[{v} s{sd}] reuse ckpt", flush=True)
        else:
            res, best_va, ep = r09.train_res(df, sd, v, mode, use_anchor, fast, out=out)
            res.eval()
            tr = {"val_mse": round(best_va, 4), "ep": ep, "reused": False}
        with torch.no_grad():
            r, preds, truths = r09.rollout_res(model0, res, df, START, n_roll, mode, use_anchor)
        rec = {"seed": sd, "train": tr,
               "rollout": {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}}
        if n_roll == t02.ROLL_STEPS:
            rec["strat"] = r09.strat_rollout(preds, truths, pm_roll, mu_o, sd_o)
            arrs = r09.windowed_arrays_res(df, model0, res, mode, use_anchor)
            rec["windowed"] = r09.layer_agg(*arrs)
            rec["_arrs"] = arrs  # (errs_main, errs_sh1, preds_main, pm0) 仅供图, 不入 json
        if not fast:
            np.savez(os.path.join(OUT, f"rollout_res_{v}_seed{sd}.npz"),
                     preds=preds, truths=truths)
        d = rec["strat"]["dry"]["rmse_main"] if "strat" in rec else float("nan")
        w = rec["strat"]["wet"]["rmse_main"] if "strat" in rec else float("nan")
        print(f"[{v} s{sd}] rollout={r['rmse_main']:.3f} dry={d:.2f} wet={w:.2f}", flush=True)
        recs.append(rec)
    return recs


def agg(recs, layer="dry", key="rmse_main"):
    vals = [r["strat"][layer][key] for r in recs]
    return float(np.mean(vals)), float(np.std(vals)), [float(x) for x in vals]


def collect_lam(df, model0, res, n_win=240):
    """qspl 诊断: 在测试窗上收集学到的分配系数 λ (每段)."""
    Xte, Yte, Ite, Ite_T = t02.e0_build_windows(df, START, len(df) - 1, 10)
    hist = [[] for _ in range(3)]
    with torch.no_grad():
        for b in range(0, min(len(Xte), n_win), 64):
            xb = torch.from_numpy(Xte[b: b + 64]).to(DEVICE)
            ib = Ite[b: b + 64]
            pm = torch.from_numpy(ib[:, 2]).to(DEVICE)
            p_out = torch.from_numpy(ib[:, 7]).to(DEVICE)
            p0 = pm + (p_out - pm) / 3.0
            p1 = pm + 2.0 * (p_out - pm) / 3.0
            obs = torch.from_numpy(Ite_T[b: b + 64]).to(DEVICE)
            h0 = t02.h_of_pT(p0, obs[:, 0])
            h1 = t02.h_of_pT(p1, obs[:, 2])
            h2 = t02.h_of_pT(p_out, obs[:, 4])
            h = torch.stack([h0, h1, h2])
            ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
            rB = torch.from_numpy(ib[:, 1]).to(DEVICE).clone()
            Tm = (ts + model0.k_of(pm) * rB[None, :] / 3600.0 / model0.tri("UA")[:, None]
                  + model0.tri("dTm")[:, None])
            lam_list = []
            r09.integrate_res(model0, res, xb, h, Tm, rB, xb.shape[1],
                              None, "qspl", lam_list=lam_list)
            for lm in lam_list:  # (3,B)
                for j in range(3):
                    hist[j].append(lm[j].cpu().numpy())
    return [np.concatenate(x) for x in hist]


def load_res_ckpt(v, seed=0):
    mode, use_anchor, out, _ = VSPEC[v]
    F_in = 20 if use_anchor else 13
    scale = r09.Q_SCALE if mode != "dk" else r09.K_SCALE
    res = r09.ResMLP(F_in, scale, out).to(DEVICE)
    res.load_state_dict(torch.load(os.path.join(OUT, f"model_res_{v}_seed{seed}.pt"),
                                   map_location=DEVICE, weights_only=True))
    res.eval()
    return res


# ---------------- 判定 ----------------


def judge_repro(recs):
    rb = recs["rb"]
    drys = [r["strat"]["dry"]["rmse_main"] for r in rb]
    wets = [r["strat"]["wet"]["rmse_main"] for r in rb]
    R1 = all(d <= 3.0 for d in drys) and all(w <= 4.0 for w in wets)
    R2 = bool(np.std(drys) <= 0.6)
    return {"R1": bool(R1), "R2": bool(R2),
            "rb_dry_vals": [round(x, 3) for x in drys],
            "rb_wet_vals": [round(x, 3) for x in wets],
            "rb_dry_mean": round(float(np.mean(drys)), 3),
            "rb_dry_std": round(float(np.std(drys)), 3)}


def judge_inject(recs):
    vs = ["rb", "qtm", "qh", "qcon", "qspl"]
    score = {v: (recs[v][0]["strat"]["dry"]["rmse_main"]
                 + recs[v][0]["strat"]["wet"]["rmse_main"]) / 2.0 for v in vs}
    winner = min(vs, key=lambda v: score[v])
    rb_dry = recs["rb"][0]["strat"]["dry"]["rmse_main"]
    I1 = bool(recs[winner][0]["strat"]["dry"]["rmse_main"] <= rb_dry + 0.5)
    I2 = bool(winner in ("qcon", "qspl") and score["rb"] - score[winner] >= 0.3)
    return {"I1": I1, "I2": I2, "winner": winner,
            "score": {v: round(score[v], 3) for v in vs},
            "winner_dry": round(recs[winner][0]["strat"]["dry"]["rmse_main"], 3),
            "rb_dry": round(rb_dry, 3)}


def judge_seg0(recs, rb_ref):
    q0_wet_m, q0_wet_s, _ = agg(recs["q0"], "wet")
    q0_dry_m, q0_dry_s, _ = agg(recs["q0"], "dry")
    S1 = bool(q0_wet_m <= rb_ref["wet"] - 0.5)
    S2 = bool(q0_dry_m <= rb_ref["dry"] + 0.5)
    return {"S1": S1, "S2": S2,
            "q0_wet_mean": round(q0_wet_m, 3), "q0_wet_std": round(q0_wet_s, 3),
            "q0_dry_mean": round(q0_dry_m, 3), "q0_dry_std": round(q0_dry_s, 3),
            "rb_ref": rb_ref}


def judge_confirm(recs):
    qspl = [r["rollout"]["rmse_main"] for r in recs["qspl"]]
    qh = [r["rollout"]["rmse_main"] for r in recs["qh"]]
    qspl_dry = [r["strat"]["dry"]["rmse_main"] for r in recs["qspl"]]
    qspl_wet = [r["strat"]["wet"]["rmse_main"] for r in recs["qspl"]]
    qh_dry = [r["strat"]["dry"]["rmse_main"] for r in recs["qh"]]
    qh_wet = [r["strat"]["wet"]["rmse_main"] for r in recs["qh"]]
    C1 = bool(np.mean(qspl) < BASE["v2"]["all"])
    C2 = bool(np.mean(qh) < BASE["v2"]["all"])
    C3 = bool(all(v <= 3.2 for v in qspl + qh))
    return {"C1": C1, "C2": C2, "C3": C3,
            "qspl_mean": round(float(np.mean(qspl)), 3),
            "qspl_std": round(float(np.std(qspl)), 3),
            "qh_mean": round(float(np.mean(qh)), 3),
            "qh_std": round(float(np.std(qh)), 3),
            "qspl_dry": round(float(np.mean(qspl_dry)), 3),
            "qspl_wet": round(float(np.mean(qspl_wet)), 3),
            "qh_dry": round(float(np.mean(qh_dry)), 3),
            "qh_wet": round(float(np.mean(qh_wet)), 3)}


# ---------------- 图 ----------------


def _setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig10_repro(recs, judge):
    plt = _setup_plot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    layers = [("dry", "rollout dry rmse", BASE["v2"]["dry"]),
              ("wet", "rollout wet rmse", BASE["v2"]["wet"]),
              ("all", "rollout overall rmse", BASE["v2"]["all"])]
    for k, (layer, title, v2v) in enumerate(layers):
        ax = axes[k]
        for vi, v in enumerate(("ra", "rb", "rc")):
            if layer == "all":
                vals = [r["rollout"]["rmse_main"] for r in recs[v]]
            else:
                vals = [r["strat"][layer]["rmse_main"] for r in recs[v]]
            x = np.arange(3) + vi * 0.24
            ax.bar(x, vals, width=0.2, color=C[v], label=f"{v} ({VSPEC[v][3]})")
            ax.text(x.mean(), max(vals) + 0.06, f"{np.mean(vals):.2f}±{np.std(vals):.2f}",
                    ha="center", fontsize=7)
        ax.axhline(v2v, color="0.4", ls="--", lw=1.2, label=f"v2 {v2v}")
        ax.axhline(BASE["e0"][layer], color=C["e0"], ls=":", lw=1.2,
                   label=f"e0-post {BASE['e0'][layer]}")
        ax.set_title(title)
        ax.set_xticks(np.arange(3) + 0.24, ["s0", "s1", "s2"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Step① 3-seed repro — R1={'PASS' if judge['R1'] else 'FAIL'} "
                 f"R2={'PASS' if judge['R2'] else 'FAIL'}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig10_repro.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig10_repro.png", flush=True)


def fig11_inject(recs, judge, lam_hist):
    plt = _setup_plot()
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0, :])
    vs = ["rb", "qtm", "qh", "qcon", "qspl"]
    x = np.arange(len(vs))
    dry = [recs[v][0]["strat"]["dry"]["rmse_main"] for v in vs]
    wet = [recs[v][0]["strat"]["wet"]["rmse_main"] for v in vs]
    ax1.bar(x - 0.2, dry, width=0.38, color="#4472c4", label="dry")
    ax1.bar(x + 0.2, wet, width=0.38, color="#ed7d31", label="wet")
    for xi, (d, w) in enumerate(zip(dry, wet)):
        ax1.text(xi - 0.2, d + 0.05, f"{d:.2f}", ha="center", fontsize=8)
        ax1.text(xi + 0.2, w + 0.05, f"{w:.2f}", ha="center", fontsize=8)
    ax1.axhline(BASE["v2"]["dry"], color="#4472c4", ls="--", lw=1, label="v2 dry")
    ax1.axhline(BASE["v2"]["wet"], color="#ed7d31", ls="--", lw=1, label="v2 wet")
    ax1.set_xticks(x, [f"{v}: {VSPEC[v][3]}" for v in vs], fontsize=9)
    ax1.set_ylabel("rollout rmse_main (°C)")
    ax1.set_title(f"Step② injection-point ablation (seed0) — winner={judge['winner']} "
                  f"I1={'PASS' if judge['I1'] else 'FAIL'} I2={'PASS' if judge['I2'] else 'FAIL'}")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # worst-dry case 窗口曲线 (按 rb 误差选窗)
    ax2 = fig.add_subplot(gs[1, 0])
    steps = np.arange(r09.SEQ)
    em, _, preds_main, pm0 = recs["rb"][0]["_arrs"]
    mask = pm0 > P_CRIT
    win_err = np.abs(em).mean(1)
    win_err[~mask] = -1
    wi = int(np.argmax(win_err))
    truth = preds_main[wi] - em[wi]
    s = int(np.floor(wi * 10 + START))
    ax2.plot(steps, truth, color="0.3", lw=1.4, label="truth")
    for v in vs:
        p = recs[v][0]["_arrs"][2][wi]
        ax2.plot(steps, p, color=C[v], lw=1.0,
                 label=f"{v} (rmse={np.sqrt(np.mean((p - truth) ** 2)):.1f})")
    ax2.set_title(f"worst dry window @row {s} (pm0={pm0[wi]:.1f} MPa), picked by rb")
    ax2.set_xlabel("steps (×10 s)")
    ax2.set_ylabel("main steam (°C)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # λ 分布
    ax3 = fig.add_subplot(gs[1, 1])
    if lam_hist is not None:
        for j, hh in enumerate(lam_hist):
            ax3.hist(hh, bins=40, alpha=0.55, label=f"seg{j} λ (mean={hh.mean():.2f})")
        ax3.axvline(0.5, color="k", ls=":", lw=1)
        ax3.set_xlabel("λ (1=all→Tm, 0=all→h)")
        ax3.set_ylabel("count")
        ax3.legend(fontsize=8)
        ax3.set_title("learned split λ on test windows")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig11_inject.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig11_inject.png", flush=True)


def fig12_seg0(recs, judge, rb_recs):
    plt = _setup_plot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    steps = np.arange(r09.SEQ)
    # (a) 干湿 bars
    ax = axes[0]
    q0 = recs["q0"]
    drys = [r["strat"]["dry"]["rmse_main"] for r in q0]
    wets = [r["strat"]["wet"]["rmse_main"] for r in q0]
    rbd = [r["strat"]["dry"]["rmse_main"] for r in rb_recs]
    rbw = [r["strat"]["wet"]["rmse_main"] for r in rb_recs]
    x = np.arange(2)
    ax.bar(x - 0.2, [np.mean(rbd), np.mean(drys)], width=0.38, color="#4472c4",
           label="dry", yerr=[np.std(rbd), np.std(drys)], capsize=3)
    ax.bar(x + 0.2, [np.mean(rbw), np.mean(wets)], width=0.38, color="#ed7d31",
           label="wet", yerr=[np.std(rbw), np.std(wets)], capsize=3)
    ax.axhline(BASE["v2"]["dry"], color="#4472c4", ls="--", lw=1)
    ax.axhline(BASE["v2"]["wet"], color="#ed7d31", ls="--", lw=1)
    ax.set_xticks(x, ["rb (all segs)", "q0 (seg0 only)"])
    ax.set_ylabel("rollout rmse_main (°C)")
    ax.set_title(f"rollout — S1={'PASS' if judge['S1'] else 'FAIL'} "
                 f"S2={'PASS' if judge['S2'] else 'FAIL'}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) 湿态窗 sh1_in 偏差漂移
    ax = axes[1]
    for v, rec in (("rb", rb_recs[0]), ("q0", q0[0])):
        _, errs_sh1, _, pm0w = rec["_arrs"]
        mask = pm0w <= P_CRIT
        m = errs_sh1[mask].mean(0)
        se = errs_sh1[mask].std(0) / np.sqrt(mask.sum())
        ax.plot(steps, m, color=C[v], lw=1.5, label=f"{v} (n={mask.sum()})")
        ax.fill_between(steps, m - se, m + se, color=C[v], alpha=0.12)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("wet windows — sh1_in bias vs step")
    ax.set_xlabel("steps (×10 s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) 湿态窗 main 误差漂移
    ax = axes[2]
    for v, rec in (("rb", rb_recs[0]), ("q0", q0[0])):
        errs_main, _, _, pm0w = rec["_arrs"]
        mask = pm0w <= P_CRIT
        m = errs_main[mask].mean(0)
        se = errs_main[mask].std(0) / np.sqrt(mask.sum())
        ax.plot(steps, m, color=C[v], lw=1.5, label=f"{v} (n={mask.sum()})")
        ax.fill_between(steps, m - se, m + se, color=C[v], alpha=0.12)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("wet windows — main err vs step")
    ax.set_xlabel("steps (×10 s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig12_seg0.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig12_seg0.png", flush=True)


# ---------------- 主流程 ----------------


def fig13_confirm(recs, judge):
    plt = _setup_plot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    layers = [("all", "rollout overall rmse", BASE["v2"]["all"]),
              ("dry", "rollout dry rmse", BASE["v2"]["dry"]),
              ("wet", "rollout wet rmse", BASE["v2"]["wet"])]
    for k, (layer, title, v2v) in enumerate(layers):
        ax = axes[k]
        for vi, v in enumerate(("qh", "qspl")):
            if layer == "all":
                vals = [r["rollout"]["rmse_main"] for r in recs[v]]
            else:
                vals = [r["strat"][layer]["rmse_main"] for r in recs[v]]
            x = np.arange(3) + vi * 0.24
            ax.bar(x, vals, width=0.2, color=C[v], label=f"{v} ({VSPEC[v][3]})")
            ax.text(x.mean(), max(vals) + 0.05, f"{np.mean(vals):.2f}±{np.std(vals):.2f}",
                    ha="center", fontsize=7)
        ax.axhline(v2v, color="0.4", ls="--", lw=1.2, label=f"v2 {v2v}")
        ax.set_title(title)
        ax.set_xticks(np.arange(3) + 0.24, ["s0", "s1", "s2"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Step④ confirm 3seeds — C1={'PASS' if judge['C1'] else 'FAIL'} "
                 f"C2={'PASS' if judge['C2'] else 'FAIL'} C3={'PASS' if judge['C3'] else 'FAIL'}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig13_confirm.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig13_confirm.png", flush=True)


def _dump(summ, step, fast):
    tag = ".fast" if fast else ""
    with open(os.path.join(OUT, f"refine_{step}_summary{tag}.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(f"[json] out/refine_{step}_summary{tag}.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["repro", "inject", "seg0", "confirm", "smoke"])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    df, model0, mu_o, sd_o, pm_roll, n_roll = load_ctx(args)
    summ = {"step": args.step, "date": time.strftime("%Y-%m-%d %H:%M"),
            "variants": {}, "judge": {}}
    recs = {}

    if args.step == "smoke":
        # 冒烟: 仅新模式 2ep + rollout 120 步, 崩溃检测用 (rollout 截断仅此模式)
        for v in ("qtm", "qh", "qcon", "qspl", "q0"):
            run_variant(df, model0, v, [0], True, n_roll, mu_o, sd_o, pm_roll, reuse=False)
        print("[smoke] all new modes OK", flush=True)
        return

    if args.step == "repro":
        specs = {"ra": [0, 1, 2], "rb": [0, 1, 2], "rc": [0, 1, 2]}
        for v, seeds in specs.items():
            recs[v] = run_variant(df, model0, v, seeds, args.fast, n_roll, mu_o, sd_o, pm_roll,
                                  reuse=not args.fast)
        summ["variants"] = {v: [{k: x for k, x in r.items() if not k.startswith("_")}
                                for r in recs[v]] for v in recs}
        judge = judge_repro(recs)
        summ["judge"] = judge
        _dump(summ, "repro", args.fast)
        if not args.fast:
            fig10_repro(recs, judge)
        print("=== Step① 判定 ===")
        print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    elif args.step == "inject":
        for v in ("rb", "qtm", "qh", "qcon", "qspl"):
            recs[v] = run_variant(df, model0, v, [0], args.fast, n_roll, mu_o, sd_o, pm_roll,
                                  reuse=not args.fast)
        summ["variants"] = {v: [{k: x for k, x in r.items() if not k.startswith("_")}
                                for r in recs[v]] for v in recs}
        judge = judge_inject(recs)
        summ["judge"] = judge
        _dump(summ, "inject", args.fast)
        if not args.fast:
            lam_hist = collect_lam(df, model0, load_res_ckpt("qspl"))
            fig11_inject(recs, judge, lam_hist)
        print("=== Step② 判定 ===")
        print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)

    elif args.step == "seg0":
        recs["q0"] = run_variant(df, model0, "q0", [0, 1, 2], args.fast, n_roll,
                                 mu_o, sd_o, pm_roll, reuse=not args.fast)
        # rb 参考: 优先取 repro summary, 否则重评 rb seed0
        p = os.path.join(OUT, "refine_repro_summary.json")
        rb_recs = None
        if os.path.exists(p):
            with open(p) as f:
                rs = json.load(f)
            rb_ref = {"dry": rs["judge"]["rb_dry_mean"],
                      "wet": float(np.mean(rs["judge"]["rb_wet_vals"]))}
            print(f"[seg0] rb ref from repro summary: {rb_ref}", flush=True)
        else:
            rb_recs = run_variant(df, model0, "rb", [0], args.fast, n_roll,
                                  mu_o, sd_o, pm_roll, reuse=not args.fast)
            rb_ref = {"dry": rb_recs[0]["strat"]["dry"]["rmse_main"],
                      "wet": rb_recs[0]["strat"]["wet"]["rmse_main"]}
        summ["variants"] = {"q0": [{k: x for k, x in r.items() if not k.startswith("_")}
                                   for r in recs["q0"]]}
        judge = judge_seg0(recs, rb_ref)
        summ["judge"] = judge
        _dump(summ, "seg0", args.fast)
        if not args.fast:
            if rb_recs is None:
                rb_recs = run_variant(df, model0, "rb", [0], False, n_roll,
                                      mu_o, sd_o, pm_roll, reuse=True)
            fig12_seg0(recs, judge, rb_recs)
        print("=== Step③ 判定 ===")
        print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)


    elif args.step == "confirm":
        for v in ("qh", "qspl"):
            recs[v] = run_variant(df, model0, v, [0, 1, 2], args.fast, n_roll,
                                  mu_o, sd_o, pm_roll, reuse=not args.fast)
        summ["variants"] = {v: [{k: x for k, x in r.items() if not k.startswith("_")}
                                for r in recs[v]] for v in recs}
        judge = judge_confirm(recs)
        summ["judge"] = judge
        _dump(summ, "confirm", args.fast)
        if not args.fast:
            fig13_confirm(recs, judge)
        print("=== Step④ 判定 ===")
        print(json.dumps(judge, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
