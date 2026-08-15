#!/usr/bin/env python3
"""06_compare.py: 修复A前后分层对比 + 预注册门槛判决。

用法:
  python 06_compare.py --pre out/strat_pre_fixA.json --post out/strat_post_fixA.json \
                       --fig out/figs/fig4_fixA_compare.png

判决: P1 rollout rmse_main≤2.48 (3 seeds 均值) | P2 band≤0.5% | P3 viol_phys≤46.11%
      （门槛与 step1_summary.json 预注册一致；rollout 指标从 results_e0_seed*.json 读）
"""
import argparse
import json
import os

import numpy as np

P1 = 2.48
P2 = 0.005
P3 = 0.4611
NAMES = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]


def load_results():
    """读 3 seeds 的 results_e0_seed*.json 聚合（与 step1_summary 同口径）。"""
    rows = []
    for s in (0, 1, 2):
        with open(f"out/results_e0_seed{s}.json") as f:
            rows.append(json.load(f))
    agg = {
        "rmse_main": round(float(np.mean([r["rmse_main"] for r in rows])), 3),
        "rmse_all": round(float(np.mean([r["rmse_all"] for r in rows])), 3),
        "band_viol_frac": round(float(np.mean([r["band_viol_frac"] for r in rows])), 4),
        "single_step_rmse_main_C": round(float(np.mean([r["single_step_rmse_main_C"] for r in rows])), 3),
        "window60_rmse_main_C": round(float(np.mean([r["window60_rmse_main_C"] for r in rows])), 3),
        "seeds_rmse_main": [round(r["rmse_main"], 3) for r in rows],
        "seeds_band": [round(r["band_viol_frac"], 4) for r in rows],
    }
    return agg, rows


def verdict(agg, viol_phys_agg):
    p1 = agg["rmse_main"] <= P1
    p2 = agg["band_viol_frac"] <= P2
    p3 = viol_phys_agg <= P3
    print("\n=== 预注册门槛判决（3 seeds 均值） ===")
    print(f"P1 rollout rmse_main: {agg['rmse_main']} (≤{P1}) -> {'PASS' if p1 else 'FAIL'}")
    print(f"P2 band_viol_frac:    {agg['band_viol_frac']:.4f} (≤{P2}) -> {'PASS' if p2 else 'FAIL'}")
    print(f"P3 viol_phys:         {viol_phys_agg:.4f} (≤{P3}) -> {'PASS' if p3 else 'FAIL'}")
    print(f"整体: {'PASS' if (p1 and p2 and p3) else 'FAIL'}")
    return p1 and p2 and p3


def make_fig(pre, post, agg, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    x = np.arange(5)
    ax = axes[0]
    for k, (label, d) in enumerate((("wet pre", pre["rollout"]["wet"]),
                                    ("wet post", post["rollout"]["wet"]),
                                    ("dry pre", pre["rollout"]["dry"]),
                                    ("dry post", post["rollout"]["dry"]))):
        if d is None:
            continue
        b = np.array(d["bias_5"])
        colors = {"wet pre": "#9dc3e6", "wet post": "#2e75b6",
                  "dry pre": "#f4b183", "dry post": "#c55a11"}
        ax.bar(x + (k - 1.5) * 0.2, b, 0.2, label=label, color=colors[label], alpha=0.95)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("Rollout bias (°C)")
    ax.set_title("Fix A (mode-dependent k0): per-mode rollout bias, pre vs post")
    ax.legend(fontsize=8, ncol=4)
    ax = axes[1]
    xw = np.arange(2)
    for k, (label, d) in enumerate((("pre", pre["windowed"]), ("post", post["windowed"]))):
        vals = [d[m]["win60_rmse_main"] if d[m] else np.nan for m in ("wet", "dry")]
        ax.bar(xw + (k - 0.5) * 0.32, vals, 0.3, label=label,
               color=["#9dc3e6", "#2e75b6"][k], alpha=0.95)
        for i, v in enumerate(vals):
            ax.text(xw[i] + (k - 0.5) * 0.32, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xw)
    ax.set_xticklabels(["wet windows", "dry windows"])
    ax.set_ylabel("Window-60 RMSE main (°C)")
    ax.set_title(f"Windowed eval by mode — post-fix overall rmse_main={agg['rmse_main']} "
                 f"(P1≤2.48), band={agg['band_viol_frac']:.1%} (P2≤0.5%)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[fig] {fig_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--fig", required=True)
    args = ap.parse_args()
    pre = json.load(open(args.pre))
    post = json.load(open(args.post))
    agg, rows = load_results()
    viol = np.mean([r["order_viol_any_frac"] for r in rows])  # 旧对子（信息用）
    print("=== 3 seeds 聚合（results_e0_*，修复后） ===")
    print(f"rmse_main: {agg['seeds_rmse_main']} (mean {agg['rmse_main']}) | "
          f"single_step: {agg['single_step_rmse_main_C']} | win60: {agg['window60_rmse_main_C']}")
    print(f"band: {agg['seeds_band']} (mean {agg['band_viol_frac']}) | order_viol(旧对子): {viol:.4f}")
    phys_agg = float(np.mean([post["rollout"][m]["viol_phys_frac"] for m in ("wet", "dry")
                              if post["rollout"][m] is not None]))
    # 物理对违例的 rollout 全段口径（含湿+干，与 pre 同口径对比）
    pre_phys = float(np.mean([pre["rollout"][m]["viol_phys_frac"] for m in ("wet", "dry")
                              if pre["rollout"][m] is not None]))
    print(f"viol_phys(rollout全段, 物理对): pre={pre_phys:.4f} post={phys_agg:.4f} (P3≤{P3})")
    print("=== 分层对比（pre → post） ===")
    for m in ("wet", "dry"):
        pr, po = pre["rollout"][m], post["rollout"][m]
        if pr and po:
            print(f"[{m}] rmse_main {pr['rmse_main']} -> {po['rmse_main']} | "
                  f"band {pr['band_viol_frac']:.3f} -> {po['band_viol_frac']:.3f} | "
                  f"bias_5 {pr['bias_5']} -> {po['bias_5']}")
        pw, ww = pre["windowed"][m], post["windowed"][m]
        if pw and ww:
            print(f"[{m}窗] win60_rmse {pw['win60_rmse_main']} -> {ww['win60_rmse_main']} | "
                  f"首步bias sh1_in {pw['first_bias_sh1in']} -> {ww['first_bias_sh1in']}")
    make_fig(pre, post, agg, args.fig)
    # 学习参数对比（湿/干 k0 分离度）
    import json as _j
    p0 = _j.load(open("out/params_e0_seed0.json"))["learned"]
    print(f"\nk0(wet)={p0['k0']/1e6:.3f}e6 k0d(dry)={p0['k0d']/1e6:.3f}e6 "
          f"ratio={p0['k0']/p0['k0d']:.3f} (分层需求比≈557/450={557/450:.3f})")
    verdict(agg, phys_agg)


if __name__ == "__main__":
    main()
