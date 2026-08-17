#!/usr/bin/env python3
"""画 rollout 对比图"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
truth = None
colors = {"v0": "#c44", "v1": "#c90", "v2": "#287"}
for v in ["v0", "v1", "v2"]:
    d = np.load(os.path.join(OUT, f"rollout_{v}.npz"))
    p, t = d["preds"], d["truths"]
    if truth is None:
        truth = t
    main_p, main_t = p[:, 4], t[:, 4]
    tmin = np.arange(len(main_t)) / 6.0  # 10s → 分钟
    axes[0].plot(tmin, main_p, color=colors[v], lw=1.2, label=f"{v} pred")
    axes[1].plot(tmin, main_p - main_t, color=colors[v], lw=1.0, label=f"{v} err")
axes[0].plot(tmin, truth[:, 4], color="k", lw=0.9, ls="--", label="truth")
axes[0].axhspan(557.75, 572.13, color="g", alpha=0.08, label="band p1-p99")
axes[0].set_ylabel("main steam T (°C)")
axes[1].set_ylabel("pred - truth (°C)")
axes[1].set_xlabel("rollout time (min, 5h total)")
for ax in axes:
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.3)
fig.suptitle("1800-step recursive rollout: raw (v0) vs +phys features (v1) vs +invariant losses (v2)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rollout_compare.png"), dpi=130)
print("saved", os.path.join(OUT, "rollout_compare.png"))

# 汇总表
rows = []
for v in ["v0", "v1", "v2"]:
    r = json.load(open(os.path.join(OUT, f"results_{v}.json")))
    rows.append([v, r["single_rmse_main_C"], round(r["rmse_main"], 2),
                 round(r["maxerr_main"], 2), round(r["band_viol_frac"] * 100, 2),
                 round(r["rmse_all"], 2)])
print("\nvariant | single-step RMSE(°C) | rollout RMSE(°C) | max err(°C) | band-viol(%) | rmse_all")
for r in rows:
    print(" | ".join(str(x) for x in r))
