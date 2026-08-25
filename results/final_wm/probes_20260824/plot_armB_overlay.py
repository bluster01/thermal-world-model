"""armB overlay + lever comparison (2026-08-25)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/final_wm/probes_20260824/retrain_probe"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

z0 = np.load(ROOT / "results/final_wm/probes_20260824/plots_mainsteam/predictions_cache.npz")
za = np.load(OUT / "armA_budget/preds_armA_budget.npz")
zb = np.load(OUT / "armB_batch/preds_armB_batch.npz")
phys0, actual, bb = z0["phys_pred"], z0["actual"], z0["bb_pred"]
physA, physB = za["pred"], zb["pred"]
loads, days = z0["loads"], z0["days"]

mae0 = np.abs(phys0 - actual).mean(axis=1)
maeA = np.abs(physA - actual).mean(axis=1)
maeB = np.abs(physB - actual).mean(axis=1)
mbb = np.abs(bb - actual).mean(axis=1)
print(f"prod {mae0.mean():.3f} | armA {maeA.mean():.3f} | armB {maeB.mean():.3f} | bb {mbb.mean():.3f}")

# scatter armA vs armB
fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
ax.scatter(maeA, maeB, s=14, color="0.65", zorder=2)
lim = (0, max(maeA.max(), maeB.max()) * 1.05)
ax.plot([0, lim[1]], [0, lim[1]], "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("armA budget (120/20) H18 MAE (°C)")
ax.set_ylabel("armB batch64 H18 MAE (°C)")
ax.set_title("Per-window H18 MAE\nabove diagonal = armA better")
fig.tight_layout(); fig.savefig(OUT / "scatter_armA_vs_armB.png"); plt.close(fig)

# overlays with all four curves on the top-3 windows
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
gen = torch.Generator().manual_seed(50_000)
done = 0; batches = []
while done < 256:
    bsz = min(32, 256 - done)
    batches.append(sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen))
    done += bsz
hist_tail = torch.cat([b.history.obs[:, :, 4] for b in batches]).numpy()

order = np.argsort(-mae0)
W, H = 96, 18
for k, idx in enumerate(order[:3]):
    xh = np.arange(-W + 1, 1); xf = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(7.6, 3.3), dpi=150)
    ax.plot(xh, hist_tail[idx], color="0.6", lw=1.0, zorder=1)
    ax.plot(xf, actual[idx], color="black", lw=1.6, zorder=4)
    ax.plot(xf, phys0[idx], color="0.45", lw=1.2, ls="--", zorder=3)
    ax.plot(xf, physB[idx], color="purple", lw=1.2, ls=":", zorder=3)
    ax.plot(xf, physA[idx], color="tab:blue", lw=1.5, zorder=3)
    ax.plot(xf, bb[idx], color="tab:orange", lw=1.1, ls="-.", zorder=2)
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
    ax.axvspan(-W + 1, 0.5, color="0.5", alpha=0.05)
    lo = min(actual[idx].min(), phys0[idx].min(), physA[idx].min(),
             physB[idx].min(), bb[idx].min())
    hi = max(actual[idx].max(), phys0[idx].max(), physA[idx].max(),
             physB[idx].max(), bb[idx].max())
    pad = max(0.3, 0.06 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Step (10 s); forecast 18 steps = 180 s")
    ax.set_ylabel("Main steam temperature (°C)")
    ax.set_title(f"Window #{idx} | day {int(days[idx])} | flow {loads[idx]:.0f} kg/s")
    ax.legend(["Actual", f"prod seed0 ({mae0[idx]:.2f})",
               f"armB batch64 ({maeB[idx]:.2f})",
               f"armA budget ({maeA[idx]:.2f})",
               f"iTransformer ({mbb[idx]:.2f})"],
              frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f"overlay4way_worst{k+1}_win{idx}.png")
    plt.close(fig)
    print(OUT / f"overlay4way_worst{k+1}_win{idx}.png")
