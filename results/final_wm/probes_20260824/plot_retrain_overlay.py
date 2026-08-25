"""Overlay original seed0 vs armA-retrained curves (2026-08-25)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/final_wm/probes_20260824/retrain_probe"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

z0 = np.load(ROOT / "results/final_wm/probes_20260824/plots_mainsteam/predictions_cache.npz")
za = np.load(OUT / "armA_budget/preds_armA_budget.npz")
phys0, actual, bb = z0["phys_pred"], z0["actual"], z0["bb_pred"]
physA = za["pred"]
loads, days = z0["loads"], z0["days"]

mae0 = np.abs(phys0 - actual).mean(axis=1)
maeA = np.abs(physA - actual).mean(axis=1)
print(f"orig overall {mae0.mean():.3f} | armA overall {maeA.mean():.3f} | "
      f"bb overall {np.abs(bb-actual).mean(axis=1).mean():.3f}")

# ---- 1. val curves ----
rows0 = [json.loads(l) for l in open(ROOT / "artifacts/final_wm/ledger.jsonl")]
c0 = [(x["epoch"], x["val_nll"]) for x in rows0
      if x.get("run_id") == "t1_closure_cons_norew_seed0" and "val_nll" in x]
rowsA = [json.loads(l) for l in open(OUT / "armA_budget/ledger.jsonl")]
cA = [(x["epoch"], x["val_nll"]) for x in rowsA
      if x.get("run_id") == "t1_closure_cons_norew_seed0" and "val_nll" in x]
fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=150)
ax.plot([e for e, _ in c0], [v for _, v in c0], "o-", ms=3, color="0.45",
        label="production seed0 (epochs 60 / patience 10, stopped @41)")
ax.plot([e for e, _ in cA], [v for _, v in cA], "o-", ms=3, color="tab:blue",
        label="armA retrain (epochs 120 / patience 20, stopped @72)")
b0 = min((v for _, v in c0))
bA = min((v for _, v in cA))
ax.axhline(b0, color="0.45", ls=":", lw=0.8)
ax.axhline(bA, color="tab:blue", ls=":", lw=0.8)
ax.text(2, b0 + 0.03, f"best {b0:.3f}", color="0.45", fontsize=8)
ax.text(2, bA + 0.03, f"best {bA:.3f}", color="tab:blue", fontsize=8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Val NLL")
ax.set_title("t1_closure_cons_norew seed0: val curve, production vs armA budget retrain")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "val_curve_prod_vs_armA.png"); plt.close(fig)

# ---- 2. per-window scatter ----
fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
ax.scatter(mae0, maeA, s=14, color="0.65", zorder=2)
lim = (0, max(mae0.max(), maeA.max()) * 1.05)
ax.plot([0, lim[1]], [0, lim[1]], "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Production seed0 H18 MAE (°C)")
ax.set_ylabel("armA retrain H18 MAE (°C)")
ax.set_title("Per-window H18 MAE (256 windows)\nbelow diagonal = retrain better")
fig.tight_layout(); fig.savefig(OUT / "per_window_scatter_armA.png"); plt.close(fig)

# ---- 3. original-worst windows overlaid ----
order = np.argsort(-mae0)
W = 96; H = 18
hist_bank = []  # history tail per window: recompute from canonical record
import torch
sys.path.insert(0, str(ROOT))
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.analysis import STEAM_FLOW_INDEX
record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
gen = torch.Generator().manual_seed(50_000)
done = 0; batches = []
while done < 256:
    bsz = min(32, 256 - done)
    batches.append(sample_windows(record, SPLIT_VAL, bsz, W, H, gen))
    done += bsz
hist_tail = torch.cat([b.history.obs[:, :, 4] for b in batches]).numpy()  # (256,96)

def plot_overlay(idx, path):
    xh = np.arange(-W + 1, 1); xf = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=150)
    ax.plot(xh, hist_tail[idx], color="0.6", lw=1.0, zorder=1)
    ax.plot(xf, actual[idx], color="black", lw=1.6, zorder=4)
    ax.plot(xf, phys0[idx], color="0.45", lw=1.3, ls="--", zorder=3)
    ax.plot(xf, physA[idx], color="tab:blue", lw=1.5, zorder=3)
    ax.plot(xf, bb[idx], color="tab:orange", lw=1.2, ls="-.", zorder=2)
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
    ax.axvspan(-W + 1, 0.5, color="0.5", alpha=0.05)
    lo = min(actual[idx].min(), phys0[idx].min(), physA[idx].min(), bb[idx].min())
    hi = max(actual[idx].max(), phys0[idx].max(), physA[idx].max(), bb[idx].max())
    pad = max(0.3, 0.06 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Step (10 s); forecast 18 steps = 180 s")
    ax.set_ylabel("Main steam temperature (°C)")
    ax.set_title(f"Window #{idx} | day {int(days[idx])} | flow {loads[idx]:.0f} kg/s")
    ax.legend(["Actual",
               f"prod seed0 (MAE {mae0[idx]:.2f})",
               f"armA retrain (MAE {maeA[idx]:.2f})",
               f"iTransformer (MAE {np.abs(bb[idx]-actual[idx]).mean():.2f})"],
              frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

made = [plot_overlay(int(i), OUT / f"overlay_worst{k+1}_win{i}.png")
        for k, i in enumerate(order[:6])]

# ---- 4. armA's own worst windows ----
orderA = np.argsort(-maeA)
made += [plot_overlay(int(i), OUT / f"overlay_armAworst{k+1}_win{i}.png")
         for k, i in enumerate(orderA[:3])]
print("armA worst:", [(int(i), round(float(maeA[i]), 2)) for i in orderA[:6]])
for p in made:
    print(p)
