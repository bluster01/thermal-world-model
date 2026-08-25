"""When does the blackbox win vs lose? Window-level condition analysis (2026-08-25)."""
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

from src.final_wm.analysis import SPRAY_FLOW_INDEX, STEAM_FLOW_INDEX
from src.final_wm.contracts import BOUNDARY_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows

COAL = BOUNDARY_ELEMENTS.index("coal_command")
SEPT = BOUNDARY_ELEMENTS.index("separator_temperature")

z0 = np.load(ROOT / "results/final_wm/probes_20260824/plots_mainsteam/predictions_cache.npz")
za = np.load(OUT / "armA_budget/preds_armA_budget.npz")
phys, actual, bb = za["pred"], z0["actual"], z0["bb_pred"]
phys0 = z0["phys_pred"]
loads, days = z0["loads"], z0["days"]

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
gen = torch.Generator().manual_seed(50_000)
done = 0; batches = []
while done < 256:
    bsz = min(32, 256 - done)
    batches.append(sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen))
    done += bsz

hist_tail = torch.cat([b.history.obs[:, :, 4] for b in batches]).numpy()
v1 = torch.cat([torch.cat([b.history.actions[:, :, 0], b.future_actions[:, :, 0]], 1)
                for b in batches]).numpy()
v2 = torch.cat([torch.cat([b.history.actions[:, :, 1], b.future_actions[:, :, 1]], 1)
                for b in batches]).numpy()
flow = torch.cat([torch.cat([b.history.boundary[:, :, STEAM_FLOW_INDEX],
                             b.future_boundary[:, :, STEAM_FLOW_INDEX]], 1)
                  for b in batches]).numpy()
spray = torch.cat([torch.cat([b.history.boundary[:, :, SPRAY_FLOW_INDEX],
                              b.future_boundary[:, :, SPRAY_FLOW_INDEX]], 1)
                   for b in batches]).numpy()
coal = torch.cat([torch.cat([b.history.boundary[:, :, COAL],
                             b.future_boundary[:, :, COAL]], 1)
                  for b in batches]).numpy()
sept = torch.cat([torch.cat([b.history.boundary[:, :, SEPT],
                             b.future_boundary[:, :, SEPT]], 1)
                  for b in batches]).numpy()

maeP = np.abs(phys - actual).mean(axis=1)
maeB = np.abs(bb - actual).mean(axis=1)
mae0 = np.abs(phys0 - actual).mean(axis=1)
diff = maeP - maeB
bb_win = diff > 0.1
ph_win = diff < -0.1
tie = ~(bb_win | ph_win)
print(f"bb wins (diff>0.1): {bb_win.sum()} | tie: {tie.sum()} | phys wins (diff<-0.1): {ph_win.sum()}")
print(f"mean MAE: bb {maeB.mean():.3f} | armA {maeP.mean():.3f} | prod {mae0.mean():.3f}")

def feats(m):
    fut = np.arange(96, 114)
    return dict(
        n=int(m.sum()),
        maeP=maeP[m].mean(), maeB=maeB[m].mean(),
        load=loads[m].mean(), dflow=np.abs(flow[m][:, -1] - flow[m][:, 96]).mean(),
        dv1=np.abs(v1[m][:, -1] - v1[m][:, 96]).mean(),
        dv2=np.abs(v2[m][:, -1] - v2[m][:, 96]).mean(),
        dspray=np.abs(spray[m][:, -1] - spray[m][:, 96]).mean(),
        dcoal=np.abs(coal[m][:, -1] - coal[m][:, 96]).mean(),
        dsept=np.abs(sept[m][:, -1] - sept[m][:, 96]).mean(),
        hist_trend=(hist_tail[m][:, -1] - hist_tail[m][:, 0]).mean(),
        act_move=(actual[m][:, -1] - actual[m][:, 0]).mean(),
        act_range=(actual[m].max(axis=1) - actual[m].min(axis=1)).mean(),
        spray_level=spray[m][:, 95].mean(),
    )

print("\n== condition profile (means) ==")
hdr = f"{'':14s} | {'bb wins':>10s} | {'tie':>10s} | {'phys wins':>10s}"
print(hdr)
for k in ("n", "maeP", "maeB", "load", "dflow", "dv1", "dv2", "dspray", "dcoal",
          "dsept", "hist_trend", "act_move", "act_range", "spray_level"):
    f = {g: feats(m)[k] for g, m in (("b", bb_win), ("t", tie), ("p", ph_win))}
    print(f"{k:14s} | {f['b']:10.2f} | {f['t']:10.2f} | {f['p']:10.2f}")

# phys-win windows detail
order_p = np.argsort(diff)  # most phys-favorable first
print("\n== top phys-win windows ==")
for i in order_p[:8]:
    print(f"  win{i} day={int(days[i])} load={loads[i]:.0f} | phys {maeP[i]:.2f} bb {maeB[i]:.2f} "
          f"| dflow={abs(flow[i,-1]-flow[i,96]):.1f} dv1={abs(v1[i,-1]-v1[i,96]):.3f} "
          f"dv2={abs(v2[i,-1]-v2[i,96]):.3f} dspray={abs(spray[i,-1]-spray[i,96]):.1f} "
          f"act_move={actual[i,-1]-actual[i,0]:+.2f}")

# load-quintile win rates
edges = np.quantile(loads, np.linspace(0, 1, 6)[1:-1])
q = np.clip(np.digitize(loads, edges), 0, 4)
print("\n== by load quintile ==")
for b in range(5):
    m = q == b
    print(f"  Q{b+1} n={m.sum():3d} | phys {maeP[m].mean():.3f} | bb {maeB[m].mean():.3f} | "
          f"bb-win rate {(diff[m] > 0.1).mean():.2f} | phys-win rate {(diff[m] < -0.1).mean():.2f}")

# plots: top 4 phys-win windows
for k, idx in enumerate(order_p[:4]):
    xh = np.arange(-95, 1); xf = np.arange(1, 19)
    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=150)
    ax.plot(xh, hist_tail[idx], color="0.6", lw=1.0)
    ax.plot(xf, actual[idx], color="black", lw=1.6, label="actual")
    ax.plot(xf, phys[idx], color="tab:blue", lw=1.5, label=f"armA ({maeP[idx]:.2f})")
    ax.plot(xf, bb[idx], color="tab:orange", lw=1.2, ls="-.", label=f"bb ({maeB[idx]:.2f})")
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":"); ax.axvspan(-95, 0.5, color="0.5", alpha=0.05)
    lo = min(actual[idx].min(), phys[idx].min(), bb[idx].min())
    hi = max(actual[idx].max(), phys[idx].max(), bb[idx].max())
    pad = max(0.3, 0.06 * (hi - lo)); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Step (10 s)"); ax.set_ylabel("Main steam (°C)")
    ax.set_title(f"Phys-beats-bb window #{idx} | day {int(days[idx])} | flow {loads[idx]:.0f} kg/s")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"physwin{k+1}_win{idx}.png"); plt.close(fig)
    print(f"[plot] physwin{k+1}_win{idx}.png")
