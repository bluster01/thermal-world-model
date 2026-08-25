"""Protocol-asymmetry quantification + worst-window input forensics (2026-08-25).

Questions: (1) how much of the iTransformer advantage is the InstanceNorm
persistence shortcut? (2) what inputs drive the physics arm's violent
responses in the worst windows? (3) does physics error track action/boundary
movement more than blackbox error?
"""
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
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows

z0 = np.load(ROOT / "results/final_wm/probes_20260824/plots_mainsteam/predictions_cache.npz")
za = np.load(OUT / "armA_budget/preds_armA_budget.npz")
phys0, actual, bb = z0["phys_pred"], z0["actual"], z0["bb_pred"]
physA = za["pred"]

# ---- 1. InstanceNorm-persistence baseline ----
record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
gen = torch.Generator().manual_seed(50_000)
done = 0; batches = []
while done < 256:
    bsz = min(32, 256 - done)
    batches.append(sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen))
    done += bsz
hist_tail = torch.cat([b.history.obs[:, :, 4] for b in batches]).numpy()   # (256,96)
persist = hist_tail.mean(axis=1, keepdims=True) * np.ones((256, 18))
mae_persist = np.abs(persist - actual).mean(axis=1)
mae_bb = np.abs(bb - actual).mean(axis=1)
maeA = np.abs(physA - actual).mean(axis=1)
mae0 = np.abs(phys0 - actual).mean(axis=1)
print(f"[1] InstanceNorm-persistence (const=hist mean): overall {mae_persist.mean():.3f}")
print(f"    vs bb {mae_bb.mean():.3f} | armA {maeA.mean():.3f} | prod {mae0.mean():.3f}")
print(f"    bb beats persistence on {(mae_bb < mae_persist).sum()}/256 windows, "
      f"mean improvement {(mae_persist - mae_bb).mean():.3f}")

# ---- 2+3. inputs per window ----
v1_all = torch.cat([torch.cat([b.history.actions[:, :, 0], b.future_actions[:, :, 0]], dim=1)
                    for b in batches]).numpy()        # (256, 114)
v2_all = torch.cat([torch.cat([b.history.actions[:, :, 1], b.future_actions[:, :, 1]], dim=1)
                    for b in batches]).numpy()
flow_all = torch.cat([torch.cat([b.history.boundary[:, :, STEAM_FLOW_INDEX],
                                 b.future_boundary[:, :, STEAM_FLOW_INDEX]], dim=1)
                      for b in batches]).numpy()
spray_all = torch.cat([torch.cat([b.history.boundary[:, :, SPRAY_FLOW_INDEX],
                                  b.future_boundary[:, :, SPRAY_FLOW_INDEX]], dim=1)
                       for b in batches]).numpy()
loads = z0["loads"]; days = z0["days"]

# per-window input movement in the forecast horizon (steps 97..114 = future)
def horizon_movement(x):
    fut = x[:, 96:]                       # (N,18)
    return np.abs(fut[:, -1] - fut[:, 0])

dv1, dv2 = horizon_movement(v1_all), horizon_movement(v2_all)
dflow = horizon_movement(flow_all)
dspray = horizon_movement(spray_all)

def bin_table(dv, mae, name, n=4):
    q = np.quantile(dv, np.linspace(0, 1, n + 1))
    lab = np.clip(np.digitize(dv, q[:-1]) - 1, 0, n - 1)
    print(f"  [{name}] movement bins -> physics armA MAE / bb MAE / persistence MAE:")
    for b in range(n):
        m = lab == b
        print(f"    bin{b + 1} n={m.sum():3d} d={q[b]:.3f}-{q[b+1]:.3f} | "
              f"armA {maeA[m].mean():.3f} | bb {mae_bb[m].mean():.3f} | "
              f"persist {mae_persist[m].mean():.3f}")

print("[2] movement-binned MAE (movement = |val[last]-val[first]| over 18-step horizon):")
bin_table(dv1, None, "valve1")
bin_table(dv2, None, "valve2")
bin_table(dflow, None, "steam_flow")
bin_table(dspray, None, "spray_flow")

# ---- 4. worst-window input forensics ----
order = np.argsort(-maeA)
H = 18
for k, idx in enumerate(order[:3]):
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 5.6), dpi=150,
                             sharex=True, height_ratios=[1.4, 1, 1])
    xf = np.arange(1, H + 1); xh = np.arange(-95, 1)
    ax = axes[0]
    ax.plot(xh, hist_tail[idx], color="0.6", lw=1.0)
    ax.plot(xf, actual[idx], color="black", lw=1.6, label="actual")
    ax.plot(xf, physA[idx], color="tab:blue", lw=1.5, label=f"armA ({maeA[idx]:.2f})")
    ax.plot(xf, bb[idx], color="tab:orange", lw=1.2, ls="-.", label=f"bb ({mae_bb[idx]:.2f})")
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":"); ax.axvspan(-95, 0.5, color="0.5", alpha=0.05)
    ax.set_ylabel("Main steam (°C)")
    ax.set_title(f"Window #{idx} | day {int(days[idx])} | flow {loads[idx]:.0f} kg/s")
    ax.legend(frameon=False, fontsize=7.5, ncol=3)
    ax = axes[1]
    ax.plot(np.arange(-95, H + 1), v1_all[idx], label="valve1", lw=1.2)
    ax.plot(np.arange(-95, H + 1), v2_all[idx], label="valve2", lw=1.2)
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_ylabel("Valve pos (-)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax = axes[2]
    ax.plot(np.arange(-95, H + 1), flow_all[idx], color="tab:green", label="steam flow", lw=1.2)
    ax.set_ylabel("Steam flow (kg/s)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlabel("Step (10 s)")
    fig.tight_layout()
    p = OUT / f"inputs_worst{k+1}_win{idx}.png"
    fig.savefig(p); plt.close(fig)
    print(f"[4] {p}")
    print(f"    win{idx}: dv1={dv1[idx]:.3f} dv2={dv2[idx]:.3f} "
          f"dflow={dflow[idx]:.1f} dspray={dspray[idx]:.1f} | "
          f"armA {maeA[idx]:.2f} bb {mae_bb[idx]:.2f}")
