"""Plot main-steam-temperature (final_outlet_temp) prediction curves.

Exploratory probe (2026-08-25, execution-side). Protocol identical to the
accuracy-vs-blackbox pack (probe_notes_20260824 §4):
- sideA record, val split, 256 windows (sample_windows seed 50_000)
- oracle boundary, H=18 (180 s), target = main steam ch4 only
- arms: physics norew seed0 (t1_closure_cons_norew) vs v05 iTransformer seed0

Ranks windows by PHYSICS-arm H18 MAE on ch4, plots the worst ones
(individual case curves: history tail + actual + both predictions),
plus one overview (per-step MAE + per-window scatter).
Writes JSON + PNGs under plots_mainsteam/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import STEAM_FLOW_INDEX
from src.final_wm.contracts import BOUNDARY_ELEMENTS, OBSERVATION_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms
import experiments.final_wm.v05_blackbox_baselines as v05

DEVICE = "cuda"
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")  # 4
H = ms.HORIZON
W = ms.HISTORY_STEPS
N_WIN = 256
SEED = 50_000
N_WORST = 6
OUT = ROOT / "results/final_wm/probes_20260824/plots_mainsteam"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "predictions_cache.npz"

torch.backends.cuda.matmul.allow_tf32 = True
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")

if CACHE.exists():
    z = np.load(CACHE)
    phys_pred = torch.from_numpy(z["phys_pred"])
    actual = torch.from_numpy(z["actual"])
    bb_pred = torch.from_numpy(z["bb_pred"])
    loads = torch.from_numpy(z["loads"])
    days = torch.from_numpy(z["days"])
    print("[cache] loaded predictions from", CACHE)
else:
    # ---- sample the production eval windows ONCE (identical for both arms) ----
    gen = torch.Generator().manual_seed(SEED)
    batches, done = [], 0
    while done < N_WIN:
        bsz = min(32, N_WIN - done)
        batches.append(sample_windows(record, SPLIT_VAL, bsz, W, H, gen))
        done += bsz

    # ---- physics arm (norew, seed0, oracle) ----
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
    spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative_norew",
                    epochs=60, patience=10)
    phys = build_world_model(spec, props).to(DEVICE)
    phys.load_state_dict(torch.load(
        ROOT / "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed0.pt",
        map_location=DEVICE, weights_only=False)["state_dict"])
    phys.eval()

    phys_pred, actual, loads, days = [], [], [], []
    with torch.no_grad():
        for b in batches:
            hist = b.history.__class__(
                obs=b.history.obs.to(DEVICE),
                actions=b.history.actions.to(DEVICE),
                boundary=b.history.boundary.to(DEVICE),
            )
            r = phys.forecast(hist, b.future_actions.to(DEVICE),
                              boundary_mode="oracle",
                              true_future_boundary=b.future_boundary.to(DEVICE))
            phys_pred.append(r.temps_mu[:, :, CH].cpu())
            actual.append(b.future_obs[:, :, CH].cpu())
            loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX].cpu())
            days.append(b.day_ids)
    phys_pred = torch.cat(phys_pred)
    actual = torch.cat(actual)
    loads = torch.cat(loads)
    days = torch.cat(days)

    # ---- blackbox (iTransformer, seed0) ----
    mean, std = v05.channel_stats(record)
    bank = v05.build_train_bank(record, DEVICE)
    bb = v05.train_one("itransformer", v05.ITransformerForecaster(), bank,
                       record, mean, std, DEVICE, 0)
    del bank
    torch.cuda.empty_cache()

    bb_pred = []
    with torch.no_grad():
        for b in batches:
            hist = torch.cat([b.history.obs, b.history.boundary,
                              b.history.actions], dim=-1).to(DEVICE)
            fut = torch.cat([b.future_actions, b.future_boundary], dim=-1).to(DEVICE)
            hist_n = (hist - mean.to(DEVICE)) / std.to(DEVICE)
            mu_c = hist[:, :, v05.TARGET].mean(dim=1, keepdim=True)
            bb_pred.append((bb(hist_n, fut) + mu_c).cpu())
    bb_pred = torch.cat(bb_pred)
    del bb, phys
    torch.cuda.empty_cache()

    np.savez_compressed(CACHE, phys_pred=phys_pred.numpy(), actual=actual.numpy(),
                        bb_pred=bb_pred.numpy(), loads=loads.numpy(),
                        days=days.numpy())
    print("[cache] saved predictions to", CACHE)

# ---- ranking ----
phys_mae = (phys_pred - actual).abs().mean(dim=1).numpy()
bb_mae = (bb_pred - actual).abs().mean(dim=1).numpy()
order = np.argsort(-phys_mae)
worst_idx = order[:N_WORST]
med_idx = int(np.argsort(np.abs(phys_mae - np.median(phys_mae)))[0])
edges = np.quantile(loads.numpy(), np.linspace(0, 1, 6)[1:-1])
q_of = lambda v: int(np.clip(np.digitize(v, edges), 0, 4)) + 1

print("== per-window H18 MAE on main steam (ch4), physics-ranked ==")
print(f"phys overall {phys_mae.mean():.3f} C | bb overall {bb_mae.mean():.3f} C")
print(f"phys worst {phys_mae.max():.3f} | bb worst on same windows {bb_mae[worst_idx].max():.3f}")
for i in worst_idx:
    print(f"  #{i} day={int(days[i])} load={loads[i]:6.0f} kg/s Q{q_of(loads[i])} "
          f"phys={phys_mae[i]:.2f} bb={bb_mae[i]:.2f}")
print(f"  typical #{med_idx} day={int(days[med_idx])} load={loads[med_idx]:6.0f} "
      f"Q{q_of(loads[med_idx])} phys={phys_mae[med_idx]:.2f} bb={bb_mae[med_idx]:.2f}")

json.dump({
    "protocol": "sideA val 256w(seed50k) oracle H18 ch4, physics norew seed0 vs iTransformer seed0",
    "phys_overall_mae": float(phys_mae.mean()),
    "bb_overall_mae": float(bb_mae.mean()),
    "worst_windows": [{"idx": int(i), "day": int(days[i]), "load": float(loads[i]),
                       "q": q_of(loads[i]), "phys_mae": float(phys_mae[i]),
                       "bb_mae": float(bb_mae[i])} for i in worst_idx],
    "typical_window": {"idx": med_idx, "day": int(days[med_idx]),
                       "load": float(loads[med_idx]), "q": q_of(loads[med_idx]),
                       "phys_mae": float(phys_mae[med_idx]),
                       "bb_mae": float(bb_mae[med_idx])},
}, open(OUT / "window_ranking.json", "w"), indent=2)

# ---- plotting helpers ----
def ax_style(ax):
    ax.grid(alpha=0.3, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def plot_window(idx, path):
    hi = batches[idx // 32].history.obs[idx % 32, :, CH].numpy()
    xh = np.arange(-W + 1, 1)
    xf = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=150)
    ax.plot(xh, hi, color="0.6", lw=1.0, zorder=1)
    ax.plot(xf, actual[idx].numpy(), color="black", lw=1.6, zorder=4)
    ax.plot(xf, phys_pred[idx].numpy(), color="tab:blue", lw=1.4, ls="--", zorder=3)
    ax.plot(xf, bb_pred[idx].numpy(), color="tab:orange", lw=1.4, ls="-.", zorder=3)
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
    ax.axvspan(-W + 1, 0.5, color="0.5", alpha=0.05)
    lo = min(actual[idx].min().item(), phys_pred[idx].min().item(),
             bb_pred[idx].min().item())
    hi_ = max(actual[idx].max().item(), phys_pred[idx].max().item(),
              bb_pred[idx].max().item())
    pad = max(0.3, 0.06 * (hi_ - lo))
    ax.set_ylim(lo - pad, hi_ + pad)
    ax.set_xlabel("Step (10 s); history 96 steps, forecast 18 steps = 180 s")
    ax.set_ylabel("Main steam temperature (°C)")
    ax.set_title(f"Window #{idx} | day {int(days[idx])} | steam flow {loads[idx]:.0f} kg/s "
                 f"(Q{q_of(loads[idx])})")
    ax.legend([f"Actual", f"Physics WM (H18 MAE {phys_mae[idx]:.2f} °C)",
               f"iTransformer (H18 MAE {bb_mae[idx]:.2f} °C)"],
              loc="best", frameon=False, ncol=1, fontsize=8)
    ax_style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

made = [plot_window(int(i), OUT / f"worst_{k+1}_win{i}.png")
        for k, i in enumerate(worst_idx)]
made.append(plot_window(med_idx, OUT / f"typical_win{med_idx}.png"))

# ---- overview ----
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=150)
ax = axes[0]
step = np.arange(1, H + 1)
ax.plot(step, (phys_pred - actual).abs().mean(dim=0).numpy(),
        "o-", color="tab:blue", ms=3, label="Physics WM (norew)")
ax.plot(step, (bb_pred - actual).abs().mean(dim=0).numpy(),
        "s-", color="tab:orange", ms=3, label="iTransformer")
ax.set_xlabel("Forecast step (10 s)")
ax.set_ylabel("Mean |err| (°C)")
ax.set_title("Per-step MAE, main steam (256 val windows)")
ax.legend(frameon=False, fontsize=8)
ax_style(ax)
ax = axes[1]
ax.scatter(phys_mae, bb_mae, s=14, color="0.65", zorder=2, label="256 windows")
ax.scatter(phys_mae[worst_idx], bb_mae[worst_idx], s=34, marker="o",
           facecolors="none", edgecolors="crimson", lw=1.2, zorder=3,
           label="worst 6 (physics-ranked)")
ax.scatter([phys_mae[med_idx]], [bb_mae[med_idx]], s=34, marker="s",
           facecolors="none", edgecolors="seagreen", lw=1.2, zorder=3,
           label="typical")
lim = (0, max(phys_mae.max(), bb_mae.max()) * 1.05)
ax.plot([0, lim[1]], [0, lim[1]], "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Physics WM H18 MAE (°C)")
ax.set_ylabel("iTransformer H18 MAE (°C)")
ax.set_title("Per-window H18 MAE (above diagonal = physics worse)")
ax.legend(frameon=False, fontsize=7)
ax_style(ax)
fig.tight_layout()
ov = OUT / "overview.png"
fig.savefig(ov)
plt.close(fig)

print("PNG files:")
for p in made + [ov]:
    print(f"  {p}")
