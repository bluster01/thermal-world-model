"""Grid-scan response-surface probe (execution-side, exploratory).

Production arm closure_cons_norew (sideB seed0). Sweeps (steam_flow x v1)
on a 20x20 grid, constant-oracle boundary + constant action, 240-step
rollout, terminal temperature (last-30-step mean) per channel, plus a
finite-difference gain surface at +5% valve opening.

Output: /tmp/grid_out/grid_scan.npz + PNG surfaces. Not part of the frozen
matrix protocol; exploration only.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.data import CanonicalRecord, SPLIT_VAL, sample_windows
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
HIST = ms.HISTORY_STEPS
HORIZON = 240  # 40 min
N_LOAD, N_VALVE = 20, 20
DELTA_V = 0.05
CHANNELS = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main_steam"]
OUT = Path("/tmp/grid_out")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(1234)
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
spec = ms._base(
    "t1", "closure_cons_norew", 0, boundary_mode="oracle",
    initial_state_mode="hybrid", closure_mode="conservative_norew",
    epochs=60, patience=10,
)
model = build_world_model(spec, props).to(DEVICE)
ckpt = torch.load(
    ROOT / "artifacts/final_wm_sideB/checkpoints/t1_closure_cons_norew_seed0.pt",
    map_location=DEVICE, weights_only=False,
)["state_dict"]
model.load_state_dict(ckpt)
model.eval()
print("model loaded", flush=True)

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideB.npz")
gen = torch.Generator().manual_seed(777)
batch = sample_windows(record, SPLIT_VAL, 64, HIST, 2, gen)

# pick the window with the widest v1 range (wide support box)
v1_span = batch.history.actions[:, :, 0].amax(1) - batch.history.actions[:, :, 0].amin(1)
i = int(torch.argmax(v1_span).item())
win = HistoryWindow(
    obs=batch.history.obs[i:i + 1],
    actions=batch.history.actions[i:i + 1],
    boundary=batch.history.boundary[i:i + 1],
)
print(f"window idx={i}; v1 range "
      f"[{win.actions[0,:,0].min().item():.3f}, {win.actions[0,:,0].max().item():.3f}]",
      flush=True)

# ---- grids ----
val_mask = (record.split == SPLIT_VAL).numpy()
sf = record.boundary[:, 0].numpy()[val_mask]
v1_all = record.actions[:, 0].numpy()[val_mask]
load_grid = np.quantile(sf, np.linspace(0.05, 0.95, N_LOAD))
hist_v1 = win.actions[0, :, 0].numpy()
v_lo = float(hist_v1.min()) + 0.02
v_hi = float(hist_v1.max()) - 0.02
valve_grid = np.linspace(v_lo, v_hi, N_VALVE)
v2_base = float(win.actions[0, -1, 1].numpy())
print(f"load grid [{load_grid[0]:.1f}, {load_grid[-1]:.1f}] kg/s; "
      f"v1 grid [{valve_grid[0]:.3f}, {valve_grid[-1]:.3f}]; v2={v2_base:.3f}", flush=True)

B = N_LOAD * N_VALVE
loads = np.repeat(load_grid, N_VALVE)          # (B,)
valves = np.tile(valve_grid, N_LOAD)           # (B,)

hist_b = HistoryWindow(
    obs=win.obs.expand(B, -1, -1).contiguous().to(DEVICE),
    actions=win.actions.expand(B, -1, -1).contiguous().to(DEVICE),
    boundary=win.boundary.expand(B, -1, -1).contiguous().to(DEVICE),
)
bnd_last = win.boundary[0, -1].clone()  # (7,)


def make_bnd(v1_vals, load_vals):
    b = bnd_last.repeat(B, HORIZON, 1)
    b[:, :, 0] = torch.tensor(load_vals, dtype=torch.float32).unsqueeze(1)
    return b.to(DEVICE)


def make_act(v1_vals):
    a = torch.zeros(B, HORIZON, 2, device=DEVICE)
    a[:, :, 0] = torch.tensor(v1_vals, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    a[:, :, 1] = v2_base
    return a


@torch.no_grad()
def run_surface(v1_vals):
    bnd = make_bnd(v1_vals, loads)
    acts = make_act(v1_vals)
    res = model.forecast(
        hist_b, acts, boundary_mode="oracle",
        true_future_boundary=bnd,
    )
    T = res.temps_mu                       # (B, H, 5)
    Tend = T[:, -30:, :].mean(1)           # (B, 5)
    return Tend.cpu().numpy()


print("sweep A (base)...", flush=True)
T_base = run_surface(valves)

# gain sweep: v1 + DELTA_V, clamp to history max so we never invent support
v1_plus = np.minimum(valves + DELTA_V, hist_v1.max() - 1e-4)
dv_eff = v1_plus - valves
print("sweep B (+5% open)...", flush=True)
T_plus = run_surface(v1_plus)

gain = (T_plus - T_base) / np.clip(dv_eff, 1e-3, None)[:, None]  # degC per unit valve
gain_pct = (T_plus - T_base) / (np.clip(dv_eff, 1e-3, None)[:, None] * 100.0) * 100.0  # degC per 100% = slope

np.savez(
    OUT / "grid_scan.npz",
    load_grid=load_grid, valve_grid=valve_grid, T_base=T_base, T_plus=T_plus,
    gain=gain, dv_eff=dv_eff, channels=np.array(CHANNELS),
)
print("saved grid_scan.npz", flush=True)

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XX, YY = np.meshgrid(load_grid, valve_grid, indexing="ij")
SF = XX.ravel()   # load (i_load * N_VALVE + i_valve) == loads
VV = YY.ravel()

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), subplot_kw={"projection": "3d"})
for row, (ch_idx, ch_name) in enumerate([(1, "SH1 outlet"), (4, "Main steam")]):
    ax_t = axes[row, 0]
    ax_g = axes[row, 1]
    Zt = T_base[:, ch_idx].reshape(N_LOAD, N_VALVE)
    Zg = gain[:, ch_idx].reshape(N_LOAD, N_VALVE)
    for ax, Z, title, cbar_lbl in [
        (ax_t, Zt, f"{ch_name} terminal temp @40min", "degC"),
        (ax_g, Zg, f"{ch_name} gain dT/dv1 (degC per unit)", "degC/unit"),
    ]:
        surf = ax.plot_surface(XX, YY, Z, cmap="viridis", edgecolor="none", antialiased=True)
        ax.set_xlabel("Steam flow [kg/s]", fontsize=9)
        ax.set_ylabel("v1 opening", fontsize=9)
        ax.set_zlabel(cbar_lbl, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=28, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
fig.suptitle("closure_cons_norew sideB seed0 -- grid response surface (exploratory)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "grid_surfaces.png", dpi=110)
print("saved grid_surfaces.png", flush=True)

# quick sanity numbers
i_lo = np.argmin(abs(load_grid - np.median(load_grid[: N_LOAD // 3])))
i_hi = np.argmin(abs(load_grid - np.median(load_grid[2 * N_LOAD // 3:])))
print("median load T_main: low-load %.1f C, high-load %.1f C" % (
    np.median(T_base[i_lo * N_VALVE:(i_lo + 1) * N_VALVE, 4]),
    np.median(T_base[i_hi * N_VALVE:(i_hi + 1) * N_VALVE, 4])))
print("median v1 gain (main steam): low %.2f high %.2f degC/unit" % (
    np.median(gain[i_lo * N_VALVE:(i_lo + 1) * N_VALVE, 4]),
    np.median(gain[i_hi * N_VALVE:(i_hi + 1) * N_VALVE, 4])))
print("DONE", flush=True)
