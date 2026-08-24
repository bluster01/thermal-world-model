"""Ramp-load closed-loop probe (execution-side, exploratory).

Load ramp scenario: 475 -> 178 kg/s in 40 min, 20 min hold, 178 -> 475 in
40 min, 20 min hold (720 steps = 120 min). All 7 boundary channels follow
the load via conditional-mean interpolation (bin_bnd). PI on v2 tracks
main-steam 571 degC. Outputs: per-channel temperature trajectories,
valve trajectories, and a time x channel x temperature surface.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.controller import CascadePIController
from src.final_wm.contracts import ControllerConfig
from src.final_wm.data import CanonicalRecord, SPLIT_VAL
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
HIST = ms.HISTORY_STEPS
T_SP = 571.0
OUT = Path("/tmp/grid_out")

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

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideB.npz")
val_mask = (record.split == SPLIT_VAL).numpy()
sf_val = record.boundary[:, 0].numpy()[val_mask]
v2_val = record.actions[:, 1].numpy()[val_mask]

N_LOAD = 24
load_grid = np.quantile(sf_val, np.linspace(0.05, 0.95, N_LOAD))
v2_lo, v2_hi = float(np.quantile(v2_val, 0.025)), float(np.quantile(v2_val, 0.975))
eps = (load_grid[1] - load_grid[0]) / 2.0
bnd_val_all = record.boundary.numpy()[val_mask]
bin_bnd = np.zeros((N_LOAD, 7), dtype=np.float32)
for k, L in enumerate(load_grid):
    m = np.abs(sf_val - L) < eps
    bin_bnd[k] = bnd_val_all[m].mean(0)
bin_bnd[:, 0] = load_grid

# ---- ramp scenario ----
N_DOWN, N_HOLD, N_UP = 240, 120, 240
N_TOT = N_DOWN + N_HOLD + N_UP + N_HOLD
HI, LO = 475.0, 178.0
load_t = np.concatenate([
    np.linspace(HI, LO, N_DOWN, endpoint=False),
    np.full(N_HOLD, LO),
    np.linspace(LO, HI, N_UP, endpoint=False),
    np.full(N_HOLD, HI),
])
bnd_t = np.stack([np.interp(load_t, load_grid, bin_bnd[:, j]) for j in range(7)], axis=1)
bnd = torch.tensor(bnd_t, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, N_TOT, 7)

# history from the high-load end (ramp start condition)
val_pos = np.where(val_mask)[0]
m_hi = np.where(np.abs(sf_val - HI) < 15.0)[0]
idx = int(val_pos[m_hi[np.random.default_rng(9).integers(0, len(m_hi))]])
off = np.arange(HIST - 1, -1, -1)
hist = HistoryWindow(
    obs=record.obs[idx - off].unsqueeze(0).contiguous().to(DEVICE),
    actions=record.actions[idx - off].unsqueeze(0).contiguous().to(DEVICE),
    boundary=record.boundary[idx - off].unsqueeze(0).contiguous().to(DEVICE),
)
print(f"history start idx={idx}, sf={sf_val[m_hi].mean():.0f} kg/s bin", flush=True)

sp = torch.full((1, N_TOT), T_SP, device=DEVICE)
state = model._initial_state(hist)
# feedforward from the spray-demand terrain (closed_loop.npz V_ss(load))
cl = np.load(OUT / "closed_loop.npz")
ff = np.interp(load_t, cl["load_grid"], cl["V_ss"])
cl.close()
v0 = float(hist.actions[0, -1, 1].cpu().numpy())
held_v1 = hist.actions[:, -1, 0]
closure = model.closure

T = torch.zeros(1, N_TOT, 5, device=DEVICE)
V = torch.zeros(1, N_TOT, device=DEVICE)
kp, ki = 0.02, 0.002
valve = v0
integral = 0.0
with torch.no_grad():
    for t in range(N_TOT):
        b_t = bnd[:, t]
        a_t = torch.stack([held_v1.expand(1), torch.tensor([valve], device=DEVICE)], dim=-1)
        residual = closure(state, b_t)
        step = model.transition.step(state, b_t, a_t, residual)
        state = step.state
        T[:, t] = model.transition.output_temperatures(state, b_t, a_t)
        V[:, t] = valve
        err = float(T[0, t, -1].item()) - T_SP
        trial = integral + err * 10.0
        raw = ff[t] + kp * err + ki * trial
        cmd = float(np.clip(raw, v2_lo, v2_hi))
        if not ((raw >= v2_hi and err > 0) or (raw <= v2_lo and err < 0)):
            integral = trial
        valve = valve + float(np.clip(cmd - valve, -0.005, 0.005))

T_np = T[0].cpu().numpy()   # (N_TOT, 5)
V_np = V[0].cpu().numpy()
np.savez(
    OUT / "ramp.npz", load_t=load_t, T=T_np, V2=V_np, T_sp=T_SP,
    channels=np.array(["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main_steam"]),
)
print("saved ramp.npz", flush=True)
seg = [0, N_DOWN, N_DOWN + N_HOLD, N_DOWN + N_HOLD + N_UP, N_TOT]
for a, b in zip(seg[:-1], seg[1:]):
    print(f"steps {a}-{b}: load {load_t[a]:.0f}->{load_t[b-1]:.0f} kg/s | "
          f"T_main {T_np[a:b, 4].mean():.2f} C | v2 {V_np[a:b].mean():.3f}", flush=True)

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_min = np.arange(N_TOT) / 6.0
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

ax = axes[0, 0]
ax.plot(t_min, load_t, color="0.25")
ax.set_xlabel("time [min]"); ax.set_ylabel("steam flow [kg/s]")
ax.set_title("Load ramp scenario")

ax = axes[0, 1]
for c, name in enumerate(["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main_steam"]):
    ax.plot(t_min, T_np[:, c], lw=1.0, label=name)
ax.axhline(T_SP, color="crimson", ls="--", lw=1.0, label="setpoint")
ax.set_xlabel("time [min]"); ax.set_ylabel("temp [degC]")
ax.set_title("Temperatures under load ramp (PI on v2)")
ax.legend(fontsize=8, ncol=2)

ax = axes[1, 0]
ax.plot(t_min, V_np, color="teal", lw=1.2, label="v2 (controlled)")
ax.plot(t_min, np.full(N_TOT, float(held_v1[0].cpu())), color="grey", ls=":", label="v1 (held)")
ax.set_xlabel("time [min]"); ax.set_ylabel("valve opening")
ax.set_title("Spray valve trajectory")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(t_min, T_np[:, 4], color="steelblue", lw=1.0)
ax.axhline(T_SP, color="crimson", ls="--", lw=1.0)
ax.fill_between(t_min, T_SP - 2, T_SP + 2, color="crimson", alpha=0.08)
ax.set_xlabel("time [min]"); ax.set_ylabel("main steam [degC]")
ax.set_title(f"Main steam vs setpoint {T_SP:.0f} C (+-2 C band)")
band_frac = float((np.abs(T_np[:, 4] - T_SP) <= 2.0).mean())
ax.text(0.02, 0.95, f"in-band fraction: {band_frac:.0%}", transform=ax.transAxes, fontsize=9)

fig.suptitle("closure_cons_norew sideB seed0 -- load ramp closed-loop (exploratory)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "ramp_closed_loop.png", dpi=110)

# 3D space-time surface: time x channel x temperature
fig = plt.figure(figsize=(11, 6.5))
ax = fig.add_subplot(111, projection="3d")
TT, CC = np.meshgrid(t_min[::6], np.arange(5), indexing="ij")
ZZ = T_np[::6, :]
surf = ax.plot_surface(TT, CC, ZZ, cmap="viridis", edgecolor="none", antialiased=True)
ax.set_xlabel("time [min]")
ax.set_ylabel("channel (0=sh1_in ... 4=main)")
ax.set_zlabel("temp [degC]")
ax.set_yticks(np.arange(5))
ax.set_yticklabels(["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"])
ax.view_init(elev=26, azim=-58)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.06, label="degC")
ax.set_title("Space-time temperature surface under load ramp")
fig.tight_layout()
fig.savefig(OUT / "ramp_surface3d.png", dpi=110)
print("saved ramp_closed_loop.png + ramp_surface3d.png", flush=True)
print("DONE", flush=True)
