"""Closed-loop spray-demand terrain probe (execution-side, exploratory).

Production arm closure_cons_norew (sideB seed0). Per load grid point:
constant-oracle boundary at target load, PI controller on v2 tracking
main-steam 571 degC, 720-step (120 min) step-by-step loop through the
transition+closure. Outputs: valve/temp trajectories, steady spray demand
v2(load), convergence diagnostics. Exploration only, not matrix protocol.

Also verifies: does the main-steam temperature converge to 571 degC, and
does the demanded v2 stay inside the history support box?
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.controller import CascadePIController
from src.final_wm.contracts import ControllerConfig
from src.final_wm.data import CanonicalRecord, SPLIT_VAL, sample_windows
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
HIST = ms.HISTORY_STEPS
HORIZON = 720           # 120 min
T_SP = 571.0            # user-confirmed setpoint (08-24)
N_LOAD = 24
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

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideB.npz")
gen = torch.Generator().manual_seed(777)
batch = sample_windows(record, SPLIT_VAL, 64, HIST, 2, gen)
v1_span = batch.history.actions[:, :, 0].amax(1) - batch.history.actions[:, :, 0].amin(1)
i = int(torch.argmax(v1_span).item())
win = HistoryWindow(
    obs=batch.history.obs[i:i + 1],
    actions=batch.history.actions[i:i + 1],
    boundary=batch.history.boundary[i:i + 1],
)

val_mask = (record.split == SPLIT_VAL).numpy()
sf_val = record.boundary[:, 0].numpy()[val_mask]
v2_val = record.actions[:, 1].numpy()[val_mask]
load_grid = np.quantile(sf_val, np.linspace(0.05, 0.95, N_LOAD))
v1_hold = float(win.actions[0, -1, 0].numpy())
# support box from the full val split (not the single window)
v2_lo, v2_hi = float(np.quantile(v2_val, 0.025)), float(np.quantile(v2_val, 0.975))

# per-load conditional boundary means: all 7 channels move coherently with
# load (single-channel substitution caused an unphysical energy imbalance).
eps = (load_grid[1] - load_grid[0]) / 2.0
bnd_val_all = record.boundary.numpy()[val_mask]
bin_bnd = np.zeros((N_LOAD, 7), dtype=np.float32)
for k, L in enumerate(load_grid):
    m = np.abs(sf_val - L) < eps
    bin_bnd[k] = bnd_val_all[m].mean(0)
bin_bnd[:, 0] = load_grid
print(f"v2 support [{v2_lo:.3f}, {v2_hi:.3f}]; per-load bnd bins n={np.abs(sf_val-load_grid[0])<eps}..", flush=True)

B = N_LOAD
# condition-consistent histories: each load grid point gets a real window
# from its own load bin (obs/actions/boundary all coherent).
hist_idx = np.zeros(N_LOAD, dtype=np.int64)
run_span = HIST
for k, L in enumerate(load_grid):
    m = np.where(np.abs(sf_val - L) < eps)[0]
    # map val-mask positions back to record indices
    val_pos = np.where(val_mask)[0]
    candidates = val_pos[m[m >= run_span - 1]]  # enough preceding history
    if len(candidates) == 0:
        candidates = val_pos[m]
    hist_idx[k] = int(candidates[np.random.default_rng(k).integers(0, len(candidates))])
obs_h = record.obs[hist_idx[:, None] - np.arange(HIST - 1, -1, -1)[None, :]]
act_h = record.actions[hist_idx[:, None] - np.arange(HIST - 1, -1, -1)[None, :]]
bnd_h = record.boundary[hist_idx[:, None] - np.arange(HIST - 1, -1, -1)[None, :]]
hist_b = HistoryWindow(
    obs=obs_h.contiguous().to(DEVICE),
    actions=act_h.contiguous().to(DEVICE),
    boundary=bnd_h.contiguous().to(DEVICE),
)
bnd = torch.tensor(bin_bnd, device=DEVICE).unsqueeze(1).repeat(1, HORIZON, 1)
sp = torch.full((B, HORIZON), T_SP, device=DEVICE)

state_0 = model._initial_state(hist_b)
controller = CascadePIController(ControllerConfig(valve_min=v2_lo, valve_max=v2_hi))
controller.reset(hist_b.actions[:, -1, 1])
closure = model.closure
held_v1 = hist_b.actions[:, -1, 0]

T_traj = torch.zeros(B, HORIZON, 5, device=DEVICE)
V_traj = torch.zeros(B, HORIZON, device=DEVICE)
state = state_0
with torch.no_grad():
    for t in range(HORIZON):
        b_t = bnd[:, t]
        action_t = torch.stack([held_v1, controller.valve], dim=-1)
        residual = closure(state, b_t)
        step = model.transition.step(state, b_t, action_t, residual)
        state = step.state
        temp_t = model.transition.output_temperatures(state, b_t, action_t)
        T_traj[:, t] = temp_t
        V_traj[:, t] = controller.valve
        controller.step(sp[:, t], temp_t[:, -1])

T_main = T_traj[:, :, 4].cpu().numpy()     # (B, H)
V2 = V_traj.cpu().numpy()
loads = load_grid

tail = slice(-30, None)
T_ss = T_main[:, tail].mean(1)
V_ss = V2[:, tail].mean(1)
# convergence: first step after which |T-571| <= 2 and stays (allow small re-excursions)
conv = np.full(B, HORIZON)
for b in range(B):
    err = np.abs(T_main[b] - T_SP)
    inband = err <= 2.0
    for t in range(HORIZON - 30):
        if inband[t] and inband[t:t + 30].mean() > 0.9:
            conv[b] = t
            break

np.savez(
    OUT / "closed_loop.npz",
    load_grid=load_grid, T_main=T_main, V2=V2, T_sp=T_SP,
    T_ss=T_ss, V_ss=V_ss, conv_steps=conv, v2_lo=v2_lo, v2_hi=v2_hi,
    sf_val=sf_val, v2_val=v2_val,
)
print("saved closed_loop.npz", flush=True)
for b in range(0, B, 4):
    print(f"load {loads[b]:6.1f} kg/s | T_ss {T_ss[b]:6.2f} C | v2_ss {V_ss[b]:.3f} | "
          f"conv {conv[b]:4d} steps ({conv[b]/6:.0f} min)", flush=True)

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
cmap = plt.cm.viridis
norm = plt.Normalize(loads.min(), loads.max())

ax = axes[0, 0]
for b in range(B):
    ax.plot(T_main[b], color=cmap(norm(loads[b])), lw=0.8)
ax.axhline(T_SP, color="crimson", ls="--", lw=1.2, label=f"setpoint {T_SP:.0f} C")
ax.set_xlabel("step (10 s)")
ax.set_ylabel("main steam temp [degC]")
ax.set_title("PI-controlled main steam (v2), per load")
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="steam flow [kg/s]")

ax = axes[0, 1]
for b in range(B):
    ax.plot(V2[b], color=cmap(norm(loads[b])), lw=0.8)
ax.axhline(v2_hi, color="grey", ls=":", lw=0.8)
ax.axhline(v2_lo, color="grey", ls=":", lw=0.8)
ax.set_xlabel("step (10 s)")
ax.set_ylabel("v2 opening")
ax.set_title("Spray demand v2(t)")

ax = axes[1, 0]
ax.scatter(sf_val[::200], v2_val[::200], s=4, alpha=0.25, color="0.6", label="val data (subsampled)")
ax.plot(loads, V_ss, "o-", color="crimson", ms=4, label="model steady v2 demand")
ax.set_xlabel("steam flow [kg/s]")
ax.set_ylabel("v2 opening")
ax.set_title("Spray-demand terrain v2(load) vs data density")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(loads, T_ss, "o-", color="teal", ms=4, label="steady main steam")
ax.axhline(T_SP, color="crimson", ls="--", lw=1, label="setpoint")
ax2 = ax.twinx()
ax2.plot(loads, conv / 6.0, "s-", color="orange", ms=3, alpha=0.8, label="converge time")
ax.set_xlabel("steam flow [kg/s]")
ax.set_ylabel("T_ss [degC]")
ax2.set_ylabel("converge time [min]", color="orange")
ax.set_title("Tracking accuracy & convergence")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

fig.suptitle(f"closure_cons_norew sideB seed0 -- closed-loop spray terrain (setpoint {T_SP:.0f} C, PI on v2)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "closed_loop_terrain.png", dpi=110)
print("saved closed_loop_terrain.png", flush=True)
print("DONE", flush=True)
