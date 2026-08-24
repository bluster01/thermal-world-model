"""Bias-terrain vs residual-binning comparison (execution-side, exploratory).

Left: closed-loop steady bias T_ss(load) - 571 from the constant-load
closed-loop probe. Right: auditpack H18 residual bin means (load-quintile
bins) for final_outlet / sh1_outlet. Bottom: both aggregated to the same
5 load-quintile bins, min-max normalized, to compare misfit structure
across timescales (3-min prediction residual vs 120-min closed-loop bias).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/tmp/grid_out")
T_SP = 571.0

cl = np.load(OUT / "closed_loop.npz")
load_grid = cl["load_grid"]
T_ss = cl["T_ss"]
bias = T_ss - T_SP
cl.close()

ap_path = ("/home/bluster/projectA/thermal-world-model/artifacts/final_wm_sideB/"
           "auditpack_B_closure_cons_norew.json")
ap = json.loads(ap_path and Path(ap_path).read_text())
rb = ap["residual_binning"]
h18 = rb["H18"]
h1 = rb["H1"]

# quintile bin centers on the load axis (same binning rule as binning_stats)
cl2 = np.load(OUT / "closed_loop.npz")
sf_val = cl2["sf_val"]
cl2.close()
edges = np.quantile(sf_val, np.linspace(0, 1, 6)[1:-1])
bin_centers = np.concatenate([[sf_val.min()], edges, [sf_val.max()]])
bin_mid = 0.5 * (bin_centers[:-1] + bin_centers[1:])

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

# --- panel 1: closed-loop steady bias vs load ---
ax = axes[0]
ax.plot(load_grid, bias, "o-", color="teal", ms=4, label="closed-loop bias (120 min)")
ax.axhline(0, color="0.4", lw=0.8)
ax.fill_between(load_grid, bias, 0, where=bias > 0, color="crimson", alpha=0.15)
ax.fill_between(load_grid, bias, 0, where=bias < 0, color="steelblue", alpha=0.15)
ax.set_xlabel("steam flow [kg/s]")
ax.set_ylabel("T_ss - setpoint [degC]")
ax.set_title("Closed-loop steady bias vs load")
ax.legend(fontsize=8)

# --- panel 2: auditpack H18 residual bin means ---
ax = axes[1]
ax.plot(bin_mid, h18["sh1_outlet_temp"]["bin_means"], "s-", color="crimson",
        label="sh1_outlet H18")
ax.plot(bin_mid, h18["final_outlet_temp"]["bin_means"], "o-", color="0.2",
        label="final_outlet H18")
ax.set_xlabel("load quintile center [kg/s]")
ax.set_ylabel("mean abs residual [degC]")
ax.set_title("H18 residual structure (auditpack, load quintiles)")
ax.legend(fontsize=8)

# --- panel 3: same 5 bins, normalized structure comparison ---
ax = axes[2]
# aggregate closed-loop bias into the same quintile bins
bin_id = np.clip(np.digitize(load_grid, edges), 0, 4)
bias_bin = np.array([bias[bin_id == b].mean() for b in range(5)])


def norm01(x):
    x = np.asarray(x, dtype=float)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 1e-9 else np.zeros_like(x)


width = 0.3
x = np.arange(5)
ax.bar(x - width / 2, norm01(bias_bin), width, label="closed-loop bias (normalized)", color="teal")
ax.bar(x + width / 2, norm01(h18["final_outlet_temp"]["bin_means"]), width,
       label="H18 residual final_outlet (normalized)", color="0.5")
ax.plot(x, norm01(h18["sh1_outlet_temp"]["bin_means"]), "^-", color="crimson",
        label="H18 residual sh1_outlet (normalized)")
ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
ax.set_xlabel("load quintile (Q1 = lowest load)")
ax.set_ylabel("normalized magnitude")
ax.set_title("Misfit structure: 3-min residual vs 120-min bias")
ax.legend(fontsize=8)

fig.suptitle("closure_cons_norew sideB -- misfit terrain: short-horizon residual vs closed-loop bias", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "bias_vs_residual.png", dpi=110)
print("saved bias_vs_residual.png", flush=True)

print("\nbias by load quintile:", np.round(bias_bin, 2))
print("H18 final_outlet bin_means:", np.round(h18["final_outlet_temp"]["bin_means"], 3))
print("H18 sh1_outlet bin_means:", np.round(h18["sh1_outlet_temp"]["bin_means"], 3))
print("H1  sh1_outlet bin_means:", np.round(h1["sh1_outlet_temp"]["bin_means"], 3))
print("between_ratio H18: sh1_out", round(h18["sh1_outlet_temp"]["between_ratio"], 3),
      "| final_out", round(h18["final_outlet_temp"]["between_ratio"], 3))
print("DONE", flush=True)
