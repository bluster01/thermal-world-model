#!/usr/bin/env python3
"""FMTS paper figures v2 -- data sourced from audited artifacts:
  - artifacts/final_wm/auditpack_A.json     (protocolized evidence pack, commit a3ae48f)
  - artifacts/final_wm/matrix_summary_sideA.json (side-A verdicts, matrix v0.2)
  - Direct WM v2 audit table (docs/ADHOC_DIRECT_WM_V2_SUPERVISOR_AUDIT_2026-08-18.md §3)
No withdrawn numbers (25-450x / -0.005~-0.015 / -0.45~-0.87).
Run from repo root.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/fmts2026/paper/figs/"
OUT.mkdir(parents=True, exist_ok=True)

AP = json.load(open(ROOT / "artifacts/final_wm/auditpack_A.json"))
MS = json.load(open(ROOT / "artifacts/final_wm/matrix_summary_sideA.json"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.edgecolor": "0.4", "axes.linewidth": 0.6,
    "mathtext.fontset": "stix",
})
C_DEEP = "#1f4e79"   # physics WM
C_DATA = "#c26a2e"   # forecaster / plant events
C_GOLD = "#d9a441"
C_GRAY = "0.55"

# ---------------- Fig 1: tension plane (one-step magnitude, single口径) ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
# data-anchored mixing reference, v2, +2% (auditpack mixing_reference)
ax.axhspan(0.53, 1.48, color=C_GOLD, alpha=0.15)
ax.text(3.05, 1.53, "mixing ref. v2,\nper +2% (data dW/dv)", fontsize=5.8,
        color="#8a6d00", va="bottom", ha="left")
# deep forecaster: one-step responses across folds/valves (Direct WM v2 audit §3)
ax.plot([0.71], [0.006], "o", color=C_DATA, ms=8)
ax.errorbar([0.71], [0.006], yerr=[[0.006], [0.034]], fmt="none", color=C_DATA, lw=1)
ax.annotate("deep forecaster: 0.00004\u20130.040 \u00b0C,\ndirection unstable (1/8 cells 3/3)",
            (0.71, 0.006), xytext=(1.15, 0.0008), fontsize=6.3, color=C_DATA,
            arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.7))
# physics WM: one-step response ~10-30x the plant's ~0 H1 (audit §2.3)
ax.plot([3.9], [0.45], "s", color=C_DEEP, ms=8)
ax.annotate("physics WM: one-step \u224810\u201330\u00d7 plant H1;\n60-step terminal inverts (R1)",
            (3.9, 0.45), xytext=(2.2, 0.75), fontsize=6.3, color=C_DEEP,
            arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.7))
ax.set_yscale("log"); ax.set_ylim(1e-5, 8)
ax.set_xlim(0.2, 5.2)
ax.set_xlabel("18-step MAE (\u00b0C)")
ax.set_ylabel("one-step valve response |\u0394T| (\u00b0C, log)")
ax.axhline(1e-4, color="0.8", lw=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig1_tension.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 2: system + matrix (verdicts unchanged) ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
ax = axes[0]
ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.6); ax.axis("off")
for x0, y0, w, h in [(0.5, 2.8, 1.6, 0.9), (2.6, 2.8, 1.6, 0.9), (4.7, 2.8, 1.6, 0.9)]:
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04",
                                fc="#EFF4F8", ec=C_GRAY, lw=0.7))
ax.text(1.3, 3.45, "SH1", ha="center", fontsize=7.5, weight="bold")
ax.text(3.4, 3.45, "SH2", ha="center", fontsize=7.5, weight="bold")
ax.text(5.5, 3.45, "SH3", ha="center", fontsize=7.5, weight="bold")
for x0 in (0.9, 3.0, 5.1):
    ax.annotate("", xy=(x0 + 0.55, 2.8), xytext=(x0 + 0.55, 1.9),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.8))
ax.text(1.45, 2.25, "spray\nvalve A", fontsize=5.8, color=C_DATA, ha="center")
ax.text(3.55, 2.25, "spray\nvalve B", fontsize=5.8, color=C_DATA, ha="center")
for x0 in (0.5, 2.6, 4.7):
    ax.annotate("", xy=(x0 + 0.4, 2.8), xytext=(x0 + 0.4, 3.7),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.8))
for i, (x, t) in enumerate([(1.3, "obs T$_1$"), (3.4, "obs T$_2$"), (5.5, "obs T$_3$")]):
    ax.add_patch(FancyBboxPatch((x - 0.5, 0.75), 1.0, 0.55, boxstyle="round,pad=0.03",
                                fc="#FFFFFF", ec=C_GRAY, lw=0.7))
    ax.text(x, 1.02, t, ha="center", va="center", fontsize=6.5)
for i, (x, t) in enumerate([(2.15, "obs T$_4$"), (4.25, "obs T$_5$")]):
    ax.add_patch(FancyBboxPatch((x - 0.5, 0.75), 1.0, 0.55, boxstyle="round,pad=0.03",
                                fc="#FFFFFF", ec=C_GRAY, lw=0.7))
    ax.text(x, 1.02, t, ha="center", va="center", fontsize=6.5)
ax.annotate("", xy=(1.3, 1.3), xytext=(1.3, 1.85), arrowprops=dict(arrowstyle="-", color="0.7", lw=0.6))
ax.annotate("", xy=(3.4, 1.3), xytext=(3.4, 1.85), arrowprops=dict(arrowstyle="-", color="0.7", lw=0.6))
ax.text(0.15, 4.35, "660 MW once-through, supercritical dry", fontsize=6, color=C_GRAY)
ax.text(0.15, 0.35, "valve A \u2192 side B temps (cross-paired)", fontsize=6, color=C_DATA)

ax2 = axes[1]
ax2.set_xlim(0, 10); ax2.set_ylim(0, 6); ax2.axis("off")
units = [
    ("D-SYN", "identifiability\ngate", "PASS 3/3", "#E8F8F5"),
    ("O1", "init posterior", "MIXED /\nREJECTED", "#FEF9E7"),
    ("T1", "transition\nstructure", "closure SUPPORTED", "#FEF9E7"),
    ("B1", "boundary\nforecast", "REJECTED\n(persistence wins)", "#FDF2E9"),
    ("J1", "training\nscheme", "SUPPORTED 3/3", "#E8F8F5"),
    ("R1", "blindness +\nleakage + direction", "REJECTED\n(direction)", "#FDF2E9"),
]
for i, (u, sub, verdict, c) in enumerate(units):
    y = 5.2 - i * 0.92
    ax2.add_patch(FancyBboxPatch((0.4, y), 1.7, 0.75, boxstyle="round,pad=0.04",
                                 fc=c, ec=C_GRAY, lw=0.7))
    ax2.text(1.25, y + 0.47, u, ha="center", va="center", fontsize=8, weight="bold")
    ax2.text(1.25, y + 0.14, sub, ha="center", va="center", fontsize=5.6)
    ax2.text(4.9, y + 0.37, verdict, ha="left", va="center", fontsize=6.8)
    if i < 5:
        ax2.annotate("", xy=(1.25, y - 0.2), xytext=(1.25, y - 0.62),
                     arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.8))
ax2.text(4.9, 5.65, "verdict (side A)", fontsize=6.5, color=C_GRAY)
ax2.text(0.4, 0.05, "paired bootstrap CIs, 2/3-seed rule, convergence diagnostics",
         fontsize=5.8, color=C_GRAY)
fig.tight_layout()
fig.savefig(OUT / "fig2_system_matrix.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 3: evidence panel (auditpack numbers only) ----------------
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})

ax = axes[0]
h = [1, 6, 18, 60]
es_v2u = AP["event_study"]["v2"]["up"]["mean_delta"]
es_v2d = AP["event_study"]["v2"]["down"]["mean_delta"]
ax.plot(h, es_v2u, "o-", color=C_DATA, lw=1.2, ms=3.5, label="plant, v2-up events (n=22)")
ax.plot(h, es_v2d, "s--", color=C_DATA, lw=1.2, ms=3.5, label="plant, v2-down events (n=48)")
r1 = MS["units"]["r1"]["reports"]
r1_means = [r["direction"]["mean_delta_c"] for r in r1]
ax.plot([60, 60, 60], r1_means, "^", color=C_DEEP, ms=6, mew=0,
        label="physics WM, R1 +5% v2, 60-step (3 seeds)")
ax.axhspan(-0.040, 0.040, color=C_GRAY, alpha=0.18)
ax.text(2.5, 0.055, "deep forecaster band (\u00b10.04 \u00b0C)", fontsize=5.8, color=C_GRAY)
ax.axhline(0, color="0.7", lw=0.5)
ax.set_xlabel("steps (10 s)"); ax.set_ylabel("$\Delta$T terminal (\u00b0C)")
ax.legend(fontsize=5.2, loc="lower left", frameon=False)
ax.set_title("(a) real events vs model probes", fontsize=8.5)

ax = axes[1]
chans = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "final"]
pers = [AP["persistence_increment_mae"][k] for k in
        ["sh1_inlet_temp", "sh1_outlet_temp", "sh2_inlet_temp", "sh2_outlet_temp", "final_outlet_temp"]]
model = [9.40, 1.84, 0.60, 2.18, 0.16]
x = np.arange(5); w = 0.38
ax.bar(x - w/2, pers, w, color=C_GRAY, label="persistence (auditpack)")
ax.bar(x + w/2, model, w, color=C_DEEP, label="physics WM")
ax.set_yscale("log"); ax.set_ylim(0.03, 30)
ax.set_xticks(x); ax.set_xticklabels(chans, fontsize=6.5)
ax.set_ylabel("H1 MAE (\u00b0C, log)")
ax.legend(fontsize=5.8, frameon=False, loc="upper left")
ax.set_title("(b) one-step error by channel", fontsize=8.5)

ax = axes[2]
bins = ["Q1", "Q2", "Q3", "Q4", "Q5"]
final_bias = AP["residual_binning"]["H1"]["final_outlet_temp"]["bin_means"]
sh1_bias = AP["residual_binning"]["H1"]["sh1_inlet_temp"]["bin_means"]
ax.plot(bins, final_bias, "o-", color=C_GRAY, lw=1.2, ms=4, label="final stage")
ax.plot(bins, sh1_bias, "s-", color=C_DEEP, lw=1.2, ms=4, label="upstream stage")
ax.axhline(0, color="0.7", lw=0.5)
ax.set_xlabel("load quintile"); ax.set_ylabel("mean residual (\u00b0C)")
ax.legend(fontsize=5.8, frameon=False, loc="upper left")
ax.set_title("(c) residual means by load (H1)", fontsize=8.5)

fig.tight_layout()
fig.savefig(OUT / "fig3_evidence.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written to", OUT)
