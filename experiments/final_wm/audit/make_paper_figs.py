"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""FMTS 论文三图生成: fig1 张力散点 / fig2 系统+矩阵示意 / fig3 证据面板"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = str(_P(__file__).resolve().parents[3] / "docs/fmts2026/paper/figs/")
import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
})
C_DEEP = "#C0392B"   # physics line (warm)
C_DATA = "#1F618D"   # data line (cool)
C_GRAY = "#7F8C8D"
C_GOLD = "#B7950B"

# ---------------- Fig 1: tension scatter ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.axhspan(0.45, 0.87, color=C_GOLD, alpha=0.12, label="energy-balance ref.")
ax.plot([0.82], [0.01], "o", color=C_DATA, ms=8)
ax.errorbar([0.82], [0.01], yerr=[[0.004], [0.005]], fmt="none", color=C_DATA, lw=1)
ax.plot([3.9], [0.45], "s", color=C_DEEP, ms=8)
ax.annotate("deep forecaster\n(action channel ~dead)",
            (0.82, 0.01), xytext=(1.25, 0.004), fontsize=7.5, color=C_DATA,
            arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.7))
ax.annotate("physics WM\n(live at 1 step; long-run sign-inverted, R1)",
            (3.9, 0.45), xytext=(2.6, 0.5), fontsize=7.5, color=C_DEEP,
            arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=0.7))
ax.annotate("25--450\u00d7 attenuation", xy=(1.55, 0.14), xytext=(1.55, 0.26),
            fontsize=8, ha="center", color=C_GRAY,
            arrowprops=dict(arrowstyle="<->", color=C_GRAY, lw=0.8))
ax.set_yscale("log")
ax.set_ylim(0.002, 3)
ax.set_xlim(0, 4.6)
ax.set_xlabel("18-step MAE (\u00b0C)")
ax.set_ylabel("|response| per +2% valve (\u00b0C, log)")
ax.set_title("Accuracy does not transfer to action fidelity")
ax.legend(fontsize=6.5, loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT + "fig1_tension.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 2: system + matrix flow ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.05, 1.4]})
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
blocks = [
    (0.4, 3.2, "low-T\nSH", "#EAF2F8"), (2.6, 3.2, "1st\nspray", "#FEF9E7"),
    (4.2, 3.2, "platen\nSH", "#EAF2F8"), (6.0, 3.2, "2nd\nspray", "#FEF9E7"),
    (7.6, 3.2, "final\nSH", "#EAF2F8"),
]
for x, y, t, c in blocks:
    ax.add_patch(FancyBboxPatch((x, y), 1.5, 1.6, boxstyle="round,pad=0.05",
                                fc=c, ec=C_GRAY, lw=0.7))
    ax.text(x + 0.75, y + 0.8, t, ha="center", va="center", fontsize=6.3)
for x0 in (1.9, 5.0, 7.5):
    ax.annotate("", xy=(x0 + 0.62, 4.0), xytext=(x0 - 0.02, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1))
ax.text(4.8, 4.25, "steam", fontsize=6.5, color=C_GRAY, ha="center")
for i, (x, y, t, c) in enumerate([
    (1.15, 1.9, "obs T$_1$", "#FFFFFF"), (3.35, 1.9, "obs T$_2$", "#FFFFFF"),
    (5.05, 1.9, "obs T$_3$", "#FFFFFF"), (6.75, 1.9, "obs T$_4$", "#FFFFFF"),
    (8.35, 1.9, "obs T$_5$", "#FFFFFF")]):
    ax.add_patch(FancyBboxPatch((x, y), 1.0, 0.6, boxstyle="round,pad=0.04",
                                fc=c, ec=C_DEEP, lw=0.9))
    ax.text(x + 0.5, y + 0.3, t, ha="center", va="center", fontsize=6)
ax.annotate("", xy=(1.65, 2.5), xytext=(1.65, 1.9),
            arrowprops=dict(arrowstyle="-|>", color=C_DEEP, lw=0.8))
ax.annotate("", xy=(3.85, 2.5), xytext=(3.85, 1.9),
            arrowprops=dict(arrowstyle="-|>", color=C_DEEP, lw=0.8))
ax.annotate("", xy=(5.55, 2.5), xytext=(5.55, 1.9),
            arrowprops=dict(arrowstyle="-|>", color=C_DEEP, lw=0.8))
ax.annotate("", xy=(7.25, 2.5), xytext=(7.25, 1.9),
            arrowprops=dict(arrowstyle="-|>", color=C_DEEP, lw=0.8))
ax.annotate("", xy=(8.85, 2.5), xytext=(8.85, 1.9),
            arrowprops=dict(arrowstyle="-|>", color=C_DEEP, lw=0.8))
ax.text(0.15, 5.7, "valve A \u2192 side B temps (cross-paired)", fontsize=6, color=C_DATA)

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
fig.savefig(OUT + "fig2_system_matrix.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- Fig 3: evidence panel ----------------
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})

sr = np.load("/tmp/step_response.npz")
steps = np.arange(1, 19)
ax = axes[0]
ax.plot(steps, sr["v1p"][:, 1], color=C_DEEP, lw=1.4, label="physics WM, +2% valve A, T$_2$")
ax.plot(steps, sr["v2p"][:, 3], color=C_DEEP, lw=1.4, ls="--", label="physics WM, +2% valve B, T$_4$")
ax.axhline(-0.01, color=C_DATA, lw=1.4, label="deep forecaster (\u22120.005\u2026\u22120.015)")
ax.axhspan(-0.87, -0.45, color=C_GOLD, alpha=0.15)
ax.text(12.2, -0.66, "energy-balance\nreference", fontsize=6, color="#8a6d00")
ax.set_xlabel("steps (10 s)"); ax.set_ylabel("$\Delta$T (\u00b0C)")
ax.legend(fontsize=5.8, loc="lower left", frameon=False)
ax.set_title("(a) valve-step response", fontsize=8.5)
ax.axhline(0, color="0.7", lw=0.5)

ax = axes[1]
chans = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "final"]
pers = [0.246, 0.451, 0.192, 0.275, 0.088]
model = [9.40, 1.84, 0.60, 2.18, 0.155]
x = np.arange(5); w = 0.38
ax.bar(x - w/2, pers, w, color=C_GRAY, label="persistence")
ax.bar(x + w/2, model, w, color=C_DEEP, label="physics WM")
ax.set_yscale("log"); ax.set_ylim(0.03, 30)
ax.set_xticks(x); ax.set_xticklabels(chans, fontsize=6.5)
ax.set_ylabel("H1 MAE (\u00b0C, log)")
ax.legend(fontsize=5.8, frameon=False, loc="upper left")
ax.set_title("(b) one-step error by channel", fontsize=8.5)

ax = axes[2]
bins = ["Q1", "Q2", "Q3", "Q4", "Q5"]
final_bias = [-0.10, -0.10, -0.03, 0.09, 0.20]
sh1_bias = [6.04, -0.50, 0.30, 2.63, 8.75]
ax.plot(bins, final_bias, "o-", color=C_GRAY, lw=1.2, ms=4, label="final stage")
ax.plot(bins, sh1_bias, "s-", color=C_DEEP, lw=1.2, ms=4, label="upstream stage")
ax.axhline(0, color="0.7", lw=0.5)
ax.set_xlabel("load quintile"); ax.set_ylabel("mean residual (\u00b0C)")
ax.legend(fontsize=5.8, frameon=False, loc="upper left")
ax.set_title("(c) residual means by load", fontsize=8.5)

fig.tight_layout()
fig.savefig(OUT + "fig3_evidence.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written to", OUT)
