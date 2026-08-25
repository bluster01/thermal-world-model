"""Anchor probe report + figures (2026-08-25)."""
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
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "retrain_probe"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

from src.final_wm.transition import TRANSITION_PARAM_PRIORS, _SIGNED_PARAMS

prod = {0: 1.046, 1: 0.597, 2: 0.652}
unanch = {0: 0.723, 1: 0.418, 2: 0.497}
anch = {0: 0.478, 1: 0.418, 2: 0.465}   # seed1 = anchor source itself
BB = 0.354

def eff_constants(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    out = {}
    for k, v in sd.items():
        if k.startswith("transition.raw."):
            n = k.split(".")[-1]
            pr = TRANSITION_PARAM_PRIORS[n]
            out[n] = pr * np.tanh(float(v)) if n in _SIGNED_PARAMS else \
                pr * float(np.log1p(np.exp(float(v))))
    return out

anchor_src = eff_constants(OUT / "armA_budget_seed1/checkpoints/t1_closure_cons_norew_seed1.pt")
print("== constant drift: final effective constants vs anchor source (relative delta) ==")
for s in (0, 2):
    fin = eff_constants(OUT / f"armC_anchor_s1const_seed{s}/checkpoints/t1_closure_cons_norew_seed{s}.pt")
    rel = {n: abs(fin[n] - anchor_src[n]) / max(abs(anchor_src[n]), 1e-9) for n in anchor_src}
    worst = sorted(rel.items(), key=lambda kv: -kv[1])[:5]
    print(f"  armC seed{s}: mean|rel drift|={np.mean(list(rel.values())):.3f} "
          f"top5: {[(n, round(r,2)) for n,r in worst]}")

# report markdown
lines = ["# 锚定探针总报告（execution-side, 2026-08-25）", "",
         "> armC: seed0/seed2 从 armA_s1（最好盆地 0.418）的 34 个物理常数热启动，",
         "> 网络部分新鲜 init，120/20 重训。探索性；不动 verdict/冻结 checkpoint。", "",
         "## 主表（H18 主汽温 ch4，256 窗 seed50k oracle）", "",
         "| 种子 | 生产 | 无锚重训 | 锚定重训 | 锚定Δ |", "|---|---|---|---|---|"]
m_prod, m_un, m_an = [], [], []
for s in (0, 1, 2):
    m_prod.append(prod[s]); m_un.append(unanch[s]); m_an.append(anch[s])
    lines.append(f"| seed{s} | {prod[s]:.3f} | {unanch[s]:.3f} | {anch[s]:.3f} | "
                 f"{anch[s]-unanch[s]:+.3f} |")
lines.append(f"| **均值** | **{np.mean(m_prod):.3f}** | **{np.mean(m_un):.3f}** | "
             f"**{np.mean(m_an):.3f}** | **{np.mean(m_an)-np.mean(m_un):+.3f}** |")
lines.append("")
lines.append(f"跨种子极差：生产 {max(m_prod)-min(m_prod):.3f} → 无锚 {max(m_un)-min(m_un):.3f} → "
             f"锚定 **{max(m_an)-min(m_an):.3f}**（收缩 5 倍）")
lines.append("")
lines.append(f"vs 黑箱 iTransformer {BB}：生产 2.16x → 无锚 1.54x → **锚定 1.28x**")
lines.append("")
lines.append("## 结论")
lines.append("")
lines.append("1. **盆地救援成立**：seed0 0.723→0.478（-34%），seed2 0.497→0.465（-6%）。")
lines.append("   从最好盆地的常数出发，坏种子被拉回好盆地邻域。")
lines.append("2. **种子发散被共享常数压制**：三种子极差 0.30→0.06。物理常数弱可辨识是")
lines.append("   跨种子发散的主载体（init 函数种子不变+transition RMS 0.29-1.59 证据链闭合）。")
lines.append("3. 锚定后仍差 seed1 自身 0.05-0.06：可能来源=常数在训练中再漂移（见下）或")
lines.append("   observer/closure 网络的种子噪声。冻结臂（armE）可区分，但需 src 变更，")
lines.append("   留待对侧裁定。")
lines.append("")
lines.append("## 对侧待决")
lines.append("")
lines.append("1. 共享常数协议提案：多种子训练统一从已辨识的物理常数热启动（或单种子先辨识")
lines.append("   常数再冻结——armE 补丁需要 TrainSpec 新字段，指纹影响需设计侧批准）。")
lines.append("2. 预算修正（120/20）+ 常数锚定合并为 v0.4 训练协议修订包。")
(P / "anchor_report_20260825.md").write_text("\n".join(lines), encoding="utf-8")

# figures
z0 = np.load(P / "plots_mainsteam/predictions_cache.npz")
za0 = np.load(OUT / "armA_budget/preds_armA_budget.npz")
zc0 = np.load(OUT / "armC_anchor_s1const_seed0/preds_armC_anchor_s1const_seed0.npz")
zc2 = np.load(OUT / "armC_anchor_s1const_seed2/preds_armC_anchor_s1const_seed2.npz")
actual = z0["actual"]; bb = z0["bb_pred"]
m_un0 = np.abs(za0["pred"] - actual).mean(1)
m_an0 = np.abs(zc0["pred"] - actual).mean(1)
m_an2 = np.abs(zc2["pred"] - actual).mean(1)
m_bb = np.abs(bb - actual).mean(1)

# per-window: unanchored seed0 vs anchored seed0
fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=150)
ax.scatter(m_un0, m_an0, s=14, color="0.55", zorder=2)
lim = (0, max(m_un0.max(), m_an0.max()) * 1.05)
ax.plot([0, lim[1]], [0, lim[1]], "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Unanchored armA seed0 MAE (°C)")
ax.set_ylabel("Anchored seed0 MAE (°C)")
ax.set_title("Per-window H18 MAE, seed0\nbelow diagonal = anchor improves")
fig.tight_layout(); fig.savefig(OUT / "anchor_seed0_scatter.png"); plt.close(fig)

# worst windows overlay (seed0: unanchored vs anchored vs bb)
order = np.argsort(-m_un0)
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
gen = torch.Generator().manual_seed(50_000)
done = 0; batches = []
while done < 256:
    bsz = min(32, 256 - done)
    batches.append(sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen))
    done += bsz
hist_tail = torch.cat([b.history.obs[:, :, 4] for b in batches]).numpy()
for k, idx in enumerate(order[:3]):
    xh = np.arange(-95, 1); xf = np.arange(1, 19)
    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=150)
    ax.plot(xh, hist_tail[idx], color="0.6", lw=1.0)
    ax.plot(xf, actual[idx], color="black", lw=1.6, label="actual")
    ax.plot(xf, za0["pred"][idx], color="0.45", lw=1.2, ls="--",
            label=f"unanchored ({m_un0[idx]:.2f})")
    ax.plot(xf, zc0["pred"][idx], color="tab:blue", lw=1.5,
            label=f"anchored ({m_an0[idx]:.2f})")
    ax.plot(xf, bb[idx], color="tab:orange", lw=1.2, ls="-.", label=f"bb ({m_bb[idx]:.2f})")
    ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
    lo = min(actual[idx].min(), za0["pred"][idx].min(), zc0["pred"][idx].min(), bb[idx].min())
    hi = max(actual[idx].max(), za0["pred"][idx].max(), zc0["pred"][idx].max(), bb[idx].max())
    pad = max(0.3, 0.06 * (hi - lo)); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Step (10 s)"); ax.set_ylabel("Main steam (°C)")
    ax.set_title(f"Worst window #{idx} | day {int(z0['days'][idx])} | flow {z0['loads'][idx]:.0f} kg/s")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"anchor_overlay_worst{k+1}_win{idx}.png")
    plt.close(fig)

print("report + figures written")
