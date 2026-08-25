"""Compile the full retrain-probe report + summary figures (2026-08-25).

Run after all three retrained seeds (armA_budget seed0/1/2) are done.
Writes results/final_wm/probes_20260824/retrain_report_20260825.md plus
two summary figures. Discipline: exploratory evidence pack only — no
verdict blocks, no matrix_summary writes, frozen checkpoints untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "retrain_probe"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

# production per-seed overall/bins (from phys_per_segment.json, 3-seed agg per seed)
prod = {
    0: dict(overall=1.046, bins=[0.874, 0.997, 1.244, 1.201, 0.914],
            val=1.5360, stop="patience@41", best_ep=30),
    1: dict(overall=0.597, bins=[0.441, 0.499, 0.760, 0.753, 0.532],
            val=1.2397, stop="cap@60", best_ep=56),
    2: dict(overall=0.652, bins=[0.614, 0.571, 0.780, 0.703, 0.590],
            val=1.2716, stop="cap@60", best_ep=51),
}

retrain_dirs = {
    0: OUT / "armA_budget",
    1: OUT / "armA_budget_seed1",
    2: OUT / "armA_budget_seed2",
}

# bb overall from cache
z0 = np.load(P / "plots_mainsteam/predictions_cache.npz")
bb_overall = float(np.abs(z0["bb_pred"] - z0["actual"]).mean(axis=1).mean())

lines = ["# norew 预算重训探针总报告（execution-side, 2026-08-25）", "",
         "> 探索性证据包：不写 verdict、不动 matrix_summary、不动冻结 checkpoint。",
         "> 协议：sideA val, 256 窗 seed 50k, oracle boundary, H18, 主汽温(ch4) MAE [°C]。", ""]

rows = {}
mean_pred = None
n_seeds = 0
for s in (0, 1, 2):
    d = retrain_dirs[s]
    rep = OUT / "report.json"
    rep12 = OUT / "report_seeds12.json"
    if s == 0:
        data = json.load(open(rep))["armA_budget"]
    else:
        data = json.load(open(rep12))[f"armA_budget_seed{s}"]
    tr, ev = data["train"], data["eval"]
    rows[s] = (tr, ev)
    pred = np.load(d / f"preds_armA_budget_seed{s}.npz" if s else d / "preds_armA_budget.npz")
    mean_pred = pred["pred"] if mean_pred is None else mean_pred + pred["pred"]
    n_seeds += 1

mean_pred = mean_pred / n_seeds
mae_mean_w = np.abs(mean_pred - z0["actual"]).mean(axis=1)
mae_mean = float(mae_mean_w.mean())

lines.append("## 1. 主表：生产 vs 重训(120/20)")
lines.append("")
lines.append("| 种子 | 生产总体 | 重训总体 | Δ | 生产 val_nll | 重训 val_nll | 重训停止 |")
lines.append("|---|---|---|---|---|---|---|")
prod_overall, retrain_overall = [], []
for s in (0, 1, 2):
    tr, ev = rows[s]
    p = prod[s]
    d = ev["overall_h18_mae"] - p["overall"]
    prod_overall.append(p["overall"]); retrain_overall.append(ev["overall_h18_mae"])
    lines.append(f"| seed{s} | {p['overall']:.3f} | {ev['overall_h18_mae']:.3f} | "
                 f"{d:+.3f} | {p['val']:.3f} | {tr['best_val_nll']:.3f} | "
                 f"{tr['stop_reason']}@{tr['epochs_run']} (best@{tr['best_epoch']}) |")
pm, rm = np.mean(prod_overall), np.mean(retrain_overall)
lines.append(f"| **三种子均值** | **{pm:.3f}** | **{rm:.3f}** | **{rm-pm:+.3f}** | — | — | — |")
lines.append("")
lines.append("黑箱参考（iTransformer seed0, 同窗同口径）："
             f"**{bb_overall:.3f}**；重训均值/黑箱 = {rm/bb_overall:.2f}×。")
lines.append("")

lines.append("## 2. 分箱（Q1–Q5 负荷五分位）")
lines.append("")
lines.append("| 种子 | 配置 | Q1 | Q2 | Q3 | Q4 | Q5 |")
lines.append("|---|---|---|---|---|---|---|")
for s in (0, 1, 2):
    p, ev = prod[s], rows[s][1]
    lines.append(f"| seed{s} | 生产 | " + " | ".join(f"{x:.2f}" for x in p["bins"]) + " |")
    lines.append(f"| seed{s} | 重训 | " + " | ".join(f"{x:.2f}" for x in ev["bins_q1q5"]) + " |")
lines.append("")

lines.append("## 3. 跨种子发散（init 不变式 + 训练分岔）")
lines.append("")
lines.append("- 6 种子未训练 val NLL 完全一致（13709.62，零头相等）：初始化函数=纯物理先验，种子无关。")
lines.append("- 训练后 transition（物理常数）per-param RMS 距离：prod s0-s1 0.93 / s1-s2 0.29；"
             "重训 armA s0-s1 1.59。物理常数弱可辨识，closure/observer 各自补偿。")
lines.append("- 唯一种子差异=minibatch 数据流（无 dropout）→ 粗糙景观分岔到不同吸引子。")
lines.append("")

lines.append("## 4. 集成预研（负结果）")
lines.append("")
lines.append("- 全局混合 w=0.1 物理：0.348（≈无增益）；oracle 逐窗选优天花板 0.304（上限收益 0.05）。")
lines.append("- 位移阈值切换 0.463、ridge 组合器逐日留一 CV 0.642：均劣于纯黑箱。物理臂预测位移与"
             "其正确性负相关（过冲窗口恰是位移大窗口）。")
lines.append("")

lines.append("## 5. 黑箱赢/输条件")
lines.append("")
lines.append("- 黑箱胜 179/256（70%）：实际温度近不动（闭环压平），输入动作越大优势越大（喷水/煤量"
             "变化幅度 bb-win 组 5.0/5.2 vs 平局组 3.2/3.5）。")
lines.append("- 物理臂胜 38/256（15%）：实际温度真移动时（win193 +7.4°C，phys 0.37 vs bb 1.21）"
             "与小幅振荡窗（act_range 1.47 vs 1.25）。")
lines.append("")

lines.append("## 6. 对侧待决")
lines.append("")
lines.append("1. 预算修正提案：T1 训练预算 epochs 60/patience 10 → 120/20（seed1 best@98，"
             "生产预算不足最优点的 60%）。")
lines.append("2. 锚定探针立项：先验物理常数做 transition init，测种子发散收窄与精度变化。")
lines.append("3. 论文口径建议：放弃'精度平价'，改'精度接近(0.4x vs 0.35)+方向一致+反事实可用'。")
lines.append("")
lines.append("指纹不变：experiments/final_wm 树未动；冻结 checkpoint 未覆盖；matrix_summary 未写。")

(P / "retrain_report_20260825.md").write_text("\n".join(lines), encoding="utf-8")

# ---- figures ----
# val curves
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), dpi=150)
ledger_prod = [json.loads(l) for l in open(ROOT / "artifacts/final_wm/ledger.jsonl")]
for s, ax in enumerate(axes):
    c0 = [(x["epoch"], x["val_nll"]) for x in ledger_prod
          if x.get("run_id") == "t1_closure_cons_norew_seed0".replace("0", str(s))
          and "val_nll" in x]
    d = retrain_dirs[s]
    led = [json.loads(l) for l in open(d / "ledger.jsonl")]
    cA = [(x["epoch"], x["val_nll"]) for x in led if "val_nll" in x]
    ax.plot([e for e, _ in c0], [v for _, v in c0], "o-", ms=2.5, color="0.45",
            label="prod 60/10")
    ax.plot([e for e, _ in cA], [v for _, v in cA], "o-", ms=2.5, color="tab:blue",
            label="retrain 120/20")
    ax.set_title(f"seed{s}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val NLL")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("t1_closure_cons_norew val curves: production vs budget retrain", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "val_curves_3seeds.png"); plt.close(fig)

# bins bar chart prod vs retrain vs bb
fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=150)
x = np.arange(5)
w = 0.28
ax.bar(x - w, [np.mean([prod[s]["bins"][b] for s in (0, 1, 2)]) for b in range(5)],
       w, color="0.6", label="prod 3-seed mean")
ax.bar(x, [np.mean([rows[s][1]["bins_q1q5"][b] for s in (0, 1, 2)]) for b in range(5)],
       w, color="tab:blue", label="retrain 3-seed mean")
ax.axhline(bb_overall, color="tab:orange", ls="--", lw=1.2, label=f"iTransformer {bb_overall:.3f}")
ax.set_xticks(x, [f"Q{i+1}" for i in range(5)])
ax.set_xlabel("Load quintile"); ax.set_ylabel("H18 MAE (°C)")
ax.set_title("Main steam H18 MAE by load quintile (3-seed means)")
ax.legend(frameon=False)
ax.grid(alpha=0.3, linewidth=0.5, axis="y")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "bins_prod_vs_retrain.png"); plt.close(fig)

# scatter: retrained 3-seed mean vs bb
maeB = np.abs(z0["bb_pred"] - z0["actual"]).mean(axis=1)
fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=150)
ax.scatter(mae_mean_w, maeB, s=14, color="0.55", zorder=2)
lim = (0, max(mae_mean_w.max(), maeB.max()) * 1.05)
ax.plot([0, lim[1]], [0, lim[1]], "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Retrained 3-seed mean MAE (°C)")
ax.set_ylabel("iTransformer MAE (°C)")
ax.set_title("Per-window H18 MAE (above diagonal = blackbox better)")
fig.tight_layout(); fig.savefig(OUT / "scatter_retrain3s_vs_bb.png"); plt.close(fig)

print(f"report written: {P / 'retrain_report_20260825.md'}")
print(f"3-seed retrained mean: {rm:.3f} | prod mean: {pm:.3f} | bb: {bb_overall:.3f} "
      f"| ratio retrain/bb: {rm/bb_overall:.2f}")
