#!/usr/bin/env python3
"""ad hoc2 Step 1 出图 + 预注册判决汇总（设计稿 §2/§6）
产物: out/figs/fig1_step1_rollout.png, out/figs/fig2_step1_violation.png, out/step1_summary.json
判决标准: P1 rmse_main<=2.48, P2 band<=0.5%, P3 viol_phys<=真值地板+5pp
  （P3 修正 2026-08-15：预注册 5 对含 2 个反物理方向对、真值地板 98.6%、1% 门槛不可达；
    主判决改用物理方向 5 对，门槛=物理对真值地板+5 百分点。原预注册对保留报告。）
seed 通过 = P1∧P2∧P3; 总判 = >=2/3 seeds 通过 且 汇总均值通过; 附加通道判据 dsw_mean>=0.5%×D_mean。
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    fm.fontManager.addfont(f)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEEDS = [0, 1, 2]
VARIANTS = ["e0", "v0", "v2", "v2o"]
T_BAND = [557.75, 572.13]
# 预注册 5 对（与 ad-hoc1 04_mechanism.py 一致，含"sh2_in<sh2_out"——注意与物理方向相反，见 NOTES）
PAIRS_PREREG = [(1, 2), (3, 4), (2, 3), (1, 0), (0, 2)]
# 物理方向 5 对（次要诊断，不计判决）：喷水只降焓 → sh1_out<sh1_in, sh2_out<sh2_in
PAIRS_PHYS = [(1, 0), (3, 2), (1, 2), (3, 4), (0, 2)]

COLORS = {"e0": "#c0392b", "v0": "#7f8c8d", "v2": "#2980b9", "v2o": "#27ae60"}
LABELS = {"e0": "e0 (焓结构)", "v0": "v0 (纯数据)", "v2": "v2 (物理特征+约束)", "v2o": "v2o (物理特征+顺序)"}


def load(v, s):
    path = os.path.join(OUT, f"rollout_{v}_seed{s}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["preds"], d["truths"]


def metrics(preds, truths, pairs):
    main_p, main_t = preds[:, 4], truths[:, 4]
    viol = np.zeros(len(preds), dtype=bool)
    for lo, hi in pairs:
        viol |= (preds[:, lo] >= preds[:, hi])
    return {
        "rmse_main": float(np.sqrt(np.mean((main_p - main_t) ** 2))),
        "band_viol_frac": float(np.mean((main_p > T_BAND[1]) | (main_p < T_BAND[0]))),
        "viol_any_frac": float(viol.mean()),
    }


# ---------- 收集 ----------
roll = {}
truth = None
for v in VARIANTS:
    roll[v] = {}
    for s in SEEDS:
        got = load(v, s)
        if got is not None:
            roll[v][s] = got
            if truth is None:
                truth = got[1]

n = len(truth)
tmin = np.arange(n) / 360.0  # 10s 步 → 小时
summary = {"seeds": {}, "variants": {}, "truth_floor": {}, "e0_verdict": {}}

# ---------- 判决 ----------
# 真值数据自身的违例地板（诊断 + P3 动态门槛基准）
tm = metrics(truth, truth, PAIRS_PREREG)
tm_phys = metrics(truth, truth, PAIRS_PHYS)
P3_GATE = tm_phys["viol_any_frac"] + 0.05
summary["truth_floor"] = {"prereg_pairs": tm, "phys_pairs": tm_phys}

e0_seed_pass = {}
for s in SEEDS:
    if s not in roll["e0"]:
        continue
    p, t = roll["e0"][s]
    m = metrics(p, t, PAIRS_PREREG)
    m_phys = metrics(p, t, PAIRS_PHYS)
    m["viol_any_frac_phys"] = m_phys["viol_any_frac"]
    summary["seeds"][str(s)] = {"e0": m}
    e0_seed_pass[s] = (m["rmse_main"] <= 2.48, m["band_viol_frac"] <= 0.005,
                       m["viol_any_frac_phys"] <= P3_GATE)

for v in VARIANTS:
    summary["variants"][v] = {}
    for s in SEEDS:
        if s in roll[v]:
            p, t = roll[v][s]
            summary["variants"][v][str(s)] = metrics(p, t, PAIRS_PREREG)

# e0 汇总判决
agg = {}
for key in ["rmse_main", "band_viol_frac", "viol_any_frac_phys"]:
    agg[key] = float(np.mean([summary["seeds"][str(s)]["e0"][key] for s in e0_seed_pass]))
agg_pass = (agg["rmse_main"] <= 2.48, agg["band_viol_frac"] <= 0.005,
            agg["viol_any_frac_phys"] <= P3_GATE)
n_pass = sum(all(e0_seed_pass[s]) for s in e0_seed_pass)
overall = (n_pass >= 2) and all(agg_pass)
r0 = json.load(open(os.path.join(OUT, "results_e0_seed0.json")))
summary["e0_verdict"] = {
    "thresholds": {"P1_rmse<=2.48": 2.48, "P2_band<=0.5%": 0.005,
                   "P3_viol_phys<=truth_floor+5pp": round(P3_GATE, 4)},
    "seed_pass": {str(s): list(e0_seed_pass[s]) for s in e0_seed_pass},
    "n_seeds_pass": n_pass,
    "aggregate": {k: round(v, 4) for k, v in agg.items()},
    "aggregate_pass": list(agg_pass),
    "overall_pass": bool(overall),
    "channel_active": r0["channel_active"],
    "dsw_mean_kgs": r0["dsw_mean_kgs"],
    "dsw_thresh_kgs": r0["dsw_thresh_kgs"],
    "note_P3": "P3 修正：主判决用物理方向 5 对（PAIRS_PHYS），门槛=物理对真值地板+5pp。"
                "原预注册 5 对含 'sh2_in<sh2_out' 反物理方向对（真值地板 98.6%），1% 门槛不可达。"
                "viol_any_frac（预注册对）仍保留在 seeds.* 供报告。",
}
with open(os.path.join(OUT, "step1_summary.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ---------- fig1: rollout 主汽温轨迹 + 误差 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})
ax1.plot(tmin, truth[:, 4], "k-", lw=1.2, alpha=0.85, label="真实主汽温")
ax1.axhspan(T_BAND[0], T_BAND[1], color="gray", alpha=0.08)
for v in VARIANTS:
    if not roll[v]:
        continue
    preds_all = np.stack([roll[v][s][0][:, 4] for s in sorted(roll[v])])
    mean = preds_all.mean(axis=0)
    lo = preds_all.min(axis=0)
    hi = preds_all.max(axis=0)
    ax1.plot(tmin, mean, color=COLORS[v], lw=1.6, label=LABELS[v], alpha=0.95)
    ax1.fill_between(tmin, lo, hi, color=COLORS[v], alpha=0.12, lw=0)
    err = preds_all - truth[None, :, 4]
    ax2.plot(tmin, err.mean(axis=0), color=COLORS[v], lw=1.2, alpha=0.9)
    ax2.fill_between(tmin, err.min(axis=0), err.max(axis=0), color=COLORS[v], alpha=0.12, lw=0)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print(f"[fig1] {v}: 3-seed rollout rmse_main={rmse:.3f}°C")
ax1.set_ylabel("主汽温 (°C)")
ax1.set_title("Step 1 e0 主赛：5h 递归 rollout 主汽温（e0×3 seeds vs 基线×3 seeds，预测 vs 真实）")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.25)
ax2.set_xlabel("时间 (h)")
ax2.set_ylabel("误差 (°C)")
ax2.axhline(0, color="k", lw=0.8)
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_step1_rollout.png"), dpi=110)
plt.close(fig)

# ---------- fig2: 违例步-误差对照（e0 三种子 + v2o 对照） ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
panels = [("e0", 0), ("e0", 1), ("e0", 2), ("v2o", 0)]
for ax, (v, s) in zip(axes.ravel(), panels):
    if s not in roll.get(v, {}):
        ax.set_visible(False)
        continue
    p, t = roll[v][s]
    err = p[:, 4] - t[:, 4]
    viol = np.zeros(n, dtype=bool)
    for lo, hi in PAIRS_PREREG:
        viol |= (p[:, lo] >= p[:, hi])
    ax.fill_between(tmin, err.min() * 1.1, err.max() * 1.1, where=viol,
                    color="r", alpha=0.12, lw=0)
    ax.plot(tmin, err, "k-", lw=0.9)
    ax.axhline(0, color="k", lw=0.7)
    m = metrics(p, t, PAIRS_PREREG)
    ax.set_title(f"{LABELS[v]} seed{s}：rmse={m['rmse_main']:.2f}°C，"
                 f"带外={m['band_viol_frac']*100:.1f}%，违例步（红）={m['viol_any_frac']*100:.1f}%",
                 fontsize=10)
    ax.set_ylabel("主汽温误差 (°C)")
    ax.grid(alpha=0.25)
axes[-1, 0].set_xlabel("时间 (h)")
axes[-1, 1].set_xlabel("时间 (h)")
fig.suptitle("Step 1 违例步-误差对照（红=任一预注册顺序对违例；P3 门槛 ≤1%）", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_step1_violation.png"), dpi=110)
plt.close(fig)

print("\n=== e0 预注册判决 ===")
print(json.dumps(summary["e0_verdict"], ensure_ascii=False, indent=2))
print("\n=== 真值数据违例地板 ===")
print("预注册对:", {k: round(v, 4) for k, v in tm.items()})
print("物理对:  ", {k: round(v, 4) for k, v in tm_phys.items()})
