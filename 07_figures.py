#!/usr/bin/env python3
"""最终图集：4 张图（中文标注，Noto Sans CJK SC）"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK HK", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib import font_manager as _fm
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try:
        _fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]

C = {"v0": "#d1495b", "v1": "#edae49", "v2": "#2a9d8f", "v1x": "#0072b2",
     "v2x": "#9b5de5", "v2o": "#6a994e", "v2b": "#bc6c25"}
T_BAND = [557.75, 572.13]

def load(v):
    d = np.load(os.path.join(OUT, f"rollout_{v}.npz"))
    return d["preds"][:, 4], d["truths"][:, 4]

# ================= Fig 1: 5h rollout 主汽温 =================
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
tmin = np.arange(1800) / 6.0
tr = load("v0")[1]
for v in ["v0", "v1", "v2", "v1x", "v2x"]:
    p, t = load(v)
    axes[0].plot(tmin, p, color=C[v], lw=1.1, label=v)
    axes[1].plot(tmin, p - t, color=C[v], lw=0.9)
axes[0].plot(tmin, tr, color="k", lw=1.0, ls="--", label="真实值")
axes[0].axhspan(*T_BAND, color="g", alpha=0.07, label="运行带 p1–p99")
axes[0].set_ylabel("主汽温 (°C)")
axes[0].set_title("1800 步递归 rollout（5h）：特征路线 vs 驱动路线")
axes[1].set_ylabel("预测 − 真实 (°C)")
axes[1].set_xlabel("rollout 时间 (min)")
axes[0].legend(fontsize=8, ncol=3, loc="upper left")
axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_rollout_main.png"), dpi=110)
plt.close(fig)

# ================= Fig 2: 顺序违例与误差时间对应 =================
fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
OUTPUTS5 = ["一减入口", "一减出口", "二减入口", "二减出口", "主汽温"]
for ax, v in zip(axes, ["v0", "v2o"]):
    d = np.load(os.path.join(OUT, f"rollout_{v}.npz"))
    p, t = d["preds"], d["truths"]
    err = np.abs(p[:, 4] - t[:, 4])
    viol = np.zeros(len(p), bool)
    for lo, hi in [(1, 2), (3, 4), (2, 3), (1, 0), (0, 2)]:
        viol |= p[:, lo] >= p[:, hi]
    ax.fill_between(tmin, 0, err.max() * 1.1, where=viol, color="r", alpha=0.12,
                    label="沿程顺序违例时段")
    ax.plot(tmin, err, color=C[v], lw=0.8, label=f"{v} |误差|")
    ax.set_ylabel("|主汽温误差| (°C)")
    ax.set_title(f"{v}：违例步占比 {viol.mean()*100:.1f}%（v0 平台段即违例密集区）" if v == "v0"
                 else f"{v}：违例步占比 {viol.mean()*100:.1f}%")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
axes[1].set_xlabel("rollout 时间 (min)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_order_violation.png"), dpi=110)
plt.close(fig)

# ================= Fig 3: 拆解汇总 =================
fig, axes = plt.subplots(2, 1, figsize=(11, 6.4))
vs = ["v0", "v1", "v0b", "v2o", "v2b", "v2", "v1x", "v2xb", "v2x"]
rmse, mx = [], []
corr_l, bias_l, std_l = [], [], []
for v in vs:
    p, t = load(v)
    rmse.append(np.sqrt(np.mean((p - t) ** 2)))
    mx.append(np.max(np.abs(p - t)))
    corr_l.append(np.corrcoef(p, t)[0, 1])
    bias_l.append(p.mean() - t.mean())
    std_l.append(p.std() / t.std())
x = np.arange(len(vs))
ax = axes[0]
b1 = ax.bar(x - 0.2, rmse, 0.38, color="#2a9d8f", label="rollout RMSE")
b2 = ax.bar(x + 0.2, mx, 0.38, color="#d1495b", label="最大误差")
ax.set_xticks(x); ax.set_xticklabels(vs)
ax.set_ylabel("°C"); ax.set_title("各变体 rollout 误差（虚线 = v0 基线）")
ax.axhline(rmse[0], color="#2a9d8f", ls="--", lw=0.8)
ax.axhline(mx[0], color="#d1495b", ls="--", lw=0.8)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.1f}",
            ha="center", fontsize=7)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax = axes[1]
ax.plot(x, corr_l, "o-", color="#0072b2", label="形状相关 corr")
ax.plot(x, std_l, "s-", color="#edae49", label="幅度比 std(pred)/std(truth)")
ax.axhline(1.0, color="k", ls=":", lw=0.8)
ax.axhline(0, color="k", ls=":", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(vs)
ax.set_ylabel("无量纲"); ax.set_title("形状保真度与幅度还原（corr 越高形状越好；幅度比≈1 为完全还原）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_ablation_summary.png"), dpi=110)
plt.close(fig)

# ================= Fig 4: v0 平台段放大 + 残差驱动筛查 =================
fig = plt.figure(figsize=(11, 8))
ax1 = fig.add_subplot(2, 1, 1)
d = np.load(os.path.join(OUT, "rollout_v0.npz"))
p, t = d["preds"], d["truths"]
i0, i1 = 690, 760
tzoom = np.arange(i0, i1) / 6.0
names = ["一减入口", "一减出口", "二减入口", "二减出口", "主汽温"]
for j in range(5):
    ax1.plot(tzoom, p[i0:i1, j], lw=1.1, color=plt.cm.viridis(j / 4),
             label=f"预测 {names[j]}")
    ax1.plot(tzoom, t[i0:i1, j], lw=0.8, ls="--", color=plt.cm.viridis(j / 4), alpha=0.6)
ax1.set_ylabel("温度 (°C)")
ax1.set_title("v0 平台段放大（115–127 min）：预测链失序——一减出口(475)>一减入口(463)、二减出口<二减入口，物理不可能")
ax1.legend(fontsize=7, ncol=2, loc="center right")
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(2, 1, 2)
cands = [("二级减温调节阀设定", -0.570), ("再热出口汽温", -0.519),
         ("高压缸排汽至再热器温度", -0.449), ("再热器一级减温入口汽温", -0.431),
         ("再热器减温水总流量", -0.376), ("B侧主汽温", -0.374),
         ("一级减温副调设定值_B", -0.335), ("二级减温中间设定值", -0.321)]
names = [c[0] for c in cands][::-1]
vals = [c[1] for c in cands][::-1]
ax2.barh(names, vals, color=["#d1495b" if v < -0.45 else "#edae49" for v in vals])
ax2.set_xlabel("与 v1 残差的相关系数")
ax2.set_title("v1 残差(预测−真实)的候选驱动筛查：再热侧/控制器变量解释幅度偏置")
ax2.grid(alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_platform_and_drivers.png"), dpi=110)
plt.close(fig)

print("saved:", sorted(os.listdir(FIG)))
