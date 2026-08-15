#!/usr/bin/env python3
"""ad hoc2 Step 0: 数据 sanity（设计稿 §6 Step 0，图 0）
(a) 减温水总流量 vs [一级阀位, 二级阀位] 回归散点（θ 先验量级）
(b) 阀位/指令差分 vs 出口温度差分滞后相关热图（预期复现 ad-hoc1 §2.2 污染指纹）
落盘: out/step0.json + out/figs/fig0_data_sanity.png
"""
import os
import json
import numpy as np
import pandas as pd
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

CSV = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据_cleaned_10s.csv"
WIN_START, WIN = 70686, 50000
USE = ["减温水总流量", "一级减温调节门阀位", "二级减温调节门阀位",
       "一级减温喷水调节门指令", "二级减温喷水调节门指令",
       "一级减温器出口温度", "二级减温器出口温度"]
df = pd.read_csv(CSV, usecols=USE, dtype=np.float32).iloc[WIN_START:WIN_START + WIN].reset_index(drop=True)

res = {"win": [WIN_START, WIN], "n": len(df)}

# ---------- (a) 流量 vs 阀位 ----------
W = df["减温水总流量"].to_numpy()
v1 = df["一级减温调节门阀位"].to_numpy()
v2 = df["二级减温调节门阀位"].to_numpy()
c1 = df["一级减温喷水调节门指令"].to_numpy()
c2 = df["二级减温喷水调节门指令"].to_numpy()

res["valve_ranges"] = {"v1": [float(np.nanmin(v1)), float(np.nanmax(v1))],
                       "v2": [float(np.nanmin(v2)), float(np.nanmax(v2))],
                       "c1": [float(np.nanmin(c1)), float(np.nanmax(c1))],
                       "c2": [float(np.nanmin(c2)), float(np.nanmax(c2))]}
res["flow_range"] = [float(np.nanmin(W)), float(np.nanmax(W))]

def lr(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.std() == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return {"slope": float(b), "intercept": float(a), "r2": float(r2), "n": int(m.sum())}

res["flow_vs_v1"] = lr(v1, W)
res["flow_vs_v2"] = lr(v2, W)
# 双变量联合回归: W ~ a*v1 + b*v2 + c
mm = ~(np.isnan(v1) | np.isnan(v2) | np.isnan(W))
X = np.column_stack([v1[mm], v2[mm], np.ones(mm.sum())])
coef, *_ = np.linalg.lstsq(X, W[mm], rcond=None)
res["flow_vs_joint"] = {"coef_v1": float(coef[0]), "coef_v2": float(coef[1]), "intercept": float(coef[2])}
res["valve_cmd_corr"] = {"v1_c1": float(np.corrcoef(v1, c1)[0, 1]),
                         "v2_c2": float(np.corrcoef(v2, c2)[0, 1])}
res["flow_basic_stats"] = {"mean": float(np.mean(W)), "std": float(np.std(W)),
                           "q50": float(np.percentile(W, 50)), "q90": float(np.percentile(W, 90))}

# ---------- (b) 滞后相关热图 ----------
def lag_scan(drv, outT, k=6, Ls=np.arange(3, 37)):
    dW = pd.Series(drv).diff(k).to_numpy()
    thr = np.nanpercentile(np.abs(dW), 50)
    m = np.abs(dW) > thr
    rows = []
    for L in Ls:
        dT = np.full_like(outT, np.nan)
        dT[L:] = outT[L:] - outT[:-L]
        mm = m & ~np.isnan(dT)
        rows.append(float(np.corrcoef(dW[mm], dT[mm])[0, 1]))
    return rows, int(m.sum())

T1o = df["一级减温器出口温度"].to_numpy()
T2o = df["二级减温器出口温度"].to_numpy()
pairs = [("v1", v1, T1o), ("v2", v2, T2o), ("c1", c1, T1o), ("c2", c2, T2o)]
labels = ["一级阀位→一减出口", "二级阀位→二减出口", "一级指令→一减出口", "二级指令→二减出口"]
Ls = np.arange(3, 37)
heat = np.zeros((4, len(Ls)))
for i, (name, drv, outT) in enumerate(pairs):
    heat[i], nev = lag_scan(drv, outT)
    res[f"lagscan_{name}"] = {"n_events": nev, "corr_by_lag": {int(L * 10): round(v, 3) for L, v in zip(Ls, heat[i])}}

# ---------- 图 0 ----------
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.42, wspace=0.28)

# (a) 两个 hexbin
for ax, (x, name) in zip([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                          [(v1, "v1"), (v2, "v2")]):
    hb = ax.hexbin(x, W, gridsize=60, bins="log", cmap="YlOrBr", mincnt=1)
    r = res[f"flow_vs_{name}"]
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    ax.plot(xs, r["slope"] * xs + r["intercept"], "r-", lw=1.5)
    ax.set_title(f"减温水总流量 vs {name} 阀位（θ 先验：斜率 {r['slope']:.2f}，R²={r['r2']:.3f}）", fontsize=10)
    ax.set_xlabel("阀位（原始量纲）")
    ax.set_ylabel("减温水总流量（原始量纲）")
    fig.colorbar(hb, ax=ax, label="样本数 (log)")
    ax.grid(alpha=0.25)

# (b) 热图
axh = fig.add_subplot(gs[1, :])
im = axh.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
axh.set_xticks(range(len(Ls)), [str(int(L * 10)) for L in Ls])
axh.set_yticks(range(4), labels)
axh.set_xlabel("滞后 (s)，驱动差分窗 60s，事件=|dW|>中位数")
axh.set_title("阀位/指令差分 vs 出口温度差分：滞后相关热图（红=喷水增→温升=污染指纹，蓝=物理预期）", fontsize=10)
for i in range(4):
    for j in range(len(Ls)):
        axh.text(j, i, f"{heat[i, j]:+.2f}", ha="center", va="center",
                 fontsize=6.5, color="k" if abs(heat[i, j]) < 0.3 else "w")
fig.colorbar(im, ax=axh, label="相关系数 corr", fraction=0.025)
fig.suptitle("Step 0 数据 sanity：喷水流量-阀位标定 + 动作通道方向指纹（窗口 70686–120686，10s）", fontsize=12)
fig.savefig(os.path.join(FIG, "fig0_data_sanity.png"), dpi=110, bbox_inches="tight")
plt.close(fig)

with open(os.path.join(OUT, "step0.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
