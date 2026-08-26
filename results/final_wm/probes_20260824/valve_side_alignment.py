"""Valve/temp side-alignment verification vs the 381-col final data file.

2026-08-25, data-side forensics (no model involvement).
Question: the mainT per-side files' valve columns may have been cross-wired
at extraction time (user report). The all_merged 377-col file carries
explicit A/B valve labels AND left/right temp labels, so it can arbitrate:
  (1) mainT valve cols <-> 377 explicit valve cols (identity, corr~1)
  (2) 377 A/B valves <-> 377 left/right temps (physical wiring: valve open
      -> its own side's desuperheater outlet cools -> negative corr)
  (3) mainT A/B temp cols <-> 377 left/right temps (side identity of the
      per-side mainT files)
"""
import sys

import numpy as np
import pandas as pd

ALL = "/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/cleaned_data/all_merged_10s.csv"
PA = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据03_cleaned_10s.csv"
PB = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/B侧主汽温全数据03_cleaned_10s.csv"

USE_377 = [
    "time",
    "过热器一级减温器A喷水调节阀主调输出",
    "过热器一级减温器B喷水调节阀主调输出",
    "过热器二级减温器A喷水调节阀主调输出",
    "过热器二级减温器B喷水调节阀主调输出",
    "过热器一级减温器A侧喷水调节门阀位反馈",
    "过热器一级减温器B侧喷水调节门阀位反馈",
    "过热器二级减温器A侧喷水调节门阀位反馈",
    "过热器二级减温器B侧喷水调节门阀位反馈",
    "选择后左侧一过喷水减温器入口",
    "选择后左侧一过喷水减温器出口",
    "选择后右侧一过喷水减温器入口",
    "选择后右侧一过喷水减温器出口",
    "选择后二级减温器左侧入口蒸汽",
    "选择后左侧二过喷水减温器出口",
    "选择后二级减温器右侧入口蒸汽",
    "选择后右侧二过喷水减温器出口",
    "选择后左侧末级过热器出口汽温",
    "选择后右侧末级过热器出口汽温",
    "主蒸汽温度",
]

USE_T = ["date",
         "一级减温调节门阀位", "二级减温调节门阀位",
         "一级减温器入口温度", "一级减温器出口温度",
         "二级减温器入口温度", "二级减温器出口温度",
         "末级过热器出口汽温"]

print("[load] 377-col subset ...", flush=True)
g = pd.read_csv(ALL, usecols=USE_377).rename(columns={"time": "date"})
print(f"[load] 377 rows={len(g)}", flush=True)
a = pd.read_csv(PA, usecols=USE_T)
b = pd.read_csv(PB, usecols=USE_T)
print(f"[load] mainT A={len(a)} B={len(b)}", flush=True)

def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 1000:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

def show(title, rows):
    print(f"\n== {title}")
    for name, vals in rows:
        print(f"  {name:34s} " + " ".join(f"{v:+.3f}" for v in vals))

# ---- (1) identity: mainT valve cols vs 377 explicit valve cols ----
ma = pd.merge(a, g, on="date", how="inner")
mb = pd.merge(b, g, on="date", how="inner")
print(f"\n[align] overlap rows A-file x 377 = {len(ma)}, B-file x 377 = {len(mb)}", flush=True)

v377 = ["过热器一级减温器A喷水调节阀主调输出", "过热器一级减温器B喷水调节阀主调输出",
        "过热器二级减温器A喷水调节阀主调输出", "过热器二级减温器B喷水调节阀主调输出",
        "过热器一级减温器A侧喷水调节门阀位反馈", "过热器一级减温器B侧喷水调节门阀位反馈",
        "过热器二级减温器A侧喷水调节门阀位反馈", "过热器二级减温器B侧喷水调节门阀位反馈"]
for tag, m in (("mainT_A", ma), ("mainT_B", mb)):
    print(f"\n[1] {tag} valve columns vs 377 explicit valves (identity)")
    for vt in ("一级减温调节门阀位", "二级减温调节门阀位"):
        vals = [corr(m[vt].values, m[c].values) for c in v377]
        print(f"  {vt:16s}: " + " ".join(f"{v:+.3f}" for v in vals))
print("  377 columns:")
for c in v377:
    print(f"    - {c}")

# ---- (2) physical wiring: 377 A/B valve FEEDBACK vs left/right temps ----
print("\n[2] 377 valve FEEDBACK -> left/right stage outlets (negative = cooling that side)")
temps2 = ["选择后左侧一过喷水减温器出口", "选择后右侧一过喷水减温器出口",
          "选择后左侧二过喷水减温器出口", "选择后右侧二过喷水减温器出口",
          "选择后左侧末级过热器出口汽温", "选择后右侧末级过热器出口汽温"]
valves2 = ["过热器一级减温器A侧喷水调节门阀位反馈", "过热器一级减温器B侧喷水调节门阀位反馈",
           "过热器二级减温器A侧喷水调节门阀位反馈", "过热器二级减温器B侧喷水调节门阀位反馈"]
for v in valves2:
    vals = [corr(g[v].values, g[t].values) for t in temps2]
    print(f"  {v}: " + " ".join(f"{x:+.3f}" for x in vals))
print("  temps order: " + " | ".join(temps2))

# ---- (3) mainT file side identity vs 377 left/right ----
print("\n[3] mainT file temps vs 377 left/right temps (identity of A/B files)")
for tag, m, side in (("A", ma, "A"), ("B", mb, "B")):
    vals = [corr(m["末级过热器出口汽温"].values, m[t].values) for t in
            ("选择后左侧末级过热器出口汽温", "选择后右侧末级过热器出口汽温")]
    print(f"  mainT_{side} 末级过热器出口汽温 vs [左侧,右侧]: " + " ".join(f"{x:+.3f}" for x in vals))
    vals = [corr(m["一级减温器出口温度"].values, m[t].values) for t in
            ("选择后左侧一过喷水减温器出口", "选择后右侧一过喷水减温器出口")]
    print(f"  mainT_{side} 一级减温器出口温度 vs [左侧,右侧]: " + " ".join(f"{x:+.3f}" for x in vals))
    vals = [corr(m["二级减温器出口温度"].values, m[t].values) for t in
            ("选择后左侧二过喷水减温器出口", "选择后右侧二过喷水减温器出口")]
    print(f"  mainT_{side} 二级减温器出口温度 vs [左侧,右侧]: " + " ".join(f"{x:+.3f}" for x in vals))
