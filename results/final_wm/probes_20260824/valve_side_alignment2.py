"""Resolve the [2]-vs-mainT correlation contradiction + stage-1 wiring.

All correlations here on the SAME overlap window (mainT x 377 inner join).
Adds stuck ratios of the four desuperheater outlet sensors to explain
attenuated correlations on a possibly-dead side.
"""
import numpy as np
import pandas as pd

ALL = "/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/cleaned_data/all_merged_10s.csv"
PA = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据03_cleaned_10s.csv"
PB = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/B侧主汽温全数据03_cleaned_10s.csv"

V377 = ["过热器一级减温器A侧喷水调节门阀位反馈", "过热器一级减温器B侧喷水调节门阀位反馈",
        "过热器二级减温器A侧喷水调节门阀位反馈", "过热器二级减温器B侧喷水调节门阀位反馈"]
T377 = ["选择后左侧一过喷水减温器出口", "选择后右侧一过喷水减温器出口",
        "选择后左侧二过喷水减温器出口", "选择后右侧二过喷水减温器出口"]

g = pd.read_csv(ALL, usecols=["time"] + V377 + T377).rename(columns={"time": "date"})
a = pd.read_csv(PA, usecols=["date", "一级减温调节门阀位", "二级减温调节门阀位",
                             "一级减温器出口温度", "二级减温器出口温度"])
b = pd.read_csv(PB, usecols=["date", "一级减温调节门阀位", "二级减温调节门阀位",
                             "一级减温器出口温度", "二级减温器出口温度"])
ma = pd.merge(a, g, on="date", how="inner")
print("overlap rows:", len(ma))

def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

def stuck(x, tol=0.05):
    d = np.abs(np.diff(x[np.isfinite(x)]))
    return float((d < tol).mean())

print("\n[2b] overlap-window: 377 valve FEEDBACK vs L/R stage outlets (tracking confound: own side = flattest)")
print(f"{'valve':34s} | 左一过 | 右一过 | 左二过 | 右二过 | stuck")
for v in V377:
    vals = [corr(ma[v].values, ma[t].values) for t in T377]
    print(f"{v:34s} | " + " ".join(f"{x:+.3f}" for x in vals) + f" | {stuck(ma[v].values):.3f}")

print("\n[2c] overlap-window: mainT valve cols vs L/R stage outlets")
for vt in ["一级减温调节门阀位", "二级减温调节门阀位"]:
    vals = [corr(ma[vt].values, ma[t].values) for t in T377]
    print(f"  mainT_A {vt:12s}: " + " ".join(f"{x:+.3f}" for x in vals))

print("\n[2d] outlet sensor quality on overlap window (stuck ratio at 0.05)")
for t in T377:
    print(f"  {t}: stuck={stuck(ma[t].values):.3f} std={ma[t].std():.2f}")
for f, tag in ((a, "mainT_A"), (b, "mainT_B")):
    for t in ["一级减温器出口温度", "二级减温器出口温度"]:
        print(f"  {tag} {t}: stuck={stuck(f[t].values):.3f} std={f[t].std():.2f}")

print("\n[2e] valve-temp lead-lag (valve CHANGES at t vs temp CHANGES at t+k), k=+3 steps (30s)")
for vt in ["一级减温调节门阀位", "二级减温调节门阀位"]:
    dv = np.diff(ma[vt].values)
    for t in T377:
        dt = np.diff(ma[t].values)
        n = min(len(dv), len(dt)) - 3
        c = corr(dv[:n], dt[3:3 + n])
        print(f"  d({vt}) -> d({t.split('出口')[0][4:]}) lag30s: {c:+.3f}")
