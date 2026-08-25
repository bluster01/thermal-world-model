"""Recon: candidate v0.6 channels in all_merged_10s.csv (coverage/range/rate).

Single pass with usecols. Reports per-column: non-null %, min/max/mean,
frozen-value ratio (zero-diff fraction), and overall time span / sampling.
"""
import sys
import time

import numpy as np
import pandas as pd

PATH = r"C:\Users\14020\Desktop\时间预测模型\AA数据中心\伊敏12.10\merged_all_data\all_merged_10s.csv"

CANDIDATES = [
    # heat input / fuel
    "校正后总燃料量", "未校正总煤量", "燃料主控输出", "BTU输出", "水煤比",
    # mill furnace-gas (per mill 1..8 handled by prefix match below)
    "1号磨煤机高温炉烟温度(选择后）", "2号磨煤机高温炉烟温度(选择后）",
    "3号磨煤机高温炉烟温度(选择后）", "4号磨煤机高温炉烟温度(选择后）",
    "5号磨煤机高温炉烟温度(选择后）", "6号磨煤机高温炉烟温度(选择后）",
    "7号磨煤机高温炉烟温度(选择后）", "8号磨煤机高温炉烟温度(选择后）",
    "1号磨煤机高温炉烟流量(选择后）", "2号磨煤机高温炉烟流量(选择后）",
    "3号磨煤机高温炉烟流量(选择后）", "4号磨煤机高温炉烟流量(选择后）",
    "5号磨煤机高温炉烟流量(选择后）", "6号磨煤机高温炉烟流量(选择后）",
    "7号磨煤机高温炉烟流量(选择后）", "8号磨煤机高温炉烟流量(选择后）",
    # feeder coal
    "1号给煤机瞬时煤量", "2号给煤机瞬时煤量", "3号给煤机瞬时煤量", "4号给煤机瞬时煤量",
    "5号给煤机瞬时煤量", "6号给煤机瞬时煤量", "7号给煤机瞬时煤量", "8号给煤机瞬时煤量",
    # flue gas
    "烟气含氧量", "三选后A侧烟气含氧量", "三选后B侧烟气含氧量",
    "A侧反应器入口烟气流量", "B侧反应器入口烟气流量",
    "水平低温再热器入口烟气温度(A)", "水平低温再热器入口烟气温度(B)",
    "立式低温再热器入口烟气温度(A)", "立式低温再热器入口烟气温度(B)",
    # air
    "总二次风量", "B侧二次风量选择后", "A侧二次风量选择后",
    # reheater
    "选择后左侧再热器入口蒸汽温度", "选择后右侧再热器入口蒸汽温度",
    "选择后左侧再热出口汽温", "选择后右侧再热出口汽温",
    "再热冷段蒸汽压力(DEH)", "再热蒸汽压力(DEH)",
    # wall temps (diagnostics)
    "汽水分离器出口过热度",
    # spray per-side supervision
    "再热器减温水总流量", "过热器减温水总流量",
    # control context
    "机组负荷_GENERATOR_POWER", "AGC指令", "目标负荷值", "滑压设定",
    "三选后主蒸汽压力", "主汽压力设定",
]

t0 = time.time()
df = pd.read_csv(PATH, usecols=lambda c: c == "time" or c in CANDIDATES)
print(f"read {time.time()-t0:.0f}s rows={len(df)} cols={len(df.columns)}")

tm = pd.to_datetime(df["time"], errors="coerce")
print("time range:", tm.min(), "->", tm.max(), " bad_time:", int(tm.isna().sum()))
dt = tm.diff().dt.total_seconds().dropna()
print("dt seconds: median %.0f  p99 %.0f  >20s frac %.4f" % (
    dt.median(), dt.quantile(0.99), (dt > 20).mean()))

print(f"\n{'column':40s} {'ok%':>6s} {'frozen%':>8s} {'min':>10s} {'mean':>10s} {'max':>10s}")
for c in df.columns:
    if c == "time":
        continue
    v = pd.to_numeric(df[c], errors="coerce")
    ok = v.notna().mean() * 100
    frozen = (v.diff().fillna(0) == 0).mean() * 100
    print(f"{c:40s} {ok:6.1f} {frozen:8.1f} {v.min():10.2f} {v.mean():10.2f} {v.max():10.2f}")
