#!/usr/bin/env python3
"""方向因果重扫：单阀指令 vs 对应侧减温出口温度，滞后按 >=60s 中心化。
A侧文件: 一级减温喷水调节门指令=SH1A(直喷A侧), 二级减温喷水调节门指令=SH2B(交叉,注入A侧)。
"""
import os
import numpy as np
import pandas as pd

CSV = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据_cleaned_10s.csv"
WIN_START, WIN = 70686, 50000
df = pd.read_csv(CSV, usecols=["一级减温喷水调节门指令", "二级减温喷水调节门指令",
                               "一级减温器出口温度", "二级减温器出口温度", "减温水总流量"],
                 dtype=np.float32).iloc[WIN_START:WIN_START+WIN].reset_index(drop=True)

def scan(cmd_col, out_col, tag):
    cmd = df[cmd_col].to_numpy(); T = df[out_col].to_numpy()
    print(f"\n### {tag}: {cmd_col} -> {out_col}")
    for k in [1, 3, 6]:           # 指令变化窗口 (10/30/60s)
        dW = pd.Series(cmd).diff(k).to_numpy()
        thr = np.nanpercentile(np.abs(dW), 50)   # 前 50% 变化事件
        m = np.abs(dW) > thr
        print(f" dW window={k*10}s, n_events={int(m.sum())}")
        for L in [3, 6, 9, 12, 18, 24, 30, 36]:   # 滞后 30~360s
            dT = np.full_like(T, np.nan); dT[L:] = T[L:] - T[:-L]
            mm = m & ~np.isnan(dT)
            opp = float(np.mean(np.sign(dW[mm]) * np.sign(dT[mm]) < 0))
            corr = float(np.corrcoef(dW[mm], dT[mm])[0, 1])
            print(f"   lag={L*10:>3}s  opposite_rate={opp:.3f}  corr={corr:+.3f}")

scan("一级减温喷水调节门指令", "一级减温器出口温度", "SH1A直喷→一减出口(A)")
scan("二级减温喷水调节门指令", "二级减温器出口温度", "SH2B交叉→二减出口(A)")

# 对照: 总流量 (旧方案)
print("\n### 旧方案对照: 减温水总流量 -> 各出口")
for out_col in ["一级减温器出口温度", "二级减温器出口温度"]:
    T = df[out_col].to_numpy(); W = df["减温水总流量"].to_numpy()
    dW = pd.Series(W).diff(6).to_numpy()
    thr = np.nanpercentile(np.abs(dW), 50)
    m = np.abs(dW) > thr
    row = []
    for L in [6, 12, 18, 30]:
        dT = np.full_like(T, np.nan); dT[L:] = T[L:] - T[:-L]
        mm = m & ~np.isnan(dT)
        row.append(f"L{L*10}:{np.mean(np.sign(dW[mm])*np.sign(dT[mm])<0):.2f}")
    print(f"  {out_col}: " + "  ".join(row))
