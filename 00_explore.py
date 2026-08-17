#!/usr/bin/env python3
"""ad-hoc 实验 00：数据探查 + 窗口选择 + 物理先验验证。
只读数据集，输出 json 到 out/explore.json，全部使用 float32 + usecols。
"""
import json, os, time
import numpy as np
import pandas as pd

CSV = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据_cleaned_10s.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

COLS = [
    "机组负荷", "主蒸汽压力", "主蒸汽流量", "主给水流量", "总风量指令", "总二次风量",
    "燃料主控输出", "未校正总煤量", "水煤比", "减温水总流量",
    "一级减温喷水调节门指令", "二级减温喷水调节门指令",
    "一级减温调节门阀位", "二级减温调节门阀位",
    "省煤器出口给水温度", "分离器出口温度", "分离器出口压力",
    "过热器出口温度升速率", "AGC指令", "机组负荷变化率",
    "一级减温器入口温度", "二级减温器入口温度",
    "一级减温器出口温度", "二级减温器出口温度", "末级过热器出口汽温",
]
WIN = 50000  # 50K 样本 ≈ 5.8 天

t0 = time.time()
df = pd.read_csv(CSV, usecols=COLS, dtype=np.float32)
print(f"loaded {len(df)} rows in {time.time()-t0:.0f}s")
rpt = {"n_rows": len(df)}

# NaN / 异常
rpt["nan_counts"] = {c: int(df[c].isna().sum()) for c in COLS if df[c].isna().sum() > 0}
df = df.ffill().bfill()

# 负荷滚动 std 最大窗口（挑变工况段）
load = df["机组负荷"].to_numpy()
win_std = pd.Series(load).rolling(WIN, center=False).std().to_numpy()
i0 = int(np.nanargmax(win_std)) - WIN + 1
i0 = max(0, i0)
rpt["window_start"] = i0
rpt["window_end"] = i0 + WIN
rpt["load_std_in_window"] = float(np.std(load[i0:i0+WIN]))

# 温度链相关性（沿程 5 点）
temps = ["分离器出口温度", "一级减温器入口温度", "一级减温器出口温度",
         "二级减温器入口温度", "二级减温器出口温度", "末级过热器出口汽温"]
rpt["temp_chain_corr"] = {t: round(float(np.corrcoef(df[t], df["末级过热器出口汽温"])[0, 1]), 4)
                          for t in temps[:-1]}

# 方向成立率：喷水指令变化 vs 减温出口温度变化（多滞后）
W = df["减温水总流量"].to_numpy()
dW = pd.Series(W).diff(6).to_numpy()  # 60s 窗差分
dir_stats = {}
for tag, Tcol in [("sh1_out", "一级减温器出口温度"), ("sh2_out", "二级减温器出口温度")]:
    T = df[Tcol].to_numpy()
    dT = pd.Series(T).diff(6).to_numpy()
    m = (np.abs(dW) > 0.2) & ~np.isnan(dW) & ~np.isnan(dT)  # 喷水有实质变化
    agree = float(np.mean(np.sign(dW[m]) * np.sign(dT[m]) < 0))  # 期望相反号
    dir_stats[tag] = {"n_events": int(m.sum()), "opposite_sign_rate": round(agree, 4)}
rpt["spray_direction_6step"] = dir_stats

# 主汽温分位数（band 损失边界）
Tmain = df["末级过热器出口汽温"]
rpt["Tmain_quantiles"] = {q: round(float(np.percentile(Tmain, q)), 2) for q in [0.5, 1, 99, 99.5]}

# 阀位 vs 指令（执行机构响应）
rpt["valve_cmd_corr"] = {
    "sh1": round(float(np.corrcoef(df["一级减温喷水调节门指令"], df["一级减温调节门阀位"])[0, 1]), 4),
    "sh2": round(float(np.corrcoef(df["二级减温喷水调节门指令"], df["二级减温调节门阀位"])[0, 1]), 4),
}

# ===== 沿程顺序不变量（无滞后，可直接作为损失） =====
def order_rate(a, b, op):
    """op='gt': P(a>b); 'lt': P(a<b)"""
    v = (df[a] > df[b]).to_numpy() if op == "gt" else (df[a] < df[b]).to_numpy()
    return round(float(v.mean()), 4)

rpt["chain_order_rates"] = {
    "T_sh1_in > T_sep_out": order_rate("一级减温器入口温度", "分离器出口温度", "gt"),
    "T_sh1_out < T_sh1_in": order_rate("一级减温器出口温度", "一级减温器入口温度", "lt"),
    "T_sh2_in > T_sh1_out": order_rate("二级减温器入口温度", "一级减温器出口温度", "gt"),
    "T_sh2_out < T_sh2_in": order_rate("二级减温器出口温度", "二级减温器入口温度", "lt"),
    "T_main > T_sh2_out": order_rate("末级过热器出口汽温", "二级减温器出口温度", "gt"),
}
# 顺序不变量只在喷水/加热活跃时成立，检查带喷水条件时的成立率
rpt["chain_order_conditional"] = {}
for tag, Tcol in [("sh1", "一级减温器出口温度"), ("sh2", "二级减温器出口温度")]:
    pass

# ===== 喷水方向：滞后扫描 =====
dW = pd.Series(W).diff(6).to_numpy()
lag_scan = {}
for lag in [1, 3, 6, 12, 18, 30]:
    for tag, Tcol in [("sh1_out", "一级减温器出口温度"), ("sh2_out", "二级减温器出口温度")]:
        T = df[Tcol].to_numpy()
        dT = np.full_like(T, np.nan)
        dT[lag:] = T[lag:] - T[:-lag]
        m = (np.abs(dW) > 0.2) & ~np.isnan(dW) & ~np.isnan(dT)
        rate = float(np.mean(np.sign(dW[m]) * np.sign(dT[m]) < 0))
        lag_scan.setdefault(f"lag{lag}_{tag}", {}).update(
            {"n": int(m.sum()), "opposite_rate": round(rate, 4)})
rpt["spray_direction_lag_scan"] = lag_scan

# 主汽温变化率分位数（速率有界约束）
dTmain = pd.Series(Tmain).diff(1).abs().to_numpy()
rpt["dTmain_abs_quantiles"] = {q: round(float(np.nanpercentile(dTmain, q)), 3) for q in [90, 99, 99.9]}

# 窗口内子统计
w = df.iloc[i0:i0+WIN]
rpt["window_Tmain_range"] = [round(float(w["末级过热器出口汽温"].min()), 2),
                             round(float(w["末级过热器出口汽温"].max()), 2)]
rpt["window_load_range"] = [round(float(w["机组负荷"].min()), 1), round(float(w["机组负荷"].max()), 1)]

with open(os.path.join(OUT, "explore.json"), "w") as f:
    json.dump(rpt, f, ensure_ascii=False, indent=2)
print(json.dumps(rpt, ensure_ascii=False, indent=2))
