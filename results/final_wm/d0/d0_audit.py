#!/usr/bin/env python3
"""D0 数据审计 — 步骤2: 点位映射审计表 + 质量门（矩阵 §1 合同）

输入: mainT/A侧03 + B侧03 (707k 行, 41 列, 10s)
输出: results/final_wm/d0/mapping_audit.json + quality_gates.json
纪律: 映射置信度三档 (HIGH/MEDIUM/MISSING); 未闭合通道如实判 MIXED, 不自行补映射
"""
import json
import hashlib
import numpy as np
import pandas as pd

BASE = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT"
OUT = "/home/bluster/projectA/thermal-world-model/results/final_wm/d0"
A, B = f"{BASE}/A侧主汽温全数据03_cleaned_10s.csv", f"{BASE}/B侧主汽温全数据03_cleaned_10s.csv"

# ---------------- 1. 点位映射审计表 ----------------
# 来源: 既有代码映射 (不自行推断):
#   E0: adhoc2 02_train.py e0_build_windows (exo_cols 9通道顺序+单位换算+init_states 分段边界)
#   PINN: adhoc_pinn 01_experiment.py (EXO/OUTPUTS, 同 5 观测温度)
#   RM3: src/phase35/schema.py (TARGET/VALVE/SP 列) + RM3 交叉 cache (A阀→B温)
# registry channel -> (DCS tag, 单位换算, 置信度, 注记)
MAPPING = [
    # BOUNDARY (E0 exo_cols L274-276)
    ("steam_flow", "主蒸汽流量", "t/h -> kg/s (x1/3.6)", "HIGH", "E0 L278 换算 /3.6"),
    ("coal_command", "未校正总煤量", "t/h, 原单位", "HIGH", "E0 exo_cols[1] 作 uB; PINN EXO 同时含燃料主控输出, 但煤量通道选未校正总煤量"),
    ("separator_pressure", "分离器出口压力", "MPa, 原单位", "HIGH", "E0 exo_cols[2]"),
    ("separator_temperature", "分离器出口温度", "degC, 原单位", "HIGH", "E0 exo_cols[3]"),
    ("feedwater_temperature", "省煤器出口给水温度", "degC, 原单位", "HIGH", "E0 exo_cols[4]; PINN EXO 同"),
    ("outlet_pressure", "末级过热器出口压力", "MPa, 原单位", "HIGH", "E0 exo_cols[7]"),
    ("spray_flow_total", "减温水总流量", "t/h, 原单位", "HIGH", "E0 exo_cols[8]; registry 标注 unreliable (E4: 读W污染动作通道)"),
    # ACTION (E0 exo_cols[5,6], L279-280)
    ("valve1_position", "一级减温调节门阀位", "clip(lower=0)/100 -> fraction", "HIGH",
     "E0 L279; 交叉映射(用户2026-08-09确认+RM3交叉cache): A侧阀管B侧温"),
    ("valve2_position", "二级减温调节门阀位", "clip(lower=0)/100 -> fraction", "HIGH",
     "E0 L280; RM3 schema VALVE_COLUMN=二级减温调节门阀位; 交叉同上"),
    # OBSERVATION (E0 OUTPUTS L35-36; WM transition.py initial_steady_state L260-264 锚点 obs[0]/obs[2]/obs[4])
    # 汽水系统路径: 低过 -> 一级减温 -> 屏过 -> 二级减温 -> 高过(末级过热器)
    # WM 的 sh1/sh2 指减温器站 (与 E0 输出原样同序), 非过热器束; 14/14 全有 tag, 无缺失
    ("sh1_inlet_temp", "一级减温器入口温度", "degC, 原单位", "HIGH",
     "E0 OUTPUTS[0] + WM 锚点 h0=obs[0] (transition.py L261); 物理位置=低过出口/一级减温器前"),
    ("sh1_outlet_temp", "一级减温器出口温度", "degC, 原单位", "HIGH",
     "E0 OUTPUTS[1]; 物理位置=屏过入口(一级喷水控屏过出口, 见汽水系统结构)"),
    ("sh2_inlet_temp", "二级减温器入口温度", "degC, 原单位", "HIGH",
     "E0 OUTPUTS[2] + WM 锚点 h1=obs[2] (L262); 物理位置=屏过出口/二级减温器前"),
    ("sh2_outlet_temp", "二级减温器出口温度", "degC, 原单位", "HIGH",
     "E0 OUTPUTS[3]; 物理位置=高过入口(二级喷水控末过出口)"),
    ("final_outlet_temp", "末级过热器出口汽温", "degC, 原单位", "HIGH",
     "E0 OUTPUTS[4] + WM 锚点 h2=obs[4] (L263) + RM3 TARGET_COLUMN"),
]
# 矩阵文档命名对齐问题 (供 Codex 复核, 非数据缺口):
# 矩阵 §1.1 写"5个观测元素=屏过入/出口、高过入/出口、末过出口汽温";
# 实际蒸汽路径 低过->一减->屏过->二减->高过, 故"屏过入"=一减出(sh1_outlet),
# "屏过出"=二减入(sh2_inlet), "高过入"=二减出(sh2_outlet), "高过出"=末过出(final)。
# 代码层 WM 注册表 = E0 5输出原样 (transition.py 锚点同位置), 5个tag全部存在, 无 MISSING。

mapping_rows = []
for ch, tag, conv, conf, note in MAPPING:
    mapping_rows.append({
        "registry_channel": ch, "dcs_tag": tag, "unit_conversion": conv,
        "confidence": conf, "note": note,
        "closure": "CLOSED" if conf == "HIGH" else "MIXED" if conf in ("MEDIUM", "MISSING") else conf,
    })

# ---------------- 2. 质量门（A03+B03, fail-closed 判定只读） ----------------
NEEDED = ["date", "主蒸汽流量", "未校正总煤量", "燃料主控输出", "分离器出口压力", "分离器出口温度",
          "省煤器入口给水温度", "省煤器出口给水温度", "末级过热器出口压力", "减温水总流量",
          "一级减温调节门阀位", "二级减温调节门阀位",
          "一级减温器入口温度", "一级减温器出口温度", "二级减温器入口温度", "二级减温器出口温度",
          "末级过热器出口汽温"]

def quality_gates(path, side):
    df = pd.read_csv(path, usecols=NEEDED)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    n = len(df)
    # 时间戳连续性: 相邻 diff == 10s 占比
    dt = np.diff(df["date"].values.astype("int64"))  # ns
    ok_10s = float(np.mean(dt == 10_000_000_000))
    gap_rate = 1.0 - ok_10s
    # 饱和/卡死段: 连续 >30min (180 步) 零方差 —— 用滚动窗差分近似: 每列检测 180 步窗内全同值
    stuck_cols = {}
    for c in NEEDED[1:]:
        v = df[c].to_numpy(dtype=np.float32)
        nan_frac = float(np.isnan(v).mean())
        if nan_frac > 0:
            v = np.nan_to_num(v, nan=np.nanmedian(v) if not np.all(np.isnan(v)) else 0.0)
        # 180 步窗内 min==max 检测 (向量化: 步长为180的窗口极差)
        if n >= 180:
            rng = np.max(v, axis=0) if v.ndim == 0 else None
            # 分块计算避免 707k^2: 用 stride 采样窗口极差
            idx = np.arange(0, n - 180, 90)  # 每 90 步采样一个窗
            windows = np.stack([v[i:i + 180] for i in idx[:4000]])  # 最多 4000 窗
            stuck_frac = float(np.mean(np.ptp(windows, axis=1) == 0))
        else:
            stuck_frac = 0.0
        stuck_cols[c] = {"stuck_30min_frac": round(stuck_frac, 4), "nan_frac": round(nan_frac, 4)}
    # 阀位非饱和覆盖: v1/v2 不同时贴 0/100 的占比
    v1 = df["一级减温调节门阀位"].to_numpy(dtype=np.float32)
    v2 = df["二级减温调节门阀位"].to_numpy(dtype=np.float32)
    both_sat = ((v1 <= 0.5) | (v1 >= 99.5)) & ((v2 <= 0.5) | (v2 >= 99.5))
    non_sat_frac = float(np.mean(~both_sat))
    # 连续时长
    span_hours = (df["date"].iloc[-1] - df["date"].iloc[0]).total_seconds() / 3600
    # 数据 SHA
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    gates = {
        "side": side, "n_rows": n,
        "start": str(df["date"].iloc[0]), "end": str(df["date"].iloc[-1]),
        "span_hours": round(span_hours, 1), "span_days_equiv": round(span_hours / 24, 1),
        "sample_period_10s_fraction": ok_10s,
        "gate_gap_rate_lt_1pct": bool(gap_rate < 0.01),
        "gate_stuck_all_channels_lt_5pct": all(s["stuck_30min_frac"] < 0.05 for s in stuck_cols.values()),
        "gate_valve_non_saturation_ge_60pct": bool(non_sat_frac >= 0.60),
        "gate_duration_ge_30days": bool(span_hours / 24 >= 30),
        "valve_non_saturation_fraction": round(non_sat_frac, 4),
        "stuck_by_channel": stuck_cols,
        "file_sha256": sha,
    }
    return gates

gates = [quality_gates(A, "A"), quality_gates(B, "B")]

with open(f"{OUT}/mapping_audit.json", "w") as f:
    json.dump({"mapping": mapping_rows,
               "mixed_channels": [r["registry_channel"] for r in mapping_rows if r["closure"] != "CLOSED"],
               "note": "执行侧如实标注, MIXED 由本地审计最终裁定"}, f, indent=2, ensure_ascii=False)
with open(f"{OUT}/quality_gates.json", "w") as f:
    json.dump({"gates": gates}, f, indent=2, ensure_ascii=False)

for g in gates:
    print(f"[{g['side']}] rows={g['n_rows']} span={g['span_days_equiv']}d gap_rate={1-g['sample_period_10s_fraction']:.4f} "
          f"non_sat={g['valve_non_saturation_fraction']:.3f} "
          f"gates: gap<1%={g['gate_gap_rate_lt_1pct']} stuck<5%={g['gate_stuck_all_channels_lt_5pct']} "
          f"valve>=60%={g['gate_valve_non_saturation_ge_60pct']} dur>=30d={g['gate_duration_ge_30days']}")
print("\nMIXED channels:", [r["registry_channel"] for r in mapping_rows if r["closure"] != "CLOSED"])
