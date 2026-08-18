#!/usr/bin/env python3
"""D0 数据审计 — 步骤3: canonical record 构建（矩阵 §1.3）

- A03+B03 按 date 内连接 (10s 对齐, 无缺口掩码需求)
- 通道按注册表序打包: boundary×7 + action×4(v1A,v2A,v1B,v2B) + observation×10(A5,B5)
- 单位换算: steam_flow t/h->kg/s (/3.6); 阀位 %->fraction (/100)
- 时间顺序切分提案: train 80% / validation 20% (test 锁定, 本记录不含 test)
- 交叉映射 (用户确认+RM3 先例): valve_A 管控 obsB, valve_B 管控 obsA — 记录于 manifest
- npz 不入 git (矩阵 §1: 数据不入仓); SHA 与全部元数据写入 manifest 入库
"""
import hashlib
import json
import numpy as np
import pandas as pd

BASE = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT"
OUT = "/home/bluster/projectA/thermal-world-model/results/final_wm/d0"
A, B = f"{BASE}/A侧主汽温全数据03_cleaned_10s.csv", f"{BASE}/B侧主汽温全数据03_cleaned_10s.csv"

COLS = ["date", "主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
        "省煤器出口给水温度", "末级过热器出口压力", "减温水总流量",
        "一级减温调节门阀位", "二级减温调节门阀位",
        "一级减温器入口温度", "一级减温器出口温度", "二级减温器入口温度",
        "二级减温器出口温度", "末级过热器出口汽温"]

a = pd.read_csv(A, usecols=COLS)
b = pd.read_csv(B, usecols=COLS)
for df in (a, b):
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    df.rename(columns={"date": "_date"}, inplace=True)

m = pd.merge(a, b, on="_date", suffixes=("_A", "_B"), how="inner")
n = len(m)
print(f"merged rows={n} (A={len(a)}, B={len(b)})")

date_ns = m["_date"].values.astype("int64")
boundary = np.stack([
    m["主蒸汽流量_A"].values / 3.6,            # steam_flow kg/s
    m["未校正总煤量_A"].values,                 # coal_command t/h
    m["分离器出口压力_A"].values,               # MPa
    m["分离器出口温度_A"].values,               # degC
    m["省煤器出口给水温度_A"].values,           # degC
    m["末级过热器出口压力_A"].values,           # MPa
    m["减温水总流量_A"].values,                 # t/h oracle
], axis=1).astype(np.float32)
action = np.stack([
    np.clip(m["一级减温调节门阀位_A"].values, 0, None) / 100.0,
    np.clip(m["二级减温调节门阀位_A"].values, 0, None) / 100.0,
    np.clip(m["一级减温调节门阀位_B"].values, 0, None) / 100.0,
    np.clip(m["二级减温调节门阀位_B"].values, 0, None) / 100.0,
], axis=1).astype(np.float32)
obsA = np.stack([
    m["一级减温器入口温度_A"].values, m["一级减温器出口温度_A"].values,
    m["二级减温器入口温度_A"].values, m["二级减温器出口温度_A"].values,
    m["末级过热器出口汽温_A"].values,
], axis=1).astype(np.float32)
obsB = np.stack([
    m["一级减温器入口温度_B"].values, m["一级减温器出口温度_B"].values,
    m["二级减温器入口温度_B"].values, m["二级减温器出口温度_B"].values,
    m["末级过热器出口汽温_B"].values,
], axis=1).astype(np.float32)

# 时间顺序切分提案: train 80% / validation 20%; test 锁定不参与
split_train = int(n * 0.80)
arrays = {"date_ns": date_ns, "boundary": boundary, "action": action,
          "obsA": obsA, "obsB": obsB, "split_train": np.array([split_train], dtype=np.int64)}
np.savez_compressed(f"{OUT}/canonical_record.npz", **arrays)

with open(f"{OUT}/canonical_record.npz", "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

manifest = {
    "canonical_record": "canonical_record.npz (本地, 不入git; 矩阵§1: 数据不入仓)",
    "sha256": sha, "n_rows": n, "n_channels_boundary": 7, "n_channels_action": 4,
    "n_channels_observation": 10, "sample_period_s": 10,
    "start": str(m["_date"].iloc[0]), "end": str(m["_date"].iloc[-1]),
    "split": {"train": [0, split_train], "validation": [split_train, n],
              "note": "时间顺序切分提案 80/20; test 锁定不参与; 切分点由本地审计最终冻结"},
    "cross_side_mapping": {
        "valve1_A / valve2_A": "管控 B 侧汽温 (obsB)", "valve1_B / valve2_B": "管控 A 侧汽温 (obsA)",
        "source": "用户2026-08-09确认 + RM3 交叉 cache 先例"},
    "registry_order": {
        "boundary": ["steam_flow", "coal_command", "separator_pressure", "separator_temperature",
                     "feedwater_temperature", "outlet_pressure", "spray_flow_total"],
        "action": ["valve1_A", "valve2_A", "valve1_B", "valve2_B"],
        "observation_A": ["sh1_inlet(缺失见mapping)", "sh1_outlet", "sh2_inlet", "sh2_outlet", "final_outlet"],
        "observation_B": ["同上B侧"]},
    "gap_mask": "无缺口 (gap_rate=0.0000, 见 quality_gates.json)",
    "source_files": {"A": A, "B": B},
    "units": {"steam_flow": "kg/s (=t/h /3.6)", "action": "fraction 0-1 (=%/100)", "其余": "原单位"},
}
with open(f"{OUT}/canonical_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"canonical_record.npz sha256={sha[:16]}... split_train={split_train}/{n}")
