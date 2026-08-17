#!/usr/bin/env python3
"""06: 合并 mainT(A/B) + reheat 关键列 → merged_aug.csv (对齐 A侧 行序)"""
import pandas as pd

DATA = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data"
OUT = "/home/bluster/.hermes/workspace/adhoc_pinn_features/data"
import os
os.makedirs(OUT, exist_ok=True)

A = pd.read_csv(f"{DATA}/mainT/A侧主汽温全数据_cleaned_10s.csv")
B = pd.read_csv(f"{DATA}/mainT/B侧主汽温全数据_cleaned_10s.csv")
RH = pd.read_csv(f"{DATA}/reheat/A侧再热汽温全数据_cleaned_10s.csv")

# A侧补充控制器内部/设定值列 (mainT 文件里有但之前没用)
A_extra = A[["date", "二级减温调节阀设定", "二级减温中间设定值", "一级减温副调设定值", "一级减温温度设定偏值"]]
# B侧: 主汽温 + 两侧喷水指令
B_extra = B[["date", "末级过热器出口汽温", "一级减温喷水调节门指令", "二级减温喷水调节门指令"]]
B_extra.columns = ["date", "末级过热器出口汽温_B", "一级减温喷水调节门指令_B", "二级减温喷水调节门指令_B"]
# 再热侧
RH_extra = RH[["date", "再热出口汽温", "再热器减温水总流量", "高压缸排汽至再热器温度",
               "再热器一级减温入口汽温", "立式低温再热器入口烟气温度", "水平低温再热器入口烟气温度"]]

m = A[["date"]].merge(A_extra, on="date").merge(B_extra, on="date").merge(RH_extra, on="date")
print(f"merged rows: {len(m)} (A {len(A)}, B {len(B)}, RH {len(RH)})")
print("NaN counts:")
print(m.isna().sum()[m.isna().sum() > 0])
m = m.ffill().bfill()
m.to_csv(os.path.join(OUT, "merged_aug.csv"), index=False)
print("saved", os.path.join(OUT, "merged_aug.csv"), m.shape)
