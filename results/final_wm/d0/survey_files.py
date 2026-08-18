#!/usr/bin/env python3
"""D0 数据审计 — 步骤1: 候选文件盘点（列/行数/日期范围/关键通道单位）"""
import pandas as pd
import numpy as np

BASE = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data"
FILES = [
    "mainT/A侧主汽温全数据_cleaned_10s.csv",
    "mainT/B侧主汽温全数据_cleaned_10s.csv",
    "mainT/A侧主汽温全数据03_cleaned_10s.csv",
    "mainT/B侧主汽温全数据03_cleaned_10s.csv",
    "reheat/A侧再热汽温全数据_cleaned_10s.csv",
    "reheat/B侧再热汽温全数据_cleaned_10s.csv",
]

for f in FILES:
    try:
        df_head = pd.read_csv(f"{BASE}/{f}", nrows=5)
        cols = list(df_head.columns)
        n_rows = sum(1 for _ in open(f"{BASE}/{f}")) - 1
        d0 = str(df_head["date"].iloc[0])
        d1 = str(df_head["date"].iloc[-1])
        print(f"\n{f}")
        print(f"  rows={n_rows} ncols={len(cols)} date_head={d0} .. {d1}")
        print(f"  cols={cols}")
    except Exception as e:
        print(f"\n{f} ERROR {e}")
