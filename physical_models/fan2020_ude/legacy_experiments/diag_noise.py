import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CSV = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据_cleaned_10s.csv"
WIN_START, WIN = 70686, 50000
TRAIN_N, VAL_N = 30000, 10000

# 1) 主汽温 10s 差分 std（噪声地板估计）
df = pd.read_csv(CSV, usecols=["末级过热器出口汽温", "分离器出口压力", "一级减温调节门阀位"],
                 dtype=np.float32).iloc[WIN_START: WIN_START + WIN]
seg = df.iloc[TRAIN_N + VAL_N: TRAIN_N + VAL_N + 1800]
d = seg["末级过热器出口汽温"].diff().dropna()
print("test 段主汽温 10s 差分 std:", round(float(d.std()), 2), "°C  | mean:", round(float(seg['末级过热器出口汽温'].mean()), 1),
      "| range:", float(seg.min()['末级过热器出口汽温']), "-", float(seg.max()['末级过热器出口汽温']))
print("test 段 pm range:", float(seg['分离器出口压力'].min()), "-", float(seg['分离器出口压力'].max()), "MPa")
print("test 段 v1 mean:", round(float(seg['一级减温调节门阀位'].mean()), 2), "(一级喷水是否基本关闭)")

# 2) e0 seed0 rollout 逐对违例
d = np.load(os.path.join(OUT, "rollout_e0_seed0.npz"))
p = d["preds"]
PAIRS = {"sh1_out<sh2_in": (1, 2), "sh2_out<main": (3, 4), "sh2_in<sh2_out": (2, 3),
         "sh1_out<sh1_in": (1, 0), "sh1_in<sh2_in": (0, 2)}
print("\n逐对违例占比（pred，>= 计违例）:")
for name, (lo, hi) in PAIRS.items():
    print(f"  {name}: {(p[:, lo] >= p[:, hi]).mean()*100:.1f}%")

# 3) 沿程预测 vs 真实 均值
print("\n沿程温度 预测均值 vs 真实均值:")
cols = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "main"]
for i, c in enumerate(cols):
    print(f"  {c}: pred={p[:, i].mean():.1f} truth={d['truths'][:, i].mean():.1f}")

# 4) 训练段 vs 测试段 主汽温分布
tr = df["末级过热器出口汽温"].iloc[:TRAIN_N]
print(f"\n训练段主汽温: mean={tr.mean():.1f} std={tr.std():.2f} | 测试段: mean={seg['末级过热器出口汽温'].mean():.1f} std={seg['末级过热器出口汽温'].std():.2f}")
