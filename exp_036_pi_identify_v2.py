#!/usr/bin/env python3
"""
exp_036_pi_identify_v2.py — 改进 PI 模拟器辨识 (联合优化前置)
==============================================================
之前: 增量PI (Δe, e) → R²=0.06 太差
改进: 加协变量 (负荷变化率/蒸汽流量/阀位自回归/一级阀位) + MLP
目标: R² 能否到 0.5+ 决定联合优化(SP→PI→阀位→WM)离线可行性
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from experiments.phase1_dynamics.exp_025_unified_benchmark import train_raw, test_raw, SP_IDX, VALVE_IDX, TARGET_IDX

# 找负荷/蒸汽流量列 (csv idx 1-40 中与阀位强相关的)
import pandas as pd
df = pd.read_csv(cfg.DATA_DIR + '/' + cfg.TRAIN_FILE, nrows=20000)
cols = list(df.columns)
print(f"列: {[f'{i}:{c}' for i, c in enumerate(cols[:8])]}...")

# 二级阀 (主汽温 PI 执行机构)
a2 = test_raw[:, VALVE_IDX[1]]
a1 = test_raw[:, VALVE_IDX[0]]
sp = test_raw[:, SP_IDX]
pv = test_raw[:, TARGET_IDX]
e = sp - pv
de = np.diff(e)
da2 = np.diff(a2)
# 协变量: 负荷 (col 1), 蒸汽流量, 温度变化率
load = test_raw[:, 0]       # 机组负荷 (数据第0列)
dload = np.diff(load)
dt_pv = np.diff(pv)

# 特征矩阵 (对齐 da2[t+1] ↔ 特征[t])
X = np.stack([
    de[1:],                    # Δe(t+1): 偏差变化
    e[1:-1],                   # e(t): 偏差
    dload[1:],                 # 负荷变化率
    dt_pv[1:],                 # 温度变化率
    a2[1:-1],                  # 阀位自回归
    a1[1:-1],                  # 一级阀位 (耦合)
    np.ones(len(de)-1),        # 常数
], 1)
y = da2[1:]
mask = np.abs(y) < 30
X, y = X[mask], y[mask]

# 1) 线性 (加协变量)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ beta
r2_lin = 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)
print(f"\n线性+协变量: R² = {r2_lin:.4f}")
print(f"  系数: de={beta[0]:.4f} e={beta[1]:.4f} dload={beta[2]:.4f} dt_pv={beta[3]:.4f} a2={beta[4]:.4f} a1={beta[5]:.4f}")

# 2) MLP (torch)
import torch, torch.nn as nn
X_t = torch.FloatTensor(X); y_t = torch.FloatTensor(y)
n_train = int(0.8 * len(X_t))
Xtr, Xte, ytr, yte = X_t[:n_train], X_t[n_train:], y_t[:n_train], y_t[n_train:]
mlp = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
for ep in range(30):
    opt.zero_grad()
    loss = nn.functional.mse_loss(mlp(Xtr).squeeze(1), ytr)
    loss.backward(); opt.step()
with torch.no_grad():
    yhat_te = mlp(Xte).squeeze(1).numpy()
r2_mlp = 1 - np.sum((yte.numpy() - yhat_te)**2) / np.sum((yte.numpy() - np.mean(yte.numpy()))**2)
print(f"MLP (30ep, 80/20分): R² = {r2_mlp:.4f}")

# 3) 滚动仿真对比 (MLP 版 vs 真实)
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
for ep in range(30):
    opt.zero_grad()
    loss = nn.functional.mse_loss(mlp(X_t).squeeze(1), y_t)
    loss.backward(); opt.step()

def roll_mlp(sp_seq, pv_seq, load_seq, a1_seq, a0):
    a = np.zeros(len(sp_seq)); a[0] = a0
    e_prev = sp_seq[0] - pv_seq[0]
    for t in range(1, len(sp_seq)):
        e_t = sp_seq[t] - pv_seq[t]
        feat = np.array([e_t - e_prev, e_prev, load_seq[t]-load_seq[t-1], pv_seq[t]-pv_seq[t-1],
                         a[t-1], a1_seq[t-1], 1.0])
        with torch.no_grad():
            a[t] = a[t-1] + mlp(torch.FloatTensor(feat)).item()
        a[t] = np.clip(a[t], 0, 100)
        e_prev = e_t
    return a

np.random.seed(1)
errs = []
for _ in range(5):
    i = np.random.randint(5000, len(test_raw) - 600)
    a_sim = roll_mlp(sp[i:i+500], pv[i:i+500], load[i:i+500], a1[i:i+500], a2[i])
    errs.append(np.abs(a_sim - a2[i:i+500]).mean())
print(f"\nMLP 滚动仿真 MAE (5段×500步): {np.mean(errs):.3f} 阀位单位 (相对 std {a2.std():.2f}, {np.mean(errs)/a2.std()*100:.0f}%)")

json.dump({'r2_lin': float(r2_lin), 'r2_mlp': float(r2_mlp), 'roll_mae': float(np.mean(errs))},
          open("results/exp_036_pi_v2.json", 'w'), indent=2)
print("\nSaved: results/exp_036_pi_v2.json")
