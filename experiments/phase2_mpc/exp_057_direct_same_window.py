#!/usr/bin/env python3
"""验证: 同一固定窗口下, 4个不同H_OUT模型的 direct 预测逐步误差
排除闭环回填/起点差异 — 纯模型训练差异 (loss=H步平均β-NLL, 早停统一看第5步)
500 个固定窗口 (seed 42), direct 预测, 对比 1-6 步 (全部模型都有值) + 各模型全长
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import config as cfg
import experiments.phase1_dynamics.exp_025_unified_benchmark as E
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CK_DIR = 'results/exp_048_horizon/checkpoints'
W = cfg.WINDOW_SIZE
N = len(test_raw)
N_WIN = 500

def load_wm(h):
    E.H_OUT = h
    model = build_model('M7').to(DEVICE).eval()
    ck = torch.load(f'{CK_DIR}/h{h:02d}.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    return model

np.random.seed(42)
idxs = np.random.randint(0, N - W - 18, N_WIN)  # 同一批固定窗口

print("=== 同一 500 窗口 direct 预测: per-step MAE (°C) ===")
print(f"{'step':>4} | " + ' | '.join([f"{'H='+str(h*10)+'s':>9}" for h in [6, 9, 12, 18]]))
data = {}
for h in [6, 9, 12, 18]:
    wm = load_wm(h)
    errs = np.zeros((N_WIN, h))
    for i, idx in enumerate(idxs):
        xh = torch.FloatTensor(test_raw[idx:idx + W]).unsqueeze(0).to(DEVICE)
        af = torch.FloatTensor(test_raw[idx + W:idx + W + h, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(xh, af)
        tgt = test_raw[idx + W:idx + W + h, TARGET_IDX]
        errs[i] = np.abs(mu[0].cpu().numpy() - tgt)
    data[h] = errs
    print(f"  H={h*10:>3}s 加载完成")

for k in range(6):
    row = [f"{k+1:>4} |"]
    for h in [6, 9, 12, 18]:
        row.append(f"{data[h][:, k].mean():>9.3f}")
    print(' '.join(row))
print("--- 各模型全长 (超视野部分) ---")
for h in [9, 12, 18]:
    row = [f"{h*10:>3}s |"]
    for k in range(6, h):
        row.append(f"{data[h][:, k].mean():>6.3f}")
    print(' '.join(row))

# 配对检验: 同窗口第1步差异
from scipy import stats
print("\n=== 同窗口第 1 步配对 Wilcoxon ===")
for a, b in [(6, 9), (6, 18), (9, 18), (9, 12), (12, 18)]:
    p = stats.wilcoxon(data[a][:, 0], data[b][:, 0]).pvalue
    print(f"  H={a*10}s vs H={b*10}s: Δ={data[b][:,0].mean()-data[a][:,0].mean():+.4f}  p={p:.2e}")
