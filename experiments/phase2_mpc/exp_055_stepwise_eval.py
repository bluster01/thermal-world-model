#!/usr/bin/env python3
"""
exp_055_stepwise_eval.py — 评估协议审核 + 逐步扫描 0-18 步
==============================================================
审核目标:
  1. 逐步 (per-step) 扫描 4 个 checkpoint (60/90/120/180s) 的闭环预测误差 1..18 步
  2. 复现检查: h=6 模型在 exp_048 的 h 依赖起点集 (B) 上是否重现 2.510
     → 验证 exp_048 vs exp_054 数值差异 (2.510 vs 1.284) 是否纯起点集造成
  3. 完整表格: 公平起点集 (A) 下逐步 MAE 1-18 步 + 20/30/60/120 步

协议 (与 exp_048/054 一致): 真实阀位动作 + 温度预测回填, 其余列真实, 50轨迹 seed42
起点集 A (公平): range(N-W-H_OUT-120-50)  ← exp_054 修正版
起点集 B (h依赖): range(N-W-h-120-50)     ← exp_048 缺陷版 (仅复现检查用)
用法: python exp_055_stepwise_eval.py
"""
import os, sys, json
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
N_TRACKS = 50
N_STEPS = 120
OUT = 'results/exp_055_stepwise'
os.makedirs(OUT, exist_ok=True)
W = cfg.WINDOW_SIZE
N = len(test_raw)
H_OUT_FIX = 18  # 公平起点集上界: 固定常量 (不能读 E.H_OUT — load_wm(h) 会把它改成 h!)

def load_wm(h):
    E.H_OUT = h
    model = build_model('M7').to(DEVICE).eval()
    ck = torch.load(f'{CK_DIR}/h{h:02d}.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    return model

def sim_rollout_errors(wm, track_idx, h, n_steps):
    """返回 per-step 绝对误差 [n_steps] (预测温度 vs 真实温度, 逐步记录)"""
    win = torch.FloatTensor(test_raw[track_idx:track_idx + W]).unsqueeze(0).to(DEVICE)
    errs = []
    for k in range(n_steps):
        gi = track_idx + W + k
        a_real = torch.FloatTensor(test_raw[gi:gi + h, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(win, a_real)
            y1 = mu[0, 0].item()
        real = test_raw[gi, TARGET_IDX]
        errs.append(abs(y1 - real))
        next_row = torch.FloatTensor(test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
    return np.array(errs)

def get_starts(h=None, seed=42, fixed=True):
    np.random.seed(seed)
    if fixed:
        hi = N - W - H_OUT_FIX - N_STEPS - 50  # 公平: 固定 18 (与模型 h 无关)
    else:
        hi = N - W - h - N_STEPS - 50        # h 依赖 (exp_048 缺陷版)
    return np.random.choice(range(hi), N_TRACKS, replace=False)

# ============ 1. 公平起点集 A: 逐步扫描 4 模型 ============
print("=== 起点集 A (公平固定): per-step 闭环 MAE ===")
results = {}
for h in [6, 9, 12, 18]:
    wm = load_wm(h)
    starts = get_starts(fixed=True)
    err_mat = np.zeros((N_TRACKS, N_STEPS))
    for i, s in enumerate(starts):
        err_mat[i] = sim_rollout_errors(wm, s, h, N_STEPS)
    results[f'h{h*10}'] = {'starts_A': starts.tolist(),
                           'per_step_mae': err_mat.mean(0).tolist(),
                           'per_step_std': err_mat.std(0).tolist(),
                           'rollout_1200s': float(err_mat.mean())}
    print(f"  H={h*10:>3}s: 1200s MAE {err_mat.mean():.3f}")

# ============ 2. 复现检查: h=6 在起点集 B (exp_048 缺陷版) ============
print("\n=== 复现检查: h=6 起点集 B (h依赖, exp_048 版) ===")
wm6 = load_wm(6)
startsB = get_starts(h=6, fixed=False)
errB = np.zeros((N_TRACKS, N_STEPS))
for i, s in enumerate(startsB):
    errB[i] = sim_rollout_errors(wm6, s, 6, N_STEPS)
print(f"  H=60s 起点集B: 1200s MAE {errB.mean():.3f}  (exp_048 报告 2.510)")
print(f"  H=60s 起点集A: 1200s MAE {results['h60']['rollout_1200s']:.3f}  (exp_054 报告 1.284)")
print(f"  → 差异 {'纯起点集造成' if abs(errB.mean()-2.510)<0.2 else '起点集不能完全解释, 需进一步查'}")

# ============ 3. 完整表格 ============
print("\n=== 完整表格: 公平起点集 A, 逐步 MAE (°C) ===")
print(f"{'step':>4} {'10s':>6} {'H=60s':>8} {'H=90s':>8} {'H=120s':>8} {'H=180s':>8}")
steps = list(range(0, 18)) + [19, 29, 59, 119]
for k in steps:
    label = f'{k+1}' if k < 18 else f'{k+1}*'
    row = [f'{label:>4}']
    for h in [6, 9, 12, 18]:
        row.append(f"{results[f'h{h*10}']['per_step_mae'][k]:>8.3f}")
    print(' '.join(row))
print("(* 聚合点: 20/30/60/120 步)")

json.dump(results, open(f'{OUT}/stepwise_A.json', 'w'), indent=2)
json.dump({'starts_B': startsB.tolist(), 'rollout_1200s': float(errB.mean()),
           'per_step_mae': errB.mean(0).tolist()}, open(f'{OUT}/repro_B_h60.json', 'w'), indent=2)
print(f"\nSaved: {OUT}/")
