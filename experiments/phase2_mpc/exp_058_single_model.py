#!/usr/bin/env python3
"""
exp_058_single_model.py — 单模型 (M7 正式版, H_OUT=18) 1-18 步完整曲线
=========================================================================
用户需求: 不用 4 个训练长度模型, 用 18 步一个模型测 1-18 步不同 horizon
1. direct: 同 500 窗口, 1-18 步逐步 MAE (exp_057 同款, 但用正式 M7)
2. 闭环: 3 起点集 (seed 42/7/13 × 50 轨迹) 逐步 MAE 1-18 步 (同 exp_056 协议)
3. 与 exp_048 训练的 h18 模型对比 (验证正式 M7 与专用 180s 模型一致性)
用法: python exp_058_single_model.py [--smoke]
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

SMOKE = '--smoke' in sys.argv
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRACKS = 5 if SMOKE else 50
N_WIN = 100 if SMOKE else 500
N_STEPS = 120
SEEDS = [42] if SMOKE else [42, 7, 13]
OUT = 'results/exp_058_single_model'
os.makedirs(OUT, exist_ok=True)
W = cfg.WINDOW_SIZE
N = len(test_raw)
H = 18

def load_m7(ck_path):
    E.H_OUT = H
    model = build_model('M7').to(DEVICE).eval()
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    return model

wm = load_m7('results/exp_025_M7/checkpoints/best_model.pth')
print("M7 正式版加载完成")

# ============ 1. direct: 同 500 窗口 1-18 步 ============
np.random.seed(42)
idxs = np.random.randint(0, N - W - H, N_WIN)
direct = np.zeros((N_WIN, H))
for i, idx in enumerate(idxs):
    xh = torch.FloatTensor(test_raw[idx:idx + W]).unsqueeze(0).to(DEVICE)
    af = torch.FloatTensor(test_raw[idx + W:idx + W + H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu, _ = wm(xh, af)
    tgt = test_raw[idx + W:idx + W + H, TARGET_IDX]
    direct[i] = np.abs(mu[0].cpu().numpy() - tgt)
print("direct 1-18 步 MAE:", ' '.join(f'{direct[:,k].mean():.3f}' for k in range(18)))

# ============ 2. 闭环: 3 起点集逐步 ============
def sim_rollout_errors(track_idx, n_steps):
    win = torch.FloatTensor(test_raw[track_idx:track_idx + W]).unsqueeze(0).to(DEVICE)
    errs = []
    for k in range(n_steps):
        gi = track_idx + W + k
        a_real = torch.FloatTensor(test_raw[gi:gi + H, VALVE_IDX]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = wm(win, a_real)
            y1 = mu[0, 0].item()
        errs.append(abs(y1 - test_raw[gi, TARGET_IDX]))
        next_row = torch.FloatTensor(test_raw[gi]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        next_row[0, 0, TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
    return np.array(errs)

per_step = np.zeros((len(SEEDS), N_STEPS))
for si, seed in enumerate(SEEDS):
    np.random.seed(seed)
    starts = np.random.choice(range(N - W - H - N_STEPS - 50), N_TRACKS, replace=False)
    err_mat = np.zeros((N_TRACKS, N_STEPS))
    for i, s in enumerate(starts):
        err_mat[i] = sim_rollout_errors(s, N_STEPS)
    per_step[si] = err_mat.mean(0)
    print(f"  seed {seed}: 1200s MAE {err_mat.mean():.3f}")

# ============ 3. 输出表 ============
print("\n=== M7 正式版: 1-18 步 (direct 单次 / 闭环 3-seed 均值±std) ===")
print(f"{'step':>4} | {'direct':>8} | {'闭环':>14} | {'exp_048h18(闭环)':>16}")
exp056 = json.load(open('results/exp_056_horizon_curves/curves.json'))
h18_closed = np.mean(exp056['fixed']['h180']['per_step'], 0)
for k in range(18):
    print(f"{k+1:>4} | {direct[:,k].mean():>8.3f} | "
          f"{per_step[:,k].mean():>7.3f}±{per_step[:,k].std():<5.3f} | {h18_closed[k]:>16.3f}")

out = {'direct_per_step': direct.mean(0).tolist(),
       'closed_per_step': per_step.mean(0).tolist(),
       'closed_per_step_std': per_step.std(0).tolist(),
       'closed_1200s': per_step.mean(1).tolist()}
json.dump(out, open(f'{OUT}/m7_curves.json', 'w'), indent=2)
print(f"\nSaved: {OUT}/m7_curves.json")
