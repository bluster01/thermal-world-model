#!/usr/bin/env python3
"""
exp_056_horizon_curves.py — 0-18步完整逐horizon曲线 × 3起点集平均
====================================================================
需求: 每个 horizon 步 (1..18步=10s..180s) 的闭环预测 MAE 完整曲线,
      公平性: 3 个不同起点集 (seed 42/7/13) 各跑 50 轨迹, 取平均±std

两个协议都做:
  fixed: 公平固定集 range(N−W−18−120−50)   ← exp_054/055 修正版
  hdep : h依赖集 range(N−W−h−120−50)       ← exp_048 缺陷版 (验证 2.510 是抽样运气还是系统性)

输出:
  1. 公平协议逐步表: 1-18 步 + 20/30/60/120 步, 3-seed 均值±std
  2. 两协议总 MAE per seed (验证起点集效应)
  3. 合并 150 轨迹配对 Wilcoxon (90/120/180s 两两)
用法: python exp_056_horizon_curves.py [--smoke]
"""
import os, sys, json
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import config as cfg
import experiments.phase1_dynamics.exp_025_unified_benchmark as E
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX)

SMOKE = '--smoke' in sys.argv
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CK_DIR = 'results/exp_048_horizon/checkpoints'
N_TRACKS = 5 if SMOKE else 50
N_STEPS = 120
SEEDS = [42] if SMOKE else [42, 7, 13]
OUT = 'results/exp_056_horizon_curves'
os.makedirs(OUT, exist_ok=True)
W = cfg.WINDOW_SIZE
N = len(test_raw)
H_OUT_FIX = 18  # 公平起点集上界: 固定常量 (不能读 E.H_OUT — load_wm(h) 会改成 h)

def load_wm(h):
    E.H_OUT = h
    model = build_model('M7').to(DEVICE).eval()
    ck = torch.load(f'{CK_DIR}/h{h:02d}.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    return model

def sim_rollout_errors(wm, track_idx, h, n_steps):
    """返回 per-step 绝对误差 [n_steps] (预测温度 vs 真实温度)"""
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

def get_starts(seed, h=None, fixed=True):
    np.random.seed(seed)
    if fixed:
        hi = N - W - H_OUT_FIX - N_STEPS - 50
    else:
        hi = N - W - h - N_STEPS - 50
    return np.random.choice(range(hi), N_TRACKS, replace=False)

# ============ 公平协议: 3 seeds × 4 模型 逐步扫描 (同时收集 per-track) ============
print("=== 公平固定集: 3 seeds × 4 模型 ===")
fixed_data = {}   # {'h60': {'per_step': [3,120], 'total': [3], 'per_track': [150]}}
for h in [6, 9, 12, 18]:
    wm = load_wm(h)
    per_step = np.zeros((len(SEEDS), N_STEPS))
    total = []
    per_track_all = []
    for si, seed in enumerate(SEEDS):
        starts = get_starts(seed, fixed=True)
        err_mat = np.zeros((N_TRACKS, N_STEPS))
        for i, s in enumerate(starts):
            err_mat[i] = sim_rollout_errors(wm, s, h, N_STEPS)
        per_step[si] = err_mat.mean(0)
        total.append(float(err_mat.mean()))
        per_track_all.extend(err_mat.mean(1).tolist())
    fixed_data[f'h{h*10}'] = {'per_step': per_step, 'total': total,
                              'per_track_1200s': per_track_all}
    print(f"  H={h*10:>3}s: 1200s MAE per-seed {['%.3f'%t for t in total]} → 均值 {np.mean(total):.3f} ± {np.std(total):.3f}")

print("\n=== 配对 Wilcoxon (公平集, 合并150轨迹) ===")
tracks = {k: np.array(v['per_track_1200s']) for k, v in fixed_data.items()}
for a, b in [('h90', 'h120'), ('h120', 'h180'), ('h90', 'h180'), ('h60', 'h180'), ('h60', 'h90')]:
    p = stats.wilcoxon(tracks[a], tracks[b]).pvalue
    d = tracks[b].mean() - tracks[a].mean()
    print(f"  {a} vs {b}: Δ={d:+.3f}  p={p:.4f} {'**' if p<0.05 else ''}")

# ============ h 依赖协议 (exp_048 版): 3 seeds 验证起点集效应 ============
print("\n=== h 依赖集 (exp_048 版): 3 seeds 验证 2.510 是否系统性 ===")
hdep_data = {}
for h in [6, 9, 12, 18]:
    wm = load_wm(h)
    per_step = np.zeros((len(SEEDS), N_STEPS))
    total = []
    for si, seed in enumerate(SEEDS):
        starts = get_starts(seed, h=h, fixed=False)
        err_mat = np.zeros((N_TRACKS, N_STEPS))
        for i, s in enumerate(starts):
            err_mat[i] = sim_rollout_errors(wm, s, h, N_STEPS)
        per_step[si] = err_mat.mean(0)
        total.append(float(err_mat.mean()))
    hdep_data[f'h{h*10}'] = {'per_step': per_step, 'total': total}
    print(f"  H={h*10:>3}s: 1200s MAE per-seed {['%.3f'%t for t in total]} → 均值 {np.mean(total):.3f} ± {np.std(total):.3f}")

# ============ 完整表格: 公平集 1-18 步 ============
print("\n=== 完整表格: 公平集 逐步 MAE (°C), 3-seed 均值±std ===")
print(f"{'step':>4} | " + ' | '.join([f"{'H='+str(h*10)+'s':>14}" for h in [6,9,12,18]]))
for k in range(18):
    row = [f"{k+1:>4} |"]
    for h in [6, 9, 12, 18]:
        ps = fixed_data[f'h{h*10}']['per_step'][:, k]
        row.append(f"{ps.mean():>7.3f}±{ps.std():<6.3f}")
    print(' '.join(row))
print("--- 聚合点 ---")
for k in [19, 29, 59, 119]:
    row = [f"{k+1:>4} |"]
    for h in [6, 9, 12, 18]:
        ps = fixed_data[f'h{h*10}']['per_step'][:, k]
        row.append(f"{ps.mean():>7.3f}±{ps.std():<6.3f}")
    print(' '.join(row))

# 存 JSON (per_step 转 list)
def tojson(d):
    return {k: {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv) for kk, vv in v.items()}
            for k, v in d.items()}
json.dump({'fixed': tojson(fixed_data), 'hdep': tojson(hdep_data), 'seeds': SEEDS},
          open(f'{OUT}/curves.json', 'w'), indent=2)
print(f"\nSaved: {OUT}/curves.json")
