#!/usr/bin/env python3
"""exp_047_mstep_scan.py — M_STEP 敏感性扫描 (Phase 2.5 任务4a)
m ∈ {1,3,6,12,18}: 主评测 10 条, 观察 RMSE/std/TV 随执行周期变化
验证: 执行周期须匹配过程时标 (60-120s), 过短→动作效应丢失, 过长→反应迟钝
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
from exp_027_dwm_mpc import load_wm, W, H_OUT, DEVICE, test_raw, VALVE_IDX, SP_IDX, TARGET_IDX
import exp_027_dwm_mpc as M
sys.argv = _argv

wm = load_wm()
M.SP_TRAJ = 0
N = len(test_raw)
np.random.seed(42)
starts = np.random.choice(range(N - W - H_OUT - 120), 10, replace=False)

results = {}
for m in [1, 3, 6, 12, 18]:
    M.M_STEP = m
    rmses, stds, tvs = [], [], []
    for s in starts:
        mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad', n_steps=120)
        rmses.append(float(np.sqrt(np.mean((np.array(mpc_t) - np.array(tset))**2))))
        stds.append(float(np.std(mpc_t)))
        tvs.append(float(np.abs(np.diff(np.array(mpc_a)[:, 1])).sum()))
    results[m] = {'rmse': float(np.mean(rmses)), 'std': float(np.mean(stds)), 'tv': float(np.mean(tvs))}
    print(f"M_STEP={m:>2}: RMSE {results[m]['rmse']:.3f} | std {results[m]['std']:.3f} | TV {results[m]['tv']:.1f}")

json.dump(results, open("results/exp_047_mstep.json", 'w'), indent=2, default=float)
print("\nSaved: results/exp_047_mstep.json")
