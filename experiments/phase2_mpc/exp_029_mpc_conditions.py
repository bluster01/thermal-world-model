#!/usr/bin/env python3
"""
exp_029_mpc_conditions.py — Phase 2b: MPC 敏感性复验 + 11 工况对比
==================================================================
1. MPC 动作敏感性: 在 MPC 规划的动作分布下, WM 对二级阀首步扰动的响应
   (验证 -0.41 敏感性在新动作分布下保持物理方向)
2. 11 工况 MPC vs PID: 复用 exp_019 工况分类, 每工况 RMSE/波动/TV
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from exp_027_dwm_mpc import (
    load_wm, simulate, metrics, W, H_OUT, DEVICE, test_raw, VALVE_IDX, TARGET_IDX,
    H_PLAN, ALPHA)

TCN_ROOT = '/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU'
sys.path.insert(0, TCN_ROOT)
sys.path.insert(0, os.path.join(TCN_ROOT, 'utils'))
from evaluate_by_condition import classify_conditions

csv_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE)
df = pd.read_csv(csv_path)
if 'date' in df.columns:
    df.set_index('date', inplace=True)
print("工况分类中...")
condition_labels, _, _ = classify_conditions(df)
n_total = len(test_raw) + int(len(pd.read_csv(csv_path)) * 0.15)
n_val_end = int(len(pd.read_csv(csv_path)) * 0.85)
test_start = n_val_end

# ===== 1. MPC 敏感性复验 =====
print("=" * 70)
print("1. MPC 动作敏感性复验 (WM 对二级阀首步扰动的响应)")
print("   (验证 MPC 动作分布下敏感性仍物理: 开阀应降温)")
wm = load_wm()
np.random.seed(7)
idxs = np.random.choice(range(len(test_raw) - W - H_OUT), 100, replace=False)
dT = {s: [] for s in [1, 3, 8, 12]}
for i in idxs:
    xh = torch.FloatTensor(test_raw[i:i+W]).unsqueeze(0).to(DEVICE)
    # 用 MPC 规划的典型动作 (真实阀位作为基线, 首步扰动 ±10)
    a_base = torch.FloatTensor(test_raw[i+W:i+W+H_OUT, VALVE_IDX]).unsqueeze(0).to(DEVICE)
    t_base, _ = wm(xh, a_base)
    for d in [10.0]:
        a_p = a_base.clone(); a_p[0, 0, 1] = torch.clamp(a_p[0, 0, 1] + d, 0, 100)  # 二级阀首步 +10
        t_p, _ = wm(xh, a_p)
        for s in [1, 3, 8, 12]:
            dT[s].append((t_p[0, s-1] - t_base[0, s-1]).item())
print(f"  二级阀首步+10 响应 (100样本):")
for s in [1, 3, 8, 12]:
    v = np.mean(dT[s])
    print(f"    t{s}: {v:+.4f} °C {'✅物理' if v < 0 else '❌反物理'}")

# ===== 2. 11 工况 MPC vs PID =====
print("=" * 70)
print("2. 11 工况 MPC vs PID (grad, H=10, α=0.5, 每工况最多 15 条轨迹)")
N = len(test_raw)
np.random.seed(42)
idxs = np.random.choice(range(N - W - H_OUT - 120), 300, replace=False)

def sample_condition(global_start):
    labels = condition_labels[global_start + W : global_start + W + 18]
    cnt = Counter(labels)
    non_steady = [lbl for lbl in cnt if lbl != 'steady']
    if non_steady:
        return max(non_steady, key=lambda x: cnt[x])
    return 'steady'

per_cond = {}
count = Counter()
for k, i in enumerate(idxs):
    cond = sample_condition(test_start + i)
    if count[cond] >= 15:
        continue
    count[cond] += 1
    mpc_t, pid_t, tset, mpc_a, pid_a = simulate(wm, i, 'grad', n_steps=120)
    m = metrics(mpc_t, pid_t, tset, mpc_a, pid_a)
    per_cond.setdefault(cond, []).append(m)
    if (k+1) % 50 == 0:
        print(f"  {k+1}/300")

print(f"\n{'工况':<12}{'n':>4}{'RMSE_mpc':>10}{'RMSE_pid':>10}{'↓%':>7}{'TV_mpc':>8}{'TV_pid':>8}")
COND_CN = {
    'steady': '稳态', 'wcr_mismatch': '水煤比失配', 'load_reversal': '负荷反转',
    'ramp_up': '升负荷', 'ramp_down': '降负荷', 'soot_blow': '锅炉吹灰',
    'aux_steam': '辅汽/抽汽', 'fast_ramp_up': '快速升负荷', 'fast_ramp_down': '快速降负荷',
    'spray_oscillation': '减温水波动', 'large_oscillation': '大幅波动',
}
rows = []
for cond, lst in sorted(per_cond.items(), key=lambda x: -len(x[1])):
    r = {k: float(np.mean([m[k] for m in lst])) for k in lst[0]}
    r['n'] = len(lst)
    rows.append({'condition': cond, **r})
    imp = (1 - r['rmse_mpc']/r['rmse_pid']) * 100
    print(f"{COND_CN.get(cond, cond):<12}{r['n']:>4}{r['rmse_mpc']:>10.3f}{r['rmse_pid']:>10.3f}"
          f"{imp:>6.1f}%{r['act_tv_mpc']:>8.3f}{r['act_tv_pid']:>8.3f}")

json.dump({'sens_mpc': {s: float(np.mean(v)) for s, v in dT.items()},
           'per_condition': rows},
          open("results/exp_029_mpc_conditions.json", 'w'), indent=2, default=float)
print("\nSaved: results/exp_029_mpc_conditions.json")
