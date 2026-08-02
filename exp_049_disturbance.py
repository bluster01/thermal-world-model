#!/usr/bin/env python3
"""exp_049_disturbance.py — 过程扰动响应验证 (用户质疑: MPC曲线是直线=奖励函数?)
================================================================================
机制假设: WM期望预测平滑 + 虚拟世界无扰动 → MPC"不动最优" → 直线
验证: 虚拟世界注入过程扰动 (随机游走, 模拟负荷/燃料扰动):
  - MPC (滚动优化, 每周期看到新状态) vs PID-WM (固定动作) vs 无控制 (动作不动)
  - 若 MPC std << 无控制 → 奖励函数正确 (会响应扰动, 直线=世界无扰动)
  - 若 MPC std ≈ 无控制 → 奖励函数有问题 (不响应扰动)
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

N_TRACKS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DIST_AMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3  # 扰动幅度 (°C 随机游走步)

wm = load_wm()
M.SP_TRAJ = 0
N = len(test_raw)
np.random.seed(42)
starts = np.random.choice(range(N - W - H_OUT - 120), N_TRACKS, replace=False)

def sim_dist(track_idx, controller, dist_amp, n_steps=120, m_step=M.M_STEP):
    """虚拟世界 + 过程扰动: 每步温度预测注入随机游走扰动 (真实扰动信号)"""
    rng = np.random.default_rng(100 + track_idx)
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W]).unsqueeze(0).to(DEVICE)
    a_last = torch.FloatTensor(test_raw[track_idx+W, VALVE_IDX]).to(DEVICE)
    temps, dists = [], []
    d_state = 0.0  # 扰动状态 (随机游走)
    for k in range(0, n_steps, m_step):
        gi = track_idx + W + k
        t_set = torch.tensor(float(test_raw[gi, SP_IDX]), device=DEVICE)
        if controller == 'mpc':
            a_plan, _ = M.plan_grad(wm, win, t_set, a_last, None, None)
        elif controller == 'pid':
            a_plan = torch.FloatTensor(test_raw[gi:gi+M.H_PLAN, VALVE_IDX]).to(DEVICE)
        else:  # no-control: 保持初始动作
            a_plan = a_last.unsqueeze(0).repeat(M.H_PLAN, 1)
        with torch.no_grad():
            a_full = torch.cat([a_plan, a_plan[-1:].repeat(H_OUT - M.H_PLAN, 1)], 0) if M.H_PLAN < H_OUT else a_plan[:H_OUT]
            mu, _ = wm(win, a_full.reshape(1, -1))
        n_exec = min(m_step, len(a_plan), len(mu[0]))
        for j in range(n_exec):
            gij = gi + j
            if gij >= track_idx + W + n_steps: break
            # 过程扰动: 随机游走 (自相关扰动, 模拟负荷/燃料)
            d_state = 0.9 * d_state + rng.normal(0, dist_amp)
            y_j = mu[0, j].item() + d_state
            next_row = torch.FloatTensor(test_raw[gij]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            next_row[0, 0, TARGET_IDX] = y_j
            win = torch.cat([win[:, 1:, :], next_row], 1)
            temps.append(y_j); dists.append(d_state)
        a_last = a_plan[n_exec - 1]
    return np.array(temps), np.array(dists)

results = {c: [] for c in ['mpc', 'pid', 'none']}
for s in starts:
    tset = test_raw[s+W:s+W+120, SP_IDX]
    for c in results:
        t, _ = sim_dist(s, c, DIST_AMP)
        results[c].append(float(np.sqrt(np.mean((t - tset)**2))))

print(f"\n===== 过程扰动世界 (随机游走 σ={DIST_AMP}°C/步) =====")
print(f"{'控制器':>10} {'RMSE':>10} {'vs无控制':>10}")
base = np.mean(results['none'])
for c in ['mpc', 'pid', 'none']:
    m = np.mean(results[c])
    print(f"{c:>10} {m:>10.3f} {(1-m/base)*100:>+9.1f}%")

json.dump({c: {'rmse': float(np.mean(v)), 'per': [float(x) for x in v]} for c, v in results.items()},
          open("results/exp_049_disturbance.json", 'w'), indent=2)
print("\nSaved: results/exp_049_disturbance.json")
