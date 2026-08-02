#!/usr/bin/env python3
"""exp_048_robustness.py — 鲁棒性分析 (Phase 2.5 任务4b)
WM 闭环预测加噪声 σ∈{0, 0.1, 0.3, 0.5}°C (模拟模型失配/传感器噪声):
1. MPC 性能退化 (RMSE/std/TV 随 σ)
2. 动作退化 (TV 增幅 = 控制器对抗噪声的代价)
3. 与 PID-WM 在相同噪声下对比 (谁对噪声更鲁棒)
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
SIGMAS = [0.0, 0.1, 0.3, 0.5]
wm = load_wm()
M.SP_TRAJ = 0
N = len(test_raw)
np.random.seed(42)
starts = np.random.choice(range(N - W - H_OUT - 120), N_TRACKS, replace=False)

def sim_with_noise(track_idx, sigma, controller='mpc', n_steps=120, m_step=M.M_STEP):
    """闭环仿真, 每步 WM 预测温度加高斯噪声 σ (注入虚拟世界)"""
    rng = np.random.default_rng(42 + track_idx)
    win = torch.FloatTensor(test_raw[track_idx:track_idx+W]).unsqueeze(0).to(DEVICE)
    a_last = torch.FloatTensor(test_raw[track_idx+W, VALVE_IDX]).to(DEVICE)
    temps = []
    for k in range(0, n_steps, m_step):
        gi = track_idx + W + k
        t_set = torch.tensor(float(test_raw[gi, SP_IDX]), device=DEVICE)
        if controller == 'mpc':
            a_plan, _ = M.plan_grad(wm, win, t_set, a_last, None, None)
            with torch.no_grad():
                a_full = torch.cat([a_plan, a_plan[-1:].repeat(H_OUT - M.H_PLAN, 1)], 0) if M.H_PLAN < H_OUT else a_plan[:H_OUT]
                mu, _ = wm(win, a_full.reshape(1, -1))
        else:  # pid: 真实动作
            a_plan = torch.FloatTensor(test_raw[gi:gi+M.H_PLAN, VALVE_IDX]).to(DEVICE)
            with torch.no_grad():
                a_full = torch.cat([a_plan, a_plan[-1:].repeat(H_OUT - M.H_PLAN, 1)], 0) if M.H_PLAN < H_OUT else a_plan[:H_OUT]
                mu, _ = wm(win, a_full.reshape(1, -1))
        n_exec = min(m_step, len(a_plan), len(mu[0]))
        for j in range(n_exec):
            gij = gi + j
            if gij >= track_idx + W + n_steps: break
            y_j = mu[0, j].item()
            if sigma > 0:
                y_j = y_j + rng.normal(0, sigma)  # 注入噪声 (模拟失配)
            next_row = torch.FloatTensor(test_raw[gij]).unsqueeze(0).unsqueeze(0).to(DEVICE)
            next_row[0, 0, TARGET_IDX] = y_j
            win = torch.cat([win[:, 1:, :], next_row], 1)
            temps.append(y_j)
        a_last = a_plan[n_exec - 1]
    return np.array(temps)

results = {c: {s: [] for s in SIGMAS} for c in ['mpc', 'pid']}
for s in starts:
    tset = test_raw[s+W:s+W+120, SP_IDX]
    for sig in SIGMAS:
        for c in ['mpc', 'pid']:
            t = sim_with_noise(s, sig, c)
            results[c][sig].append(float(np.sqrt(np.mean((t - tset)**2))))

print("\n===== 鲁棒性: WM 预测噪声 σ 下 RMSE 退化 =====")
print(f"{'σ(°C)':>7} {'MPC RMSE':>10} {'PID RMSE':>10} {'MPC退化':>8} {'PID退化':>8}")
base_m, base_p = np.mean(results['mpc'][0.0]), np.mean(results['pid'][0.0])
for sig in SIGMAS:
    m = np.mean(results['mpc'][sig]); p = np.mean(results['pid'][sig])
    dm = (m/base_m - 1) * 100; dp = (p/base_p - 1) * 100
    print(f"{sig:>7.1f} {m:>10.3f} {p:>10.3f} {dm:>+7.1f}% {dp:>+7.1f}%")

json.dump({c: {str(s): {'rmse': float(np.mean(v)), 'rmse_per': [float(x) for x in v]} for s, v in d.items()}
           for c, d in results.items()},
          open("results/exp_048_robustness.json", 'w'), indent=2)
print("\nSaved: results/exp_048_robustness.json")
