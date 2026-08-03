#!/usr/bin/env python3
"""
exp_065_h1h18_trajectories.py — H1-18 每个 H 的 MPC vs PID 轨迹对比图 + 模型物理响应测试
===========================================================================================
图1: 18 子图, 每格一条轨迹 (seed42 starts[0]), 显示 MPC/PID 温度 vs SP + 动作
图2: 物理响应测试 — 同一输入窗口 + 阀位阶跃, 4 个训练长度模型 (h06/09/12/18) 的温度响应,
     检查响应时标 (应 60-120s 滞后, 若立即响应=因果替代假说成立)
用法: python exp_065_h1h18_trajectories.py
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ 图1: H1-18 轨迹对比 ============
M.SP_TRAJ = 0; M.DIST_AMP = 0.3; M.M_STEP = 6
M.FIX_MODE = 'hard5'; M.HARD_DELTA = 0.0; M.BENCH_SP_EACH = True
wm = M.load_wm()
N = len(M.test_raw)
np.random.seed(42)
starts = np.random.choice(range(N - M.W - M.H_OUT - 120), 10, replace=False)
s = int(starts[0])

fig, axes = plt.subplots(6, 3, figsize=(15, 16), sharex=True)
rows = []
for h in range(1, 19):
    M.H_PLAN = h
    mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
    rows.append((mpc_t, pid_t, tset, mpc_a, pid_a))

t_axis = np.arange(len(rows[0][0])) * 10  # s
for i, (mpc_t, pid_t, tset, mpc_a, pid_a) in enumerate(rows):
    hp = i + 1  # 修正: 原误用循环外残留变量 h (=18), 标题全显示 H_PLAN=18
    ax = axes[i // 3, i % 3]
    ax.plot(t_axis, mpc_t, color='#C0392B', lw=1.6, label='DWM-MPC')
    ax.plot(t_axis, pid_t, color='#2E86C1', lw=1.2, alpha=0.85, label='PID')
    ax.plot(t_axis, tset, color='gray', ls='--', lw=1, label='SP')
    r1 = float(np.sqrt(np.mean((mpc_t - tset) ** 2)))
    r2 = float(np.sqrt(np.mean((pid_t - tset) ** 2)))
    ax.set_title(f'H_PLAN={hp} (RMSE: MPC {r1:.2f} / PID {r2:.2f})', fontsize=10)
    if i == 0:
        ax.legend(fontsize=8, loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(t_axis, mpc_a[:, 0], color='#C0392B', ls=':', lw=0.8, alpha=0.5)
    ax2.plot(t_axis, pid_a[:, 0], color='#2E86C1', ls=':', lw=0.8, alpha=0.5)
    ax2.set_ylim(-15, 15)
    ax2.set_yticks([])
    if i % 3 == 0:
        ax.set_ylabel('Temp (°C)')
fig.supxlabel('Time (s)')
fig.suptitle(f'MPC vs PID closed-loop trajectories (track {s}, disturbed world) — H_PLAN = 1..18', y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig('figures/fig_h1h18_trajectories.png', bbox_inches='tight')
print('Saved: figures/fig_h1h18_trajectories.png')

# ============ 图2: 物理响应测试 (动作阶跃) ============
import experiments.phase1_dynamics.exp_025_unified_benchmark as E
from experiments.phase1_dynamics.exp_025_unified_benchmark import (
    build_model, test_raw, VALVE_IDX, TARGET_IDX, SP_IDX)

# 取一个稳定窗口 (温度平稳段, 动作小幅波动)
np.random.seed(7)
W_ = M.W
for _ in range(200):
    i = np.random.randint(W_ + 50, N - 200)
    win_t = test_raw[i:i + W_, TARGET_IDX]
    if abs(win_t[-1] - win_t[-W_//2]).mean() < 0.3:  # 平稳窗口
        break

base_a = test_raw[i + W_, VALVE_IDX]           # 当前阀位
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
print(f"窗口起点 i={i} | 温度 {test_raw[i+W_-1, TARGET_IDX]:.2f} | 阀位 {base_a}")
for hi, (h, ax) in enumerate(zip([6, 9, 12, 18], axes.flat)):
    E.H_OUT = h
    model = build_model('M7').to(DEVICE).eval()
    ck = torch.load(f'results/exp_048_horizon/checkpoints/h{h:02d}.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck['model_state_dict'])
    xh = torch.FloatTensor(test_raw[i:i + W_]).unsqueeze(0).to(DEVICE)
    # 阶跃: 阀位 +5% (开大减温水) 从 t=0 起
    a_step = np.tile(base_a, (h, 1)).copy()
    a_step[:] = base_a + 5.0
    af = torch.FloatTensor(a_step).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu, _ = model(xh, af)
    resp = mu[0].cpu().numpy()
    base = test_raw[i + W_, TARGET_IDX]
    ax.plot(np.arange(h) * 10, resp - base, 'o-', color='#C0392B', lw=1.6)
    # 无阶跃对照 (基线动作)
    a0 = torch.FloatTensor(np.tile(base_a, (h, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu0, _ = model(xh, a0)
    resp0 = mu0[0].cpu().numpy()
    ax.plot(np.arange(h) * 10, resp0 - base, 's--', color='#2E86C1', lw=1.2, alpha=0.7)
    ax.set_title(f'H_OUT={h*10}s', fontsize=11)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('Prediction horizon (s)')
    ax.set_ylabel('ΔT vs base (°C)')
    # 响应时标: 阶跃响应达最大斜率的时间
    d = np.diff(resp - resp0)
    t_max_slope = (np.argmax(np.abs(d)) + 1) * 10 if len(d) else float('nan')
    print(f"  H_OUT={h*10}s: 阶跃响应最大变化率在 {t_max_slope}s | 终值 Δ = {resp[-1]-resp0[-1]:+.3f}°C")
axes[0, 0].legend(['+5% valve step', 'baseline'])
fig.suptitle('Model response to a +5% valve (attemperation water) step — physics time scale check', fontsize=13)
fig.tight_layout()
fig.savefig('figures/fig_action_response.png', bbox_inches='tight')
print('Saved: figures/fig_action_response.png')
