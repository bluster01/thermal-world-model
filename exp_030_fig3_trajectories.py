#!/usr/bin/env python3
"""
exp_030_fig3_trajectories.py — 论文图3: 反事实轨迹对比
========================================================
典型 3 条轨迹: PID 真实温度 vs MPC 闭环温度 vs 设定值
+ 动作对比 (MPC 动作 vs PID 动作)
输出: figures/fig3_mpc_trajectories.png (300dpi)
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from exp_027_dwm_mpc import load_wm, simulate, W, H_OUT, DEVICE, test_raw

os.makedirs("figures", exist_ok=True)
wm = load_wm()
N = len(test_raw)

# 选 3 条轨迹: 用固定 seed 重现 grad 50 条的起始索引
np.random.seed(42)
starts = np.random.choice(range(N - W - H_OUT - 120), 50, replace=False)
pick = [starts[0], starts[15], starts[30]]  # 均匀采样 3 条

fig, axes = plt.subplots(3, 2, figsize=(11, 9))
for row, i in enumerate(pick):
    mpc_t, pid_t, tset, mpc_a, pid_a = simulate(wm, i, 'grad', n_steps=120)
    t = np.arange(len(mpc_t)) * 10  # 秒
    # 左: 温度
    ax = axes[row, 0]
    ax.plot(t, pid_t, 'k-', lw=1.2, label='PID (真实)')
    ax.plot(t, mpc_t, 'C0-', lw=1.2, label='DWM-MPC (闭环)')
    ax.plot(t, tset, 'r--', lw=0.8, alpha=0.7, label='设定值 (窗口均值)')
    ax.set_ylabel('主汽温 (°C)')
    ax.legend(fontsize=8)
    ax.set_title(f'轨迹 {row+1} (起点 {i})', fontsize=9)
    # 右: 动作 (二级阀)
    ax = axes[row, 1]
    ax.plot(t, pid_a[:, 1], 'k-', lw=1.2, label='PID 二级阀')
    ax.plot(t, mpc_a[:, 1], 'C1-', lw=1.2, label='MPC 二级阀')
    ax.set_ylabel('二级减温阀 (%)')
    ax.legend(fontsize=8)
    ax.set_title(f'执行器动作 {row+1}', fontsize=9)
    rmse_mpc = np.sqrt(np.mean((mpc_t - tset)**2)); rmse_pid = np.sqrt(np.mean((pid_t - tset)**2))
    ax.text(0.02, 0.95, f'RMSE: MPC {rmse_mpc:.2f} vs PID {rmse_pid:.2f}',
            transform=ax.transAxes, fontsize=8, va='top')

for ax in axes[:, 0]:
    ax.set_xlabel('时间 (s)')
for ax in axes[:, 1]:
    ax.set_xlabel('时间 (s)')
plt.tight_layout()
plt.savefig('figures/fig3_mpc_trajectories.png', dpi=300)
print("Saved: figures/fig3_mpc_trajectories.png")
