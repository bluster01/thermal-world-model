#!/usr/bin/env python3
"""
exp_066_multi_track_plots.py — 多 track 对比: MPC/PID温度 vs 真实温度 vs 真实阀位
====================================================================================
主协议 (H_PLAN=18, ovl05_hard5, M_STEP=6, DIST_AMP=0.3, 新基准)
每条 track 2 面板: 上=温度 (MPC/PID/真实/SP), 下=动作 (MPC/真实阀位)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 140})

M.SP_TRAJ = 0; M.DIST_AMP = float(os.environ.get('DIST_AMP', 0.3)); M.M_STEP = 6; M.H_PLAN = 18
M.FIX_MODE = 'overlap'; M.LAMBDA3 = 0.05; M.HARD_DELTA = 5.0
M.BENCH_SP_EACH = True
M.SIM_COLLECT_SIGMA = True
wm = M.load_wm()

N = len(M.test_raw)
np.random.seed(42)
starts = np.random.choice(range(N - M.W - M.H_OUT - 120), 10, replace=False)
TRACKS = [0, 1, 4, 5, 6]  # 5条: 覆盖好/差轨迹 (RMSE 分布从 exp_064 看: 中位1.99, max14)

fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
for r, ti in enumerate(TRACKS):
    s = int(starts[ti])
    mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, s, 'grad')
    mpc_sig = np.array(M.SIM_SIGMA_BUF)  # 每步 σ (物理空间)
    real_t = M.test_raw[s + M.W:s + M.W + len(mpc_t), M.TARGET_IDX]
    real_a = M.test_raw[s + M.W:s + M.W + len(mpc_a), M.VALVE_IDX]  # 真实阀位 (PID实际)
    t_ax = np.arange(len(mpc_t)) * 10
    rm = float(np.sqrt(np.mean((mpc_t - tset) ** 2)))
    rp = float(np.sqrt(np.mean((pid_t - tset) ** 2)))
    rr = float(np.sqrt(np.mean((real_t - tset) ** 2)))
    # 温度面板
    ax = axes[r, 0]
    ax.plot(t_ax, mpc_t, color='#C0392B', lw=1.8, label='MPC temp')
    ax.fill_between(t_ax, mpc_t - 2 * mpc_sig, mpc_t + 2 * mpc_sig,
                    color='#C0392B', alpha=0.15, label='MPC ±2σ')
    ax.plot(t_ax, pid_t, color='#2E86C1', lw=1.4, alpha=0.9, label='PID temp (model)')
    ax.plot(t_ax, real_t, color='#27AE60', lw=1.2, alpha=0.8, label='Real temp')
    ax.plot(t_ax, tset, color='gray', ls='--', lw=1, label='SP')
    ax.set_ylabel('Temp (°C)')
    ax.set_title(f'Track {ti} (start={s}): RMSE MPC {rm:.2f} / PID {rp:.2f} / Real {rr:.2f}', fontsize=10)
    if r == 0:
        ax.legend(fontsize=8, ncol=2)
    # 动作面板
    ax = axes[r, 1]
    ax.plot(t_ax, mpc_a[:, 0], color='#C0392B', ls=':', lw=1.2, label='MPC action')
    ax.plot(t_ax, real_a[:, 0], color='#2E86C1', ls='--', lw=1.2, alpha=0.9, label='Real valve (PID)')
    ax.set_ylabel('Valve 1 (% open)')
    ax.set_title(f'Track {ti}: actions', fontsize=10)
    if r == 0:
        ax.legend(fontsize=8)
    ax.set_xlabel('Time (s)')
fig.suptitle('Multi-track comparison: MPC vs PID vs Real (H_PLAN=18, ovl05_hard5, '
              + ('disturbed world)' if M.DIST_AMP > 0 else 'NO-disturbance world (model pure prediction)'), y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig('figures/fig_multi_track_compare.png', bbox_inches='tight')
print('Saved: figures/fig_multi_track_compare.png')
