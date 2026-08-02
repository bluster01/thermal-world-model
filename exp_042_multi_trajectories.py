#!/usr/bin/env python3
"""
exp_042_multi_trajectories.py — 多样本轨迹图 (9 条典型轨迹)
=============================================================
用途: 用户审查 MPC 闭环行为 (温度/动作物理一致性)
- 9 条轨迹 (覆盖不同 RMSE 档位)
- 每行: 温度 (MPC虚拟 vs PID真实 vs SP) + 二级阀动作 (MPC vs PID)
- 中文字体: Noto Serif CJK SC
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体 (TTC 需显式 addfont)
import glob
for f in glob.glob('/usr/share/fonts/**/*.ttc', recursive=True) + glob.glob('/usr/share/fonts/**/*.otf', recursive=True):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import exp_027_dwm_mpc as M

M.SP_TRAJ = 0  # 标量目标 (真实SP)
wm = M.load_wm()
N = len(M.test_raw)
os.makedirs("figures", exist_ok=True)

def sim_pid_wm(track_idx, n_steps=120):
    """PID 动作 (真实阀位) + WM 闭环预测温度 — 公平对照 (同一虚拟世界)"""
    win = torch.FloatTensor(M.test_raw[track_idx:track_idx+M.W]).unsqueeze(0).to(M.DEVICE)
    temps = []
    for k in range(n_steps):
        gi = track_idx + M.W + k
        a_real = torch.FloatTensor(M.test_raw[gi:gi+M.H_OUT, M.VALVE_IDX]).unsqueeze(0).to(M.DEVICE)
        with torch.no_grad():
            mu, _ = wm(win, a_real)
            y1 = mu[0, 0].item()
        next_row = torch.FloatTensor(M.test_raw[gi]).unsqueeze(0).unsqueeze(0).to(M.DEVICE)
        next_row[0, 0, M.TARGET_IDX] = y1
        win = torch.cat([win[:, 1:, :], next_row], 1)
        temps.append(y1)
    return np.array(temps)

# 选 9 条: 从 50 条主评测起点中均匀采样
np.random.seed(42)
starts = np.random.choice(range(N - M.W - M.H_OUT - 120), 50, replace=False)
pick = starts[[0, 8, 16, 24, 32, 40, 45, 47, 49]]
results = []
for i in pick:
    mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, i, 'grad', n_steps=120)
    pid_wm_t = sim_pid_wm(i, 120)
    t_real = M.test_raw[i+M.W:i+M.W+120, M.TARGET_IDX]
    rm = float(np.sqrt(np.mean((mpc_t - tset)**2)))
    rp_wm = float(np.sqrt(np.mean((pid_wm_t - tset)**2)))
    rp_real = float(np.sqrt(np.mean((t_real - tset)**2)))
    results.append({'start': int(i), 'rmse_mpc': rm, 'rmse_pidwm': rp_wm, 'rmse_pidreal': rp_real,
                    'temp_mpc': mpc_t.tolist(), 'temp_pidwm': pid_wm_t.tolist(),
                    'temp_real': t_real.tolist(), 'sp': tset.tolist(),
                    'act_mpc': mpc_a[:, 1].tolist(), 'act_pid': pid_a[:, 1].tolist()})
    print(f"轨迹 {i}: RMSE MPC {rm:.2f} | PID-WM {rp_wm:.2f} | PID真实 {rp_real:.2f}")

# 保存数据 (供后续重画)
json.dump(results, open("results/exp_042_trajectories.json", 'w'), indent=2, default=float)

# ===== 绘图: 9 行 × 2 列 =====
fig, axes = plt.subplots(9, 2, figsize=(12, 24))
for row, r in enumerate(results):
    t = np.arange(len(r['temp_mpc'])) * 10  # 秒
    # Left: temperature (three views)
    ax = axes[row, 0]
    ax.plot(t, r['temp_real'], 'k-', lw=1.0, alpha=0.85, label='Actual (physical)')
    ax.plot(t, r['temp_pidwm'], 'gray', lw=1.2, ls='--', label='PID-WM (fair baseline)')
    ax.plot(t, r['temp_mpc'], 'C0-', lw=1.4, label='DWM-MPC (WM closed-loop)')
    ax.plot(t, r['sp'], 'r--', lw=0.8, alpha=0.5, label='Setpoint SP')
    ax.set_ylabel('Main steam temp (°C)', fontsize=9)
    ax.legend(fontsize=6.5, loc='best', ncol=2)
    ax.set_title(f"Track {row+1} (start {r['start']})  RMSE: MPC {r['rmse_mpc']:.2f} / PID-WM {r['rmse_pidwm']:.2f} / Actual {r['rmse_pidreal']:.2f}", fontsize=9)
    ax.grid(alpha=0.3)
    # Right: secondary valve action
    ax = axes[row, 1]
    ax.plot(t, r['act_pid'], 'k-', lw=1.0, label='PID valve (actual)')
    ax.plot(t, r['act_mpc'], 'C1-', lw=1.2, label='MPC valve')
    ax.set_ylabel('Secondary attemp. valve (%)', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.set_title(f"Actuator {row+1}", fontsize=9)
    ax.grid(alpha=0.3)
for ax in axes[:, 0]:
    ax.set_xlabel('Time (s)', fontsize=9)
for ax in axes[:, 1]:
    ax.set_xlabel('Time (s)', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig_multi_trajectories.png', dpi=150)
print("\nSaved: figures/fig_multi_trajectories.png")
