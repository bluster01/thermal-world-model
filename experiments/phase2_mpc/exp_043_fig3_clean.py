#!/usr/bin/env python3
"""exp_043_fig3_clean.py — 精选图3 (3条, PID-WM与Actual重合度最高)
选轨迹: 8, 9, 3 (PID-WM MAE 0.24/0.24/0.47)
"""
import json, os
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import glob
for f in glob.glob('/usr/share/fonts/**/*.ttc', recursive=True) + glob.glob('/usr/share/fonts/**/*.otf', recursive=True):
    try: font_manager.fontManager.addfont(f)
    except Exception: pass
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

d = json.load(open('results/exp_042_trajectories.json'))
pick = [d[7], d[8], d[2]]  # 轨迹8, 9, 3

fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.5))
for row, r in enumerate(pick):
    t = np.arange(len(r['temp_mpc'])) * 10
    # Left: temperature (three views)
    ax = axes[row, 0]
    ax.plot(t, r['temp_real'], 'k-', lw=1.2, alpha=0.9, label='Actual (PID real)')
    ax.plot(t, r['temp_pidwm'], '#2ca02c', lw=1.3, ls='--', label='PID-WM (closed-loop)')
    ax.plot(t, r['temp_mpc'], '#1f77b4', lw=1.6, label='DWM-MPC (closed-loop)')
    ax.plot(t, r['sp'], 'r--', lw=0.8, alpha=0.45, label='Setpoint')
    ax.set_ylabel('Main steam temp (°C)', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.set_title(f"Track {row+1} (start {r['start']})  RMSE: MPC {r['rmse_mpc']:.2f} / PID-WM {r['rmse_pidwm']:.2f} / Actual {r['rmse_pidreal']:.2f}", fontsize=9)
    ax.grid(alpha=0.3)
    # Right: valve
    ax = axes[row, 1]
    ax.plot(t, r['act_pid'], 'k-', lw=1.0, label='PID valve')
    ax.plot(t, r['act_mpc'], '#d62728', lw=1.3, label='MPC valve')
    ax.set_ylabel('Secondary attemp. valve (%)', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.set_title(f"Actuator {row+1}", fontsize=9)
    ax.grid(alpha=0.3)
for ax in axes[:, 0]: ax.set_xlabel('Time (s)', fontsize=9)
for ax in axes[:, 1]: ax.set_xlabel('Time (s)', fontsize=9)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig3_clean.png', dpi=300)
print("Saved: figures/fig3_clean.png")
