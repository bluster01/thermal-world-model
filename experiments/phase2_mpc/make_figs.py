#!/usr/bin/env python3
"""论文对照图: Fig1 H_PLAN全扫曲线 + Fig2 平滑方案对比 (全英文, Applied Energy 风格)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/bluster/projectA/thermal-world-model'
os.chdir(ROOT)
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150})

# ============ Fig 1: H_PLAN sweep ============
H = list(range(1, 19))
rmse, iae, itae, tv, jump, ovt = [], [], [], [], [], []
for h in H:
    d = json.load(open(f'results/exp_059b_hplan_newbench/h{h}.json'))
    a = d['agg']
    rmse.append(a['rmse_mpc']); iae.append(a['iae_mpc']); itae.append(a['itae_mpc'])
    tv.append(a['act_tv_mpc']); jump.append(a['jump_mean']); ovt.append(a['overtemp_int_mpc'])
H_s = [f'{h*10}s' for h in H]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
ax = axes[0, 0]
ax.plot(H, rmse, 'o-', color='#C0392B', lw=2, ms=5)
ax.axvline(7, color='gray', ls='--', lw=1, alpha=0.7)
ax.annotate('M_STEP transition\n(H=7)', xy=(7, rmse[6]), xytext=(8.5, 2.45),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax.plot(18, rmse[-1], 'o', color='#C0392B', ms=9, mfc='white', mew=2.5)
ax.annotate(f'min {rmse[-1]:.3f}', xy=(18, rmse[-1]), xytext=(15.5, 2.06),
            fontsize=10, arrowprops=dict(arrowstyle='->', lw=1))
ax.set_ylabel('RMSE (°C)')
ax.set_title('(a) Tracking error vs planning horizon', fontsize=11)

ax = axes[0, 1]
ax.plot(H, tv, 's-', color='#2E86C1', lw=2, ms=5, label='Total variation')
ax.plot(H, jump, '^-', color='#7D3C98', lw=2, ms=5, label='Boundary jump')
ax.axvline(7, color='gray', ls='--', lw=1, alpha=0.7)
ax.set_ylabel('Action variation (°C)')
ax.legend(fontsize=9)
ax.set_title('(b) Actuator smoothness vs planning horizon', fontsize=11)

ax = axes[1, 0]
ax.plot(H, ovt, 'd-', color='#E67E22', lw=2, ms=5)
ax.axvline(7, color='gray', ls='--', lw=1, alpha=0.7)
ax.set_ylabel('Overtemperature integral (°C·10 s)')
ax.set_title('(c) Overtemperature risk vs planning horizon', fontsize=11)

ax = axes[1, 1]
ax.plot(H, iae, 'o-', color='#27AE60', lw=2, ms=5, label='IAE')
ax.plot(H, np.array(itae)/50, 's-', color='#16A085', lw=2, ms=5, label='ITAE/50')
ax.axvline(7, color='gray', ls='--', lw=1, alpha=0.7)
ax.set_ylabel('Integral error (°C·10 s)')
ax.legend(fontsize=9)
ax.set_title('(d) Integral errors vs planning horizon', fontsize=11)

for ax in axes.flat:
    ax.set_xticks(H)
    ax.set_xticklabels(H_s, rotation=45, fontsize=8)
    ax.set_xlabel('Planning horizon H_PLAN')
fig.tight_layout()
fig.savefig('figures/fig_hplan_sweep.png', bbox_inches='tight')
print('Saved: figures/fig_hplan_sweep.png')

# ============ Fig 2: Smooth modes ============
modes = ['hard5', 'hard2', 'ovl05_hard5', 'inert05', 'inert025']
cols = {'hard5': '#C0392B', 'hard2': '#E67E22', 'ovl05_hard5': '#7D3C98',
        'inert05': '#2E86C1', 'inert025': '#27AE60'}
rm, tvv, jmp = [], [], []
for m in modes:
    d = json.load(open(f'results/exp_063_smooth_scan/{m}.json'))
    a = d['agg']
    rm.append(a['rmse_mpc']); tvv.append(a['act_tv_mpc']); jmp.append(a['jump_mean'])
base_rm = rm[0]; base_tv = tvv[0]; base_j = jmp[0]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(modes)); w = 0.26
b1 = ax.bar(x - w, [r/base_rm*100 - 100 for r in rm], w, label='ΔRMSE (%)', color=cols['hard5'])
b2 = ax.bar(x, [t/base_tv*100 - 100 for t in tvv], w, label='ΔTV (%)', color=cols['hard2'])
b3 = ax.bar(x + w, [j/base_j*100 - 100 for j in jmp], w, label='ΔBoundary jump (%)', color=cols['inert05'])
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(modes)
ax.set_ylabel('Relative change vs hard5 (%)')
ax.set_title('Smoothness fixes: cost (RMSE) vs benefit (TV / boundary jump)')
ax.legend(fontsize=9)
for xi, (r, t, j) in enumerate(zip(rm, tvv, jmp)):
    ax.text(xi - w, (r/base_rm*100-100) + 0.5, f'{r:.3f}', ha='center', fontsize=8)
    ax.text(xi, (t/base_tv*100-100) + 0.5, f'{t:.3f}', ha='center', fontsize=8)
    ax.text(xi + w, (j/base_j*100-100) + 0.5, f'{j:.3f}', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig('figures/fig_smooth_modes.png', bbox_inches='tight')
print('Saved: figures/fig_smooth_modes.png')
