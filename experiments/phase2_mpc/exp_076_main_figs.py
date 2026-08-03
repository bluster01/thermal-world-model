#!/usr/bin/env python3
"""
exp_076_main_figs.py — P6: 论文主图 (baseline 对比)
=====================================================
(a) 扰动/无扰动 RMSE bar (9 方法, 分组)
(b) 扰动超温时间 (log 轴)
(c) 扰动 TV
英文图表 (Applied Energy 风格)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

METHODS = ['PID', 'Linear MPC\n(ΔT-ARX)', 'PETS-style\n(Ens+CEM)', 'Ensemble\n+grad', 'Det. WM\n(M5)', 'DWM-MPC\n(M7)', 'TD3+BC', 'IQL', 'SAC\n(in WM)']
RMSE_D = [2.661, 5.499, 2.603, 2.383, 2.423, 2.502, 9.310, 9.311, 9.120]
RMSE_N = [1.704, 1.692, 1.132, 1.057, 1.051, 1.106, 1.803, 1.803, 1.825]
OVT_D  = [1, 1682, 156, 112, 182, 1, 2756, 2754, 2685]
TV_D   = [0.338, 0.244, 0.648, 0.246, 0.208, 0.278, 18.5/119, 18.6/119, 69.0/119]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
x = np.arange(len(METHODS)); w = 0.38
# (a) RMSE
ax = axes[0]
ax.bar(x - w/2, RMSE_D, w, label='Disturbed world', color='#c0504d')
ax.bar(x + w/2, RMSE_N, w, label='No disturbance', color='#4f81bd')
ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=7.5)
ax.set_ylabel('RMSE vs setpoint (°C)')
ax.set_title('(a) Tracking RMSE')
ax.legend(fontsize=7.5)
ax.set_ylim(0, 10.5)
for i, v in enumerate(RMSE_D):
    ax.text(i - w/2, v + 0.15, f'{v:.2f}', ha='center', fontsize=6.5)
for i, v in enumerate(RMSE_N):
    ax.text(i + w/2, v + 0.15, f'{v:.2f}', ha='center', fontsize=6.5)
# (b) overtemp (log)
ax = axes[1]
ax.bar(x, OVT_D, w * 1.5, color='#c0504d')
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=7.5)
ax.set_ylabel('Overtime above 575°C (s)')
ax.set_title('(b) Overtemperature time (disturbed)')
for i, v in enumerate(OVT_D):
    ax.text(i, v * 1.15, f'{v:.0f}', ha='center', fontsize=6.5)
# (c) TV
ax = axes[2]
ax.bar(x, TV_D, w * 1.5, color='#4f81bd')
ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=7.5)
ax.set_ylabel('Action TV per step')
ax.set_title('(c) Action total variation (disturbed)')
for i, v in enumerate(TV_D):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=6.5)

fig.tight_layout()
fig.savefig('figures/fig_baseline_compare.png', dpi=200, bbox_inches='tight')
print('Saved: figures/fig_baseline_compare.png')

# ============ 英文表格 (LaTeX 源) ============
rows = [
    ('PID', '—', '—', 2.661, 1.704, 0.338, 1, 0.0),
    ('Linear MPC (ΔT-ARX)', 'ARX-ident.', 'grad', 5.499, 1.692, 0.244, 1682, 0.01),
    ('PETS-style (Ens+CEM)', '3×M7 ens.', 'CEM', 2.603, 1.132, 0.648, 156, 27),
    ('Ensemble + grad', '3×M7 ens.', 'grad', 2.383, 1.057, 0.246, 112, 27),
    ('Det. WM (M5) + grad', 'MSE det.', 'grad', 2.423, 1.051, 0.208, 182, 10),
    ('DWM-MPC (M7, ours)', 'β-NLL prob.', 'grad', 2.502, 1.106, 0.278, 1, 10),
    ('TD3+BC (offline)', '—', 'policy', 9.310, 1.803, '0.31*', 2756, 45),
    ('IQL (offline)', '—', 'policy', 9.311, 1.803, '0.31*', 2754, 40),
    ('SAC in learned WM', '—', 'policy', 9.120, 1.825, '1.15*', 2685, 25),
]
with open('docs/table_baseline.tex', 'w') as f:
    f.write('% Main comparison table (Applied Energy style)\n')
    f.write('% *: policy TV per step (total TV / 119 steps); MPC TV = total variation per trajectory\n')
    f.write('\\begin{table}[t]\n\\centering\\small\n')
    f.write('\\caption{Closed-loop comparison on main steam temperature control (150 trajectories from 3 start sets, '
            'paired Wilcoxon $p<10^{-12}$ for MPC-family vs PID; disturbed world: process disturbance $\\sigma=0.3$). '
            'Bold: best per metric.}\n')
    f.write('\\label{tab:baseline}\n')
    f.write('\\begin{tabular}{lccc cccc}\n\\toprule\n')
    f.write('Method & WM type & Controller & RMSE$_{dist}$ & RMSE$_{clean}$ & TV$_{dist}$ & Overtemp (s) & Train (min)\\\\\n\\midrule\n')
    for r in rows:
        f.write(' & '.join([str(v) for v in r]) + ' \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
print('Saved: docs/table_baseline.tex')
