#!/usr/bin/env python3
"""增益校准效果图: gain_180 vs λ, 与 SP-IV 真值区间对比。"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = 'results/exp_201_valve_action'
data = [
    ('baseline (no calib)', 0.0, -1.5, 0.953, '#95a5a6'),
    ('λ=0.1', 0.1, -65.4, 1.194, '#e67e22'),
    ('λ=0.2', 0.2, -96.2, 1.359, '#c0392b'),
    ('λ=0.5', 0.5, -0.4, 1.529, '#7f8c8d'),
]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=150)

ax = axes[0]
ax.axhspan(-130, -90, color='#2ecc71', alpha=0.18, label='SP-IV truth range')
for name, lam, g, mae, c in data:
    ax.plot(lam, abs(g), 'o', ms=8, color=c, label=name)
    ax.text(lam, abs(g) * 1.15, f'{abs(g):.0f}', ha='center', fontsize=8, color=c)
ax.set_yscale('log')
ax.set_xlabel('λ_gain (calibration weight)')
ax.set_ylabel('|gain@180s| (m°C/%, log)')
ax.set_title('(a) Gain calibration lifts model gain\nto SP-IV truth range', fontsize=10)
ax.legend(fontsize=7, loc='lower right')
ax.grid(alpha=0.3, which='both')

ax = axes[1]
lams = [d[1] for d in data[1:]]
maes = [d[3] for d in data[1:]]
gains = [abs(d[2]) for d in data[1:]]
ax.plot(lams, maes, 'o-', color='#2980b9', lw=1.6, label='MAE')
ax.set_xlabel('λ_gain')
ax.set_ylabel('MAE (°C)')
ax.set_title('(b) Accuracy cost of calibration', fontsize=10)
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(lams, gains, 's--', color='#c0392b', lw=1.5, label='|gain|')
ax2.set_ylabel('|gain@180s| (m°C/%)')
ax2.set_yscale('log')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')

plt.tight_layout()
out = 'results/exp_201_valve_action/fig_gain_calib.png'
plt.savefig(out, bbox_inches='tight')
print('saved:', out)
