#!/usr/bin/env python3
"""SP-IV vs 模型增益对比图 (log 轴): 真实 plant 增益 vs 模型扰动增益。"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_proj = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, 'experiments', 'phase1_dynamics'))

from exp_025_unified_benchmark import data_all, NUMERIC_COLS

raw = data_all
I_V2 = NUMERIC_COLS.index('二级减温调节门阀位')
LAYERS = [(0, 10), (10, 20), (20, 30), (30, 45)]
VC = np.array([5, 15, 25, 37.5])

# SP-IV 中位数真值 (180s, 来自 exp_201_sp_iv2, 方向过滤后)
g_iv_med = {0: -127.1, 10: -90.9, 20: -96.1}   # m°C/%, 0-10/10-20/20-30
g_iv_n = {0: 10, 10: 17, 20: 22}
# 模型 (flow noff 600s 口径, 来自 gain_diag)
g_model = np.array([1.01, 1.57, 2.30, 1.39])   # m°C/%, 全层

fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), dpi=150)
xs = np.arange(3)
vals_iv = np.abs([g_iv_med[0], g_iv_med[10], g_iv_med[20]])
ax.bar(xs - 0.18, vals_iv, width=0.36, color='#c0392b', alpha=0.85,
       label='SP-IV truth (180s, median, |gain|)')
ax.bar(xs + 0.18, g_model[:3], width=0.36, color='#2980b9', alpha=0.85,
       label='Model flow noff (600s)')
for i, n in zip(xs, [g_iv_n[0], g_iv_n[10], g_iv_n[20]]):
    ax.text(i - 0.18, vals_iv[i] * 1.15, f'n={n}', ha='center', fontsize=8, color='#c0392b')
for i in xs:
    ax.text(i + 0.18, g_model[i] * 1.15, f'{g_model[i]:.1f}', ha='center', fontsize=8, color='#2980b9')
ax.set_yscale('log')
ax.set_xticks(xs)
ax.set_xticklabels(['0–10%', '10–20%', '20–30%'])
ax.set_ylabel('|Gain| (m°C per % opening, log)')
ax.set_title('Plant gain: SP-IV truth vs model\nmodel under-gain ≈ ×50–100', fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
out = 'results/exp_201_valve_action/fig_spiv_vs_model.png'
plt.savefig(out, bbox_inches='tight')
print('saved:', out)
