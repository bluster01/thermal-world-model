#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig for spray ablation (execution-side probe). Recomputes means from summary.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
s = json.loads((HERE / 'full/summary.json').read_text())

seeds = ['seed0', 'seed1', 'seed2']
per_seed = {
    'control': [s['arms']['control'][k]['H18'] for k in seeds],
    'retrain, no future spray': [s['arms']['masked_train'][k]['H18'] for k in seeds],
    'eval-only mask, no retrain': [s['arms']['masked_eval'][k]['H18'] for k in seeds],
}
labels = list(per_seed.keys())
means = [float(np.mean(v)) for v in per_seed.values()]
stds = [float(np.std(v, ddof=1)) for v in per_seed.values()]

fig, ax = plt.subplots(figsize=(4.6, 3.1), dpi=200)
x = np.arange(len(labels))
ax.bar(x, means, yerr=stds, width=0.58, capsize=4,
       color=['#4C72B0', '#55A868', '#C44E52'])
rng = np.random.default_rng(0)
for i, (lab, v) in enumerate(per_seed.items()):
    ax.scatter(i + rng.uniform(-0.09, 0.09, len(v)), v, s=14, color='0.25', zorder=3)
    ax.text(i, means[i] + stds[i] + 0.0035, f'{means[i]:.4f}', ha='center', fontsize=7.5)
ax.axhline(0.4431, ls='--', lw=1.0, color='0.25')
ax.text(len(labels) - 0.55, 0.4458, 'GRU hybrid (reference) 0.4431',
        fontsize=6.8, ha='right', color='0.25')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.2)
ax.set_ylabel('H18 cumulative MAE (degC)', fontsize=8)
ax.set_ylim(0.34, 0.465)
ax.tick_params(axis='y', labelsize=7.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Black-box input ablation: future spray channel\n(256 val windows, 13 days, 3 seeds)',
             fontsize=8.2)
fig.tight_layout()
out = HERE / 'fig_spray_ablation.png'
fig.savefig(out)
print('saved', out)
