#!/usr/bin/env python3
"""画 ff10 best_mae vs best_causal case 图 — 3 事件 × 双 ckpt"""
import os, sys
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

import causal_eval as CE
import causal_arch as CA

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","DejaVu Sans"],
    "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W, H, N_FEAT = E.cfg.WINDOW_SIZE, 60, 40
TARGET_IDX = E.TARGET_IDX
raw = E.data_all
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V2 = E.NUMERIC_COLS.index('二级减温调节门阀位')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40

# 3 case events (同 v2/v3)
CASES = {
    'A: 大幅阶跃': 659852,
    'B: 中幅阶跃': 57860,
    'C: 平稳段':   326938,
}

def load_a1phys(ckpt_path):
    m = CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='phys',
                            cumsum_out=False, probabilistic=True).to(DEVICE).eval()
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd['model_state_dict'])
    return CE.ModelWrapper(m, raw, raw41, W, H, I_DSP, DEVICE)

BASE = 'results/exp_106_causal_arch'
models = [
    ('ff10 best_mae',    load_a1phys(f'{BASE}/A1phys_s0_ff10/checkpoints/best_mae.pth')),
    ('ff10 best_causal', load_a1phys(f'{BASE}/A1phys_s0_ff10/checkpoints/best_causal.pth')),
    ('ff10+lg0.5 best_mae',    load_a1phys(f'{BASE}/A1phys_s0_ff10_lg0.5/checkpoints/best_mae.pth')),
    ('ff10+lg0.5 best_causal', load_a1phys(f'{BASE}/A1phys_s0_ff10_lg0.5/checkpoints/best_causal.pth')),
    ('baseline best_mae',load_a1phys(f'{BASE}/A1phys_s0/checkpoints/best_mae.pth')),
]
colors = {
    'ff10 best_mae': '#2196F3', 'ff10 best_causal': '#FF5722',
    'ff10+lg0.5 best_mae': '#009688', 'ff10+lg0.5 best_causal': '#E91E63',
    'baseline best_mae': '#4CAF50',
}
# Plot
fig, axes = plt.subplots(3, len(models), figsize=(5*len(models), 7), sharex='col', sharey='row')

for ci, (case_name, onset) in enumerate(CASES.items()):
    start = max(0, onset - W - 200)
    end = min(len(raw), onset + H + 300)
    t = np.arange(end - start)
    t_W = np.arange(W + H)
    
    # Ground truth
    T_gt = raw[start:end, I_T]
    SP_gt = raw[start:end, I_SP]
    valve_gt = raw[start:end, I_V2]
    onset_rel = onset - start
    W_rel = onset_rel - W
    
    for mi, (name, wrap) in enumerate(models):
        ax = axes[ci, mi]
        ax.plot(t, T_gt, 'k-', lw=0.8, alpha=0.5, label='true')
        
        # Prediction from onset-W
        pred_do = wrap.predict(onset - W)       # with action
        pred_0  = wrap.predict(onset - W, 0.0)   # zero action (counterfactual)
        
        t_pred = np.arange(onset, onset + H) - start
        ax.plot(t_pred, pred_do, color=colors[name], lw=1.5, label='do(a)')
        ax.plot(t_pred, pred_0, color=colors[name], lw=1.0, ls='--', alpha=0.6, label='do(0)')
        
        # Fill: counterfactual difference
        diff = pred_do - pred_0
        ax.fill_between(t_pred, pred_0, pred_do, color=colors[name], alpha=0.15)
        
        # SP onset marker
        ax.axvline(onset_rel, color='gray', ls=':', lw=0.6)
        
        # MAE annotation
        mae_do = np.abs(pred_do - raw[onset:onset+H, I_T]).mean()
        mae_0  = np.abs(pred_0 - raw[onset:onset+H, I_T]).mean()
        gain = diff.mean()
        ax.text(0.02, 0.98, f'MAE do={mae_do:.3f}\ngain={gain:+.3f}', 
                transform=ax.transAxes, fontsize=5.5, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if ci == 0:
            ax.set_title(name, fontsize=8, fontweight='bold', color=colors[name])
        if mi == 0:
            ax.set_ylabel(f'{case_name}\nT [°C]', fontsize=7)
        ax.set_xlim(0, len(t))

fig.tight_layout()
outdir = 'figures/ff10_cases'
os.makedirs(outdir, exist_ok=True)
for fmt in ['png', 'svg']:
    fig.savefig(f'{outdir}/ff10_comparison.{fmt}', dpi=200, bbox_inches='tight')
print(f"Saved: {outdir}/ff10_comparison.{{png,svg}}")
