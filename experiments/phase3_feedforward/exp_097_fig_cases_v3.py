#!/usr/bin/env python3
"""
exp_097_fig_cases_v3.py — case 图 v3: M9DSP H=60 (因果主线, 2026-08-05)
=========================================================================
v3 改动 (主模型切换 M9DSP H=60, exp_101):
  - 模型: TimeXer 动作 cross-attn (180s 因果方向 89%, exp_101)
  - 预测段 600s 全长 + 灰色带标 180s 监督窗口
  - 同 v2 事件 (659852 大 / 57860 中 / 326938 平稳) → 与 v2 (M5-DSP) 同事件对比
  - MAE 标注: 180s 监督窗 MAE + 600s 末点方向正确性
布局: 主面板 (温度/SP/阀位) + ΔSP 子面板 (同 v2)
导出: SVG/PDF/TIFF(600dpi)/PNG(300dpi)
用法: python exp_097_fig_cases_v3.py
"""
import os, sys
import numpy as np
import torch, torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E.cfg.WINDOW_SIZE
H = 60
E.H_OUT = H
raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V2 = E.NUMERIC_COLS.index('二级减温调节门阀位')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)
I_DSP = 40

# M9DSP (同 exp_101)
class M9DSP(E.TimeXerWM):
    def __init__(self):
        super().__init__(probabilistic=True, beta_mode='fixed')
        d = E.cfg.D_MODEL
        self.act_lin = nn.Linear(H, d)
    def forward(self, x_hist, a_future=None):
        mu, lv = super().forward(x_hist, a_future)
        if lv is not None:
            lv = torch.clamp(lv, -6., 20.)
        return mu, lv

ck = torch.load('results/exp_101_m9dsp_h60/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
model = M9DSP().to(DEVICE).eval()
model.load_state_dict(ck['model_state_dict'])
print(f"[load] M9DSP (exp_101, ep{ck['epoch']})")

def predict(s):
    if s < 0 or s + W + H >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    a = np.diff(raw41[s+W-1:s+W+H, I_DSP])
    a_f = torch.FloatTensor(a).reshape(1, H, 1).to(DEVICE)
    with torch.no_grad():
        mu, _ = model(win, a_f)
    return mu[0].cpu().numpy()

CASES = [
    dict(onset=659852, tag='a', title='Large SP step',  dsp_ref=3.0),
    dict(onset=57860,  tag='b', title='Medium SP step', dsp_ref=3.0),
    dict(onset=326938, tag='c', title='Calm baseline',  dsp_ref=0.0),
]

C_FIELD = '#111111'; C_HIST = '#9aa0a6'; C_DSP_MOD = '#c0392b'
C_SP = '#2e8b57'; C_VALVE = '#d4a017'; C_DSPBAR = '#5b8db8'
C_SUP = '#f2f2f2'

fig = plt.figure(figsize=(7.2, 3.4))
gs = GridSpec(2, 3, height_ratios=[3.2, 1.0], hspace=0.38, wspace=0.55,
              left=0.075, right=0.985, top=0.93, bottom=0.10)

for i, cs in enumerate(CASES):
    o = cs['onset']
    t_hist = np.arange(-W, 0) * 10
    t_pred = np.arange(0, H) * 10
    p = predict(o - W)
    actual = raw[o:o+H, I_T]
    mae18 = np.abs(p[:18] - actual[:18]).mean()
    dir600 = np.sign(p[-1] - raw[o-1, I_T]) == np.sign(actual[-1] - raw[o-1, I_T])

    ax = fig.add_subplot(gs[0, i])
    ax.axvspan(180, 600, color=C_SUP, zorder=0)
    ax.axvline(180, color='#bbbbbb', lw=0.6, ls=':')
    ax.plot(t_hist, raw[o-W:o, I_T], color=C_HIST, lw=0.9, label='History (960 s)')
    ax.plot(t_pred, actual, color=C_FIELD, lw=1.6, label='Field (actual)')
    ax.plot(t_pred, p, color=C_DSP_MOD, lw=1.3, ls='--', label='WM prediction')
    sp_full = np.concatenate([raw[o-W:o, I_SP], raw[o:o+H, I_SP]])
    ax.plot(np.concatenate([t_hist, t_pred]), sp_full, color=C_SP, lw=1.0, ls='-.', alpha=0.9,
            label='SP trajectory')
    ax.axvline(0, color='#888888', lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(t_hist, raw[o-W:o, I_V2], color=C_VALVE, lw=0.7, alpha=0.75)
    ax2.plot(t_pred, raw[o:o+H, I_V2], color=C_VALVE, lw=0.9, alpha=0.9)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelsize=6, colors=C_VALVE)
    ax2.spines['left'].set_visible(False); ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_VALVE)
    for s in ax2.spines.values(): s.set_linewidth(0.6)
    ax.set_xlim(-W*10, H*10)
    ax.set_xticks(np.arange(-900, 601, 180))
    ax.set_xticklabels([])
    ax.set_ylabel('Temp (°C)', fontsize=6.5)
    ax.tick_params(labelsize=6)
    dstr = '✓' if dir600 else '✗'
    ax.text(0.02, 0.04, f'MAE 0-180s {mae18:.2f}°C | 600s dir {dstr}',
            transform=ax.transAxes, fontsize=6, color=C_DSP_MOD)
    ax.set_title(f"{cs['tag']}  {cs['title']} (|ΔSP|={cs['dsp_ref']:.1f}°C)", fontsize=7, loc='left', fontweight='bold')

    axd = fig.add_subplot(gs[1, i])
    dsp_full = np.concatenate([dsp[o-W:o], dsp[o:o+H]])
    axd.bar(np.concatenate([t_hist, t_pred]), dsp_full, width=10, color=C_DSPBAR, alpha=0.85)
    axd.axvline(0, color='#888888', lw=0.7)
    axd.axhline(0, color='black', lw=0.5)
    axd.set_xlim(-W*10, H*10)
    axd.set_xticks(np.arange(-900, 601, 180))
    axd.tick_params(labelsize=6)
    axd.set_ylabel('ΔSP\n(°C)', fontsize=6)
    if i == 1:
        axd.set_xlabel('Time since SP step (s)', fontsize=6.5)
    else:
        axd.set_xticklabels([])

h, l = [], []
h_, l_ = fig.axes[0].get_legend_handles_labels()
h, l = h_, l_
h.append(plt.Line2D([0], [0], color=C_VALVE, lw=1.0, label='Valve (%)'))
h.append(plt.Rectangle((0, 0), 1, 1, fc=C_SUP, ec='none', label='Beyond 180 s window'))
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=6.2, frameon=False)

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig_cases_sandbox_v3.svg', bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox_v3.pdf', bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox_v3.tiff', dpi=600, bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox_v3.png', dpi=300, bbox_inches='tight')
print('Saved: figures/fig_cases_sandbox_v3.{svg,pdf,tiff,png}')
