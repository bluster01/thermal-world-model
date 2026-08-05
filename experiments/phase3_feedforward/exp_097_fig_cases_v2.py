#!/usr/bin/env python3
"""
exp_097_fig_cases_v2.py — case 图 v2 (预测器+误差补偿叙事, 含 ΔSP 轨迹)
=========================================================================
v2 改动 (用户反馈: ①没看到 ΔSP ②趋势对不上):
  - 每面板底部加 ΔSP 柱状子面板 (阶跃尖峰直接可见)
  - 主面板加 SP 绝对轨迹 (绿点划线, 历史960s+预测段, 阶梯形态)
  - 叙事: 预测器=状态轨迹预测 (M5-DSP vs 实际, 180s)
3 面板: (a) 大动作 |ΔSP|>3 | (b) 中动作 2-3 | (c) 平稳对照
事件: 同 v1 (层内方向正确+MAE接近中位: onset 659852 / 57860 / 326938)
导出: SVG/PDF/TIFF(600dpi)/PNG(300dpi)
用法: python exp_097_fig_cases_v2.py
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
H_OUT = E.H_OUT
raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_V2 = E.NUMERIC_COLS.index('二级减温调节门阀位')
I_LD = E.NUMERIC_COLS.index('机组负荷')
dsp = np.diff(raw[:, I_SP], prepend=raw[0, I_SP])
raw41 = np.concatenate([raw, dsp[:, None]], 1)

class M5DSP(E.DirectWM):
    def __init__(self):
        super().__init__(use_action=True, use_patch=True, per_variable=True,
                         use_varattn=True, probabilistic=False)
        d = E.cfg.D_MODEL
        self.action_enc = nn.Sequential(
            nn.Linear(H_OUT, d * 2), nn.GELU(), nn.Dropout(E.cfg.DROPOUT))

ck = torch.load('results/exp_096_dsp_wm/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
model = M5DSP().to(DEVICE).eval()
model.load_state_dict(ck['model_state_dict'])
print(f"[load] M5-DSP (exp_096, ep{ck['epoch']})")

def predict(s):
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    a = np.diff(raw41[s+W-1:s+W+H_OUT, 40])
    a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
    with torch.no_grad():
        mu, _ = model(win, a_f)
    return mu[0].cpu().numpy()

# 事件: 同 v1 (用户已审过的代表性事件)
CASES = [
    dict(onset=659852, tag='a', title='Large SP step',  dsp_ref=3.0),
    dict(onset=57860,  tag='b', title='Medium SP step', dsp_ref=3.0),
    dict(onset=326938, tag='c', title='Calm baseline',  dsp_ref=0.0),
]

C_FIELD = '#111111'; C_HIST = '#9aa0a6'; C_DSP_MOD = '#c0392b'
C_SP = '#2e8b57'; C_VALVE = '#d4a017'; C_DSPBAR = '#5b8db8'

fig = plt.figure(figsize=(7.2, 3.4))
gs = GridSpec(2, 3, height_ratios=[3.2, 1.0], hspace=0.38, wspace=0.55,
              left=0.075, right=0.985, top=0.93, bottom=0.10)

for i, cs in enumerate(CASES):
    o = cs['onset']
    t_hist = np.arange(-W, 0) * 10
    t_pred = np.arange(0, H_OUT) * 10
    p = predict(o - W)
    actual = raw[o:o+H_OUT, I_T]
    mae = np.abs(p - actual).mean()

    # ---- 主面板 ----
    ax = fig.add_subplot(gs[0, i])
    ax.plot(t_hist, raw[o-W:o, I_T], color=C_HIST, lw=0.9, label='History (960 s)')
    ax.plot(t_pred, actual, color=C_FIELD, lw=1.6, label='Field (actual)')
    ax.plot(t_pred, p, color=C_DSP_MOD, lw=1.3, ls='--', label='WM prediction')
    # SP 绝对轨迹 (历史 + 预测段实际)
    sp_full = np.concatenate([raw[o-W:o, I_SP], raw[o:o+H_OUT, I_SP]])
    ax.plot(np.concatenate([t_hist, t_pred]), sp_full, color=C_SP, lw=1.0, ls='-.', alpha=0.9,
            label='SP trajectory')
    ax.axvline(0, color='#888888', lw=0.7)
    # 阀位右轴
    ax2 = ax.twinx()
    ax2.plot(t_hist, raw[o-W:o, I_V2], color=C_VALVE, lw=0.7, alpha=0.75)
    ax2.plot(t_pred, raw[o:o+H_OUT, I_V2], color=C_VALVE, lw=0.9, alpha=0.9)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelsize=6, colors=C_VALVE)
    ax2.spines['left'].set_visible(False); ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_VALVE)
    for s in ax2.spines.values(): s.set_linewidth(0.6)
    ax.set_xlim(-W*10, H_OUT*10)
    ax.set_xticks(np.arange(-900, 181, 180))
    ax.set_xticklabels([])
    ax.set_ylabel('Temp (°C)', fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.text(0.02, 0.04, f'MAE {mae:.2f}°C (180 s)', transform=ax.transAxes, fontsize=6, color=C_DSP_MOD)
    ax.set_title(f"{cs['tag']}  {cs['title']} (|ΔSP|={cs['dsp_ref']:.1f}°C)", fontsize=7, loc='left', fontweight='bold')

    # ---- ΔSP 子面板 ----
    axd = fig.add_subplot(gs[1, i])
    dsp_full = np.concatenate([dsp[o-W:o], dsp[o:o+H_OUT]])
    axd.bar(np.concatenate([t_hist, t_pred]), dsp_full, width=10, color=C_DSPBAR, alpha=0.85)
    axd.axvline(0, color='#888888', lw=0.7)
    axd.axhline(0, color='black', lw=0.5)
    axd.set_xlim(-W*10, H_OUT*10)
    axd.set_xticks(np.arange(-900, 181, 180))
    axd.tick_params(labelsize=6)
    axd.set_ylabel('ΔSP\n(°C)', fontsize=6)
    if i == 1:
        axd.set_xlabel('Time since SP step (s)', fontsize=6.5)
    else:
        axd.set_xticklabels([])

# 合并图例 (从第一面板收集)
h, l = [], []
for ax_ in [fig.axes[0]]:
    h_, l_ = ax_.get_legend_handles_labels()
    h, l = h_, l_
h.append(plt.Line2D([0], [0], color=C_VALVE, lw=1.0, label='Valve (%)'))
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=6.2, frameon=False)

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig_cases_sandbox_v2.svg', bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox_v2.pdf', bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox_v2.tiff', dpi=600, bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox_v2.png', dpi=300, bbox_inches='tight')
print('Saved: figures/fig_cases_sandbox_v2.{svg,pdf,tiff,png}')
