#!/usr/bin/env python3
"""
exp_097_fig_cases.py — 正式版 case 图 (nature-figure 规范, Python backend)
==========================================================================
3 面板: (a) 大动作 |ΔSP|>3 (沙盒价值) | (b) 中动作 2-3 | (c) 平稳对照
每面板: 历史温度 + 现场实际 + M5-DSP 沙盒预测 (+M5 对照) + SP 水平 + 阀位 (右轴)
case 选择: 层内方向正确且 MAE 最接近层中位 (代表性, 非最优)
导出: SVG/PDF/TIFF(600dpi)/PNG(预览) — Applied Energy 单栏
用法: python exp_097_fig_cases.py
"""
import os, sys
import numpy as np
import torch, torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt

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
I_V1 = E.NUMERIC_COLS.index('一级减温调节门阀位')
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

ck_m5 = torch.load('results/exp_025_M5/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
m5 = E.build_model('M5').to(DEVICE).eval()
m5.load_state_dict(ck_m5['model_state_dict'])
ck_dsp = torch.load('results/exp_096_dsp_wm/checkpoints/best_model.pth', map_location=DEVICE, weights_only=True)
m5dsp = M5DSP().to(DEVICE).eval()
m5dsp.load_state_dict(ck_dsp['model_state_dict'])
print('[load] M5 + M5-DSP OK')

# ===== 事件筛选 (同 exp_097) =====
dsp_abs = np.abs(np.diff(raw[:, I_SP]))
onsets = []
for i in np.where(dsp_abs > 1.0)[0] + 1:
    if not onsets or i - onsets[-1] >= 60:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and
          np.abs(np.diff(raw[max(0, o-20):min(N, o+20), I_LD])).max() <= 3.0]
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.3]
print(f"[events] {len(kept)}")

def predict(model, s, kind):
    if s < 0 or s + W + H_OUT >= N:
        return None
    win = torch.FloatTensor(raw[s:s+W]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if kind == 'm5':
            a = raw[s+W:s+W+H_OUT, I_V1:I_V2+1]  # 一二级阀位
            a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
            mu, _ = model(win, a_f)
        else:
            a = np.diff(raw41[s+W-1:s+W+H_OUT, 40])
            a_f = torch.FloatTensor(a).reshape(1, -1).to(DEVICE)
            mu, _ = model(win, a_f)
    return mu[0].cpu().numpy()

# ===== 分层选 case: 方向正确 + MAE 最接近层中位 =====
def pick_case(pool):
    best = []
    for o in pool:
        p_dsp = predict(m5dsp, o - W, 'dsp')
        p_m5 = predict(m5, o - W, 'm5')
        if p_dsp is None or p_m5 is None:
            continue
        actual = raw[o:o+H_OUT, I_T]
        prev_T = raw[o-1, I_T]
        mae_dsp = np.abs(p_dsp - actual).mean()
        mae_m5 = np.abs(p_m5 - actual).mean()
        d_act = actual[-1] - prev_T
        d_pred = p_dsp[-1] - prev_T
        if np.sign(d_pred) != np.sign(d_act) or abs(d_act) < 0.05:
            continue
        best.append((mae_dsp, o, mae_m5))
    best.sort(key=lambda x: x[0])
    med = best[len(best)//2]
    return med, best

large = [o for o in kept if abs(raw[o, I_SP] - raw[o-1, I_SP]) > 3]
mid   = [o for o in kept if 2 < abs(raw[o, I_SP] - raw[o-1, I_SP]) <= 3]
small = [o for o in kept if 1 < abs(raw[o, I_SP] - raw[o-1, I_SP]) <= 2]
rng = np.random.default_rng(42)
calm = []
for _ in range(200):
    while True:
        c = int(rng.integers(W + 60, N - 60))
        if np.abs(np.diff(raw[c-20:c+20, I_SP])).max() <= 0.15 and c not in kept:
            calm.append(c); break

picks = {}
for name, pool in (('large', large), ('mid', mid), ('calm', calm)):
    _, cands = pick_case(pool)
    o = cands[len(cands)//2][1] if name != 'calm' else cands[len(cands)//2][1]
    # 重新取该事件的预测/指标
    p_dsp = predict(m5dsp, o - W, 'dsp'); p_m5 = predict(m5, o - W, 'm5')
    actual = raw[o:o+H_OUT, I_T]
    mae_dsp = np.abs(p_dsp - actual).mean(); mae_m5 = np.abs(p_m5 - actual).mean()
    picks[name] = dict(o=o, p_dsp=p_dsp, p_m5=p_m5,
                       mae_dsp=mae_dsp, mae_m5=mae_m5,
                       dsp=raw[o, I_SP] - raw[o-1, I_SP])
    print(f"[pick] {name:5s} onset={o} |ΔSP|={picks[name]['dsp']:.2f}°C | MAE M5-DSP {mae_dsp:.3f} vs M5 {mae_m5:.3f}")

# ===== 绘图 =====
C_FIELD = '#111111'; C_HIST = '#9aa0a6'; C_DSP = '#c0392b'; C_M5 = '#2c6fbb'; C_SP = '#2e8b57'; C_VALVE = '#d4a017'
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
for ax, (name, meta) in zip(axes, picks.items()):
    o = meta['o']
    t_hist = np.arange(-W, 0) * 10
    t_pred = np.arange(0, H_OUT) * 10
    ax.plot(t_hist, raw[o-W:o, I_T], color=C_HIST, lw=0.9, label='History (960 s)')
    ax.plot(t_pred, raw[o:o+H_OUT, I_T], color=C_FIELD, lw=1.6, label='Field (actual)')
    ax.plot(t_pred, meta['p_dsp'], color=C_DSP, lw=1.3, ls='--', label='Sandbox (M5-DSP)')
    ax.plot(t_pred, meta['p_m5'], color=C_M5, lw=1.1, ls=':', label='Sandbox (M5)')
    sp_val = raw[o, I_SP]
    ax.plot([-W*10, H_OUT*10], [sp_val, sp_val], color=C_SP, lw=0.8, ls='-.', label=f'SP ({sp_val:.1f}°C)')
    ax.axvline(0, color='#888888', lw=0.7)
    ax2 = ax.twinx()
    ax2.plot(t_hist, raw[o-W:o, I_V2], color=C_VALVE, lw=0.7, alpha=0.75)
    ax2.plot(t_pred, raw[o:o+H_OUT, I_V2], color=C_VALVE, lw=0.9, alpha=0.9)
    ax2.set_ylim(0, 100); ax2.set_ylabel('Valve (%)', fontsize=6.5, color=C_VALVE)
    ax2.tick_params(axis='y', labelsize=6, colors=C_VALVE)
    ax2.spines['left'].set_visible(False); ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_VALVE)
    for s in ax2.spines.values(): s.set_linewidth(0.6)
    ax.set_xlim(-W*10, H_OUT*10)
    ax.set_xlabel('Time since SP step (s)', fontsize=6.5)
    ax.set_ylabel('Outlet temp (°C)', fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.text(0.02, 0.03, f'MAE {meta["mae_dsp"]:.2f}°C', transform=ax.transAxes,
            fontsize=6, color=C_DSP)
    if name == 'large':
        lab = f'a  Large SP step (|ΔSP|={abs(meta["dsp"]):.1f}°C)'
    elif name == 'mid':
        lab = f'b  Medium SP step (|ΔSP|={abs(meta["dsp"]):.1f}°C)'
    else:
        lab = 'c  Calm baseline'
    ax.set_title(lab, fontsize=7, loc='left', fontweight='bold')
axes[0].legend(fontsize=5.5, loc='upper left', ncol=1)
fig.tight_layout(w_pad=1.2)
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig_cases_sandbox.svg', bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox.pdf', bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox.tiff', dpi=600, bbox_inches='tight')
fig.savefig('figures/fig_cases_sandbox.png', dpi=300, bbox_inches='tight')
print('Saved: figures/fig_cases_sandbox.{svg,pdf,tiff,png}')
