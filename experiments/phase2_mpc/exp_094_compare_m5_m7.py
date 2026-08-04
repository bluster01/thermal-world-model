#!/usr/bin/env python3
"""
exp_094_compare_m5_m7.py — DWM(M7,概率) vs 确定性M5 差距可视化 (2026-08-04)
============================================================================
主表同口径下 M5 扰动 2.105 vs M7 2.443 (−14%), 无扰动 0.781 vs 0.972 (−20%)。
本脚本: 1) per-track 配对散点+差值分布 (150 轨迹)  2) 典型 track 温度轨迹对比
用法: python exp_094_compare_m5_m7.py
"""
import os, sys
import numpy as np
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase2_mpc'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_027_dwm_mpc.py']
import exp_027_dwm_mpc as M
sys.argv = _argv

# ===== 配置 (同 exp_086 最终协议) =====
M.M_STEP = 6; M.H_PLAN = 18
M.FIX_MODE = 'overlap'; M.LAMBDA3 = 0.05; M.HARD_DELTA = 5.0
M.BENCH_SP_EACH = True
M.LAMBDA1 = 0.0; M.LAMBDA2 = 0.0; M.LAMBDA1_2ND = 0.0
M.EXEC_KF = 0.01; M.EXEC_SMA = 6
N_TRACKS = 50; SEEDS = [42, 7, 13]

# ===== 加载模型 =====
print('[load] M5 (确定性) ...')
wm5 = M.build_model('M5').to(M.DEVICE).eval()
ck5 = torch.load('results/exp_025_M5/checkpoints/best_model.pth', map_location=M.DEVICE, weights_only=True)
wm5.load_state_dict(ck5['model_state_dict'])
print('[load] M7 (概率) ...')
wm7 = M.load_wm(); wm7.eval()

# ===== 1. per-track 配对分析 =====
m7 = json.load(open('results/exp_086_final_main/ovl05_hard5_kf_sma6_dist.json'))
m5 = json.load(open('results/exp_086_maintable_m5/dist.json'))
rm7 = np.array([r['rmse_mpc'] for r in m7]); rm5 = np.array([r['rmse_mpc'] for r in m5])
rp7 = np.array([r['rmse_pid'] for r in m7]); rp5 = np.array([r['rmse_pid'] for r in m5])
d = rm7 - rm5  # >0: M5 更好
print(f"[stat] 扰动 M7 {rm7.mean():.3f} vs M5 {rm5.mean():.3f} (Δ={d.mean():+.3f})")
print(f"       M5 更优的 track: {(d > 0.05).sum()}/150 | M7 更优: {(d < -0.05).sum()}/150 | 持平: {((np.abs(d)<=0.05)).sum()}/150")
print(f"       Wilcoxon M7 vs M5: p={stats.wilcoxon(rm7, rm5).pvalue:.2e}")
print(f"       差值分位: p10 {np.percentile(d,10):+.3f} | p50 {np.percentile(d,50):+.3f} | p90 {np.percentile(d,90):+.3f}")

# ===== 2. 典型 track 反推 (seed, start) =====
def track_map():
    """json 顺序 → (seed, start) 列表"""
    out = []
    for seed in SEEDS:
        np.random.seed(seed)
        starts = np.random.choice(range(len(M.test_raw) - M.W - M.H_OUT - 120), N_TRACKS, replace=False)
        for s in starts:
            out.append((seed, s))
    return out
tmap = track_map()
# 差距最大/中位/最小
idx_worst = int(np.argmax(d)); idx_med = int(np.argsort(d)[len(d)//2]); idx_best = int(np.argmin(d))

# ===== 3. 重跑典型 track 拿轨迹 =====
def run_track(wm, seed, start):
    mpc_t, pid_t, tset, mpc_a, pid_a = M.simulate(wm, start, 'grad', n_steps=120, seed=seed)
    return np.array(mpc_t), np.array(pid_t), np.array(tset), np.array(mpc_a), np.array(pid_a)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
# (a) 散点
ax = fig.add_subplot(gs[0, 0])
ax.scatter(rm5, rm7, s=14, alpha=0.6, color='#4f81bd')
lim = [min(rm5.min(), rm7.min()) - 0.2, max(rm5.max(), rm7.max()) + 0.2]
ax.plot(lim, lim, 'k--', lw=1.0, label='Equal')
ax.plot(lim, [x - 0.5 for x in lim], 'r:', lw=0.8, label='M5 −0.5°C')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Deterministic M5 per-track RMSE (°C)')
ax.set_ylabel('Probabilistic M7 per-track RMSE (°C)')
ax.set_title(f'(a) Per-track RMSE (disturbed, n=150): M5 {rm5.mean():.3f} vs M7 {rm7.mean():.3f}')
ax.legend(fontsize=8)
# (b) 差值直方图
ax = fig.add_subplot(gs[0, 1])
ax.hist(d, bins=25, color='#4f81bd', alpha=0.85)
ax.axvline(0, color='k', lw=1.0)
ax.axvline(d.mean(), color='r', ls='--', lw=1.0, label=f'mean {d.mean():+.3f}')
ax.set_xlabel('RMSE diff (M7 − M5, °C); >0 = M5 better')
ax.set_ylabel('Count'); ax.set_title('(b) Difference distribution'); ax.legend(fontsize=8)
# (c) 典型轨迹
ax = fig.add_subplot(gs[1, :])
colors = {'M7': '#c0504d', 'M5': '#4f81bd', 'PID': '#888888'}
for tag, idx, lbl in [('worst', idx_worst, 'largest gap (M5 better)'),
                      ('median', idx_med, 'median gap'),
                      ('best', idx_best, 'largest gap (M7 better)')]:
    seed, start = tmap[idx]
    t7, p7, sp7, a7, _ = run_track(wm7, seed, start)
    t5, p5, sp5, a5, _ = run_track(wm5, seed, start)
    t_ax = np.arange(len(t7)) * 10
    off = {'worst': 0.0, 'median': -3.0, 'best': -6.0}[tag]
    ax.plot(t_ax, t7 + off, color=colors['M7'], lw=1.1, label=f'M7 {lbl}' if tag == 'worst' else None)
    ax.plot(t_ax, t5 + off, color=colors['M5'], lw=1.1, label=f'M5 {lbl}' if tag == 'worst' else None)
    ax.plot(t_ax, sp7 + off, '--', color='gray', lw=0.8)
    ax.text(0.02, 0.97 - {'worst': 0, 'median': 0.10, 'best': 0.20}[tag], lbl,
            transform=ax.transAxes, fontsize=8, va='top')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Outlet temp (°C, offset for clarity)')
ax.set_title('(c) Typical disturbed tracks: M7 vs M5 (offset −3°C per panel)')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig('figures/fig_m5_vs_m7_gap.png', dpi=170, bbox_inches='tight')
print('\nSaved: figures/fig_m5_vs_m7_gap.png')
