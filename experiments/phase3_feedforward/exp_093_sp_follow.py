#!/usr/bin/env python3
"""
exp_093_sp_follow.py — SP 阶跃温度跟随率 (历史数据中的"自然阶跃试验", 2026-08-04)
==============================================================================
监督模式关键验证: 现场 SP 阶跃后温度是否跟随到新 SP (闭环杠杆是否=1)?
  - 筛选: |ΔSP|>1°C 大阶跃 + 工况相对稳定 (负荷变化率小, 近似阶跃试验条件)
  - 指标: 跟随率 (300/600s 内 |T−SP|<0.3°C) | 跟随时标 | 稳态偏差 | 超调
  - 输出: 分布表 + 跟随曲线图 (全部英文)
用法: python exp_093_sp_follow.py [--smoke]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase2_mpc'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

SMOKE = '--smoke' in sys.argv
cols = E.NUMERIC_COLS
I_SP  = cols.index('二级减温调节阀设定')
I_T   = cols.index('末级过热器出口汽温')
I_LOAD = cols.index('机组负荷')
I_V2  = cols.index('二级减温调节门阀位')
raw = E.data_all
N = len(raw)

THR = 1.0            # |ΔSP| 阈值 (°C)
GAP = 60             # 事件间隔 (步) — 需留足跟随时间
STABLE = 3.0         # 工况稳定: |Δ负荷| ≤ 3MW/10s (阶跃前后 20 步内)
H_FOLLOW = 60        # 跟随观察窗 (600s)
H_SETTLE = 30        # 稳态窗 (300s)

dsp = np.abs(np.diff(raw[:, I_SP]))
idxs = np.where(dsp > THR)[0] + 1
onsets = []
for i in idxs:
    if not onsets or i - onsets[-1] >= GAP:
        onsets.append(i)
print(f"[events] |ΔSP|>{THR}°C 且间隔≥{GAP}步: {len(onsets)}")

# 稳定工况筛选
stable_ons = []
for o in onsets:
    w0 = max(0, o - 20); w1 = min(N, o + 20)
    dload = np.abs(np.diff(raw[w0:w1, I_LOAD])).max()
    if dload <= STABLE:
        stable_ons.append(o)
print(f"[events] 工况稳定 (Δ负荷≤{STABLE}MW): {len(stable_ons)}")

# 跟随分析
follow300, follow600, settle_err, t63s, overshoot, dsp_v, dt_v = [], [], [], [], [], [], []
trajs = []
for o in stable_ons[:600 if not SMOKE else 40]:
    if o + H_FOLLOW >= N:
        continue
    sp0 = raw[o - 1, I_SP]; sp1 = raw[o, I_SP]
    d = sp1 - sp0
    if abs(d) < 0.5:
        continue
    t0 = raw[o - 1, I_T]
    # 跟随: 300s/600s 后温度与 SP1 的偏差
    e300 = raw[o + 30, I_T] - sp1
    e600 = raw[o + H_FOLLOW, I_T] - sp1
    # 稳态偏差: 300-600s 平均
    settle = np.mean(raw[o + H_SETTLE:o + H_FOLLOW, I_T]) - sp1
    # 时标: 首次 |T−SP1|<0.3°C 的时间
    t63 = None
    for k in range(5, H_FOLLOW):
        if abs(raw[o + k, I_T] - sp1) < 0.3:
            t63 = k * 10; break
    # 超调: 跟随方向的反向最大偏移
    traj = raw[o:o + H_FOLLOW, I_T] - sp1
    ovs = -traj.min() if d > 0 else traj.max()
    follow300.append(1 if abs(e300) < 0.3 else 0)
    follow600.append(1 if abs(e600) < 0.3 else 0)
    settle_err.append(settle); t63s.append(t63); overshoot.append(ovs)
    dsp_v.append(d); dt_v.append(t0 - sp0)
    trajs.append(traj)

follow300 = np.array(follow300); follow600 = np.array(follow600)
settle_err = np.array(settle_err); t63s = np.array([t for t in t63s if t is not None])
overshoot = np.array(overshoot); dsp_v = np.array(dsp_v)
print(f"\n===== SP 阶跃温度跟随 (n={len(follow300)}) =====")
print(f"  跟随率: 300s {follow300.mean()*100:.0f}% | 600s {follow600.mean()*100:.0f}%")
print(f"  稳态偏差 (300-600s, T−SP1): 中位 {np.median(settle_err):+.3f}°C | p25-p75 [{np.percentile(settle_err,25):+.3f}, {np.percentile(settle_err,75):+.3f}]")
print(f"  跟随时标 (|T−SP|<0.3°C, n={len(t63s)}): 中位 {np.median(t63s):.0f}s | p25-p75 [{np.percentile(t63s,25):.0f}, {np.percentile(t63s,75):.0f}]")
print(f"  超调 (跟随方向反向峰值): 中位 {np.median(overshoot):.2f}°C | p90 {np.percentile(overshoot,90):.2f}°C")
# 方向正确率
dir_ok = (np.sign(settle_err) == np.sign(dsp_v)).mean() if len(settle_err) else 0
print(f"  稳态偏差与阶跃同向比例 (杠杆方向): {dir_ok*100:.0f}%")

# 图: 归一化跟随轨迹 (对齐 SP 阶跃, 画 30 条)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
t_ax = np.arange(H_FOLLOW) * 10
for traj in trajs[:30]:
    ax[0].plot(t_ax, traj, lw=0.8, alpha=0.35, color='#4f81bd')
ax[0].axhline(0, color='k', lw=1.0, ls='--', label='New SP')
ax[0].axhline(-0.3, color='gray', lw=0.6, ls=':'); ax[0].axhline(0.3, color='gray', lw=0.6, ls=':')
ax[0].set_title(f'Temp minus new SP after SP step (n={len(trajs)})')
ax[0].set_xlabel('Time since step (s)'); ax[0].set_ylabel('T − SP1 (°C)'); ax[0].legend(fontsize=8)
ax[1].hist(settle_err, bins=30, color='#4f81bd', alpha=0.8)
ax[1].axvline(0, color='k', lw=1.0)
ax[1].set_title('Steady-state error distribution (T − SP1, 300-600s)')
ax[1].set_xlabel('°C')
fig.tight_layout()
fig.savefig('figures/fig_sp_follow.png', dpi=170, bbox_inches='tight')
print('Saved: figures/fig_sp_follow.png')
