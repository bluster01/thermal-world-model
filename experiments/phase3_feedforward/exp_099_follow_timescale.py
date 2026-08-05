#!/usr/bin/env python3
"""
exp_099_follow_timescale.py — SP 跟随时标 vs 180s 预测窗口 (2026-08-05)
========================================================================
用户质疑: exp_093 显示真实跟随 600s 后才 98%, 预测窗口只有 180s —
180s 内 ΔSP 的物理因果响应可能本就很小, 模型 0.06°C 响应未必是缺陷。
输出: 归一化跟随轨迹 (T−SP1) 在 60/120/180/240/300/420/600s 的响应比例分布,
      180s 时平均响应比例 = 物理基准修正值。
用法: python exp_099_follow_timescale.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')

THR, GAP, STABLE = 1.0, 60, 3.0
H_FOLLOW = 60   # 600s

dsp = np.abs(np.diff(raw[:, I_SP]))
idxs = np.where(dsp > THR)[0] + 1
onsets = []
for i in idxs:
    if not onsets or i - onsets[-1] >= GAP:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and
          np.abs(np.diff(raw[max(0, o-20):min(N, o+20), I_LD])).max() <= STABLE]
# 保持 SP (同 exp_093: 61 步内 |ΔSP|≤0.5)
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.5]
print(f"[events] {len(kept)} (exp_093 协议)")

# 归一化轨迹: 温度相对新 SP 的偏差 (T−SP1), 及相对阶跃的归一化响应
# resp_frac[k] = (T[k] − T0) / ΔSP  — 1.0 = 完全跟随, 0 = 未动
trajs, fracs = [], []
for o in kept:
    sp0, sp1 = raw[o-1, I_SP], raw[o, I_SP]
    d = sp1 - sp0
    if abs(d) < 0.5:
        continue
    T0 = raw[o-1, I_T]
    T = raw[o:o+H_FOLLOW, I_T]
    trajs.append(T - sp1)
    fracs.append((T - T0) / d)   # 响应比例 (含方向)
trajs = np.array(trajs); fracs = np.array(fracs)
print(f"[traj] n={len(fracs)}")

print("\n===== 响应比例 (温度变化/ΔSP) 随时间 ===== (1.0=完全跟随)")
for k, lab in ((3, '30s'), (6, '60s'), (12, '120s'), (18, '180s'), (24, '240s'),
               (30, '300s'), (42, '420s'), (60, '600s')):
    if k >= fracs.shape[1]:
        continue
    f = fracs[:, k]
    med = np.median(f)
    ok = (np.sign(f) == np.sign(1)).mean() * 100
    print(f"  {lab:5s} | 响应比例 中位 {med:+.3f} | 均值 {f.mean():+.3f} | 同向比例 {ok:.0f}% | "
          f"|f|>0.3 比例 {(np.abs(f)>0.3).mean()*100:.0f}%")
print(f"  600s 响应比例 中位 {np.median(fracs[:,59]):+.3f}")

print("\n===== 180s 时响应比例分布 (直方简表) =====")
f180 = fracs[:, 18]
bins = [-10, -0.8, -0.3, 0.3, 0.8, 10]
h, _ = np.histogram(f180, bins=bins)
for b, c in zip(bins[:-1], h):
    print(f"  ({b:+.1f}, {bins[bins.index(b)+1]:+.1f}): {c} 事件")

print(f"\n  180s 响应比例中位 {np.median(f180):+.3f} | 均值 {f180.mean():+.3f}")
print(f"  推论: ΔSP=2.07°C 时 180s 物理响应 ≈ {np.median(f180)*2.07:+.2f}°C (中位口径)")

# 300s vs 600s 跟随率 (exp_093 口径)
follow300 = (np.abs(trajs[:, 30]) < 0.3).mean() * 100
follow600 = (np.abs(trajs[:, 60]) < 0.3).mean() * 100
print(f"\n  跟随率 (|T−SP1|<0.3): 300s {follow300:.0f}% | 600s {follow600:.0f}%")
