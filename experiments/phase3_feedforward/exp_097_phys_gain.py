#!/usr/bin/env python3
"""
exp_097_phys_gain.py — ΔSP→温度 物理增益事件研究 (2026-08-05)
================================================================
问题: M5-DSP 对 ΔSP 动作响应仅 0.05°C, 是模型没学到还是物理效应本就小?
方法 (差分事件研究, 控制共因):
  1. 134 个 SP 阶跃事件: ΔSP vs 180s 温度净变化 (含共因, 上界)
  2. 差分: 阶跃后 180s 温度斜率 vs 阶跃前 180s 斜率 (ΔT_after−ΔT_before)
     — 扣除趋势延续, 近似 ΔSP 的独立贡献
  3. 分层 (大/中/小) + 回归斜率
  4. 对照: 平稳事件的相同差分 (零假设分布)
用法: python exp_097_phys_gain.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

W = E.cfg.WINDOW_SIZE
H_OUT = E.H_OUT
raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')

dsp_abs = np.abs(np.diff(raw[:, I_SP]))
onsets = []
for i in np.where(dsp_abs > 1.0)[0] + 1:
    if not onsets or i - onsets[-1] >= 60:
        onsets.append(i)
stable = [o for o in onsets if o + 60 < N and
          np.abs(np.diff(raw[max(0, o-20):min(N, o+20), I_LD])).max() <= 3.0]
kept = [o for o in stable if np.abs(raw[o:o+61, I_SP] - raw[o, I_SP]).max() <= 0.5]
print(f"[events] {len(kept)} (SP保持阈值放宽到0.5以增加样本)")

rng = np.random.default_rng(42)
calm = []
for _ in range(len(kept)):
    while True:
        c = int(rng.integers(W + 60, N - 60))
        if np.abs(np.diff(raw[c-20:c+20, I_SP])).max() <= 0.15 and c not in kept:
            calm.append(c); break

def slope(T):  # 线性斜率 °C/步 (10s)
    x = np.arange(len(T), dtype=float)
    return np.polyfit(x, T, 1)[0]

print("\n===== ΔSP 阶跃前后 180s 温度变化 (差分事件研究) =====")
print(f"{'组':10s} {'n':>4s} {'ΔSP':>7s} {'ΔT净变化':>9s} {'ΔT斜率差':>9s} {'响应/ΔSP':>9s}")
all_ret = {}
for name, pool in (('全部', kept), ('大|ΔSP|>3', [o for o in kept if abs(raw[o,I_SP]-raw[o-1,I_SP])>3]),
                   ('中2-3', [o for o in kept if 2<abs(raw[o,I_SP]-raw[o-1,I_SP])<=3]),
                   ('小1-2', [o for o in kept if 1<abs(raw[o,I_SP]-raw[o-1,I_SP])<=2]),
                   ('平稳对照', calm)):
    dsp_v, dt_net, dt_diff, resp = [], [], [], []
    for o in pool:
        ds = raw[o, I_SP] - raw[o-1, I_SP]
        if abs(ds) < 0.05 and name != '平稳对照':
            continue
        if o - H_OUT < 0 or o + H_OUT >= N:
            continue
        T_before = raw[o-H_OUT:o, I_T]
        T_after  = raw[o:o+H_OUT, I_T]
        dt_net.append(T_after[-1] - T_before[-1])       # 净变化 (含共因)
        dt_diff.append((T_after[-1]-T_after[0]) - (T_before[-1]-T_before[0]))  # 差分
        dsp_v.append(ds)
    dsp_v, dt_net, dt_diff = map(np.array, (dsp_v, dt_net, dt_diff))
    gain = float('nan')
    if len(dsp_v) > 5 and dsp_v.max() - dsp_v.min() > 1e-9:
        gain = np.polyfit(dsp_v, dt_diff, 1)[0]
    all_ret[name] = dict(dsp=dsp_v, net=dt_net, diff=dt_diff)
    print(f"{name:10s} {len(dsp_v):4d} {np.abs(dsp_v).mean():7.2f} {dt_net.mean():+9.3f} {dt_diff.mean():+9.3f} {gain:+9.4f}")

print("\n===== 斜率分析 (180s 窗口) =====")
for name, pool in (('全部阶跃', kept), ('大', [o for o in kept if abs(raw[o,I_SP]-raw[o-1,I_SP])>3]),
                   ('中', [o for o in kept if 2<abs(raw[o,I_SP]-raw[o-1,I_SP])<=3]),
                   ('小', [o for o in kept if 1<abs(raw[o,I_SP]-raw[o-1,I_SP])<=2]),
                   ('平稳', calm)):
    s_b, s_a = [], []
    for o in pool:
        s_b.append(slope(raw[o-H_OUT:o, I_T]))
        s_a.append(slope(raw[o:o+H_OUT, I_T]))
    s_b, s_a = np.array(s_b), np.array(s_a)
    print(f"  {name:6s} 阶跃前斜率 {s_b.mean():+.4f}°C/步 | 阶跃后 {s_a.mean():+.4f} | 差 {s_a.mean()-s_b.mean():+.4f}")

# 与模型响应对比
# ===== 模型响应 vs 物理响应对比 =====
print("\n===== 符号分析: ΔSP>0 vs ΔSP<0 事件 (共因 vs 因果) =====")
for name, pool in (('全部', kept), ('大|ΔSP|>3', [o for o in kept if abs(raw[o,I_SP]-raw[o-1,I_SP])>3])):
    for sgn, lab in ((1, 'ΔSP>0 (上调)'), (-1, 'ΔSP<0 (下调)')):
        dsp_v, dt_diff = [], []
        for o in pool:
            ds = raw[o, I_SP] - raw[o-1, I_SP]
            if np.sign(ds) != sgn or abs(ds) < 0.05:
                continue
            T_before = raw[o-H_OUT:o, I_T]; T_after = raw[o:o+H_OUT, I_T]
            dsp_v.append(ds)
            dt_diff.append((T_after[-1]-T_after[0]) - (T_before[-1]-T_before[0]))
        dsp_v, dt_diff = np.array(dsp_v), np.array(dt_diff)
        if len(dsp_v) > 3:
            print(f"  {name} {lab:10s} n={len(dsp_v):3d} | ΔSP {dsp_v.mean():+.2f} | 差分ΔT {dt_diff.mean():+.3f} | 增益 {np.polyfit(dsp_v, dt_diff, 1)[0]:+.4f}°C/°C")

phys = all_ret['全部']
print(f"  物理: ΔSP 平均 {np.abs(phys['dsp']).mean():.2f}°C → 差分温度变化 {phys['diff'].mean():+.3f}°C (增益 {np.polyfit(phys['dsp'], phys['diff'], 1)[0]:+.4f}°C/°C)")
print(f"  模型: M5-DSP 平均响应 0.0506°C (诊断1) → 增益 ~{0.0506/3.0:.4f}°C/°C (假设ΔSP≈3)")
