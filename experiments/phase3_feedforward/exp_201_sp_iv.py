#!/usr/bin/env python3
"""exp_201 SP-IV 增益真值: 用 SP 阶跃事件构造 plant 增益 dT/dV。

工具变量逻辑: SP 阶跃是外生的 (运行人员设定, 不由温度误差驱动),
SP 只通过 PID→阀位→喷水→温度影响 T (排除性)。
  dT/dV_plant = (dT/dSP) / (dV/dSP)
事件级:
  ΔV_init = V(t+30) − V(t)   # SP 阶跃后 30s 的 PID 初始响应 (温度尚未动, 阀位变化由 SP 驱动)
  ΔT_600  = T(t+600) − T(t) − 前趋势外推   # 600s 温度响应
  g_i = ΔT_600 / ΔV_init    # 应为负 (阀关→升温)
分层: 按事件前开度 (等百分比非线性检验) — 与 flow 模型分层增益对比。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_proj = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, 'experiments', 'phase1_dynamics'))

from exp_025_unified_benchmark import data_all, NUMERIC_COLS, TARGET_IDX

raw = data_all
I_V2 = NUMERIC_COLS.index('二级减温调节门阀位')
I_T = TARGET_IDX
n_val_end = 601566

d = np.load('results/cfe_groundtruth_p2/did_response.npz', allow_pickle=True)
onsets = d['onsets18'].astype(int)
dsp = d['dsp18']

rows = []
for t, dsp_i in zip(onsets, dsp):
    if t - 18 < 0 or t + 600 >= len(raw):
        continue
    v0 = raw[t, I_V2]
    # SP 阶跃后的 PID 初始阀位响应 (30s)
    dv_init = raw[t + 3, I_V2] - raw[t, I_V2]
    dv_30 = raw[t + 30, I_V2] - raw[t, I_V2]
    dv_600 = raw[t + 600, I_V2] - raw[t, I_V2]
    # 温度 600s 响应, 减前 180s 趋势外推
    slope_pre = (raw[t, I_T] - raw[t - 18, I_T]) / 18.0
    dT_600 = (raw[t + 600, I_T] - raw[t, I_T]) - slope_pre * 600
    rows.append(dict(t=t, dsp=dsp_i, v0=v0, dv_init=dv_init, dv_30=dv_30,
                     dv_600=dv_600, dT_600=dT_600))

rows = [r for r in rows if r['dsp'] != 0]
print(f'n events: {len(rows)}')
# 方向检查: SP↑ → 阀应关 (dv 与 dsp 反号)
opp = sum(1 for r in rows if r['dv_init'] * r['dsp'] < 0)
print(f'SP↑→阀关 (30s内反号): {opp/len(rows):.1%}')

LAYERS = [(0, 10), (10, 20), (20, 30), (30, 45)]
print(f'\n{"layer":>12} | {"n":>3} | {"dT/dSP":>9} | {"dV30/dSP":>9} | {"dVinit/dSP":>10} | '
      f'{"g_plant=dT/dV30":>15} | {"g_plant=dT/dVinit":>16} | {"model noff":>11}')
for lo, hi in LAYERS:
    sub = [r for r in rows if lo <= r['v0'] < hi]
    if len(sub) < 5:
        print(f'[{lo:2d},{hi:2d}) | n={len(sub)} too few')
        continue
    dsp = np.array([r['dsp'] for r in sub])
    dT = np.array([r['dT_600'] for r in sub])
    dv30 = np.array([r['dv_30'] for r in sub])
    dvinit = np.array([r['dv_init'] for r in sub])
    # 每事件比再平均 (避免大事件主导)
    g30 = np.mean(dT / dv30)
    ginit = np.mean(dT / dvinit)
    # 模型 noff 分层增益 (来自 gain_diag: m°C/%)  — 手动填, 见下方注
    print(f'[{lo:2d},{hi:2d}) | {len(sub):3d} | {np.mean(dT/dsp):8.3f} | '
          f'{np.mean(dv30/dsp):8.3f} | {np.mean(dvinit/dsp):9.4f} | '
          f'{g30*1000:13.2f} m°C/% | {ginit*1000:14.2f} m°C/% | (见下)')

print('\n注: 模型 noff 分层增益 (m°C/%) 来自 exp_201_gain_diag: '
      '0-10:1.01, 10-20:1.57, 20-30:2.30, 30-45:1.39')
print('对照: 期望模型增益与 g_plant 同量级且同为负方向。')
