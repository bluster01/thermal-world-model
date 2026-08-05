#!/usr/bin/env python3
"""
exp_109_p2_expand.py — P2: 放宽事件筛选 + 扩样本 DiD 真值 (2026-08-05)
=====================================================================
P1 review 发现 n_ev=15 是致命短板。P2 对策:
  - THR_DSP 1.0→0.6   (纳入中小幅阶跃, ΔSP 幅度作分层变量)
  - SP_HOLD 0.3→0.5    (放宽 SP 保持要求)
  - LOAD_STABLE 3.0→5.0 (放宽负荷稳定要求)
  - val+test 合并做评测集 (lo=n_train)
目标: n_ev >= 60
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

import causal_eval as CE

OUT = 'results/cfe_groundtruth_p2'
os.makedirs(OUT, exist_ok=True)

raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T  = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')
W = E.cfg.WINDOW_SIZE
n_train, n_val_end = 495407, 601566

# P2 放宽参数
THR = 0.6; SP_H = 0.5; LD_S = 5.0

print(f"[P2] THR_DSP={THR} SP_HOLD={SP_H} LOAD_STABLE={LD_S}")
print(f"[P2] val+test range: [{n_train}, {N})")

# ===== Part 1: 处理组 (val+test)
print("\n" + "=" * 70)
print("Part 1 — 处理组: SP 阶跃事件 (val+test, 放宽筛选)")
print("=" * 70)

events60, dsp60 = CE.select_events(raw, I_SP, I_LD, H=60,
                                    thr=THR, sp_hold=SP_H, load_stable=LD_S,
                                    lo=n_train, hi=N, W=W)
events18, dsp18 = CE.select_events(raw, I_SP, I_LD, H=18,
                                    thr=THR, sp_hold=SP_H, load_stable=LD_S,
                                    lo=n_train, hi=N, W=W)
print(f"  H=60 事件: n={len(events60)}")
print(f"  H=18 事件: n={len(events18)}")

# 分层统计
for lab, ev, dv in [('H=60', events60, dsp60), ('H=18', events18, dsp18)]:
    abs_dv = np.abs(dv)
    print(f"  {lab} |ΔSP|分布: min={abs_dv.min():.2f} med={np.median(abs_dv):.2f} "
          f" [{np.sum(abs_dv<1):d}<1, {np.sum((abs_dv>=1)&(abs_dv<2)):d} 1-2, "
          f"{np.sum(abs_dv>=2):d}>=2]")

# ===== Part 2: 对照组
print("\n" + "=" * 70)
print("Part 2 — 对照组: 平稳段候选 (val+test)")
print("=" * 70)

controls = CE.select_controls(raw, I_SP, I_LD, W, H=60, stride=37, load_stable=LD_S)
controls = [c for c in controls if c >= n_train and c + 60 < N]
print(f"  对照组候选 n={len(controls)}")

# ===== Part 3: CEM 匹配
print("\n" + "=" * 70)
print("Part 3 — CEM 粗化精确匹配")
print("=" * 70)

matched60 = CE.match_controls(raw, events60, controls, I_T, I_LD, n_match=20)
matched18 = CE.match_controls(raw, events18, controls, I_T, I_LD, n_match=20)
n60 = sum(1 for v in matched60.values() if len(v) >= 3)
n18 = sum(1 for v in matched18.values() if len(v) >= 3)
print(f"  匹配成功 (≥3 对照): H=60 {n60}/{len(events60)}, H=18 {n18}/{len(events18)}")

# ===== Part 4: DiD
print("\n" + "=" * 70)
print("Part 4 — DiD 响应 + bootstrap CI + ceiling")
print("=" * 70)

did60 = CE.did_response(raw, events60, dsp60, matched60, I_T, H=60, n_boot=2000, min_ctrl=3)
did18 = CE.did_response(raw, events18, dsp18, matched18, I_T, H=18, n_boot=2000, min_ctrl=3)

print(f"  H=60: n={did60['n_ev']} | R_true[600s]={did60['R_true'][-1]:+.4f} "
      f"CI=[{did60['ci_lo'][-1]:+.4f}, {did60['ci_hi'][-1]:+.4f}]")
print(f"  H=18: n={did18['n_ev']} | R_true[180s]={did18['R_true'][-1]:+.4f} "
      f"CI=[{did18['ci_lo'][-1]:+.4f}, {did18['ci_hi'][-1]:+.4f}]")

PROFILE60 = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s'),
             (29, '300s'), (41, '420s'), (59, '600s')]
print(f"\n  {'时刻':>6} | {'R_true':>9} | {'CI_lo':>9} {'CI_hi':>9} | {'SGN_ceil':>8} {'GAIN_ceil':>8}")
for k, lab in PROFILE60:
    rt=did60['R_true'][k]; lo=did60['ci_lo'][k]; hi=did60['ci_hi'][k]
    sc=did60['sgn_ceiling'][k]; gc=did60['gain_ceiling'][k]
    print(f"  {lab:>6} | {rt:+9.4f} | {lo:+9.4f} {hi:+9.4f} | {sc:8.4f} {gc:8.4f}")

# Save
np.savez_compressed(f'{OUT}/did_response.npz',
    r60=did60['r'], onsets60=did60['onsets'], dsp60=did60['dsp'],
    R_true60=did60['R_true'], ci_lo60=did60['ci_lo'], ci_hi60=did60['ci_hi'],
    sgn_ceiling60=did60['sgn_ceiling'], gain_ceiling60=did60['gain_ceiling'],
    r18=did18['r'], onsets18=did18['onsets'], dsp18=did18['dsp'],
    R_true18=did18['R_true'], ci_lo18=did18['ci_lo'], ci_hi18=did18['ci_hi'],
    sgn_ceiling18=did18['sgn_ceiling'], gain_ceiling18=did18['gain_ceiling'])
print(f"\nSaved: {OUT}/did_response.npz")
