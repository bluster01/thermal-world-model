#!/usr/bin/env python3
"""
exp_104_did_groundtruth.py — L1 DiD 真值 + ceiling (2026-08-05)
=================================================================
设计稿: docs/causal_eval_framework.md §L1

用差分中差分 (DiD) 估计观测因果响应, 替代 sign(ΔSP) 作为"方向正确率"的真值。
这一步本身是独立贡献: 主汽温对 ΔSP 干预的观测因果响应曲线及其时标。

处理组: |ΔSP| > 1.0, 事件间隔 ≥ 60, 负荷稳定, SP 保持
对照组: CEM 粗化精确匹配 (同负荷分箱 × 同 onset 前温度趋势分箱)

产出: results/cfe_groundtruth/did_response.json + did_response.npz
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

OUT = 'results/cfe_groundtruth'
os.makedirs(OUT, exist_ok=True)

raw = E.data_all
N = len(raw)
I_SP = E.NUMERIC_COLS.index('二级减温调节阀设定')
I_T = E.NUMERIC_COLS.index('末级过热器出口汽温')
I_LD = E.NUMERIC_COLS.index('机组负荷')
W = E.cfg.WINDOW_SIZE
n_train, n_val_end = 495407, 601566

print(f"[data] N={N} | I_SP={I_SP} I_T={I_T} I_LD={I_LD} | W={W}")
print(f"[data] test range: [{n_val_end}, {N})")

# ===================================================== Part 1: 处理组事件
# 限定 test 区间 + 历史窗口不进训练集
print("\n" + "=" * 70)
print("Part 1 — 处理组: SP 阶跃事件 (test 区间)")
print("=" * 70)

events, dsp_vals = CE.select_events(raw, I_SP, I_LD, H=60, lo=n_val_end, hi=N, W=W)
print(f"  事件数 n={len(events)} (test-only)")

# 补 H=18 事件集 (用于 18 步模型对照)
events18, dsp18 = CE.select_events(raw, I_SP, I_LD, H=18, lo=n_val_end, hi=N, W=W)
print(f"  H=18 事件数 n={len(events18)}")

# ===================================================== Part 2: 对照组
print("\n" + "=" * 70)
print("Part 2 — 对照组: 平稳段候选 (test 区间)")
print("=" * 70)

controls = CE.select_controls(raw, I_SP, I_LD, W, H=60, stride=37)
# 过滤到 test 区间
controls = [c for c in controls if c >= n_val_end and c + 60 < N]
print(f"  对照组候选 n={len(controls)}")

# ===================================================== Part 3: CEM 匹配
print("\n" + "=" * 70)
print("Part 3 — CEM 粗化精确匹配")
print("=" * 70)

matched60 = CE.match_controls(raw, events, controls, I_T, I_LD, n_match=20)
matched18 = CE.match_controls(raw, events18, controls, I_T, I_LD, n_match=20)

n_matched60 = sum(1 for v in matched60.values() if len(v) >= 3)
n_matched18 = sum(1 for v in matched18.values() if len(v) >= 3)
print(f"  匹配成功 (≥3 对照): H=60 {n_matched60}/{len(events)}, H=18 {n_matched18}/{len(events18)}")

# ===================================================== Part 4: DiD 真值
print("\n" + "=" * 70)
print("Part 4 — DiD 响应 + bootstrap CI + split-half ceiling")
print("=" * 70)

did60 = CE.did_response(raw, events, dsp_vals, matched60, I_T, H=60, n_boot=2000, min_ctrl=3)
did18 = CE.did_response(raw, events18, dsp18, matched18, I_T, H=18, n_boot=2000, min_ctrl=3)

print(f"\n  H=60: n={did60['n_ev']} | R_true 600s = {did60['R_true'][-1]:+.4f}")
print(f"  H=18: n={did18['n_ev']} | R_true 180s = {did18['R_true'][-1]:+.4f}")

# 关键时标报告
PROFILE60 = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s'),
             (29, '300s'), (41, '420s'), (59, '600s')]
PROFILE18 = [(2, '30s'), (5, '60s'), (11, '120s'), (17, '180s')]

print(f"\n  {'时刻':>6} | {'R_true':>9} | {'CI_lo':>9} {'CI_hi':>9} | {'SGN_ceil':>9} {'GAIN_ceil':>9}")
for k, lab in PROFILE60:
    rt = did60['R_true'][k]
    lo, hi = did60['ci_lo'][k], did60['ci_hi'][k]
    sc = did60['sgn_ceiling'][k]
    gc = did60['gain_ceiling'][k]
    print(f"  {lab:>6} | {rt:+9.4f} | {lo:+9.4f} {hi:+9.4f} | {sc:8.4f} {gc:8.4f}")

# ===================================================== Save
print(f"\nSaving to {OUT}/")

# JSON (可读)
results = {
    'H60': {
        'n_ev': int(did60['n_ev']),
        'R_true': did60['R_true'].tolist(),
        'ci_lo': did60['ci_lo'].tolist(),
        'ci_hi': did60['ci_hi'].tolist(),
        'sgn_ceiling': did60['sgn_ceiling'].tolist(),
        'gain_ceiling': did60['gain_ceiling'].tolist(),
        'noise_floor': did60['noise_floor'].tolist(),
        'profile': {lab: dict(
            R_true=float(did60['R_true'][k]),
            ci_lo=float(did60['ci_lo'][k]),
            ci_hi=float(did60['ci_hi'][k]),
            sgn_ceiling=float(did60['sgn_ceiling'][k]),
            gain_ceiling=float(did60['gain_ceiling'][k]),
        ) for k, lab in PROFILE60},
    },
    'H18': {
        'n_ev': int(did18['n_ev']),
        'R_true': did18['R_true'].tolist(),
        'ci_lo': did18['ci_lo'].tolist(),
        'ci_hi': did18['ci_hi'].tolist(),
        'sgn_ceiling': did18['sgn_ceiling'].tolist(),
        'gain_ceiling': did18['gain_ceiling'].tolist(),
        'noise_floor': did18['noise_floor'].tolist(),
        'profile': {lab: dict(
            R_true=float(did18['R_true'][k]),
            ci_lo=float(did18['ci_lo'][k]),
            ci_hi=float(did18['ci_hi'][k]),
            sgn_ceiling=float(did18['sgn_ceiling'][k]),
            gain_ceiling=float(did18['gain_ceiling'][k]),
        ) for k, lab in PROFILE18},
    },
}
with open(f'{OUT}/did_response.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# NPZ (逐事件数据, 供 exp_105/106 使用)
np.savez_compressed(f'{OUT}/did_response.npz',
                    r60=did60['r'], onsets60=did60['onsets'], dsp60=did60['dsp'],
                    R_true60=did60['R_true'], ci_lo60=did60['ci_lo'], ci_hi60=did60['ci_hi'],
                    sgn_ceiling60=did60['sgn_ceiling'], gain_ceiling60=did60['gain_ceiling'],
                    r18=did18['r'], onsets18=did18['onsets'], dsp18=did18['dsp'],
                    R_true18=did18['R_true'], ci_lo18=did18['ci_lo'], ci_hi18=did18['ci_hi'],
                    sgn_ceiling18=did18['sgn_ceiling'], gain_ceiling18=did18['gain_ceiling'])

print(f"  did_response.json  ({os.path.getsize(f'{OUT}/did_response.json')} bytes)")
print(f"  did_response.npz   ({os.path.getsize(f'{OUT}/did_response.npz')} bytes)")
print("\nDone — L1 DiD 真值就绪, 所有后续模型评测共用。")
