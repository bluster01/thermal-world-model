#!/usr/bin/env python3
"""exp_201 SP-IV v2: 用 CEM 匹配后的 DiD 响应 (r18/r60) 构造 plant 增益真值。

g_plant = ΔT_did(匹配后) / ΔV_init(SP 阶跃后 30s 阀位响应, 由 SP 驱动)
对照: flow 模型同层增益 (180s 与 600s 双口径)。
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

LAYERS = [(0, 10), (10, 20), (20, 30), (30, 45)]


def event_gain(onsets, dsp, resp, step_idx, label):
    """resp: [n_ev, n_steps] 匹配后响应; 增益 = resp[-1] / dV30, 按事件前开度分层。"""
    out = {}
    for lo, hi in LAYERS:
        gs, ns = [], []
        for t, s, r in zip(onsets, dsp, resp):
            v0 = raw[t, I_V2]
            if not (lo <= v0 < hi):
                continue
            dv30 = raw[t + 3, I_V2] - raw[t, I_V2]
            # 方向过滤: SP↑→阀关 (dv 与 dsp 反号), 排除混杂事件
            if dv30 * s >= 0:
                continue
            gs.append(r[step_idx] / dv30)
            ns.append(1)
        if len(gs) >= 3:
            out[(lo, hi)] = (len(gs), float(np.mean(gs)), float(np.median(gs)))
    return out


if __name__ == '__main__':
    p2 = np.load('results/cfe_groundtruth_p2/did_response.npz', allow_pickle=True)
    old = np.load('results/cfe_groundtruth/did_response.npz', allow_pickle=True)

    print('=== 180s 口径 (P2, 79 事件, 匹配后 r18) ===')
    g180 = event_gain(p2['onsets18'], p2['dsp18'], p2['r18'], -1, 'r18')
    print(f'{"layer":>12} | {"n":>3} | {"g_plant mean":>12} | {"median":>10}')
    for lo, hi in LAYERS:
        if (lo, hi) in g180:
            n, m, med = g180[(lo, hi)]
            print(f'[{lo:2d},{hi:2d}) | {n:3d} | {m*1000:9.1f} m°C/% | {med*1000:8.1f}')

    print('\n=== 600s 口径 (老版, 15 事件, 匹配后 r60) ===')
    g600 = event_gain(old['onsets60'], old['dsp60'], old['r60'], -1, 'r60')
    print(f'{"layer":>12} | {"n":>3} | {"g_plant mean":>12} | {"median":>10}')
    for lo, hi in LAYERS:
        if (lo, hi) in g600:
            n, m, med = g600[(lo, hi)]
            print(f'[{lo:2d},{hi:2d}) | {n:3d} | {m*1000:9.1f} m°C/% | {med*1000:8.1f}')

    print('\n=== 模型 flow noff 增益 (gain_diag 实测) ===')
    print('  180s 口径待测; 600s 口径: 0-10:1.01, 10-20:1.57, 20-30:2.30, 30-45:1.39 m°C/%')
    print('\n=== R_true 全局 (不分层) ===')
    print(f'  r18 末点均值: {p2["r18"][:, -1].mean():.3f}°C @180s (n=79)')
    print(f'  r60 末点均值: {old["r60"][:, -1].mean():.3f}°C @600s (n=15)')
    print(f'  对应 dV30 平均: {np.mean([abs(raw[t+3,I_V2]-raw[t,I_V2]) for t in p2["onsets18"]]):.2f}%')
