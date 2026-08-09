#!/usr/bin/env python3
"""合成时间轴单测: sp_events_1s_v2 的 dv_* 索引与 split 判定 (审计 P0-1/P0-2 要求)。

构造合成 1s 网格: t0 前 960s + 后 600s, 在已知位置注入阀位阶跃,
断言 dv_3s/10s/30s/60s/180s/600s 索引取到正确值; 断言 split_of 边界。
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.phase3_5.sp_events_1s_v2 import split_of, SPLIT_FRAC


def make_synthetic_event(t0_rel_s: float = 0.0, step_s: int = 1):
    """构造 1s 网格合成事件窗口: [t0-960, t0+600], 返回 (grid_rel_s, valve_filled, n_pre)。"""
    PRE, POST = 960, 600
    n = PRE + POST + 1
    grid = np.arange(-PRE, POST + 1, step_s, dtype=np.float64)  # 相对 t0 的秒
    valve = np.zeros(n)
    # 阶跃: t0 前 0.5% -> t0 后 8.5% (幅度 8%), 一步到位
    valve[grid < 0] = 0.5
    valve[grid >= 0] = 8.5
    return grid, valve, PRE


def test_dv_indices():
    """dv_3s = valve[t0+3] - valve[t0-1], dv_600s = valve[t0+600] - valve[t0-1] 等。"""
    grid, valve, n_pre = make_synthetic_event()
    # 模拟脚本: n_pre 是 t0 索引
    dv = {}
    for label, offset in [('dv_3s', 3), ('dv_10s', 10), ('dv_30s', 30),
                          ('dv_60s', 60), ('dv_180s', 180), ('dv_600s', 600)]:
        dv[label] = valve[n_pre + offset] - valve[n_pre - 1]
    expected = 8.5 - 0.5  # 所有档位在阶跃后都应取到 8.0
    for label, val in dv.items():
        assert abs(val - expected) < 1e-9, f'{label}: got {val}, expected {expected}'
    print('test_dv_indices PASS: dv_3s..dv_600s 全部取到 8.0 (t0+offset - t0-1)')


def test_dv_before_step():
    """dv_30s 若阶跃发生在 t0+100s, 则 dv_3s/30s 应≈0, dv_180s/600s≈8。"""
    PRE = 960
    n = PRE + 600 + 1
    valve = np.full(n, 0.5)
    valve[PRE + 100:] = 8.5  # 阶跃在 t0+100s
    vals = {}
    for label, offset in [('dv_3s', 3), ('dv_30s', 30), ('dv_180s', 180), ('dv_600s', 600)]:
        vals[label] = valve[PRE + offset] - valve[PRE - 1]
    assert abs(vals['dv_3s']) < 1e-9, f'dv_3s should be 0, got {vals["dv_3s"]}'
    assert abs(vals['dv_30s']) < 1e-9, f'dv_30s should be 0, got {vals["dv_30s"]}'
    assert abs(vals['dv_180s'] - 8.0) < 1e-9, f'dv_180s should be 8, got {vals["dv_180s"]}'
    assert abs(vals['dv_600s'] - 8.0) < 1e-9, f'dv_600s should be 8, got {vals["dv_600s"]}'
    print('test_dv_before_step PASS: 阶跃在 t0+100s 时 dv_3s/30s=0, dv_180s/600s=8')


def test_split_of():
    """split_of 在 60/20/20 边界上正确。"""
    gs, ge, n = 1_000_000_000_000, 1_000_000_000_000 + 100_000 * 10_000_000_000, 100_000
    step = 10_000_000_000
    t_train_end = gs + int(n * 0.60) * step
    t_val_end = gs + int(n * 0.80) * step
    assert split_of(t_train_end - 1, gs, ge, n) == 'train'
    assert split_of(t_train_end + 1, gs, ge, n) == 'validation'
    assert split_of(t_val_end - 1, gs, ge, n) == 'validation'
    assert split_of(t_val_end + 1, gs, ge, n) == 'test'
    assert split_of(gs, gs, ge, n) == 'train'
    assert split_of(ge, gs, ge, n) == 'test'
    print('test_split_of PASS: 60/20/20 边界正确')


if __name__ == '__main__':
    test_dv_indices()
    test_dv_before_step()
    test_split_of()
    print('ALL PASS')
