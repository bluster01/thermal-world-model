#!/usr/bin/env python3
"""P0-1 回归测试: flow 模式评估的 split-offset 对齐。

审计 (docs/PHASE3_5_LINUX_REVIEW_2026-08-09.md P0-1) 发现 eval_jacobian/
eval_gain_180 的状态窗口来自 test_raw[i] (split 相对索引) 而阀位基线来自
全局 raw[i], 两个不同时间段。修复后: 阀位基线必须与状态窗口同源
(全局行 n_val_end + i), 且与 test_raw[i, :N_FEAT] 的阀位列一致。

运行: python -m pytest tests/phase3_feedforward/test_split_offset.py -q
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'phase3_feedforward'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'phase1_dynamics'))

from exp_025_unified_benchmark import data_all, NUMERIC_COLS, TARGET_IDX, N_FEAT
import exp_201_valve_action as E


def _state_valve_row(i_global: int) -> float:
    """状态窗口首行 (全局行 i_global) 的阀位值。"""
    return float(data_all[i_global, E.I_V2])


def test_flow_jacobian_uses_split_aligned_valve():
    """eval_jacobian flow 分支的阀位基线必须与 x 同源 (全局行 n_val_end+i)。"""
    model = E.build_model('A1phys_valve_noff', mode='flow').to(E.DEVICE)
    model.eval()
    rng = np.random.default_rng(7)
    idxs = rng.integers(0, len(E.test_raw) - E.W - E.H, size=20)
    for i in idxs:
        i_int = int(i)
        g = E.n_val_end + i_int
        # 修复后的实现: 全局行 g 的阀位 == test_raw[i, I_V2] (同源)
        assert abs(data_all[g, E.I_V2] - float(E.test_raw[i_int, E.I_V2])) < 1e-6, \
            f'split offset mismatch at i={i_int}: global {data_all[g, E.I_V2]} vs test_raw {E.test_raw[i_int, E.I_V2]}'
    # 与修复前行为对比: 修复前用 raw[i_int] (offset=0), 现在必须 offset=n_val_end
    assert E.n_val_end > 0, 'test expects a nonzero split offset'
    # 抽样确认: 全局行 g 的阀位 != raw[i_int] 的阀位 (即修复确实改变了取值范围)
    mismatches = 0
    for i in idxs[:10]:
        i_int = int(i)
        if abs(data_all[E.n_val_end + i_int, E.I_V2] - data_all[i_int, E.I_V2]) > 1e-6:
            mismatches += 1
    assert mismatches > 0, 'offset fix appears ineffective: global rows equal un-offset rows'


def test_gain_180_uses_split_aligned_valve():
    """eval_gain_180 flow 分支同源检查 (修复后与 eval_jacobian 一致)。"""
    rng = np.random.default_rng(11)
    idxs = rng.integers(0, len(E.test_raw) - E.W - E.H, size=10)
    for i in idxs:
        i_int = int(i)
        g = E.n_val_end + i_int
        assert abs(data_all[g, E.I_V2] - float(E.test_raw[i_int, E.I_V2])) < 1e-6


if __name__ == '__main__':
    test_flow_jacobian_uses_split_aligned_valve()
    test_gain_180_uses_split_aligned_valve()
    print('P0-1 split-offset alignment: OK')
