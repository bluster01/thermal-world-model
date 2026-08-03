#!/usr/bin/env python3
"""
exp_071_rl_data.py — P3: offline RL 数据准备 (transitions)
============================================================
状态 s = 当前时刻 14 维原始观测 | 动作 a = 阀位2维 | r = −|e| − λ_a·|Δa| | s' = 下一时刻
存 npy (train/val 来自 train_raw, 评测在 M7 仿真器内做)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_argv = sys.argv
sys.argv = ['exp_025_unified_benchmark.py']
from experiments.phase1_dynamics import exp_025_unified_benchmark as E
sys.argv = _argv

LAMBDA_A = float(os.environ.get('LAMBDA_A', 0.5))
OUT = 'results/exp_071_rl_data'
os.makedirs(OUT, exist_ok=True)

def build(raw, name):
    cols = E.NUMERIC_COLS
    TIDX = E.TARGET_IDX; SPIDX = E.SP_IDX
    vidx = [cols.index(c) for c in ['一级减温调节门阀位', '二级减温调节门阀位']]
    n = len(raw)
    s = raw[:-1, :len(cols)].astype(np.float32)          # 当前 14 维观测
    a = raw[:-1, vidx].astype(np.float32)                # 阀位动作
    sp = raw[:-1, SPIDX]
    e = raw[:-1, TIDX] - sp
    da = np.abs(np.diff(raw[:, vidx], axis=0)).sum(1)    # |Δa| (a_t − a_{t-1})
    r = (-np.abs(e) - LAMBDA_A * da).astype(np.float32)  # 奖励
    s_next = raw[1:, :len(cols)].astype(np.float32)
    np.savez(f'{OUT}/{name}.npz', s=s, a=a, r=r, s_next=s_next)
    print(f"{name}: {len(s)} transitions | 状态 {s.shape[1]}维 | 动作 {a.shape[1]}维 | "
          f"r 均值 {r.mean():.4f} | e 均值 {e.mean():.2f}°C | |Δa| 均值 {da.mean():.4f}")

if __name__ == '__main__':
    build(E.train_raw, 'train')
    build(E.val_raw, 'val')
    print(f"Saved: {OUT}/")
