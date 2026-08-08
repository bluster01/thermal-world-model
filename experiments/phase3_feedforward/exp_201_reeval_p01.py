#!/usr/bin/env python3
"""P0-1 修复后重评估: 复用已有 checkpoint, 用对齐后的 eval_jacobian/eval_gain_180。

对比修复前后: 方向 jac:neg 与 gain_180 是否保持。
"""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'phase3_feedforward'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'phase1_dynamics'))

from exp_025_unified_benchmark import N_FEAT, TARGET_IDX
import causal_arch as CA
import exp_201_valve_action as E

DEVICE = E.DEVICE
H = E.H

def build(mode):
    return CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='phys', cumsum_out=False,
                               probabilistic=True, n_lag=2, free_head_type='mlp',
                               alpha_init=0.0, clamp_interv=15.0,
                               k_init=0.05 if mode == 'flow' else 0.01,
                               integrate=(mode == 'delta')).to(DEVICE)

RUNS = [
    # (目录名, mode, 检查点)
    ('A1phys_valve_s0_ff10_flow', 'flow', 'best_cfi.pth'),
    ('A1phys_valve_s1_ff10_flow', 'flow', 'best_cfi.pth'),
    ('A1phys_valve_s2_ff10_flow', 'flow', 'best_cfi.pth'),
    ('A1phys_valve_noff_s0_flow', 'flow', 'best_cfi.pth'),
    ('A1phys_valve_noff_s1_flow', 'flow', 'best_cfi.pth'),
    ('A1phys_valve_noff_s2_flow', 'flow', 'best_cfi.pth'),
    ('A1phys_valve_noff_s0_flow_lg0.15', 'flow', 'best_gain.pth'),
    ('A1phys_valve_noff_s0_flow_lg0.2', 'flow', 'best_gain.pth'),
    ('A1phys_valve_noff_s1_flow_lg0.2', 'flow', 'best_gain.pth'),
    ('A1phys_valve_noff_s2_flow_lg0.2', 'flow', 'best_gain.pth'),
]

print(f'{"run":<36} | {"jac_neg":>8} | {"gain_180":>9}')
rows = []
for d, mode, ck in RUNS:
    p = os.path.join('results/exp_201_valve_action', d, ck)
    if not os.path.exists(p):
        print(f'{d:<36} | missing {ck}')
        continue
    m = build(mode)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    m.eval()
    j = E.eval_jacobian(m, H, n=100, seed=99, delta=5.0, mode=mode)
    g = E.eval_gain_180(m, n=100, seed=99, mode=mode)
    print(f'{d:<36} | {j["neg"]:7.1%} | {g*1000:8.1f}')
    rows.append({'run': d, 'ckpt': ck, 'jac_neg': j['neg'], 'gain_180_mC_pct': g * 1000})

with open('results/exp_201_valve_action/p0_1_reeval.json', 'w') as f:
    json.dump(rows, f, indent=2)
print('saved: results/exp_201_valve_action/p0_1_reeval.json')
