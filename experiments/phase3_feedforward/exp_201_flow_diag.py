#!/usr/bin/env python3
"""exp_201 flow 诊断: K(x) 增益分布 + 分层方向。

1) K: InterventionPhysics 学到的增益 (流量空间) — 应为负 (流量↑→T↓) 且量级合理。
2) 分层: 有偏采样按开度层测方向 — flow 是否修好高开度层。
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_proj = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, 'experiments', 'phase1_dynamics'))

import causal_arch as CA
from exp_025_unified_benchmark import cfg as E_cfg, data_all, N_FEAT, TARGET_IDX, NUMERIC_COLS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E_cfg.WINDOW_SIZE
H = 60
n_val_end = 601566
I_V2 = NUMERIC_COLS.index('二级减温调节门阀位')

raw = data_all
R_FLOW = 50.0
v_pct = np.clip(raw[:, I_V2], 0.0, 100.0) / 100.0
flow_col = (R_FLOW ** (v_pct - 1.0) - 1.0 / R_FLOW) / (1.0 - 1.0 / R_FLOW)
flow_med_train = float(np.median(flow_col[:495407]))
test_raw = np.concatenate([raw, flow_col[:, None]], 1)[n_val_end:]
N_MAX = len(test_raw) - W - H


def valve_to_flow(v):
    v_pct = np.clip(np.asarray(v, dtype=np.float64), 0.0, 100.0) / 100.0
    f = (R_FLOW ** (v_pct - 1.0) - 1.0 / R_FLOW) / (1.0 - 1.0 / R_FLOW)
    return (f - flow_med_train).astype(np.float32)


def load_model(variant_dir):
    m = CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='phys', cumsum_out=False,
                            probabilistic=True, n_lag=2, free_head_type='mlp',
                            alpha_init=0.0, clamp_interv=15.0, k_init=0.05,
                            integrate=False).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(variant_dir, 'best_cfi.pth'), map_location=DEVICE))
    m.eval()
    return m


def biased_windows(vmin, vmax, n, seed):
    rng = np.random.default_rng(seed)
    found = []
    cand = rng.integers(0, N_MAX, size=40000)
    for i in cand:
        vm = float(raw[i:i + W + H, I_V2].mean())
        if vmin <= vm < vmax:
            found.append(int(i))
        if len(found) >= n:
            break
    return found[:n]


if __name__ == '__main__':
    root = 'results/exp_201_valve_action'
    for d in [f'{root}/A1phys_valve_noff_s0_flow', f'{root}/A1phys_valve_s0_ff10_flow']:
        print(f'=== {d} ===')
        model = load_model(d)

        # 1) K 分布: 100 随机窗口
        rng = np.random.default_rng(1)
        idxs = rng.integers(0, N_MAX, size=100)
        Ks = []
        for i in idxs:
            x = torch.from_numpy(test_raw[i:i + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, s_flat = model.encode(x)
                K, tau = model.interv.params(s_flat)
            Ks.append(K.item())
        Ks = np.array(Ks)
        print(f'  K: mean={Ks.mean():.5f} median={np.median(Ks):.5f} '
              f'min={Ks.min():.5f} max={Ks.max():.5f} neg_frac={(Ks < 0).mean():.1%}')
        print(f'  tau (n_lag=2): 采样 5 窗口均值:', end=' ')

        # 2) 分层方向
        for vmin, vmax in [(0, 10), (10, 20), (20, 30), (30, 45)]:
            idxs = biased_windows(vmin, vmax, 80, seed=99)
            if len(idxs) < 30:
                print(f'  valve[{vmin:2d},{vmax:2d}): n={len(idxs)} (insufficient)')
                continue
            neg = pos = 0
            for i in idxs:
                x = torch.from_numpy(test_raw[i:i + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
                v_win = raw[i:i + W + H, I_V2]
                a_up = valve_to_flow(np.clip(v_win[W:], 0, 100) + 5.0)
                a_dn = valve_to_flow(np.clip(v_win[W:], 0, 100) - 5.0)
                with torch.no_grad():
                    mu_up, _ = model(x, torch.from_numpy(a_up).float().reshape(1, H, 1).to(DEVICE))
                    mu_dn, _ = model(x, torch.from_numpy(a_dn).float().reshape(1, H, 1).to(DEVICE))
                diff = (mu_up - mu_dn).mean().item()
                if diff < 0: neg += 1
                else: pos += 1
            print(f'  valve[{vmin:2d},{vmax:2d}): n={len(idxs):3d} jac_neg={neg/len(idxs):.1%}')
