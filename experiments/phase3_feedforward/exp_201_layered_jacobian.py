#!/usr/bin/env python3
"""exp_201 分层 jacobian 诊断: 按窗口平均阀位工作点分层, 测方向 + 响应幅度。

回答: abs 模式方向不稳(65%)是否工作点相关 —— 高开度层(流量非线性区)
是否方向更稳/增益更大; delta 模式是否有同样的分层结构。
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
dvalve_col = np.diff(raw[:, I_V2], prepend=raw[0, I_V2])
v_med_train = float(np.median(raw[:495407, I_V2]))
absvalve_col = raw[:, I_V2] - v_med_train
raw42 = np.concatenate([raw, dvalve_col[:, None], absvalve_col[:, None]], 1)
I_DVALVE, I_AVALVE = 40, 41
test_raw = raw42[n_val_end:]


def build_action(s, mode):
    icol = I_DVALVE if mode == 'delta' else I_AVALVE
    return test_raw[s + W:s + W + H, icol].astype(np.float32)


def load_model(variant_dir, mode):
    m = CA.ResidualCausalWM(N_FEAT, TARGET_IDX, H, intervention='phys', cumsum_out=False,
                            probabilistic=True, n_lag=2, free_head_type='mlp',
                            alpha_init=0.0, clamp_interv=15.0,
                            k_init=0.01 if mode == 'delta' else 0.002,
                            integrate=(mode == 'delta')).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(variant_dir, 'best_cfi.pth'), map_location=DEVICE))
    m.eval()
    return m


def layered_jacobian(model, mode, n=200, seed=42, delta=None, layers=((0, 10), (10, 30), (30, 100))):
    if delta is None:
        delta = 0.1 if mode == 'delta' else 5.0
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(test_raw) - W - H, size=n)
    stats = {lo: dict(neg=0, pos=0, zero=0, n=0, mag=[]) for lo, hi in layers}
    for i in idxs:
        i_int = int(i)
        v_win = raw[i_int:i_int + W + H, I_V2]
        v_mean = float(np.mean(v_win[~np.isnan(v_win)])) if np.isnan(v_win).any() else float(v_win.mean())
        layer = None
        for lo, hi in layers:
            if lo <= v_mean < hi:
                layer = lo; break
        if layer is None: continue
        x = torch.from_numpy(test_raw[i_int:i_int + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
        a = build_action(i_int, mode)
        a_up = a.copy(); a_up += delta
        a_dn = a.copy(); a_dn -= delta
        with torch.no_grad():
            mu_up, _ = model(x, torch.from_numpy(a_up).float().reshape(1, H, 1).to(DEVICE))
            mu_dn, _ = model(x, torch.from_numpy(a_dn).float().reshape(1, H, 1).to(DEVICE))
        diff = (mu_up - mu_dn).mean().item()
        s = stats[layer]
        s['n'] += 1
        s['mag'].append(abs(diff))
        if abs(diff) < 1e-6: s['zero'] += 1
        elif diff < 0: s['neg'] += 1
        else: s['pos'] += 1
    print(f'  mode={mode}:')
    for lo, s in stats.items():
        if s['n'] == 0:
            print(f'    valve[{lo:2d},...): n=0'); continue
        print(f'    valve[{lo:2d},...): n={s["n"]:3d} jac_neg={s["neg"]/s["n"]:.1%} '
              f'pos={s["pos"]/s["n"]:.1%} mean|dT|={np.mean(s["mag"]):.4f}°C')
    return stats


if __name__ == '__main__':
    root = 'results/exp_201_valve_action'
    for mode, d in [('delta', f'{root}/A1phys_valve_s0_ff10'),
                    ('abs', f'{root}/A1phys_valve_s0_ff10_abs')]:
        print(f'=== {d} ({mode}) ===')
        model = load_model(d, mode)
        layered_jacobian(model, mode, n=300, seed=99)
