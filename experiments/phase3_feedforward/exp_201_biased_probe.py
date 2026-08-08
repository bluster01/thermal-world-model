#!/usr/bin/env python3
"""exp_201 高开度有偏采样诊断: 方向 + 响应幅度 vs 阀位工作点。

假说 (阀门流量非线性): 等百分比特性下 dF/dV ∝ R^(V-1), 同样 +5% 开度,
高开度层流量增量更大 → 温度响应 |dT| 应随开度单调增大。
若 abs 模型在该层方向更稳/响应更大 → 学到了非线性; 否则仅学到平均增益。
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
N_MAX = len(test_raw) - W - H


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


def biased_windows(vmin, vmax, n, seed):
    """有偏采样: 窗口 (起点起 W+H 步) 阀位均值 ∈ [vmin, vmax) 的测试窗口。"""
    rng = np.random.default_rng(seed)
    found = []
    # 先随机抽 20000 个候选, 筛出满足阀位区间的, 不足则暴力扫描
    cand = rng.integers(0, N_MAX, size=20000)
    for i in cand:
        v_win = raw[i:i + W + H, I_V2]
        vm = float(v_win.mean())
        if vmin <= vm < vmax:
            found.append(int(i))
        if len(found) >= n:
            break
    return found[:n]


def probe(model, idxs, mode, delta, n):
    neg = pos = zero = 0; mags = []
    for i_int in idxs:
        x = torch.from_numpy(test_raw[i_int:i_int + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
        a = build_action(i_int, mode)
        a_up = a.copy(); a_up += delta
        a_dn = a.copy(); a_dn -= delta
        with torch.no_grad():
            mu_up, _ = model(x, torch.from_numpy(a_up).float().reshape(1, H, 1).to(DEVICE))
            mu_dn, _ = model(x, torch.from_numpy(a_dn).float().reshape(1, H, 1).to(DEVICE))
        diff = (mu_up - mu_dn).mean().item()
        mags.append(abs(diff))
        if abs(diff) < 1e-6: zero += 1
        elif diff < 0: neg += 1
        else: pos += 1
    return dict(neg=neg / n, pos=pos / n, mag_mean=float(np.mean(mags)),
                mag_p50=float(np.median(mags)), n=n)


if __name__ == '__main__':
    root = 'results/exp_201_valve_action'
    jobs = [
        ('abs', f'{root}/A1phys_valve_noff_s0_abs', 5.0,
         [(0, 10), (10, 20), (20, 30), (30, 45), (45, 100)]),
        ('abs', f'{root}/A1phys_valve_s0_ff10_abs', 5.0,
         [(0, 10), (10, 20), (20, 30), (30, 45), (45, 100)]),
        ('delta', f'{root}/A1phys_valve_s0_ff10', 0.1,
         [(0, 10), (10, 20), (20, 30), (30, 45), (45, 100)]),
    ]
    for mode, d, delta, layers in jobs:
        print(f'=== {d} ({mode}, δ={delta}) ===')
        model = load_model(d, mode)
        for vmin, vmax in layers:
            idxs = biased_windows(vmin, vmax, 120, seed=99)
            if len(idxs) < 30:
                print(f'  valve[{vmin:2d},{vmax:2d}): n={len(idxs)} (insufficient)')
                continue
            r = probe(model, idxs, mode, delta, len(idxs))
            print(f'  valve[{vmin:2d},{vmax:2d}): n={r["n"]:3d} jac_neg={r["neg"]:.1%} '
                  f'mean|dT|={r["mag_mean"]:.4f}°C p50={r["mag_p50"]:.4f}°C')
