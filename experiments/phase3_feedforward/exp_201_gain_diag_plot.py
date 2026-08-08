#!/usr/bin/env python3
"""exp_201 增益诊断图: 模型增益形状 vs 等百分比理论 + DiD 事件方向散点。"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_proj = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, 'experiments', 'phase1_dynamics'))

import causal_arch as CA
from exp_025_unified_benchmark import cfg as E_cfg, data_all, N_FEAT, TARGET_IDX, NUMERIC_COLS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = E_cfg.WINDOW_SIZE; H = 60; n_val_end = 601566
I_V2 = NUMERIC_COLS.index('二级减温调节门阀位'); I_T = TARGET_IDX
raw = data_all
R_FLOW = 50.0
v_pct = np.clip(raw[:, I_V2], 0.0, 100.0) / 100.0
flow_col = (R_FLOW ** (v_pct - 1.0) - 1.0 / R_FLOW) / (1.0 - 1.0 / R_FLOW)
flow_med_train = float(np.median(flow_col[:495407]))
test_raw = np.concatenate([raw, flow_col[:, None]], 1)[n_val_end:]
N_MAX = len(test_raw) - W - H
LAYERS = [(0, 10), (10, 20), (20, 30), (30, 45)]
VC = np.array([5, 15, 25, 37.5])


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


def model_gain(model, idxs, delta=5.0):
    mags = []
    for i in idxs:
        x = torch.from_numpy(test_raw[i:i + W, :N_FEAT]).float().unsqueeze(0).to(DEVICE)
        v_win = raw[i:i + W + H, I_V2]
        a_up = valve_to_flow(np.clip(v_win[W:], 0, 100) + delta)
        a_dn = valve_to_flow(np.clip(v_win[W:], 0, 100) - delta)
        with torch.no_grad():
            mu_up, _ = model(x, torch.from_numpy(a_up).float().reshape(1, H, 1).to(DEVICE))
            mu_dn, _ = model(x, torch.from_numpy(a_dn).float().reshape(1, H, 1).to(DEVICE))
        mags.append((mu_up - mu_dn).mean().item())
    mags = np.array(mags)
    return float(np.mean(np.abs(mags)) / (2 * delta))


def find_valve_events(vmin, vmax, thr=2.0, n_max=600, seed=7):
    rng = np.random.default_rng(seed)
    cand = rng.integers(0, len(raw) - W - H - 18, size=200000)
    ev = []
    for t in cand:
        v0 = raw[t, I_V2]
        if not (vmin <= v0 < vmax):
            continue
        vseg = raw[t:t + H, I_V2]
        dv = vseg[-1] - vseg[0]
        if abs(dv) < thr:
            continue
        d = np.diff(vseg)
        mono = max((d > 0).mean(), (d < 0).mean())
        if mono < 0.8:
            continue
        slope_pre = (raw[t, I_T] - raw[max(t - 18, 0), I_T]) / 18.0
        dT = (raw[t + H, I_T] - raw[t, I_T]) - slope_pre * H
        ev.append((t, float(dv), float(dT)))
        if len(ev) >= n_max:
            break
    return ev


if __name__ == '__main__':
    root = 'results/exp_201_valve_action'
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)

    # ── 左: 增益形状 (归一化到 10-20 层) ──
    ax = axes[0]
    dFdV = np.log(R_FLOW) * R_FLOW ** (VC / 100.0 - 1.0) / 100.0
    ax.plot(VC, dFdV / dFdV[1], 'k--o', lw=1.6, ms=5, label='Equal-percentage theory (R=50)')
    for k, d, c in [('flow noff', f'{root}/A1phys_valve_noff_s0_flow', '#c0392b'),
                    ('flow ff10', f'{root}/A1phys_valve_s0_ff10_flow', '#2980b9')]:
        m = load_model(d)
        g = np.array([model_gain(m, biased_windows(lo, hi, 100, seed=99)) for lo, hi in LAYERS])
        ax.plot(VC, g / g[1], 'o-', lw=1.6, ms=5, color=c, label=f'Model {k}')
    ax.set_xlabel('Valve opening (%)')
    ax.set_ylabel('Relative gain (norm. to 10–20% layer)')
    ax.set_title('(a) Gain shape vs opening\nmodel learns the nonlinearity', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

    # ── 右: DiD 事件散点 (0-10% 层) ──
    ax = axes[1]
    ev = find_valve_events(0, 10)
    dv = np.array([e[1] for e in ev]); dT = np.array([e[2] for e in ev])
    ax.scatter(dv, dT, s=14, alpha=0.55, color='#7f8c8d', edgecolor='none')
    ax.axhline(0, color='k', lw=0.7)
    ax.axvline(0, color='k', lw=0.7)
    correct = ((dT < 0) == (dv > 0)).mean()
    ax.text(0.05, 0.95, f'n={len(ev)}\ndirection-consistent: {correct:.0%}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(fc='white', ec='#c0392b', alpha=0.9))
    ax.set_xlabel('Valve step ΔV over 600s (%)')
    ax.set_ylabel('Net ΔT at 600s (°C)')
    ax.set_title('(b) Real valve-step events (0–10% layer)\nconfounding dominates → no reliable gain truth',
                 fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = 'results/exp_201_valve_action/fig_gain_diag.png'
    plt.savefig(out, bbox_inches='tight')
    print('saved:', out)
