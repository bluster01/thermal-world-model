#!/usr/bin/env python3
"""exp_201 增益诊断: 模型增益 vs 真实事件 DiD vs 等百分比理论 (按开度层)。

目的: 判断 under-gain (~1/6) 是"真收缩"(模型系统性低估)还是
"基准错配"(0.006°C/% 的 DiD 地板来自高开度大事件, 常规区真实增益本就小)。

输出: 每开度层 — 模型扰动增益(°C/%) / DiD 事件增益(°C/%) / 理论相对斜率。
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
W = E_cfg.WINDOW_SIZE   # 96
H = 60
n_val_end = 601566
I_V2 = NUMERIC_COLS.index('二级减温调节门阀位')
I_T = TARGET_IDX

raw = data_all
R_FLOW = 50.0
v_pct = np.clip(raw[:, I_V2], 0.0, 100.0) / 100.0
flow_col = (R_FLOW ** (v_pct - 1.0) - 1.0 / R_FLOW) / (1.0 - 1.0 / R_FLOW)
flow_med_train = float(np.median(flow_col[:495407]))
test_raw = np.concatenate([raw, flow_col[:, None]], 1)[n_val_end:]
N_MAX = len(test_raw) - W - H
LAYERS = [(0, 10), (10, 20), (20, 30), (30, 45)]


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
    """模型扰动增益: ±delta% 开度 → mean|ΔT| (反归一化°C), 换算 °C/%."""
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
    return float(np.mean(np.abs(mags)) / (2 * delta))   # °C/%(开度)


def find_valve_events(vmin, vmax, thr=2.0, n_max=600, seed=7):
    """阀位阶跃事件: [t, t+60] 净变化 ≥ thr%, 单调性 ≥80%, 事件前开度 ∈ 层。
    DiD 响应: [T(t+60)-T(t)] 减 180s 前趋势外推 (exp_097 模式)。"""
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
        # 单调性: 步间同号率
        d = np.diff(vseg)
        mono = max((d > 0).mean(), (d < 0).mean())
        if mono < 0.8:
            continue
        # 温度响应 (600s), 减 180s 前趋势外推
        slope_pre = (raw[t, I_T] - raw[max(t - 18, 0), I_T]) / 18.0
        dT = (raw[t + H, I_T] - raw[t, I_T]) - slope_pre * H
        ev.append((t, float(dv), float(dT)))
        if len(ev) >= n_max:
            break
    return ev


if __name__ == '__main__':
    root = 'results/exp_201_valve_action'
    models = {
        'flow_ff10': f'{root}/A1phys_valve_s0_ff10_flow',
        'flow_noff': f'{root}/A1phys_valve_noff_s0_flow',
    }

    # 理论: 等百分比 dF/dV ∝ lnR · R^(V/100-1) / 100, 归一化到 10-20 层 = 1
    Vc = np.array([5, 15, 25, 37.5])
    dFdV = np.log(R_FLOW) * R_FLOW ** (Vc / 100.0 - 1.0) / 100.0
    theo = dFdV / dFdV[1]

    print(f'{"layer":>12} | {"theory":>7} | ' + ' | '.join(f'{k:>18}' for k in models) + ' | DiD_gain')
    did_all = {}
    for (lo, hi) in LAYERS:
        row = [f'[{lo:2d},{hi:2d})', f'{theo[LAYERS.index((lo,hi))]:6.2f}']
        for k, d in models.items():
            m = load_model(d)
            idxs = biased_windows(lo, hi, 100, seed=99)
            g = model_gain(m, idxs)
            row.append(f'{g*1000:.4f}e-3'.replace('e-3', ' m°C/%'))
            # 统一单位: °C/% * 1000 = m°C/%
            row[-1] = f'{g*1000:8.2f} m°C/%'
        ev = find_valve_events(lo, hi)
        if len(ev) >= 20:
            dv = np.array([e[1] for e in ev]); dT = np.array([e[2] for e in ev])
            # 事件增益: 响应/幅度 (°C/%), 加权平均
            g_did = float(np.mean(dT / dv))
            did_all[(lo, hi)] = (len(ev), g_did)
            row.append(f'{g_did*1000:8.2f} m°C/% (n={len(ev)})')
        else:
            row.append(f'n={len(ev)} too few')
        print(' | '.join(row))

    # DiD 方向正确率 (阀位↑ 事件应 dT<0)
    print('\nDiD 事件方向 (阀位↑应降温):')
    for (lo, hi), (n, g) in did_all.items():
        ev = find_valve_events(lo, hi)
        dT = np.array([e[2] for e in ev]); dv = np.array([e[1] for e in ev])
        neg = ((dT < 0) == (dv > 0)).mean()
        print(f'  [{lo:2d},{hi:2d}): n={n} 方向正确率={neg:.1%} 平均|dv|={np.abs(dv).mean():.2f}%')
