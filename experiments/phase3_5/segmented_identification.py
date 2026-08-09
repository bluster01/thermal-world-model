#!/usr/bin/env python3
"""分段辨识正式脚本: 导前区 FOPDT + 惰性区差分脉冲响应 (交叉喷水确认)。

用法: python experiments/phase3_5/segmented_identification.py --csv <路径> [--out <json>]
方法依据: docs/papers/IDENTIFICATION_METHODS_SUMMARY_2026-08-09.md (Cao2021/PoliMi)
物理配置: 左=A, 右=B; A阀→右二过(交叉), B阀→左二过(交叉); 惰性区直连(金属导热)。
输出: JSON (配对矩阵: K<0率/峰值/能量/时滞) + 控制台表格。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

COLS = ['过热器二级减温器A侧喷水调节门阀位反馈', '过热器二级减温器B侧喷水调节门阀位反馈',
        '选择后左侧二过喷水减温器出口', '选择后右侧二过喷水减温器出口',
        '选择后左侧末级过热器出口汽温', '选择后右侧末级过热器出口汽温']

PRE, GAP = 60, 60  # 600s 前窗安静, 600s 事件间隔


def load(csv_path):
    df = pd.read_csv(csv_path, usecols=COLS, low_memory=False)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = {k: df[k].to_numpy() for k in COLS}
    out['n'] = len(df)
    return out


def valve_events(v, n, thr=0.8):
    d = np.diff(v)
    idx = np.where(np.abs(d) >= thr)[0]
    evs = []
    for k in idx:
        t0 = k + 1
        if t0 < PRE or t0 + 360 >= n:
            continue
        pre = v[t0 - PRE:t0]
        if not np.isfinite(pre).all() or np.nanmax(pre) - np.nanmin(pre) > 0.3:
            continue
        seg = v[t0:t0 + 3]
        if not np.isfinite(seg).all() or np.max(np.abs(seg - v[t0])) > 0.3 * abs(d[k]):
            continue
        if evs and t0 - evs[-1][0] < GAP:
            continue
        evs.append((t0, float(d[k])))
    return evs


def fopdt(t, K, T, tau):
    return np.where(t >= tau, K * (1 - np.exp(-(t - tau) / max(T, 1e-6))), 0.0)


def fit_fopdt(t, y, u):
    from scipy.optimize import curve_fit
    if len(t) < 10 or not np.isfinite(y).all():
        return None
    try:
        y0 = np.median(y[:3])
        ys = (y - y0) / max(abs(u), 1e-9)
        popt, _ = curve_fit(fopdt, t, ys, p0=[ys[-1], 100.0, 30.0],
                            bounds=([-np.inf, 1.0, 0.0], [np.inf, 600.0, 300.0]),
                            maxfev=50000)
        resid = ys - fopdt(t, *popt)
        r2 = 1 - np.sum(resid ** 2) / max(np.sum((ys - np.mean(ys)) ** 2), 1e-12)
        return (*popt, r2)
    except Exception:
        return None


def leading_zone(v_act, T_cross, T_direct, evs):
    """导前区: 阀位阶跃 → 二过出口 0-300s FOPDT。"""
    t_axis = np.arange(0, 301, 10)
    res = {}
    for resp_name, T_resp in [('cross', T_cross), ('direct', T_direct)]:
        fits = []
        for t0, dv in evs:
            y = np.array([T_resp[min(t0 + h, len(T_resp) - 1)] - T_resp[t0 - 1] for h in t_axis])
            if np.isfinite(y).all():
                f = fit_fopdt(t_axis, y, dv)
                if f:
                    fits.append(f)
        if len(fits) < 5:
            res[resp_name] = {'n_fit': len(fits)}
            continue
        fits = np.array(fits)
        K, T, tau, r2 = fits[:, 0], fits[:, 1], fits[:, 2], fits[:, 3]
        neg = float(np.mean(K < 0) * 100)
        good = fits[K < 0]
        if len(good) >= 5:
            res[resp_name] = {
                'K_neg_rate_pct': round(neg, 1), 'n_fit': len(fits),
                'K_med': round(float(np.median(good[:, 0])), 3),
                'T_med_s': round(float(np.median(good[:, 1])), 0),
                'tau_med_s': round(float(np.median(good[:, 2])), 0),
                'R2_med': round(float(np.median(good[:, 3])), 2)}
        else:
            res[resp_name] = {'K_neg_rate_pct': round(neg, 1), 'n_fit': len(fits)}
    return res


def impulse_response(T_in, T_out, evs, n, L=180):
    """惰性区: 事件窗差分脉冲响应。"""
    H = 360
    hs = []
    for t0, dv in evs:
        i0 = t0 - 1
        u = T_in[i0:i0 + H]
        y = T_out[i0:i0 + H]
        du = np.diff(u)
        dy = np.diff(y)
        m = np.isfinite(du) & np.isfinite(dy)
        if m.sum() < 100:
            continue
        du = du[m]
        dy = dy[m]
        var_u = np.var(du)
        if var_u < 1e-6:
            continue
        h = np.zeros(L + 1)
        for l in range(L + 1):
            a, b = (du, dy) if l == 0 else (du[:-l], dy[l:])
            mm = np.isfinite(a) & np.isfinite(b)
            if mm.sum() < 50:
                break
            h[l] = np.mean(a[mm] * b[mm]) / var_u
        hs.append(h)
    if not hs:
        return None
    return np.median(np.array(hs), axis=0)


def summarize_ir(name, h):
    if h is None:
        return {'status': 'no_data'}
    h_abs = np.abs(h)
    pk = int(np.argmax(h_abs))
    e60 = float(np.sum(h_abs[:7] ** 2) / max(np.sum(h_abs ** 2), 1e-12))
    return {'peak_t_s': pk * 10, 'peak_h': round(float(h[pk]), 4),
            'energy_60s_pct': round(e60 * 100, 0),
            'h0': round(float(h[0]), 4), 'h60s': round(float(h[6]), 4),
            'h300s': round(float(h[30]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    D = load(args.csv)
    n = D['n']
    vA = D['过热器二级减温器A侧喷水调节门阀位反馈']
    vB = D['过热器二级减温器B侧喷水调节门阀位反馈']
    T2L = D['选择后左侧二过喷水减温器出口']
    T2R = D['选择后右侧二过喷水减温器出口']
    TmL = D['选择后左侧末级过热器出口汽温']
    TmR = D['选择后右侧末级过热器出口汽温']

    evA = valve_events(vA, n)
    evB = valve_events(vB, n)

    result = {
        'data': {'rows': n, 'csv': os.path.basename(args.csv)},
        'source': 'experiments/phase3_5/segmented_identification.py',
        'cross_config': 'A_valve->right_2nd_out (cross), B_valve->left_2nd_out (cross)',
        'leading_zone': {
            'A_valve': {'n_events': len(evA),
                        **leading_zone(vA, T2R, T2L, evA)},
            'B_valve': {'n_events': len(evB),
                        **leading_zone(vB, T2L, T2R, evB)},
        },
        'inertia_zone': {
            'A_events': {
                'n_events': len(evA),
                'right2nd_to_right_ms [direct]': summarize_ir(
                    'direct', impulse_response(T2R, TmR, evA, n)),
                'right2nd_to_left_ms [negative]': summarize_ir(
                    'neg', impulse_response(T2R, TmL, evA, n)),
            },
            'B_events': {
                'n_events': len(evB),
                'left2nd_to_left_ms [direct]': summarize_ir(
                    'direct', impulse_response(T2L, TmL, evB, n)),
                'left2nd_to_right_ms [negative]': summarize_ir(
                    'neg', impulse_response(T2L, TmR, evB, n)),
            },
        },
    }

    print(json.dumps(result, indent=1, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        print(f'\n[written] {args.out}', flush=True)


if __name__ == '__main__':
    main()
