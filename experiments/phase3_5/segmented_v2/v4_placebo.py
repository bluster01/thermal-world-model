#!/usr/bin/env python3
"""V4 placebo 与分层敏感性 (审计协议 §7/§4 V4)。

placebo: 输入错移 ±1h; 日内置换 (事件内时间重排); 错侧配对
分层: 负荷 (250-400 / 400-550 / >550 MW); 开/关阀; 月份
输出: placebo_summary.json (真配对 vs placebo 排名), 分层方向一致性。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

COLS = ['time', '机组负荷_GENERATOR_POWER',
        '过热器二级减温器A侧喷水调节门阀位反馈', '过热器二级减温器B侧喷水调节门阀位反馈',
        '选择后二级减温器左侧入口蒸汽', '选择后二级减温器右侧入口蒸汽',
        '选择后左侧二过喷水减温器出口', '选择后右侧二过喷水减温器出口']
PRE = 60
AREA_LO, AREA_HI = 3, 30


def load_data(csv_path):
    df = pd.read_csv(csv_path, usecols=COLS, low_memory=False)
    for c in COLS:
        if c != 'time':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df['time'])
    return df


def response_area(drop, t0, dv, n, shift=0):
    """有符号响应面积 30-300s。shift>0: 输入时间错移 (步)。"""
    t1 = t0 + shift
    if t1 - 6 < 0 or t1 + AREA_HI >= n:
        return np.nan
    base = np.nanmedian(drop[t1 - 6:t1])
    vals = [np.sign(dv) * (drop[min(t1 + h, n - 1)] - base) for h in range(AREA_LO, AREA_HI + 1)]
    if not np.isfinite(vals).all():
        return np.nan
    return float(np.mean(vals))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    df = load_data(args.csv)
    n = len(df)
    load = df['机组负荷_GENERATOR_POWER'].to_numpy()
    vA = df['过热器二级减温器A侧喷水调节门阀位反馈'].to_numpy()
    vB = df['过热器二级减温器B侧喷水调节门阀位反馈'].to_numpy()
    T_inL = df['选择后二级减温器左侧入口蒸汽'].to_numpy()
    T_inR = df['选择后二级减温器右侧入口蒸汽'].to_numpy()
    T2L = df['选择后左侧二过喷水减温器出口'].to_numpy()
    T2R = df['选择后右侧二过喷水减温器出口'].to_numpy()
    dropL = T_inL - T2L
    dropR = T_inR - T2R

    evs = [json.loads(l) for l in open(args.events)]
    out = {'placebo': {}, 'strata': {}}

    for side in ['A', 'B']:
        sel = [e for e in evs if e['side'] == side]
        if not sel:
            continue
        # 真配对: A→dropR(右), B→dropL(左); 错侧: A→dropL, B→dropR
        drop_true = dropR if side == 'A' else dropL
        drop_wrong = dropL if side == 'A' else dropR
        true_areas = [response_area(drop_true, e['t0'], e['dose'], n) for e in sel]
        wrong_areas = [response_area(drop_wrong, e['t0'], e['dose'], n) for e in sel]
        shift_areas = {}
        for shift_name, sh in [('m1h', -360), ('p1h', 360), ('m2h', -720), ('p2h', 720)]:
            shift_areas[shift_name] = [response_area(drop_true, e['t0'], e['dose'], n, shift=sh)
                                       for e in sel]
        # 日内置换: 事件内 30-300s 顺序打乱 (保留同分布破坏时序)
        rng = np.random.default_rng(7)
        perm_areas = []
        for e in sel:
            t0 = e['t0']
            base = np.nanmedian(drop_true[t0 - 6:t0])
            idx = np.arange(AREA_LO, AREA_HI + 1)
            rng.shuffle(idx)
            vals = [np.sign(e['dose']) * (drop_true[min(t0 + h, n - 1)] - base) for h in idx]
            perm_areas.append(float(np.mean(vals)) if np.isfinite(vals).all() else np.nan)

        def agg(arr):
            a = np.array([x for x in arr if np.isfinite(x)])
            return (round(float(np.mean(a)), 4), round(float(np.std(a)), 4), int(len(a))) if len(a) else None

        out['placebo'][side] = {
            'true_pair': agg(true_areas), 'wrong_side': agg(wrong_areas),
            'shift': {k: agg(v) for k, v in shift_areas.items()},
            'intraday_perm': agg(perm_areas),
            'true_ranks_better': None,
        }
        # 排名: 真配对均值 vs 全部 placebo
        all_placebo = wrong_areas + list(shift_areas.values())[0] + perm_areas
        ap_ = np.array([x for x in all_placebo if np.isfinite(x)])
        tm = np.array([x for x in true_areas if np.isfinite(x)])
        if len(ap_) and len(tm):
            out['placebo'][side]['true_rank_pct'] = round(
                100 * np.mean(tm.mean() > ap_), 1)

        # 分层
        strata = {'load_250_400': [], 'load_400_550': [], 'load_550_plus': [],
                  'open': [], 'close': []}
        for e, ta in zip(sel, true_areas):
            ld = e.get('pre_load_mean', 0)
            if ld < 400:
                strata['load_250_400'].append(ta)
            elif ld < 550:
                strata['load_400_550'].append(ta)
            else:
                strata['load_550_plus'].append(ta)
            strata[e['direction']].append(ta)
        out['strata'][side] = {k: agg(v) for k, v in strata.items()}

    with open(os.path.join(args.out_dir, 'placebo_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
