#!/usr/bin/env python3
"""V1 导前区 2×2 有符号温降响应 (审计协议 §6)。

R_s(h) = sign(delta_v) * [drop_s(t+h) - mean(drop_s, pre-window)]
  drop_left  = T_in_left  - T_out2_left   (近端物理温降)
  drop_right = T_in_right - T_out2_right
配对主对比: A阀: R_right - R_left;  B阀: R_left - R_right
主终点: 30-300s 响应面积; 60/120/180/300s 轨迹诊断
CI: UTC 日聚类 bootstrap (事件按日聚合, 不能当独立样本)
注意: V0 判定 INCONCLUSIVE (事件<30/回路), 本输出仅 exploratory, 不用于确认。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

COLS = [
    'time', '过热器二级减温器A侧喷水调节门阀位反馈', '过热器二级减温器B侧喷水调节门阀位反馈',
    '选择后二级减温器左侧入口蒸汽', '选择后二级减温器右侧入口蒸汽',
    '选择后左侧二过喷水减温器出口', '选择后右侧二过喷水减温器出口',
]
PRE = 60          # 前 600s
H_RESP = 60       # 响应窗 600s (主终点 30-300s = 步 3-30)
AREA_LO, AREA_HI = 3, 30


def load(csv_path):
    df = pd.read_csv(csv_path, usecols=COLS, low_memory=False)
    for c in COLS:
        if c != 'time':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df['time'])
    return df


def day_bootstrap_cis(diffs_by_day, n_boot=2000, alpha=0.05):
    """按日聚类 bootstrap: 输入 {day: [diff值]}, 输出 (mean, ci_lo, ci_hi)。"""
    days = list(diffs_by_day.keys())
    if not days:
        return None, None, None, 0
    day_means = np.array([np.mean(diffs_by_day[d]) for d in days])
    m = np.mean(day_means)
    if len(days) < 2:
        return m, m, m, len(days)
    rng = np.random.default_rng(42)
    boot = np.array([np.mean(rng.choice(day_means, len(day_means), replace=True))
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, 100 * alpha / 2), np.percentile(boot, 100 * (1 - alpha / 2))
    return m, lo, hi, len(days)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    df = load(args.csv)
    n = len(df)
    t_sec = df['time'].astype('int64').to_numpy() // 10**9
    T_inL = df['选择后二级减温器左侧入口蒸汽'].to_numpy()
    T_inR = df['选择后二级减温器右侧入口蒸汽'].to_numpy()
    T_out2L = df['选择后左侧二过喷水减温器出口'].to_numpy()
    T_out2R = df['选择后右侧二过喷水减温器出口'].to_numpy()

    evs = [json.loads(l) for l in open(args.events)]
    dropL = T_inL - T_out2L
    dropR = T_inR - T_out2R

    out_rows = []
    summary = {}
    for side in ['A', 'B']:
        sel = [e for e in evs if e['side'] == side]
        summary[side] = {'n': len(sel)}
        if not sel:
            continue
        # 每事件: R_left/R_right 曲线 + 面积
        rows = []
        for e in sel:
            t0 = e['t0']
            dv = e['dose']
            baseL = np.nanmedian(dropL[t0 - 6:t0])
            baseR = np.nanmedian(dropR[t0 - 6:t0])
            rL, rR = [], []
            for h in range(H_RESP + 1):
                i = min(t0 + h, n - 1)
                rL.append(np.sign(dv) * (dropL[i] - baseL))
                rR.append(np.sign(dv) * (dropR[i] - baseR))
            rL, rR = np.array(rL), np.array(rR)
            area = np.mean(rR[AREA_LO:AREA_HI + 1]) - np.mean(rL[AREA_LO:AREA_HI + 1])
            rows.append({
                'side': side, 'time': e['time'], 'date': e['date'],
                'dose': e['dose'], 'direction': e['direction'],
                'area_R_minus_L': round(float(area), 4),
                'R_left_60': round(float(rL[6]), 4), 'R_right_60': round(float(rR[6]), 4),
                'R_left_120': round(float(rL[12]), 4), 'R_right_120': round(float(rR[12]), 4),
                'R_left_180': round(float(rL[18]), 4), 'R_right_180': round(float(rR[18]), 4),
                'R_left_300': round(float(rL[30]), 4), 'R_right_300': round(float(rR[30]), 4),
            })
            out_rows.append(rows[-1])
        # 日块 bootstrap: A阀: R_right-R_left; B阀: R_left-R_right (符号翻转)
        sign = 1 if side == 'A' else -1
        diff_by_day = {}
        for r in rows:
            diff_by_day.setdefault(r['date'], []).append(sign * r['area_R_minus_L'])
        m, lo, hi, nd = day_bootstrap_cis(diff_by_day)
        summary[side]['paired_area_mean'] = round(float(m), 4) if m is not None else None
        summary[side]['paired_area_ci95'] = [round(float(lo), 4), round(float(hi), 4)] if lo is not None else None
        summary[side]['days'] = nd
        summary[side]['open'] = sum(1 for r in rows if r['direction'] == 'open')
        summary[side]['close'] = sum(1 for r in rows if r['direction'] == 'close')
        # 轨迹均值 (跨事件)
        for h_name, h in [('60', 6), ('120', 12), ('180', 18), ('300', 30)]:
            k = f'R_left_{h_name}'
            summary[side][f'mean_{k}'] = round(float(np.mean([r[k] for r in rows])), 4)
            k2 = f'R_right_{h_name}'
            summary[side][f'mean_{k2}'] = round(float(np.mean([r[k2] for r in rows])), 4)

    with open(os.path.join(args.out_dir, 'leading_response_by_event.csv'), 'w') as f:
        cols = list(out_rows[0].keys()) if out_rows else []
        f.write(','.join(cols) + '\n')
        for r in out_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    with open(os.path.join(args.out_dir, 'leading_summary.json'), 'w') as f:
        json.dump({'v0_status': 'INCONCLUSIVE (events<30/loop)',
                   'note': 'exploratory only, not confirmatory',
                   'summary': summary}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
