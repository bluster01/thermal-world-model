#!/usr/bin/env python3
"""V0 数据审计与事件检测 (冻结协议: docs/PHASE35_SEGMENTED_IDENTIFICATION_REVIEW_2026-08-09.md §5)。

事件定义 (主分析冻结):
- 10s 连续网格; 缺口前后窗口全部排除
- 负荷 > 250 MW
- 事件前 600s: 负荷 range≤15MW; 主汽压 range≤0.4MPa; 两侧末过温度 range≤3°C;
  动作阀位 range≤0.3%
- 阀位剂量: median(post 30-60s) - median(pre 60s), 保留符号
- 主阈值 |delta_v|≥3%; 2%/5% 仅敏感性
- 另一阀同期变化 < max(1%, 0.5*|delta_v|)
- 独立事件间隔 ≥600s; held-step: 后300s 阀位保持 max(0.5%, 0.2*|delta_v|)
- 同时报开/关阀数量, 不允许单侧外推
- 不得以事件后温度/负荷/压力筛选事件
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

COLS = [
    'time', '机组负荷_GENERATOR_POWER', '主蒸汽压力',
    '过热器二级减温器A侧喷水调节门阀位反馈', '过热器二级减温器B侧喷水调节门阀位反馈',
    '选择后二级减温器左侧入口蒸汽', '选择后二级减温器右侧入口蒸汽',
    '选择后左侧二过喷水减温器出口', '选择后右侧二过喷水减温器出口',
    '选择后左侧末级过热器出口汽温', '选择后右侧末级过热器出口汽温',
    '过热器二级减温器A喷水调节阀设定', '过热器二级减温器B喷水调节阀设定',
]

PRE_STEPS = 60          # 600s
POST_DOSE_LO, POST_DOSE_HI = 3, 6   # 30-60s
HELD_STEPS = 30         # 后300s
GAP_SEC = 600           # 事件间隔
SENSITIVITY_THRESHOLDS = [2.0, 3.0, 5.0]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(csv_path):
    df = pd.read_csv(csv_path, usecols=COLS, low_memory=False)
    for c in COLS:
        if c != 'time':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # 时间解析
    df['time'] = pd.to_datetime(df['time'])
    return df


def check_grid(df):
    """检查 10s 连续网格, 返回缺口索引 (间隔>10s 的位置)。"""
    t = df['time'].astype('int64').to_numpy() // 10**9
    dt = np.diff(t)
    gap_idx = np.where(dt > 10)[0]
    return gap_idx, dt


def detect_events(df, side, thr=3.0):
    """V0 事件检测。side: 'A' 或 'B'。返回事件列表 (dict)。"""
    vA = df['过热器二级减温器A侧喷水调节门阀位反馈'].to_numpy()
    vB = df['过热器二级减温器B侧喷水调节门阀位反馈'].to_numpy()
    load = df['机组负荷_GENERATOR_POWER'].to_numpy()
    pres = df['主蒸汽压力'].to_numpy()
    TL = df['选择后左侧末级过热器出口汽温'].to_numpy()
    TR = df['选择后右侧末级过热器出口汽温'].to_numpy()
    t = df['time'].to_numpy()
    n = len(df)

    v_act = vA if side == 'A' else vB
    v_oth = vB if side == 'A' else vA
    t_sec = df['time'].astype('int64').to_numpy() // 10**9

    # 预计算滑动窗口 range (前 600s) — 向量化 (pandas rolling)
    def rolling_range(a, w):
        s = pd.Series(a)
        return (s.rolling(w).max() - s.rolling(w).min()).to_numpy()

    # 阀位阶跃点 (diff >= 阈值的一半, 先粗筛)
    d = np.diff(v_act)
    idx = np.where(np.abs(d) >= 0.5 * thr)[0]
    # 预筛: 只保留有足够上下文且负荷>250 的候选
    load_mean = pd.Series(load).rolling(PRE_STEPS).mean().to_numpy()
    pre_load_r = rolling_range(load, PRE_STEPS)
    pre_pres_r = rolling_range(pres, PRE_STEPS)
    pre_TL_r = rolling_range(TL, PRE_STEPS)
    pre_TR_r = rolling_range(TR, PRE_STEPS)
    pre_v_r = rolling_range(v_act, PRE_STEPS)
    evs = []
    for k in idx:
        t0 = k + 1
        if t0 < PRE_STEPS or t0 + max(POST_DOSE_HI, HELD_STEPS) >= n:
            continue
        if not np.isfinite(load_mean[t0 - 1]) or load_mean[t0 - 1] <= 250:
            continue
        if not np.isfinite(pre_load_r[t0 - 1]) or pre_load_r[t0 - 1] > 15:
            continue
        if not np.isfinite(pre_pres_r[t0 - 1]) or pre_pres_r[t0 - 1] > 0.4:
            continue
        if not np.isfinite(pre_TL_r[t0 - 1]) or pre_TL_r[t0 - 1] > 3.0:
            continue
        if not np.isfinite(pre_TR_r[t0 - 1]) or pre_TR_r[t0 - 1] > 3.0:
            continue
        if not np.isfinite(pre_v_r[t0 - 1]) or pre_v_r[t0 - 1] > 0.3:
            continue
        # 缺口排除: 事件前窗 + 剂量窗 + held 窗内不得有缺口
        pre_win = t_sec[t0 - PRE_STEPS:t0]
        post_win = t_sec[t0:t0 + HELD_STEPS]
        if np.any(np.diff(pre_win) > 10):
            continue
        if np.any(np.diff(post_win) > 10):
            continue
        # 剂量: median(post 30-60s) - median(pre 60s), 有符号
        dose = (np.nanmedian(v_act[t0 + POST_DOSE_LO:t0 + POST_DOSE_HI])
                - np.nanmedian(v_act[t0 - 6:t0]))
        if not np.isfinite(dose) or abs(dose) < thr:
            continue
        # 另一阀同期变化 (剂量窗)
        oth_change = (np.nanmedian(v_oth[t0 + POST_DOSE_LO:t0 + POST_DOSE_HI])
                      - np.nanmedian(v_oth[t0 - 6:t0]))
        if not np.isfinite(oth_change) or abs(oth_change) >= max(1.0, 0.5 * abs(dose)):
            continue
        # 独立事件间隔
        if evs and t0 - evs[-1]['t0'] < GAP_SEC // 10:
            continue
        # held-step: 剂量完成后 (t0+60s) 起 300s 阀位保持在 max(0.5%, 0.2*|dose|)
        held_lo = t0 + POST_DOSE_HI
        held_v = v_act[held_lo:held_lo + HELD_STEPS]
        held_ok = (np.isfinite(held_v).all()
                   and np.nanmax(held_v) - np.nanmin(held_v)
                   <= max(0.5, 0.2 * abs(dose)))
        evs.append({
            't0': int(t0),
            'time': str(t[t0]),
            'date': str(t[t0])[:10],
            'dose': round(float(dose), 3),
            'direction': 'open' if dose > 0 else 'close',
            'held_step': bool(held_ok),
            'pre_load_mean': round(float(load_mean[t0 - 1]), 1),
        })
    return evs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--sha-csv', default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_sha = args.sha_csv or sha256(args.csv)
    df = load(args.csv)
    n = len(df)

    gap_idx, dt = check_grid(df)
    data_audit = {
        'rows': n,
        'time_min': str(df['time'].min()),
        'time_max': str(df['time'].max()),
        'gap_count': int(len(gap_idx)),
        'gap_positions': gap_idx[:50].tolist(),
        'csv_sha256': csv_sha,
        'cols': COLS,
    }
    with open(os.path.join(args.out_dir, 'data_audit.json'), 'w') as f:
        json.dump(data_audit, f, indent=1)

    # 事件检测: 3% 主阈值 + 2%/5% 敏感性
    summary = {}
    all_events = []
    for side in ['A', 'B']:
        for thr in SENSITIVITY_THRESHOLDS:
            evs = detect_events(df, side, thr)
            opens = [e for e in evs if e['direction'] == 'open']
            closes = [e for e in evs if e['direction'] == 'close']
            held = [e for e in evs if e['held_step']]
            days = len(set(e['date'] for e in evs))
            summary[f'{side}_thr{thr}'] = {
                'n': len(evs), 'open': len(opens), 'close': len(closes),
                'held_step': len(held), 'days': days,
            }
        # 主阈值 3% 事件进 manifest (供 V1-V4)
        evs = detect_events(df, side, 3.0)
        for e in evs:
            e['side'] = side
        all_events.extend(evs)

    with open(os.path.join(args.out_dir, 'event_manifest.jsonl'), 'w') as f:
        for e in sorted(all_events, key=lambda x: x['time']):
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    out = {'data_audit': data_audit, 'event_summary': summary}
    with open(os.path.join(args.out_dir, 'run_manifest_v0.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
