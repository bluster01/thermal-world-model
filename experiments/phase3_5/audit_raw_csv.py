#!/usr/bin/env python3
"""TODO 任务1: A/B CSV 审计 — 表头/时间范围/关键tag/单位 (Linux 只读)。"""
import sys
import pandas as pd
import numpy as np

FILES = {
    'A': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/merged_data/A侧主汽温全数据4.csv',
    'B': '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/merged_data/B侧主汽温全数据4.csv',
}
KEY = ['机组负荷', '主蒸汽压力', '主给水流量', '未校正总煤量', '主蒸汽流量',
       '二级减温器入口温度', '二级减温器出口温度', '末级过热器出口汽温',
       '二级减温中间设定值', '二级减温喷水调节门指令', '二级减温调节阀设定',
       '二级减温调节门阀位']

for side, path in FILES.items():
    print(f'===== side {side}: {path.split("/")[-1]} =====')
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    print(f'columns: {len(cols)} | first: {cols[0]} | last: {cols[-1]}')
    print(f'BOM: {cols[0].startswith(chr(0xfeff))} | date col: {"date" in cols}')
    stats = {c: dict(n=0, vmin=np.inf, vmax=-np.inf) for c in KEY}
    tmin, tmax = None, None
    rows = 0
    for chunk in pd.read_csv(path, usecols=lambda c: c in KEY or c == 'date',
                            chunksize=2_000_000, low_memory=False):
        rows += len(chunk)
        t = pd.to_datetime(chunk['date'], utc=True, errors='coerce')
        tt = t.dropna()
        if len(tt):
            tm = tt.min(); tx = tt.max()
            tmin = tm if tmin is None else min(tmin, tm)
            tmax = tx if tmax is None else max(tmax, tx)
        for c in KEY:
            v = pd.to_numeric(chunk[c], errors='coerce').to_numpy()
            f = v[np.isfinite(v)]
            if len(f):
                stats[c]['n'] += len(f)
                stats[c]['vmin'] = min(stats[c]['vmin'], f.min())
                stats[c]['vmax'] = max(stats[c]['vmax'], f.max())
    print(f'rows(raw lines): {rows}')
    print(f'time range: {tmin} → {tmax}  (span {(tmax - tmin).total_seconds()/86400:.1f} days)')
    print(f'{"column":<22} {"nonnull":>10} {"rate":>7} {"min":>12} {"max":>12}')
    for c in KEY:
        s = stats[c]
        print(f'{c:<22} {s["n"]:>10} {s["n"]/rows:>6.1%} {s["vmin"]:>12.4g} {s["vmax"]:>12.4g}')
    print()
