#!/usr/bin/env python3
"""1s 数据 SP 阶跃事件提取 + S/D 稳态分层 + first-stage 诊断。

只收集稀疏更新列(SP/阀位/负荷/压力/温度/指令), 事件窗口内按需 LOCF,
避免全量 1s 重采样。输出事件清单 JSON, 供后续 gain/IRF 分析。
"""
import json
import sys
import numpy as np
import pandas as pd

CSV = '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/merged_data/A侧主汽温全数据4.csv'
OUT = '/home/bluster/projectA/thermal-world-model/results/phase35_sp1s_events.json'

COLS = ['date', '机组负荷', '主蒸汽压力', '末级过热器出口汽温',
        '二级减温调节阀设定', '二级减温调节门阀位', '二级减温喷水调节门指令']

SP_THR = 1.0          # °C 阶跃阈值
SP_HOLD_S = 60.0      # 阶跃后保持时长(检测用)
PRE_S = 960.0         # 事件前稳定性窗口
POST_S = 600.0        # 事件后响应窗口

# 稳态门槛 (S 层, 预注册): 事件前 PRE_S 内 range 限制
S_LOAD_RANGE = 5.0    # MW
S_PRES_RANGE = 0.2    # MPa
S_TEMP_RANGE = 1.0    # °C

def main():
    print('scanning sparse updates...', flush=True)
    upd = {c: ([], []) for c in COLS[1:]}
    first_ns = last_ns = None
    for chunk in pd.read_csv(CSV, usecols=COLS, chunksize=2_000_000, low_memory=False):
        t = pd.to_datetime(chunk['date'], utc=True, errors='coerce')
        ns = (t.astype('int64') // 1000).to_numpy()   # ns
        for c in COLS[1:]:
            v = pd.to_numeric(chunk[c], errors='coerce').to_numpy(dtype=np.float64)
            m = np.isfinite(v)
            if m.any():
                upd[c][0].extend(ns[m]); upd[c][1].extend(v[m])
        if first_ns is None:
            first_ns = ns[0]
        last_ns = ns[-1]
    print(f'scan done: {first_ns} → {last_ns}', flush=True)

    # 排序去重 (稀疏行时间戳可能乱序)
    for c in COLS[1:]:
        ts, vs = upd[c]
        order = np.argsort(ts, kind='stable')
        upd[c] = (np.array(ts)[order], np.array(vs)[order])

    sp_ts, sp_v = upd['二级减温调节阀设定']
    # 阶跃检测: |Δ| ≥ SP_THR
    d = np.diff(sp_v)
    idx = np.where(np.abs(d) >= SP_THR)[0]
    events = []
    for k in idx:
        t0 = sp_ts[k + 1]           # 阶跃后第一个点
        # 阶跃后保持: 60s 内 SP 不再反向大动
        after = sp_ts[k + 1:]
        hold = after <= t0 + SP_HOLD_S * 1e6
        if hold.sum() < 2:
            continue
        if np.abs(sp_v[k + 1 + hold.sum() - 1] - sp_v[k + 1]) > 0.5 * SP_THR:
            continue   # 未保持, 跳过
        if t0 - PRE_S * 1e6 < first_ns or t0 + POST_S * 1e6 > last_ns:
            continue
        events.append((t0, sp_v[k], sp_v[k + 1]))

    print(f'candidate SP steps: {len(events)}', flush=True)

    # 事件窗口特征: 用 searchsorted 取各列在 [t0-PRE_S, t0+POST_S] 的更新, 窗口内 LOCF
    rows = []
    for t0, sp_before, sp_after in events:
        feats = {'t0_ns': int(t0), 'dsp': float(sp_after - sp_before)}
        pre = {}
        for c in COLS[1:]:
            ts, vs = upd[c]
            lo = np.searchsorted(ts, t0 - PRE_S * 1e6, side='left')
            hi = np.searchsorted(ts, t0 + POST_S * 1e6, side='right')
            w_ts, w_v = ts[lo:hi], vs[lo:hi]
            # 前向填充到 1s 网格 (事件窗口内 ~1560 点, 可控)
            if len(w_ts) == 0:
                pre[c] = None; continue
            grid = np.arange((t0 - PRE_S * 1e6) // 1e6, (t0 + POST_S * 1e6) // 1e6 + 1) * 1e6
            pos = np.searchsorted(w_ts, grid, side='right') - 1
            filled = w_v[np.clip(pos, 0, len(w_v) - 1)]
            filled[pos < 0] = np.nan
            pre[c] = filled
        T = pre['末级过热器出口汽温']
        n_pre = int(PRE_S); n_post = int(POST_S)
        if T is None or not np.isfinite(T[:n_pre]).sum() >= 0.9 * n_pre:
            continue
        n_pre = int(PRE_S); n_post = int(POST_S)
        load = pre['机组负荷']; pres = pre['主蒸汽压力']; valve = pre['二级减温调节门阀位']
        if load is None or pres is None or valve is None:
            continue   # 事件窗口内关键协变量无观测, 跳过
        # 事件前紧邻窗口 (数组布局: [t0-960s .. t0+600s], n_pre=960 是 t0)
        feats.update({
            'load_range_60': float(np.nanmax(load[n_pre-60:n_pre]) - np.nanmin(load[n_pre-60:n_pre])),
            'pres_range_60': float(np.nanmax(pres[n_pre-60:n_pre]) - np.nanmin(pres[n_pre-60:n_pre])),
            'temp_range_60': float(np.nanmax(T[n_pre-60:n_pre]) - np.nanmin(T[n_pre-60:n_pre])),
            'load_range_600': float(np.nanmax(load[n_pre-600:n_pre]) - np.nanmin(load[n_pre-600:n_pre])),
            'pres_range_600': float(np.nanmax(pres[n_pre-600:n_pre]) - np.nanmin(pres[n_pre-600:n_pre])),
            'temp_range_600': float(np.nanmax(T[n_pre-600:n_pre]) - np.nanmin(T[n_pre-600:n_pre])),
            'load_range_960': float(np.nanmax(load[:n_pre]) - np.nanmin(load[:n_pre])),
            'pres_range_960': float(np.nanmax(pres[:n_pre]) - np.nanmin(pres[:n_pre])),
            'temp_range_960': float(np.nanmax(T[:n_pre]) - np.nanmin(T[:n_pre])),
            'dT_post_600': float(T[n_pre + 600] - T[n_pre - 1]) if np.isfinite(T[n_pre + 600]) else None,
            'valve_dv_30s': float(valve[n_pre + 3] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 3]) else None,
            'valve_dv_600s': float(valve[n_pre + 600] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 600]) else None,
        })
        rows.append(feats)

    df = pd.DataFrame(rows)
    n_s = int(((df['load_range_600'] <= S_LOAD_RANGE) & (df['pres_range_600'] <= S_PRES_RANGE)
               & (df['temp_range_600'] <= S_TEMP_RANGE)).sum())
    n_s960 = int(((df['load_range_960'] <= S_LOAD_RANGE) & (df['pres_range_960'] <= S_PRES_RANGE)
                  & (df['temp_range_960'] <= S_TEMP_RANGE)).sum())
    print(f'events with full pre-window: {len(df)}')
    print(f'S-layer (600s): {n_s}   S-layer (960s): {n_s960}')
    if len(df):
        print(df[['dsp', 'load_range_600', 'pres_range_600', 'temp_range_600',
                  'dT_post_600', 'valve_dv_30s', 'valve_dv_600s']].describe().round(3).to_string())
    with open(OUT, 'w') as f:
        json.dump({'n': len(df), 'n_s_600': n_s, 'n_s_960': n_s960,
                   'thresholds': {'sp': SP_THR, 's_load': S_LOAD_RANGE, 's_pres': S_PRES_RANGE,
                                  's_temp': S_TEMP_RANGE},
                   'events': rows}, f, ensure_ascii=False, indent=1)
    print('saved:', OUT)

if __name__ == '__main__':
    main()
