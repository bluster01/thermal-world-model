#!/usr/bin/env python3
"""1s 数据 SP 阶跃事件提取 + S/D 稳态分层 + first-stage 诊断 (审计 5A 修复版)。

修复项 (2026-08-09):
- P0-1: 显式保存 dv_3s/dv_10s/dv_30s/dv_60s/dv_180s/dv_600s, 不再用误导性 valve_dv_30s
- P0-2: 参数化 --side/--split/--input/--output, 事件写 split 标记, 输出 provenance
- P1: t0_ns 存 epoch 纳秒 (原为微秒); 补 source SHA/生成 commit/拒绝漏斗
- 只收集稀疏更新列(SP/阀位/负荷/压力/温度/指令), 事件窗口内按需 LOCF
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = '/home/bluster/Desktop/AI/时序预测/AA数据中心/伊敏12.10/merged_data/A侧主汽温全数据4.csv'
DEFAULT_OUT = '/home/bluster/projectA/thermal-world-model/results/phase35_sp1s_events_v2.json'

COLS = ['date', '机组负荷', '主蒸汽压力', '末级过热器出口汽温',
        '二级减温调节阀设定', '二级减温调节门阀位', '二级减温喷水调节门指令']

SP_THR = 1.0          # °C 阶跃阈值
SP_HOLD_S = 60.0      # 阶跃后保持时长(检测用)
PRE_S = 960.0         # 事件前稳定性窗口
POST_S = 600.0        # 事件后响应窗口
MIN_GAP_S = 600.0     # 相邻入选事件最小间隔，避免响应窗重叠

# 稳态门槛 (S 层, 预注册): 事件前 PRE_S 内 range 限制
S_LOAD_RANGE = 5.0    # MW
S_PRES_RANGE = 0.2    # MPa
S_TEMP_RANGE = 1.0    # °C

# split 比例 (与 Phase 3.5 cache 一致)
SPLIT_FRAC = (0.60, 0.20, 0.20)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'unknown'


def git_status_paths() -> list[str] | None:
    try:
        output = subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parents[2]), 'status', '--porcelain'],
            stderr=subprocess.DEVNULL, text=True,
        )
        return [line[3:].strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return None


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def split_of(t_ns: int, grid_start_ns: int, grid_end_ns: int, n_rows: int) -> str:
    """按 Phase 3.5 cache 的 60/20/20 时间边界判定 split。"""
    step_ns = 10_000_000_000
    train_end = grid_start_ns + int(n_rows * SPLIT_FRAC[0]) * step_ns
    val_end = grid_start_ns + int(n_rows * (SPLIT_FRAC[0] + SPLIT_FRAC[1])) * step_ns
    if t_ns < train_end:
        return 'train'
    if t_ns < val_end:
        return 'validation'
    return 'test'


def filter_rows_by_split(rows: list[dict], split: str) -> list[dict]:
    if split == 'all':
        return list(rows)
    return [row for row in rows if row['split'] == split]


def validate_split_access(split: str, allow_test_access: bool) -> None:
    if split in {'test', 'all'} and not allow_test_access:
        raise ValueError("test outcomes are locked; pass --allow-test-access for split=test/all")


def event_grid_ns(t0_ns: int, pre_s: int, post_s: int) -> np.ndarray:
    """Build an exact event-aligned 1 s grid without float timestamp rounding."""
    offsets = np.arange(-pre_s, post_s + 1, dtype=np.int64) * 1_000_000_000
    return np.int64(t0_ns) + offsets


def held_within(values: np.ndarray, reference: float, tolerance: float) -> bool:
    values = np.asarray(values, dtype=np.float64)
    return bool(len(values) and np.isfinite(values).all() and np.max(np.abs(values - reference)) <= tolerance)


def respects_min_gap(t0_ns: int, previous_t0_ns: int | None, min_gap_s: float) -> bool:
    return previous_t0_ns is None or t0_ns - previous_t0_ns >= min_gap_s * 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--side', choices=['A', 'B'], default='A')
    ap.add_argument('--split', choices=['train', 'validation', 'test', 'all'], default='validation',
                    help='只保留指定 split 的事件；test/all 需要显式解锁')
    ap.add_argument('--allow-test-access', action='store_true')
    ap.add_argument('--allow-dirty', action='store_true', help='仅调试；正式产物禁止从 dirty tree 生成')
    ap.add_argument('--input', default=DEFAULT_CSV)
    ap.add_argument('--output', default=DEFAULT_OUT)
    ap.add_argument('--grid-start-ns', type=int, required=True,
                    help='cache grid 起始 ns (来自 cache manifest)')
    ap.add_argument('--grid-end-ns', type=int, required=True)
    ap.add_argument('--grid-rows', type=int, required=True)
    args = ap.parse_args()
    try:
        validate_split_access(args.split, args.allow_test_access)
    except ValueError as exc:
        ap.error(str(exc))
    dirty_paths_at_start = git_status_paths()
    dirty_non_result_paths = [
        path for path in (dirty_paths_at_start or []) if not path.replace('\\', '/').startswith('results/')
    ]
    if dirty_non_result_paths and not args.allow_dirty:
        ap.error(f'working tree has uncommitted non-result files: {dirty_non_result_paths}')

    print('scanning sparse updates...', flush=True)
    upd = {c: ([], []) for c in COLS[1:]}
    first_ns = last_ns = None
    for chunk in pd.read_csv(args.input, usecols=COLS, chunksize=2_000_000, low_memory=False):
        t = pd.to_datetime(chunk['date'], utc=True, errors='coerce')
        ns = t.astype('int64').to_numpy()  # 直接存 ns, 不再 //1000
        for c in COLS[1:]:
            v = pd.to_numeric(chunk[c], errors='coerce').to_numpy(dtype=np.float64)
            m = np.isfinite(v)
            if m.any():
                upd[c][0].extend(ns[m]); upd[c][1].extend(v[m])
        if first_ns is None:
            first_ns = ns[0]
        last_ns = ns[-1]
    print(f'scan done: {first_ns} → {last_ns}', flush=True)

    for c in COLS[1:]:
        ts, vs = upd[c]
        order = np.argsort(ts, kind='stable')
        upd[c] = (np.array(ts)[order], np.array(vs)[order])

    sp_ts, sp_v = upd['二级减温调节阀设定']
    d = np.diff(sp_v)
    idx = np.where(np.abs(d) >= SP_THR)[0]
    candidates = 0
    rejected_hold = 0
    rejected_window = 0
    rejected_min_gap = 0
    events = []
    for k in idx:
        candidates += 1
        t0 = sp_ts[k + 1]
        after = sp_ts[k + 1:]
        hold = after <= t0 + SP_HOLD_S * 1e9
        if hold.sum() < 2:
            rejected_hold += 1
            continue
        held_values = sp_v[k + 1:k + 1 + hold.sum()]
        if not held_within(held_values, sp_v[k + 1], 0.5 * SP_THR):
            rejected_hold += 1
            continue
        if t0 - PRE_S * 1e9 < first_ns or t0 + POST_S * 1e9 > last_ns:
            rejected_window += 1
            continue
        events.append((t0, sp_v[k], sp_v[k + 1]))

    print(
        f'candidate SP steps: {candidates} (rejected hold={rejected_hold}, '
        f'window={rejected_window}, held-window candidates={len(events)})',
        flush=True,
    )

    rows = []
    rejected_covariate_window = 0
    last_output_t0 = None
    for t0, sp_before, sp_after in events:
        feats = {'t0_ns': int(t0), 'split': split_of(t0, args.grid_start_ns, args.grid_end_ns, args.grid_rows),
                 'dsp': float(sp_after - sp_before)}
        pre = {}
        for c in COLS[1:]:
            ts, vs = upd[c]
            lo = np.searchsorted(ts, t0 - PRE_S * 1e9, side='left')
            hi = np.searchsorted(ts, t0 + POST_S * 1e9, side='right')
            w_ts, w_v = ts[lo:hi], vs[lo:hi]
            if len(w_ts) == 0:
                pre[c] = None; continue
            grid = event_grid_ns(int(t0), int(PRE_S), int(POST_S))
            pos = np.searchsorted(w_ts, grid, side='right') - 1
            filled = w_v[np.clip(pos, 0, len(w_v) - 1)]
            filled[pos < 0] = np.nan
            pre[c] = filled
        T = pre['末级过热器出口汽温']
        n_pre = int(PRE_S); n_post = int(POST_S)
        if T is None or not np.isfinite(T[:n_pre]).sum() >= 0.9 * n_pre:
            rejected_covariate_window += 1
            continue
        load = pre['机组负荷']; pres = pre['主蒸汽压力']; valve = pre['二级减温调节门阀位']
        if load is None or pres is None or valve is None:
            rejected_covariate_window += 1
            continue
        sp_path = pre['二级减温调节阀设定']
        sp_max_dev_600 = float(np.nanmax(np.abs(sp_path[n_pre:n_pre + n_post + 1] - sp_after)))
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
            'sp_max_dev_600': sp_max_dev_600,
            'sp_held_600': bool(sp_max_dev_600 <= 0.5 * SP_THR),
            'dv_3s': float(valve[n_pre + 3] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 3]) else None,
            'dv_10s': float(valve[n_pre + 10] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 10]) else None,
            'dv_30s': float(valve[n_pre + 30] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 30]) else None,
            'dv_60s': float(valve[n_pre + 60] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 60]) else None,
            'dv_180s': float(valve[n_pre + 180] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 180]) else None,
            'dv_600s': float(valve[n_pre + 600] - valve[n_pre - 1]) if np.isfinite(valve[n_pre + 600]) else None,
        })
        if not respects_min_gap(int(t0), None if last_output_t0 is None else int(last_output_t0), MIN_GAP_S):
            rejected_min_gap += 1
            continue
        rows.append(feats)
        last_output_t0 = t0

    print(
        f'event funnel after covariates/gap: covariate={rejected_covariate_window}, '
        f'min_gap={rejected_min_gap}, kept={len(rows)}',
        flush=True,
    )

    all_df = pd.DataFrame(rows)
    filtered_rows = filter_rows_by_split(rows, args.split)
    df = pd.DataFrame(filtered_rows)
    if df.empty:
        n_s = n_s960 = 0
    else:
        n_s = int(((df['load_range_600'] <= S_LOAD_RANGE) & (df['pres_range_600'] <= S_PRES_RANGE)
                   & (df['temp_range_600'] <= S_TEMP_RANGE)).sum())
        n_s960 = int(((df['load_range_960'] <= S_LOAD_RANGE) & (df['pres_range_960'] <= S_PRES_RANGE)
                      & (df['temp_range_960'] <= S_TEMP_RANGE)).sum())

    payload = {
        'n': len(df),
        'side': args.side,
        'split_filter': args.split,
        'n_s_600': n_s,
        'n_s_960': n_s960,
        'thresholds': {'sp': SP_THR, 's_load': S_LOAD_RANGE, 's_pres': S_PRES_RANGE,
                       's_temp': S_TEMP_RANGE},
        'funnel': {'candidates': candidates, 'rejected_hold': rejected_hold,
                   'rejected_window': rejected_window, 'rejected_min_gap': rejected_min_gap,
                   'rejected_covariate_window': rejected_covariate_window,
                   'kept_full_window_all_splits': len(all_df), 'kept_after_split': len(df)},
        'provenance': {
            'source': args.input,
            'source_sha256': file_sha256(args.input),
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_sha': git_sha(),
            'git_dirty_at_start': bool(dirty_paths_at_start),
            'dirty_non_result_paths_at_start': dirty_non_result_paths,
            'script': str(Path(__file__).resolve()),
            'grid_start_ns': args.grid_start_ns,
            'grid_end_ns': args.grid_end_ns,
            'grid_rows': args.grid_rows,
            'split_frac': list(SPLIT_FRAC),
        },
        'events': filtered_rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f'saved: {out} (n={len(df)}, S600={n_s}, S960={n_s960})')
    if len(df):
        print(df[['split', 'dsp', 'dv_3s', 'dv_30s', 'dv_600s', 'dT_post_600']].describe().round(3).to_string())


if __name__ == '__main__':
    main()
