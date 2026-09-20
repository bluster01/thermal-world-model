#!/usr/bin/env python3
"""History-bypass pack builder (v0 draft, execution-side).

Builds a side-channel pack of 12 auxiliary *history-only* series for the
action-predictor bench, aligned to the original window starts of
data/screen_A_33pct.npz.  Basis: docs/plans/2026-09-20-response-focused-design.md §4.

Conventions / alignment
-----------------------
* Ref pack (screen_A_33pct.npz): bank [N, 96, 13] (train/selector) or [N, 576, 13] (eval);
  ``*_time`` = epoch seconds of the LAST history step t; epoch of window row k
  (k=0..95) = t_end - 63*10 + 10*k.  History = first 64 rows = epochs
  ``t_end - 630 + 10*k`` (k=0..63).
* New-series values are taken RAW from the merged plant table (all_merged_10s.csv)
  at the identical epochs -> [N, 64, 12] per split. History-only by construction.
* mean/scale: fit on UNIQUE history-step epochs of the TRAIN split only (ddof=0).

Modes
-----
--pilot N : identification + alignment report on a random sample of N train windows
            (no output file).  Full run (no --pilot) writes the npz + json sidecar.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CONTEXT, DT = 64, 10

PACK_DEFAULT = HERE / 'data' / 'screen_A_33pct.npz'
COLS_CSV = ROOT / 'docs' / '2026-09-20-点位清单-全量381列分类.csv'
MERGED_DEFAULT = Path.home() / 'Desktop/AI/时序预测/AA数据中心/伊敏12.10/cleaned_data/all_merged_10s.csv'

# The 12 auxiliary series (design doc §4).  (key, merged-table column name, note)
SERIES = [
    ('corr_fuel_total', '校正后总燃料量', '热负荷; key new signal'),
    ('sa_flow', '总二次风量', '热负荷'),
    ('o2_a_3sel', '三选后A侧烟气含氧量', '热负荷; alternates: A侧锅炉出口烟气氧量1/2/3'),
    ('load', '机组负荷_GENERATOR_POWER', '热负荷'),
    ('sp1_a', '一级减A副调设定值', '控制意图'),
    ('sp2_a_set', '过热器二级减温器A喷水调节阀设定', '控制意图'),
    ('agc', 'AGC指令', '控制意图'),
    ('load_rate', '机组负荷变化率', '控制意图'),
    ('b_sh1_in', '选择后右侧一过喷水减温器入口', 'B侧代理 (T1 对侧)'),
    ('b_sh_out', '选择后右侧末级过热器出口汽温', 'B侧代理 (T5 对侧)'),
    ('sh_spray_total', '过热器减温水总流量', '代理'),
    ('fuel_ctl', '燃料主控输出', '增量热输入候选'),
]

# Candidates for identifying the 13 reference channels (validation only).
IDENT_SHORTLIST = [
    '主蒸汽流量_60BKAO0312', '未校正总煤量', '选择后分离器最终出口压力', '分离器最终出口温度',
    '分离器出口温度1/2选择后', '分离器出口温度3/4选择后', '分离器贮水罐饱和温度',
    '选择后左侧一过喷水减温器入口', '选择后右侧一过喷水减温器入口',
    '选择后左侧一过喷水减温器出口', '选择后右侧一过喷水减温器出口',
    '选择后二级减温器左侧入口蒸汽', '选择后二级减温器右侧入口蒸汽',
    '选择后左侧二过喷水减温器出口', '选择后右侧二过喷水减温器出口',
    '选择后左侧末级过热器出口汽温', '选择后右侧末级过热器出口汽温', '主蒸汽温度',
    '过热器一级减温器A侧喷水调节门阀位反馈', '过热器一级减温器B侧喷水调节门阀位反馈',
    '过热器二级减温器A侧喷水调节门阀位反馈', '过热器二级减温器B侧喷水调节门阀位反馈',
    '过热器一级减温器A喷水调节阀主调输出', '过热器一级减温器B喷水调节阀主调输出',
    '过热器二级减温器A喷水调节阀主调输出', '过热器二级减温器B喷水调节阀主调输出',
    '过热器二级减温器A喷水调节阀设定', '过热器二级减温器B喷水调节阀设定',
    '选择后省煤器入口给水温度', '选择后省煤器出口给水温度',
    '选择后末级过热器左侧出口压力', '选择后末级过热器右侧出口压力',
    '主蒸汽压力', '三选后主蒸汽压力',
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_name_map(cols_csv=COLS_CSV):
    df = pd.read_csv(cols_csv, dtype=str)
    return {str(r['列名']).strip(): int(r['列号']) for _, r in df.iterrows()}


def resolve(name2num, hint):
    if hint in name2num:
        return hint
    hits = [k for k in name2num if hint in k]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f'无法解析列名: {hint} | 近邻: {hits[:8]}')


def gather_rows(merged_path, needed_epochs, keep_names, chunksize=200_000):
    needed = np.array(sorted(needed_epochs), dtype='int64')
    frames = []
    for ch in pd.read_csv(merged_path, usecols=['time'] + keep_names,
                          dtype={n: np.float32 for n in keep_names},
                          chunksize=chunksize, encoding='utf-8-sig'):
        t = (pd.to_datetime(ch['time'], utc=True).astype('int64') // 10 ** 9).to_numpy()
        m = np.isin(t, needed)
        if m.any():
            sub = ch.loc[m, keep_names].copy()
            sub['epoch'] = t[m]
            frames.append(sub)
    df = pd.concat(frames, ignore_index=True).set_index('epoch').sort_index()
    return df


def hist_epochs(t_end):
    k = np.arange(CONTEXT)
    return t_end[:, None] - (CONTEXT - 1) * DT + DT * k


def full_epochs(t_end, n_steps=96):
    k = np.arange(n_steps)
    return t_end[:, None] - (CONTEXT - 1) * DT + DT * k


def lookup(df, epochs2d):
    """epochs2d [N, T] -> values [N, T, K] (NaN where missing)."""
    epochs = df.index.to_numpy()
    pos = pd.Series(np.arange(len(epochs)), index=epochs)
    idx = pos.reindex(epochs2d.ravel()).to_numpy()
    vals = np.full((epochs2d.size, df.shape[1]), np.nan)
    ok = np.isfinite(idx)
    vals[ok] = df.to_numpy(np.float64)[idx[ok].astype(int)]
    return vals.reshape(epochs2d.shape[0], epochs2d.shape[1], df.shape[1]), ok.reshape(epochs2d.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', type=Path, default=PACK_DEFAULT)
    ap.add_argument('--merged', type=Path, default=MERGED_DEFAULT)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--pilot', type=int, default=0, help='sample N train windows for identification report')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    name2num = load_name_map()
    all_names = [s[1] for s in SERIES] + IDENT_SHORTLIST
    resolved = {n: resolve(name2num, n) for n in all_names}
    keep_names = list(resolved)  # unique by construction
    ref = {k: np.load(args.pack)[k] for k in np.load(args.pack).files}
    ref_meta = json.loads(args.pack.with_suffix('.json').read_text(encoding='utf-8'))
    ref_channels = ref_meta['channel_names']

    if args.pilot:
        t_end = ref['train_time']
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(t_end), size=min(args.pilot, len(t_end)), replace=False)
        times = full_epochs(t_end[sel])
        df = gather_rows(args.merged, set(times.ravel().tolist()), keep_names)
        vals, found = lookup(df, times)
        print(f'[pilot] 窗口 {len(sel)} × 96 步 = {times.size} 行 | 时间覆盖 {found.mean() * 100:.4f}% '
              f'| 采集列 {len(keep_names)} | 合并表命中行 {len(df)}')
        bank = ref['train'][sel]
        print(f'识别参考 13 通道（top-3 匹配）:')
        for c in range(13):
            b = bank[:, :, c].ravel()
            res = []
            for j, nm in enumerate(keep_names):
                v = vals[:, :, j].ravel()
                m = np.isfinite(b) & np.isfinite(v)
                if m.sum() < 100:
                    continue
                corr = np.corrcoef(b[m], v[m])[0, 1]
                med = float(np.median(np.abs(b[m] - v[m])))
                res.append((corr, med, nm))
            res.sort(key=lambda x: -x[0])
            print(f'  ch{c:2d} {ref_channels[c]:<18s}', end='')
            for corr, med, nm in res[:3]:
                print(f' | {corr:.6f}/{med:.4f} {nm}', end='')
            print()
        print()
        print('12 新序列采样统计:')
        for j, nm in enumerate(keep_names[:len(SERIES)]):
            v = vals[:, :, j]
            print(f'  {SERIES[j][0]:<16s} {nm:<28s} mean={np.nanmean(v):10.3f} std={np.nanstd(v):9.3f} '
                  f'nan={int(np.isnan(v).sum())}')
        return

    # ---------- full build ----------
    assert args.out, '--out required for full build'
    splits = ['train', 'selector', 'evaluation']
    hist_ep = {k: hist_epochs(ref[k + '_time']) for k in splits}
    need = set()
    for k in splits:
        need.update(hist_ep[k].ravel().tolist())
    print(f'[full] 需采集唯一时刻 {len(need)} 个；开始扫描合并表…')
    df = gather_rows(args.merged, need, keep_names)
    print(f'[full] 合并表命中 {len(df)} 行（缺失 {len(need) - len(df)}）')

    out = {}
    miss = {}
    for k in splits:
        vals, found = lookup(df, hist_ep[k])
        S = len(SERIES)
        hv = vals[:, :, :S].astype(np.float32)
        valid = np.isfinite(hv).all(axis=(1, 2))
        out[f'hist12_{k}'] = hv
        out[f'{k}_starts'] = ref[k + '_starts']
        out[f'{k}_time'] = ref[k + '_time']
        out[f'valid_{k}'] = valid
        print(f'[full] {k}: rows={len(hv)} valid={int(valid.sum())}/{len(valid)} '
              f'time-coverage={found.mean() * 100:.4f}%')

    he = hist_ep['train']
    hv = out['hist12_train']
    flat_e = he.ravel()
    flat_v = hv.reshape(-1, len(SERIES))
    _, first = np.unique(flat_e, return_index=True)
    uv = flat_v[first]
    out['mean'] = np.nanmean(uv, axis=0).astype(np.float32)
    out['scale'] = np.nanstd(uv, axis=0).astype(np.float32)
    out['channel_names'] = np.array([s[0] for s in SERIES])

    d = np.diff(hv, axis=1)
    frozen = (np.abs(d) < 1e-9).mean(axis=(0, 1))
    nancells = np.isnan(hv).sum(axis=(0, 1))
    for j, s in enumerate(SERIES):
        miss[s[0]] = {'nan_cells': int(nancells[j]), 'frozen_fraction': float(frozen[j])}

    print('[full] 对齐复核（8000 窗口抽样，逐参考通道 top-1）:')
    rng = np.random.default_rng(args.seed + 13)
    sample_idx = rng.choice(len(ref['train_time']), size=min(8000, len(ref['train_time'])), replace=False)
    vals_chk, _ = lookup(df, hist_ep['train'][sample_idx])
    check = {}
    top1 = {}
    for c in range(13):
        b = ref['train'][sample_idx, :CONTEXT, c].ravel()
        best = (0.0, None, None)
        for j, nm in enumerate(keep_names):
            v = vals_chk[:, :, j].ravel()
            m = np.isfinite(b) & np.isfinite(v)
            if m.sum() < 1000:
                continue
            corr = np.corrcoef(b[m], v[m])[0, 1]
            if corr > best[0]:
                best = (corr, float(np.median(np.abs(b[m] - v[m]))), nm)
        top1[c] = best[2]
        check[f'ch{c}_{ref_channels[c]}'] = {'top1': best[2], 'corr': best[0], 'med_abs_diff': best[1]}
        print(f'  ch{c:2d} {ref_channels[c]:<18s} -> {best[2]:<30s} corr={best[0]:.6f} med={best[1]:.4f}')

    print('[full] 12 新序列 vs 参考通道 最大 |corr|（去重复核）:')
    overlap = {}
    for s in SERIES:
        j = keep_names.index(s[1])
        v = vals_chk[:, :, j].ravel()
        best = (0.0, None)
        for c in range(13):
            nm13 = top1.get(c)
            if nm13 is None or nm13 not in keep_names:
                continue
            u = vals_chk[:, :, keep_names.index(nm13)].ravel()
            m = np.isfinite(v) & np.isfinite(u)
            if m.sum() < 1000:
                continue
            corr = abs(np.corrcoef(v[m], u[m])[0, 1])
            if corr > best[0]:
                best = (corr, f'ch{c}_{ref_channels[c]}:{nm13}')
        overlap[s[0]] = {'max_abs_corr': best[0], 'against': best[1]}
        print(f'  {s[0]:<16s} max|corr|={best[0]:.6f} vs {best[1]}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    meta = {
        'pack_id': 'hist_bypass_A_33pct_v0',
        'date': '2026-09-20',
        'produced_by': 'execution-side (Linux worker)',
        'purpose': 'history-only auxiliary series for response-focused design round '
                   '(docs/plans/2026-09-20-response-focused-design.md §4)',
        'reference_pack': str(args.pack),
        'reference_pack_sha256': sha256(args.pack),
        'reference_channel_names': ref_channels,
        'context': CONTEXT, 'dt_seconds': DT,
        'history_alignment': 'epoch = t_end - 630 + 10*k (k=0..63); t_end from reference pack *_time; '
                             'values RAW from merged plant table at identical epochs',
        'series': [{'key': k, 'name': n, 'merged_col_index': resolved[n], 'notes': note}
                   for k, n, note in SERIES],
        'duplicates_vs_reference_13': 'checked via max|corr| against identified reference columns '
                                      '(8000-window train sample); see overlap_check. '
                                      'b_sh1_in / b_sh_out are B-side counterparts of ch0 / ch4 (distinct signals).',
        'normalization': {'fit': 'unique train history-step epochs only', 'ddof': 0},
        'counts': {k: int(len(out[f'hist12_{k}'])) for k in splits},
        'validity': {k: f"{int(out[f'valid_{k}'].sum())}/{len(out[f'valid_{k}'])}" for k in splits},
        'missingness': miss,
        'alignment_validation': check,
        'overlap_check': overlap,
        'dedup_note': 'fuel_ctl (燃料主控输出) vs base-13 coal (未校正总煤量): |corr|=0.99987 — near-duplicate; '
                      'retained in pack for transparency, dedup decision to design side. '
                      'corr_fuel_total vs coal 0.9947; load/agc vs steam_flow ~0.998 (physical collinearity) — all retained.',
        'signal_notes': {
            'load_rate': 'stepwise discrete signal (frozen_fraction 0.997)',
            'sp2_a_set': 'temperature-type setpoint, piecewise constant (frozen_fraction 0.992)',
            'o2_a_3sel': 'frozen_fraction 0.122 (flat stretches from selection logic)',
            'agc': 'frozen_fraction 0.198 (stepwise AGC ramps)',
        },
        'source_merged_table': {'path': str(args.merged), 'sha256': sha256(args.merged)},
        'channel_names': [s[0] for s in SERIES],
    }
    args.out.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[full] 写入 {args.out} + .json')


if __name__ == '__main__':
    main()
