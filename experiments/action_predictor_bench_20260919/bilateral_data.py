#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bilateral (AB) pack builder — 30 history channels, 4-valve / 7-boundary future inputs.

Basis: docs/plans/2026-09-22-bilateral-response-training-spec.md §2 and
docs/plans/2026-09-22-bilateral-data-notes.md.

Alignment (same convention as build_bypass_pack.py)
---------------------------------------------------
* Reference pack ``data/screen_A_33pct_h128.npz`` fixes window starts and
  ``*_time`` = epoch seconds of the LAST history step t_end.
  Row k of a window is epoch ``t_end + (k - 63) * 10`` (k = 0..63 history,
  k = 64..63+H future temperatures, controls/boundaries on the H+1 rows
  k = 63..63+H so that the control at t drives the temperature at t+10 s).
* Values are taken RAW from the merged plant table at identical epochs.
  Valve columns are stored in PERCENT in the merged table -> multiplied by
  0.01 to match the mmap / reference-pack fraction convention.
  ``主蒸汽流量_60BKAO0312`` is t/h -> multiplied by 1/3.6 to kg/s to match
  the reference pack.

Channels (fixed order, 30)
--------------------------
0..4   A_T1..A_T5          (left chain temperatures)
5..9   B_T1..B_T5          (right chain temperatures)
10..13 u1A,u1B,u2A,u2B     (valve feedback fractions; u2A is the B-chain stage-2 valve)
14..18 steam_flow, coal, sep_p, sep_T, fw_T   (five common boundaries)
19..20 out_p_left, out_p_right
21..29 nine history-only bypass series (deduplicated: b_sh1_in/b_sh_out dropped
       because B_T1/B_T5 now live in channels 5..9)

Outputs per split: hist30 [N,64,30], future_act [N,H+1,4], future_bnd [N,H+1,7],
future_temp [N,H,10], starts/time, valid mask, protocol flags, and shared mean/scale.
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

PACK_DEFAULT = HERE / 'data' / 'screen_A_33pct_h128.npz'
BYPASS_DEFAULT = HERE / 'data' / 'hist_bypass_A_33pct_v1.npz'
COLS_CSV = ROOT / 'docs' / '2026-09-20-点位清单-全量381列分类.csv'
MERGED_DEFAULT = Path.home() / 'Desktop/AI/时序预测/AA数据中心/伊敏12.10/cleaned_data/all_merged_10s.csv'
OUT_DEFAULT = HERE / 'data' / 'bilateral_AB_33pct.npz'

# (key, merged column, multiplier, note)
CHANNELS = [
    ('A_T1', '选择后左侧一过喷水减温器入口', 1.0, ''),
    ('A_T2', '选择后左侧一过喷水减温器出口', 1.0, ''),
    ('A_T3', '选择后二级减温器左侧入口蒸汽', 1.0, ''),
    ('A_T4', '选择后左侧二过喷水减温器出口', 1.0, ''),
    ('A_T5', '选择后左侧末级过热器出口汽温', 1.0, 'main outlet, left'),
    ('B_T1', '选择后右侧一过喷水减温器入口', 1.0, ''),
    ('B_T2', '选择后右侧一过喷水减温器出口', 1.0, ''),
    ('B_T3', '选择后二级减温器右侧入口蒸汽', 1.0, ''),
    ('B_T4', '选择后右侧二过喷水减温器出口', 1.0, ''),
    ('B_T5', '选择后右侧末级过热器出口汽温', 1.0, 'main outlet, right'),
    ('u1A', '过热器一级减温器A侧喷水调节门阀位反馈', 0.01, 'percent -> fraction'),
    ('u1B', '过热器一级减温器B侧喷水调节门阀位反馈', 0.01, 'percent -> fraction'),
    ('u2A', '过热器二级减温器A侧喷水调节门阀位反馈', 0.01, 'B-chain stage-2 valve (crossed wiring)'),
    ('u2B', '过热器二级减温器B侧喷水调节门阀位反馈', 0.01, 'A-chain stage-2 valve (crossed wiring)'),
    ('steam_flow', '主蒸汽流量_60BKAO0312', 1 / 3.6, 't/h -> kg/s (common, not per-side)'),
    ('coal', '未校正总煤量', 1.0, 'common'),
    ('sep_p', '选择后分离器最终出口压力', 1.0, 'common'),
    ('sep_T', '分离器最终出口温度', 1.0, 'common'),
    ('fw_T', '选择后省煤器出口给水温度', 1.0, 'common'),
    ('out_p_left', '选择后末级过热器左侧出口压力', 1.0, ''),
    ('out_p_right', '选择后末级过热器右侧出口压力', 1.0, ''),
    ('corr_fuel_total', '校正后总燃料量', 1.0, 'history-only bypass'),
    ('sa_flow', '总二次风量', 1.0, 'history-only bypass'),
    ('o2_a_3sel', '三选后A侧烟气含氧量', 1.0, 'history-only bypass (asymmetric observation)'),
    ('load', '机组负荷_GENERATOR_POWER', 1.0, 'history-only; post-hoc grouping only'),
    ('sp1_a', '一级减A副调设定值', 1.0, 'history-only bypass'),
    ('sp2_b_set', '过热器二级减温器B喷水调节阀设定', 1.0, 'history-only bypass (stage2 crossed)'),
    ('agc', 'AGC指令', 1.0, 'history-only bypass'),
    ('load_rate', '机组负荷变化率', 1.0, 'history-only bypass (discrete)'),
    ('sh_spray_total', '过热器减温水总流量', 1.0, 'history-only bypass'),
]
NAMES = [c[0] for c in CHANNELS]
TEMP_IDX = list(range(10))
ACT_IDX = [10, 11, 12, 13]
BND_IDX = list(range(14, 21))
AUX_IDX = list(range(21, 30))
# reference pack channel -> our channel (None = not carried over)
REF_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 10, 6: 13, 7: 14, 8: 15, 9: 16, 10: 17, 11: 18, 12: 19}
SPLITS = ('train', 'selector', 'evaluation')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_name_map(cols_csv=COLS_CSV):
    df = pd.read_csv(cols_csv, dtype=str)
    return {str(r['列名']).strip(): int(r['列号']) for _, r in df.iterrows()}


def window_epochs(t_end, horizon):
    """Rows k = 0..63+H -> epochs t_end + (k-63)*10."""
    rows = CONTEXT + horizon
    k = np.arange(rows)
    return t_end[:, None] + (k - (CONTEXT - 1)) * DT


def gather_rows(merged_path, needed_epochs, keep_names, chunksize=200_000, cache_dir=None):
    """Chunked scan of the merged table; optional pickle cache keyed by needed set + columns."""
    needed = np.array(sorted(needed_epochs), dtype='int64')
    cache = None
    if cache_dir is not None:
        key = hashlib.sha256(('|'.join(keep_names) + ':' + str(len(needed)) + ':'
                              + str(int(needed[0])) + ':' + str(int(needed[-1]))).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f'gather_{key}.pkl'
        if cache.exists():
            df = pd.read_pickle(cache)
            if len(df) == len(needed):
                return df
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
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def lookup(df, epochs2d):
    epochs = df.index.to_numpy()
    pos = pd.Series(np.arange(len(epochs)), index=epochs)
    idx = pos.reindex(epochs2d.ravel()).to_numpy()
    vals = np.full((epochs2d.size, df.shape[1]), np.nan)
    ok = np.isfinite(idx)
    vals[ok] = df.to_numpy(np.float64)[idx[ok].astype(int)]
    return vals.reshape(epochs2d.shape[0], epochs2d.shape[1], df.shape[1]), ok.reshape(epochs2d.shape)


def _scaled(raw):
    return raw * np.asarray([c[2] for c in CHANNELS], dtype=np.float64)


def build(pack=PACK_DEFAULT, bypass=BYPASS_DEFAULT, merged=MERGED_DEFAULT, out=OUT_DEFAULT,
          overrides=None, verbose=True, cache_dir='/tmp/bilateral_cache'):
    import os
    overrides = overrides or {}
    merged = Path(os.environ.get('BILATERAL_MERGED', merged))
    ref = {k: np.load(pack)[k] for k in np.load(pack).files}
    ref_meta = json.loads(Path(pack).with_suffix('.json').read_text(encoding='utf-8'))
    byp = {k: np.load(bypass)[k] for k in np.load(bypass).files}
    byp_meta = json.loads(Path(bypass).with_suffix('.json').read_text(encoding='utf-8'))

    name2num = load_name_map()
    resolved = []
    for key, col, mult, note in CHANNELS:
        n = name2num if col in name2num else None
        if col not in name2num:
            hits = [k for k in name2num if col in k]
            if len(hits) != 1:
                raise KeyError(f'无法解析列名: {col} | 近邻: {hits[:8]}')
            col = hits[0]
        resolved.append(col)
    keep = list(dict.fromkeys(resolved))  # unique, stable order
    assert len(keep) == len(resolved), 'duplicate merged columns in CHANNELS'

    horizons = {'train': 128, 'selector': 128, 'evaluation': 512}
    ep = {k: window_epochs(ref[k + '_time'], horizons[k]) for k in SPLITS}
    need = set()
    for k in SPLITS:
        need.update(ep[k].ravel().tolist())
    if verbose:
        print(f'[build] 需采集唯一时刻 {len(need)} 个；扫描合并表…')
    df = gather_rows(merged, need, keep, cache_dir=cache_dir)
    if verbose:
        print(f'[build] 合并表命中 {len(df)} 行（缺失 {len(need) - len(df)}）')

    out_arrays = {}
    stats = {}
    for k in SPLITS:
        raw, found = lookup(df, ep[k])
        vals = _scaled(raw)
        H = horizons[k]
        hist = vals[:, :CONTEXT, :].astype(np.float32)
        act = vals[:, CONTEXT - 1:CONTEXT + H, ACT_IDX].astype(np.float32)
        bnd = vals[:, CONTEXT - 1:CONTEXT + H, BND_IDX].astype(np.float32)
        temp = vals[:, CONTEXT:CONTEXT + H, TEMP_IDX].astype(np.float32)
        valid = np.isfinite(hist).all(axis=(1, 2)) & np.isfinite(act).all(axis=(1, 2)) \
            & np.isfinite(bnd).all(axis=(1, 2)) & np.isfinite(temp).all(axis=(1, 2))
        valves = vals[:, :CONTEXT + H, ACT_IDX]
        all01 = ((valves >= 0) & (valves <= 1)).all(axis=(1, 2))
        near_closed = (np.abs(valves) < 0.02).any(axis=(1, 2))
        u2a_neg = (valves[:, :, 2] < 0).any(axis=1)
        out_arrays.update({
            f'hist30_{k}': hist, f'future_act_{k}': act, f'future_bnd_{k}': bnd,
            f'future_temp_{k}': temp, f'{k}_starts': ref[k + '_starts'], f'{k}_time': ref[k + '_time'],
            f'valid_{k}': valid.astype(bool), f'strict01_{k}': all01.astype(bool),
            f'near_closed_{k}': near_closed.astype(bool), f'u2a_negative_{k}': u2a_neg.astype(bool),
        })
        stats[k] = {
            'rows': int(len(hist)), 'valid': int(valid.sum()), 'strict_all01': int(all01.sum()),
            'u2a_negative': int(u2a_neg.sum()), 'near_closed': int(near_closed.sum()),
            'time_coverage': float(found.mean()),
            'u2a_min': float(np.nanmin(vals[:, :, 12])),
        }
        if verbose:
            print(f'[build] {k}: rows={stats[k]["rows"]} valid={stats[k]["valid"]} '
                  f'strict={stats[k]["strict_all01"]} u2a_neg={stats[k]["u2a_negative"]} '
                  f'u2a_min={stats[k]["u2a_min"]:.6f} coverage={stats[k]["time_coverage"]*100:.4f}%')

    # ---------- normalization ----------
    he = ep['train'][:, :CONTEXT]      # history epochs only (64 rows)
    hv = out_arrays['hist30_train']
    flat_e, first = np.unique(he.ravel(), return_index=True)
    flat_v = hv.reshape(-1, len(NAMES))[first]
    mean_new = np.nanmean(flat_v, axis=0).astype(np.float32)
    scale_new = np.nanstd(flat_v, axis=0).astype(np.float32)
    mean30 = mean_new.copy()
    scale30 = scale_new.copy()
    # old-13 channels keep the reference-pack normalization exactly
    ref_mean, ref_scale = ref['mean'], ref['scale']
    for rc, oc in REF_MAP.items():
        mean30[oc] = ref_mean[rc]
        scale30[oc] = ref_scale[rc]
    # 9 bypass channels keep the bypass-pack normalization (identical raw values)
    byp_names = list(byp_meta['channel_names'])
    keep_byp = [n for n in byp_names if n not in ('b_sh1_in', 'b_sh_out', 'fuel_ctl')]
    assert len(keep_byp) == 9, keep_byp
    for i, nm in enumerate(keep_byp):
        j = NAMES.index(nm)
        mean30[j] = byp['mean'][byp_names.index(nm)]
        scale30[j] = byp['scale'][byp_names.index(nm)]
    out_arrays['mean'] = mean30
    out_arrays['scale'] = scale30
    out_arrays['channel_names'] = np.array(NAMES)

    # ---------- verification ----------
    checks = {}
    rng = np.random.default_rng(7)
    idx = rng.choice(len(ref['train']), size=min(4000, len(ref['train'])), replace=False)
    hist_sample = out_arrays['hist30_train'][idx]
    for rc, oc in sorted(REF_MAP.items()):
        b = ref['train'][idx, :CONTEXT, rc].astype(np.float64)
        v = hist_sample[:, :, oc].astype(np.float64)
        corr = float(np.corrcoef(b.ravel(), v.ravel())[0, 1])
        mad = float(np.max(np.abs(b - v)))
        checks[f'ref_ch{rc}_{ref_meta["channel_names"][rc]}'] = {
            'our_channel': NAMES[oc], 'corr': corr, 'max_abs_diff': mad}
    # bypass channels: expect exact equality on the history block
    for i, nm in enumerate(keep_byp):
        b = byp['hist12_train'][idx, :, byp_names.index(nm)]
        v = hist_sample[:, :, NAMES.index(nm)]
        checks[f'bypass_{nm}'] = {'equal': bool(np.array_equal(b, v)),
                                  'max_abs_diff': float(np.max(np.abs(b.astype(np.float64) - v.astype(np.float64))))}
    # documented merged-table statistics (data-notes §4) reproduced from our channels
    tr = out_arrays['hist30_train']
    V = tr[:, :, ACT_IDX].reshape(-1, 4).astype(np.float64)
    checks['valve_level_corr'] = np.round(np.corrcoef(V.T), 4).tolist()
    T10 = tr[:, :, TEMP_IDX].reshape(-1, 10).astype(np.float64)
    checks['AB_mean_gap_C'] = np.round([float(np.mean(T10[:, s] - T10[:, 5 + s])) for s in range(5)], 4).tolist()
    checks['AB_stage_corr'] = np.round([float(np.corrcoef(T10[:, s], T10[:, 5 + s])[0, 1]) for s in range(5)], 4).tolist()

    meta = {
        'pack_id': Path(out).stem, 'date': '2026-09-22',
        'produced_by': 'bilateral_data.py (execution-side implementation of the 2026-09-22 spec §2)',
        'pack_sha256': '',
        'reference_pack': str(pack), 'reference_pack_sha256': sha256(pack),
        'bypass_pack': str(bypass), 'bypass_pack_sha256': sha256(bypass),
        'source_merged_table': {'path': str(merged), 'sha256': sha256(merged)},
        'channel_names': NAMES, 'context': CONTEXT, 'dt_seconds': DT,
        'horizons': horizons,
        'history_alignment': 'epoch = t_end - 630 + 10*k (k=0..63); t_end from reference pack *_time; '
                             'raw merged-table values at identical epochs',
        'future_alignment': 'future_temp rows k=64..63+H (epoch t_end+10..t_end+10H); future_act/future_bnd '
                            'rows k=63..63+H (H+1 rows) so the control at t drives T(t+10s)',
        'unit_conversions': {'valves': 'merged table percent * 0.01 -> fraction (mmap convention)',
                             'steam_flow': 't/h * 1/3.6 -> kg/s (reference-pack convention)'},
        'feedback_protocol': {
            'tolerance': 'raw feedback retained; canonical reference band [-0.02, 1] for valve validity',
            'strict_subgroup': 'strict01_* = all four valves within [0,1] over history+future rows',
            'negative_feedback': 'u2a_negative_* flags raw negative u2A readings (sensor reading, not negative spray)',
            'note': 'explicit protocol change vs the earlier strict-fraction packing; all original windows retained',
        },
        'normalization': {'fit': 'new channels: unique train rows (ddof=0); '
                                 'old-13 channels: reference pack values; bypass-9: bypass pack values'},
        'counts': {k: stats[k]['rows'] for k in SPLITS},
        'protocol_stats': stats, 'verification': checks,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    np.savez_compressed(out, **out_arrays)
    meta['pack_sha256'] = sha256(out)
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(f'[build] 写入 {out} + .json')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', type=Path, default=PACK_DEFAULT)
    ap.add_argument('--bypass', type=Path, default=BYPASS_DEFAULT)
    ap.add_argument('--merged', type=Path, default=MERGED_DEFAULT)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--pilot', type=int, default=0, help='identify channels on N sampled windows only')
    ap.add_argument('--cache-dir', default='/tmp/bilateral_cache')
    args = ap.parse_args()
    if args.pilot:
        # quick identification report, no file written
        ref = {k: np.load(args.pack)[k] for k in np.load(args.pack).files}
        name2num = load_name_map()
        cols = list(dict.fromkeys(CHANNELS[i][1] for i in range(len(CHANNELS))))
        rng = np.random.default_rng(7)
        sel = rng.choice(len(ref['train_time']), size=args.pilot, replace=False)
        times = window_epochs(ref['train_time'][sel], 128).ravel()
        df = gather_rows(args.merged, set(times.tolist()), cols)
        vals, found = lookup(df, window_epochs(ref['train_time'][sel], 128))
        scaled = _scaled(vals)
        print(f'[pilot] {len(sel)} 窗口 × {vals.shape[1]} 步 | 覆盖 {found.mean()*100:.4f}% | 命中 {len(df)} 行')
        n = min(CONTEXT, vals.shape[1])
        bank = ref['train'][sel]
        for rc, oc in sorted(REF_MAP.items()):
            b = bank[:, :n, rc].ravel()
            v = scaled[:, :n, oc].ravel()
            m = np.isfinite(b) & np.isfinite(v)
            print(f'  ref ch{rc} -> {NAMES[oc]:<12s} corr={np.corrcoef(b[m], v[m])[0,1]:.6f} '
                  f'med={np.median(np.abs(b[m]-v[m])):.5f} max={np.max(np.abs(b[m]-v[m])):.5f}')
        for j in AUX_IDX:
            v = scaled[:, :n, j]
            print(f'  {NAMES[j]:<16s} mean={np.nanmean(v):10.3f} std={np.nanstd(v):9.3f} nan={int(np.isnan(v).sum())}')
        return
    print(json.dumps(build(args.pack, args.bypass, args.merged, args.out),
                     ensure_ascii=False, indent=1)[:2000])


if __name__ == '__main__':
    main()
