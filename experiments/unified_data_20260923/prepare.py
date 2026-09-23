"""Prepare three industrial tasks from timestamped single-point exports.

No model fitting. Raw values, observation timestamps and quality flags remain
separate; only training rows determine normalization and statistical bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAT = np.iinfo(np.int64).min
MISSING, STALE, OUTSIDE_RANGE, SUSPECT_HIGH = 1, 2, 4, 8


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def causal_align(source_ns, source_values, grid_ns, max_age_s, step_s):
    """Last exported record at/before t; invalid records do not revive old values."""
    source_ns = np.asarray(source_ns, dtype=np.int64)
    source_values = np.asarray(source_values, dtype=np.float64)
    if len(source_ns) != len(source_values) or np.any(np.diff(source_ns) <= 0):
        raise ValueError('source timestamps must be unique and increasing')
    positions = np.searchsorted(source_ns, grid_ns, side='right') - 1
    found = positions >= 0
    source_time = np.full(len(grid_ns), NAT, dtype=np.int64)
    values = np.full(len(grid_ns), np.nan, dtype=np.float64)
    source_time[found] = source_ns[positions[found]]
    values[found] = source_values[positions[found]]
    age = np.full(len(grid_ns), np.inf)
    age[found] = (grid_ns[found] - source_time[found]) / 1e9
    finite = np.isfinite(values)
    fresh = found & (age < step_s) & finite
    valid = found & (age <= max_age_s) & finite
    return values, source_time, valid, fresh, age


def read_source(path):
    raw = pd.read_csv(path, usecols=['time', 'valueFloat64'])
    timestamp = pd.to_datetime(raw['time'], utc=True, errors='coerce', format='mixed')
    good_time = timestamp.notna()
    if not good_time.any():
        raise ValueError(f'No timestamps parsed: {path}')
    raw_values = pd.to_numeric(raw['valueFloat64'], errors='coerce')
    frame = pd.DataFrame({'time_ns': timestamp[good_time].astype('int64'),
                          'value': raw_values[good_time]})
    nonmonotone = int((np.diff(frame.time_ns.to_numpy()) < 0).sum())
    duplicates = int(frame.time_ns.duplicated(keep='last').sum())
    frame = frame.sort_values('time_ns', kind='mergesort').drop_duplicates('time_ns', keep='last')
    ns = frame.time_ns.to_numpy()
    gaps = np.diff(ns) / 1e9
    stats = {
        'raw_rows': len(raw), 'bad_time_rows': int((~good_time).sum()),
        'nonfinite_value_rows': int((~np.isfinite(raw_values)).sum()),
        'duplicate_time_rows': duplicates, 'backward_time_steps': nonmonotone,
        'unique_timestamps': len(frame),
        'first_utc': str(pd.Timestamp(int(ns[0]), tz='UTC')),
        'last_utc': str(pd.Timestamp(int(ns[-1]), tz='UTC')),
        'median_interval_s': float(np.median(gaps)) if len(gaps) else None,
        'max_interval_s': float(gaps.max()) if len(gaps) else None,
    }
    return ns, frame.value.to_numpy(), stats


def interval_count(mask, lo, hi):
    prefix = np.r_[0, np.cumsum(np.asarray(mask, dtype=np.int64))]
    return prefix[hi] - prefix[lo]


def make_origins(grid, split, valid, observed, source_time, channel_names, task,
                 load, load_valid, cfg, profile):
    """History ends at origin; labels start one step after it. No crossing splits."""
    history, horizon = profile['history_steps'], profile['forecast_steps']
    origin = np.arange(history - 1, len(grid) - horizon,
                       cfg['origin_stride_steps'], dtype=np.int64)
    lo, hi = origin - history + 1, origin + horizon + 1
    targets = [channel_names.index(a) for a in task['targets']]
    # Index selection uses masks, never target amplitudes or future inputs.
    history_ok = np.ones(len(origin), dtype=bool)
    label_counts = np.zeros((len(origin), len(targets)), dtype=np.int32)
    for j, channel in enumerate(targets):
        history_ok &= interval_count(valid[channel], lo, origin + 1) >= np.ceil(history * cfg['history_target_valid_fraction'])
        labels = valid[channel] & observed[channel] & load_valid & (load > cfg['operating_floor_mw'])
        label_counts[:, j] = interval_count(labels, origin + 1, hi)
    continuous_load = interval_count(load_valid, lo, hi) == (history + horizon)
    same_split = (split[lo] == split[origin]) & (split[hi - 1] == split[origin])
    eligible = history_ok & continuous_load & same_split & (load[origin] > cfg['operating_floor_mw'])
    result = {}
    cut_ns = [int(grid[0]), pd.Timestamp(cfg['cut1']).value, pd.Timestamp(cfg['cut2']).value]
    counts = {}
    for code, name in enumerate(['train', 'validation', 'historical_test']):
        selected = eligible & (split[origin] == code)
        if code:
            selected &= grid[origin] >= cut_ns[code] + cfg['split_embargo_seconds'] * 10**9
        # Training requires useful labels for every target. Evaluation keeps
        # partial channels and relies on per-target/per-step masks.
        enough = (label_counts >= np.ceil(horizon * cfg['train_label_valid_fraction'])).all(axis=1)
        if code == 0:
            selected &= enough
        else:
            selected &= (label_counts > 0).any(axis=1)
        complete = (label_counts == horizon).all(axis=1)
        result[name] = origin[selected]
        result[name + '_complete'] = origin[selected & complete]
        counts[name] = {'origins': int(selected.sum()), 'complete_label_origins': int((selected & complete).sum())}
    return result, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/processed/unified_industrial_v1_20260923')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.out
    if (out / 'manifest.json').exists():
        raise RuntimeError('Completed pack exists; choose a new version/output path')
    out.mkdir(parents=True, exist_ok=True)
    (out / '.gitignore').write_text('*\n!.gitignore\n', encoding='utf-8')
    write_json(out / 'config.json', cfg)
    step_ns = cfg['grid_seconds'] * 10**9
    start_ns, end_ns = pd.Timestamp(cfg['start_utc']).value, pd.Timestamp(cfg['end_utc']).value
    grid = start_ns + np.arange((end_ns - start_ns) // step_ns + 1, dtype=np.int64) * step_ns
    if len(grid) < 2 or grid[-1] != pd.Timestamp(cfg['end_utc']).value:
        raise ValueError('Grid endpoints must align exactly')
    split = np.where(grid < pd.Timestamp(cfg['cut1']).value, 0,
                     np.where(grid < pd.Timestamp(cfg['cut2']).value, 1, 2)).astype(np.int8)
    np.save(out / 'time_ns.npy', grid)
    np.save(out / 'split.npy', split)
    channels = cfg['channels']
    names = [c['alias'] for c in channels]
    if len(set(names)) != len(names) or len({c['name'] for c in channels}) != len(channels):
        raise ValueError('Aliases and source names must be unique')
    arrays = {}
    for name, dtype in [('values', 'float32'), ('source_time_ns', 'int64'),
                        ('valid', 'bool'), ('observed', 'bool'), ('flags', 'uint8')]:
        arrays[name] = np.lib.format.open_memmap(out / f'{name}.npy', mode='w+',
                                               dtype=dtype, shape=(len(channels), len(grid)))
    mapping_path = Path(cfg['point_mapping'])
    mapping = pd.read_csv(mapping_path).drop_duplicates(['点位中文名称', '点位编码'])
    candidates = mapping.groupby('点位中文名称')['点位编码'].agg(list).to_dict()
    sources, quality = [], []
    load_index = names.index('load')
    if load_index != 0:
        raise ValueError('Load channel must be first for training-only quality calibration')
    stats_center, stats_scale, stats_count, active = [], [], [], []
    for index, channel in enumerate(channels):
        started = time.monotonic()
        source = Path(cfg['source_root']) / (channel['name'] + '.csv')
        print(f'[{index+1}/{len(channels)}] {channel["alias"]} {channel["name"]}', flush=True)
        ns, raw, source_stats = read_source(source)
        values, times, valid, fresh, ages = causal_align(ns, raw, grid,
                                                        cfg['max_age_seconds'], cfg['grid_seconds'])
        flags = np.zeros(len(grid), dtype=np.uint8)
        flags[~np.isfinite(values)] |= MISSING
        flags[(ages > cfg['max_age_seconds']) | (times < grid[0])] |= STALE
        valid &= times >= grid[0]
        if channel.get('minimum') is not None:
            flags[values < channel['minimum']] |= OUTSIDE_RANGE
        if channel.get('maximum') is not None:
            flags[values > channel['maximum']] |= OUTSIDE_RANGE
        valid &= (flags == 0)
        if index == load_index:
            load = values.copy()
            load_valid = valid.copy()
        calibration = (split == 0) & valid & fresh & load_valid & (load > cfg['operating_floor_mw'])
        high = None
        if channel.get('statistical_high_rule') and calibration.sum() >= 2:
            q1, q3 = np.quantile(values[calibration], [.25, .75])
            high = float(q3 + 12 * max(q3 - q1, 1.0))
            flags[values > high] |= SUSPECT_HIGH
            valid &= (flags == 0)
            calibration &= valid
        fitted = values[calibration]
        if len(fitted) >= 2:
            q1, center, q3 = np.quantile(fitted, [.25, .5, .75])
            scale = max(float(q3 - q1), 1e-3)
            variable = bool(np.ptp(fitted) > 1e-8)
        else:
            center, scale, variable = 0.0, 1.0, False
        stats_center.append(float(center)); stats_scale.append(scale)
        stats_count.append(len(fitted)); active.append(variable)
        arrays['values'][index] = values.astype(np.float32)
        arrays['source_time_ns'][index] = times
        arrays['valid'][index] = valid
        arrays['observed'][index] = fresh
        arrays['flags'][index] = flags
        codes = candidates.get(channel['name'], [])
        source_info = dict(channel, path=str(source), size_bytes=source.stat().st_size,
                           sha256=digest(source), code_candidates=codes,
                           code_verified=len(codes) == 1,
                           code_identity_status='single_mapping_candidate' if len(codes) == 1 else 'duplicate_name_export_identity_unresolved',
                           **source_stats)
        sources.append(source_info)
        record = {'alias': channel['alias'], 'name': channel['name'],
                  'training_fit_count': len(fitted), 'train_variable': variable,
                  'train_median': float(center), 'train_iqr_scale': scale,
                  'train_only_high_bound': high, 'source_code_unique': len(codes) == 1,
                  'source_median_interval_s': source_stats['median_interval_s'],
                  'source_max_gap_s': source_stats['max_interval_s']}
        for code, label in enumerate(['train', 'validation', 'historical_test']):
            part = split == code
            record[label + '_available_fraction'] = float(valid[part].mean())
            record[label + '_observed_fraction'] = float((valid & fresh)[part].mean())
            record[label + '_suspect_count'] = int(((flags & SUSPECT_HIGH != 0) & part).sum())
        quality.append(record)
        write_json(out / 'progress.json', {'finished_channels': index + 1, 'total_channels': len(channels),
                                         'last_alias': channel['alias'], 'last_seconds': round(time.monotonic() - started, 2)})
        del raw, ns, values, times, valid, fresh, ages
    for array in arrays.values():
        array.flush()
    write_json(out / 'sources.json', sources)
    write_json(out / 'normalization.json', {
        'fit_scope': 'fresh accepted observations in operating training rows only',
        'aliases': names, 'center': stats_center, 'scale': stats_scale,
        'fit_count': stats_count, 'active_from_training_variation': active})
    pd.DataFrame(quality).to_csv(out / 'quality.csv', index=False, encoding='utf-8-sig')
    tasks = cfg['tasks']
    all_counts = {}
    for task_name, task in tasks.items():
        task['input_channels_requested'] = task['input_channels'][:]
        task['input_channels'] = [a for a in task['input_channels'] if active[names.index(a)] or a in task['targets']]
        task['excluded_constant_or_unavailable_inputs'] = [a for a in task['input_channels_requested'] if a not in task['input_channels']]
        all_counts[task_name] = {}
        for profile_name, profile in cfg['profiles'].items():
            positions, counts = make_origins(grid, split, arrays['valid'], arrays['observed'],
                                            arrays['source_time_ns'], names, task, load,
                                            load_valid, cfg, profile)
            np.savez_compressed(out / f'origins_{task_name}_{profile_name}.npz', **positions)
            all_counts[task_name][profile_name] = counts
    write_json(out / 'tasks.json', tasks)
    hashes = {p.name: digest(p) for p in sorted(out.iterdir())
              if p.suffix in {'.npy', '.npz', '.json', '.csv'} and p.name != 'progress.json'}
    manifest = {'dataset_id': cfg['dataset_id'], 'status': 'PREPARED_NOT_TRAINED',
                'layout': 'channel,time for values/source_time_ns/valid/observed/flags',
                'grid_rows': len(grid), 'channels': len(channels), 'aliases': names,
                'start_utc': cfg['start_utc'], 'end_utc': cfg['end_utc'],
                'grid_seconds': cfg['grid_seconds'], 'window_counts': all_counts,
                'flag_bits': {'missing_or_nonfinite': MISSING, 'stale_or_before_pack': STALE,
                              'outside_declared_range': OUTSIDE_RANGE, 'train_statistical_high': SUSPECT_HIGH},
                'labels': 'fresh record in (t-10s,t], accepted quality and operating load; no held-value labels',
                'historical_exposure': cfg['historical_exposure'],
                'timestamp_semantics': 'exported UTC timestamp used as availability; acquisition latency not recorded',
                'source_mapping_sha256': digest(mapping_path), 'output_hashes': hashes,
                'code_sha256': {p.name: digest(p) for p in [Path(__file__), HERE / 'dataset.py']},
                'scope': {'model_training': False, 'test_scoring': False}}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'rows': len(grid), 'channels': len(channels),
                      'window_counts': all_counts}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
