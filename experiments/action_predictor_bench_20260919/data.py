"""Portable, index-selected windows; no training or held-out split access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CONTEXT, TRAIN_H, LONG_H, DT = 64, 32, 512, 10


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stratified_sample(starts, count, seed):
    """One random start per chronological bin; not the first N windows."""
    if not 0 < count <= len(starts):
        raise ValueError('Sample count exceeds available windows')
    edges = np.linspace(0, len(starts), count + 1, dtype=int)
    rng = np.random.default_rng(seed)
    return np.asarray(starts)[[rng.integers(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]]


def expanded_sample(pool, existing, count, seed):
    """Retain every earlier window and stratify the additional sample in time."""
    existing = np.asarray(existing)
    if len(np.unique(existing)) != len(existing) or not np.isin(existing, pool).all():
        raise ValueError('Existing windows must be unique members of the training pool')
    if not len(existing) <= count <= len(pool):
        raise ValueError('Expansion count outside available pool')
    remaining = np.setdiff1d(pool, existing)
    extra = stratified_sample(remaining, count-len(existing), seed) if count > len(existing) else []
    return np.sort(np.concatenate((existing, extra))).astype(np.int64)


def pack(source, destination, fraction=.1, side='A', seed=20260919, include_pack=None):
    if not .1 <= fraction <= 1 / 3:
        raise ValueError('Quick screen fraction must be between 1/10 and 1/3')
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'BUILT_AUDITED':
        raise ValueError('Expected prepared full-window dataset')
    values = np.load(source / f'values_{side}.npy', mmap_mode='r')
    ts = np.load(source / 'timestamps.npy', mmap_mode='r')
    split = np.load(source / 'split.npy', mmap_mode='r')
    train_pool = np.load(source / f'windows/{side}/train/c64_h128_stride80.npy')
    val_pool = np.load(source / f'windows/{side}/validation/c64_h512_stride80.npy')
    train = stratified_sample(train_pool, int(len(train_pool) * fraction), seed)
    previous_metadata = None
    if include_pack is not None:
        previous, previous_metadata = load_pack(include_pack)
        if previous_metadata['side'] != side or previous_metadata['source_manifest_sha256'] != sha256(source / 'manifest.json'):
            raise ValueError('Expansion must use the same source and side')
        train = expanded_sample(train_pool, previous['train_starts'], len(train), seed)
    # Selector first third, reporting last two thirds, with a full long-window gap.
    edge = val_pool[len(val_pool) // 3]
    selector_pool = val_pool[val_pool + CONTEXT + LONG_H <= edge]
    report_pool = val_pool[val_pool >= edge]
    selector = stratified_sample(selector_pool, min(128, len(selector_pool)), seed + 1)
    report = stratified_sample(report_pool, min(256, len(report_pool)), seed + 2)
    arrays = {}
    for name, starts, horizon, split_id in [('train', train, TRAIN_H, 0),
                                         ('selector', selector, TRAIN_H, 1),
                                         ('evaluation', report, LONG_H, 1)]:
        rows = starts[:, None] + np.arange(CONTEXT + horizon)
        if not np.all(split[rows] == split_id) or not np.all(np.diff(ts[rows], axis=1) == DT):
            raise ValueError('Window crosses split or timestamp gap')
        bank = np.asarray(values[rows], dtype=np.float32)
        if not np.isfinite(bank).all():
            raise ValueError('Nonfinite data')
        arrays.update({name: bank, name + '_starts': starts,
                       name + '_time': np.asarray(ts[starts + CONTEXT - 1])})
    norm = manifest['normalization'][side]
    arrays['mean'] = np.asarray(norm['mean'], dtype=np.float32)
    arrays['scale'] = np.asarray(norm['scale'], dtype=np.float32)
    if include_pack is not None:
        for name in ('selector', 'evaluation', 'selector_starts', 'evaluation_starts', 'mean', 'scale'):
            if not np.array_equal(arrays[name], previous[name]):
                raise ValueError(f'Expansion changed fixed evaluation or normalization: {name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    metadata = {'source_dataset': manifest['dataset_id'], 'source_manifest_sha256': sha256(source / 'manifest.json'),
                'source_values_sha256': sha256(source / f'values_{side}.npy'),
                'pack_sha256': sha256(destination), 'side': side, 'fraction_of_training_windows': fraction,
                'fraction_note': 'Fraction of eligible stride80 windows, not unique raw rows/duration; overlapping windows.',
                'training_pool_count': len(train_pool), 'selection_seed': seed,
                'counts': {k: len(arrays[k]) for k in ('train', 'selector', 'evaluation')},
                'context': CONTEXT, 'train_horizon': TRAIN_H, 'evaluation_horizon': LONG_H,
                'normalization': 'existing unique train-row statistics; no validation fit',
                'split': 'original train/validation; selector and reporting windows time-separated; no test/extension',
                'time_alignment': 'history ends at t; u/d at t lead to T at t+1; pack also contains right-end controls for history feedback',
                'channel_names': ['T1', 'T2', 'T3', 'T4', 'T5_main', 'valve1', 'valve2',
                                  'steam_flow', 'coal', 'separator_pressure', 'separator_temperature',
                                  'feedwater_temperature', 'outlet_pressure']}
    if previous_metadata:
        metadata['contains_previous_pack_sha256'] = previous_metadata['pack_sha256']
        metadata['retained_training_windows'] = len(previous['train'])
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def load_pack(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256(path) != metadata['pack_sha256']:
        raise ValueError('Data pack checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}, metadata


def unpack(bank, device='cpu'):
    bank = torch.as_tensor(bank, dtype=torch.float32, device=device)
    # u/d have H+1 entries: final entry is for right-end history feedback only.
    return bank[:, :CONTEXT], bank[:, CONTEXT-1:, 5:7], bank[:, CONTEXT-1:, 7:], bank[:, CONTEXT:, :5]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fraction', type=float, default=.1)
    p.add_argument('--side', choices=['A', 'B'], default='A')
    p.add_argument('--include-pack', help='Keep all training windows and the fixed validation of an earlier pack')
    args = p.parse_args()
    print(json.dumps(pack(args.source, args.output, args.fraction, args.side, include_pack=args.include_pack), indent=2))
