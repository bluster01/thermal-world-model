"""Private observational windows: recorded inputs are explicit oracle conditions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CONTEXT, MAX_HORIZON, DT = 64, 128, 10
SPLITS = ('train', 'prototype_validation')
GROUPS = {'observations': 5, 'positions': 2, 'boundary': 7}
CONDITIONING = 'recorded_positions_and_boundaries_oracle'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    header = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)},
                        separators=(',', ':'), sort_keys=True).encode('ascii')
    return hashlib.sha256(header + b'\n' + array.tobytes(order='C')).hexdigest()


def text_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def logical_identity(expected):
    payload = expected['expected_receipt']
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(dataset_id=payload['dataset_id'], array_receipt_sha256=digest,
                selected_rows=payload['selected_grid_rows'])


@dataclass(frozen=True)
class WindowKey:
    side: str
    start: int
    split: str

    def as_dict(self):
        return dict(side=self.side, start=self.start, cut=self.start + CONTEXT - 1, split=self.split)


@dataclass(frozen=True)
class PlantWindowBatch:
    keys: tuple[WindowKey, ...]
    history_values: torch.Tensor
    history_available_mask: torch.Tensor
    history_present_mask: torch.Tensor
    history_age_s: torch.Tensor
    elapsed_seconds: torch.Tensor
    positions_left: torch.Tensor
    boundaries_left: torch.Tensor
    boundaries_right: torch.Tensor
    observations_right: torch.Tensor
    observed_right_mask: torch.Tensor
    observations_right_age_s: torch.Tensor
    conditioning: str = CONDITIONING


class PlantDataset:
    """Already prepared 1/10 payload; never opens the old full canonical dataset.

    Schema validation and each extracted window are fail-closed. Full array
    identity is bound by stable array hashes supplied to ``load``.
    Direct construction is intended for handcrafted software fixtures.
    """

    def __init__(self, arrays, manifest, *, identity=None):
        self.arrays, self.manifest = arrays, manifest
        self.identity = identity or {'scope': 'in_memory_software_fixture'}
        ts = arrays['timestamps_epoch_s']
        self.n = len(ts)
        if ts.dtype != np.int64 or ts.ndim != 1 or not len(ts) or not np.all(np.diff(ts) == DT):
            raise ValueError('Expected continuous int64 ten-second timestamps')
        if (manifest['dt_seconds'] != DT or manifest['window_context'] != CONTEXT or
                manifest['window_future'] != MAX_HORIZON):
            raise ValueError('Unexpected time/window contract')
        ranges = manifest['split_ranges']
        if (set(ranges) != set(SPLITS) or ranges['train'][0] != 0 or
                ranges['train'][1] != ranges['prototype_validation'][0] or
                ranges['prototype_validation'][1] != self.n):
            raise ValueError('Only disjoint train and prototype_validation partitions are allowed')
        split = arrays['prototype_split']
        expected = np.where(np.arange(self.n) < ranges['train'][1], 0, 1)
        if split.dtype != np.int8 or split.shape != (self.n,) or not np.array_equal(split, expected):
            raise ValueError('Partition metadata mismatch')
        for group, width in GROUPS.items():
            shape = (self.n, 2, width)
            for suffix, dtype in (('', np.float32), ('_available_mask', np.bool_), ('_present_mask', np.bool_),
                                   ('_age_s', np.float32), ('_source_timestamp_epoch_s', np.int64)):
                value = arrays[group + suffix]
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError(f'Invalid {group + suffix} schema')
            available, present = arrays[group + '_available_mask'], arrays[group + '_present_mask']
            age, source = arrays[group + '_age_s'], arrays[group + '_source_timestamp_epoch_s']
            if (np.any(present & ~available) or not np.isfinite(arrays[group][available]).all() or
                    not np.isfinite(age[available]).all() or np.any(age[available] < 0)):
                raise ValueError(f'Invalid availability/age values for {group}')
            timestamp = np.broadcast_to(ts[:, None, None], shape)
            if (not np.array_equal(age[available], (timestamp[available] - source[available]).astype(np.float32)) or
                    np.any(age[present] != 0)):
                raise ValueError(f'Present/held timestamp mismatch for {group}')
        for side in ('A', 'B'):
            for split_name, (lo, hi) in ranges.items():
                starts = arrays[f'{side}_{split_name}_property_input_window_starts']
                if (starts.dtype != np.int64 or starts.ndim != 1 or not len(starts) or
                        np.any(np.diff(starts) <= 0) or np.any(starts < lo) or np.any(starts + CONTEXT + MAX_HORIZON > hi)):
                    raise ValueError('Invalid or empty eligible window list')

    @classmethod
    def load(cls, directory, *, expected_identity_path):
        from analysis.world_model_plant.compare_receipts import compare, read_json
        directory = Path(directory)
        mpath, archive = directory / 'manifest.json', directory / 'plant_tenth.npz'
        expected = read_json(Path(expected_identity_path))
        preparer = Path(__file__).resolve().parents[2] / 'analysis/world_model_plant/prepare_tenth.py'
        comparison = compare(read_json(directory / 'public_receipt.json'), expected, preparer.read_bytes())
        if comparison['status'] != 'DATA_RECEIPT_MATCH':
            raise ValueError('Private dataset receipt differs from the frozen array identity')
        local_npz_sha256 = file_sha256(archive)
        manifest = json.loads(mpath.read_text(encoding='utf-8'))
        if (manifest['output_npz_sha256'] != local_npz_sha256 or
                manifest['old_validation_measurement_rows_read'] != 0 or manifest['old_test_measurement_rows_read'] != 0):
            raise ValueError('Unexpected dataset provenance')
        required = ['timestamps_epoch_s', 'prototype_split']
        required += [g + suffix for g in GROUPS for suffix in ('', '_available_mask', '_present_mask', '_age_s', '_source_timestamp_epoch_s')]
        required += [f'{side}_{split}_property_input_window_starts' for side in ('A', 'B') for split in SPLITS]
        with np.load(archive, allow_pickle=False) as data:
            arrays = {key: data[key] for key in required}
        for key, value in arrays.items():
            item = manifest['arrays'][key]
            if (list(value.shape) != item['shape'] or str(value.dtype) != item['dtype'] or
                    array_sha256(value) != item['array_sha256'] or
                    item['array_sha256'] != expected['expected_receipt']['array_hashes'][key]):
                raise ValueError(f'Array identity mismatch: {key}')
        for name in ('dataset_id', 'split_ranges', 'dt_seconds', 'window_context', 'window_future'):
            if manifest[name] != expected['expected_receipt'][name]:
                raise ValueError('Private manifest differs from the frozen data contract')
        result = cls(arrays, manifest, identity=logical_identity(expected))
        result.local_container_identity = dict(manifest_sha256=file_sha256(mpath), npz_sha256=local_npz_sha256)
        return result

    def starts(self, side, split):
        if side not in ('A', 'B') or split not in SPLITS:
            raise ValueError('Unknown side or split; no test split exists')
        return self.arrays[f'{side}_{split}_property_input_window_starts']

    def _validate_key(self, key):
        if not isinstance(key, WindowKey) or isinstance(key.start, bool) or not isinstance(key.start, int):
            raise ValueError('A typed integer WindowKey is required')
        starts = self.starts(key.side, key.split)
        location = np.searchsorted(starts, key.start)
        if location == len(starts) or starts[location] != key.start:
            raise ValueError('Window was not declared property-input eligible')
        side, sl = ('A', 'B').index(key.side), slice(key.start, key.start + CONTEXT + MAX_HORIZON)
        earliest = self.arrays['timestamps_epoch_s'][self.manifest['split_ranges'][key.split][0]]
        for group in GROUPS:
            required = slice(None, 6) if group == 'boundary' else slice(None)
            values = self.arrays[group][sl, side, required]
            present = self.arrays[group + '_present_mask'][sl, side, required]
            ages = self.arrays[group + '_age_s'][sl, side, required]
            sources = self.arrays[group + '_source_timestamp_epoch_s'][sl, side, required]
            row_ts = self.arrays['timestamps_epoch_s'][sl, None]
            if (not np.isfinite(values).all() or not self.arrays[group + '_available_mask'][sl, side, required].all() or
                    not np.isfinite(ages).all() or np.any((ages < 0) | (ages > 30)) or
                    np.any(sources < earliest) or not np.array_equal(ages, (row_ts - sources).astype(np.float32)) or
                    np.any(ages[present] != 0)):
                raise ValueError('Window contains unavailable, stale, or cross-split held input')
        o, p, b = (self.arrays[g][sl, side] for g in GROUPS)
        if (np.any((o < 300) | (o > 650)) or np.any((p < 0) | (p > 1)) or
                np.any((b[:, 3] < 300) | (b[:, 3] > 650)) or np.any((b[:, 4] < 295) | (b[:, 4] > 355)) or
                np.any((b[:, (2, 5)] < 8) | (b[:, (2, 5)] > 30)) or
                np.any(b[:, 2] < b[:, 5]) or np.any(b[:, 0] <= 0) or np.any(b[:, 1] < 0)):
            raise ValueError('Window violates the frozen property-input eligibility rule')

    def batch(self, keys, *, horizon=32, device='cpu', dtype=torch.float32):
        keys = tuple(keys)
        if not keys or isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= MAX_HORIZON:
            raise ValueError('A nonempty batch with horizon 1..128 is required')
        if len({key.split for key in keys}) != 1:
            raise ValueError('A batch cannot mix train and prototype validation')
        records = []
        for key in keys:
            self._validate_key(key)
            si, cut = ('A', 'B').index(key.side), key.start + CONTEXT - 1
            hs, left, right = slice(key.start, cut + 1), slice(cut, cut + horizon), slice(cut + 1, cut + horizon + 1)
            history = {}
            for suffix in ('', '_available_mask', '_present_mask', '_age_s'):
                history[suffix] = np.concatenate([self.arrays[g + suffix][hs, si] for g in GROUPS], -1).copy()
                # W is archived but cannot enter the encoder, query or residual.
                history[suffix][:, -1] = False if 'mask' in suffix else 0.
            bl, br = self.arrays['boundary'][left, si].copy(), self.arrays['boundary'][right, si].copy()
            bl[:, -1] = br[:, -1] = 0.
            records.append([history[''], history['_available_mask'], history['_present_mask'], history['_age_s'],
                            (self.arrays['timestamps_epoch_s'][hs] - self.arrays['timestamps_epoch_s'][cut]).astype(np.float64),
                            self.arrays['positions'][left, si], bl, br, self.arrays['observations'][right, si],
                            self.arrays['observations_present_mask'][right, si], self.arrays['observations_age_s'][right, si]])
        converted = []
        for index, column in enumerate(zip(*records)):
            array = np.stack(column)
            target_dtype = torch.bool if array.dtype == bool else torch.float64 if index == 4 else dtype
            converted.append(torch.tensor(array, device=device, dtype=target_dtype))
        return PlantWindowBatch(keys, *converted)

    def plan(self, *, updates=250, batch_size=4, sample_seed=72001, anchors_per_side=16):
        for name, value in [('updates', updates), ('batch_size', batch_size), ('anchors_per_side', anchors_per_side)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be positive integer')
        if batch_size % 2 or anchors_per_side > 16:
            raise ValueError('Require even A/B-balanced batches and at most16 anchors per side')
        rng = np.random.default_rng(sample_seed)
        def fresh_starts(side, split):
            starts = self.starts(side, split)
            fresh = self.arrays['observations_present_mask'][starts + CONTEXT - 1, ('A', 'B').index(side)].all(-1)
            selected = starts[fresh]
            if not len(selected):
                raise ValueError('No eligible windows with five fresh cut observations')
            return selected
        eligible = {(side, split): fresh_starts(side, split) for side in ('A', 'B') for split in SPLITS}
        training = []
        for _ in range(updates):
            training.append([WindowKey(side, int(rng.choice(eligible[side, 'train'])), 'train').as_dict()
                             for side in ('A', 'B') for _ in range(batch_size // 2)])
        anchors = {}
        for split in SPLITS:
            anchors[split] = []
            for side in ('A', 'B'):
                starts = eligible[side, split]
                indices = np.linspace(0, len(starts) - 1, min(anchors_per_side, len(starts)), dtype=np.int64)
                anchors[split] += [WindowKey(side, int(starts[i]), split).as_dict() for i in indices]
        return dict(schema_version=1, dataset_identity=self.identity, conditioning=CONDITIONING,
                    sampler='numpy_PCG64_balanced_AB_with_replacement_v1', sample_seed=sample_seed,
                    cut_anchor_policy='all_five_observations_present_at_cut',
                    training=training, diagnostic_anchors=anchors,
                    anchors_may_overlap=True, anchors_are_not_independent_episodes=True)


def keys_from_records(records):
    result = tuple(WindowKey(r['side'], r['start'], r['split']) for r in records)
    if any(r.get('cut', key.start + CONTEXT - 1) != key.start + CONTEXT - 1 for r, key in zip(records, result)):
        raise ValueError('Plan cut/start mismatch')
    return result
