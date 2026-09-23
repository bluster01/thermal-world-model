"""Memory-mapped, framework-neutral window reader for the prepared three tasks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class IndustrialDataset:
    def __init__(self, root, task, split='train', profile='h60m_f10m',
                 complete_labels=False, future_mode='forecast'):
        if future_mode not in {'forecast', 'recorded_replay'}:
            raise ValueError('future_mode must be forecast or recorded_replay')
        if split not in {'train', 'validation', 'historical_test'}:
            raise ValueError('Unknown split')
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.config = json.loads((self.root / 'config.json').read_text(encoding='utf-8'))
        self.task = json.loads((self.root / 'tasks.json').read_text(encoding='utf-8'))[task]
        self.profile = self.config['profiles'][profile]
        self.future_mode = future_mode
        names = self.manifest['aliases']
        self.input_names = self.task['input_channels']
        self.target_names = self.task['targets']
        self.input_indices = np.array([names.index(a) for a in self.input_names])
        self.target_indices = np.array([names.index(a) for a in self.target_names])
        self.action_indices = np.array([names.index(a) for a in self.task['measured_action_channels']], dtype=np.int64)
        self.boundary_indices = np.array([names.index(a) for a in self.task['boundary_channels']], dtype=np.int64)
        self.load_index = names.index('load')
        normal = json.loads((self.root / 'normalization.json').read_text(encoding='utf-8'))
        self.center = np.asarray(normal['center'], dtype=np.float32)
        self.scale = np.asarray(normal['scale'], dtype=np.float32)
        self.arrays = {key: np.load(self.root / f'{key}.npy', mmap_mode='r')
                       for key in ['values', 'valid', 'observed', 'flags', 'source_time_ns', 'time_ns', 'split']}
        with np.load(self.root / f'origins_{task}_{profile}.npz') as f:
            self.origins = f[split + ('_complete' if complete_labels else '')]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, item):
        origin = int(self.origins[item])
        h, f = self.profile['history_steps'], self.profile['forecast_steps']
        hist = np.arange(origin - h + 1, origin + 1)
        future = np.arange(origin + 1, origin + f + 1)
        ix = np.ix_(self.input_indices, hist)
        raw = self.arrays['values'][ix].T.copy()
        mask = self.arrays['valid'][ix].T.copy()
        source = self.arrays['source_time_ns'][ix].T.copy()
        present = self.arrays['observed'][ix].T.copy() & mask
        # Float arithmetic is done only for found timestamps (avoid int64 NAT overflow).
        age = np.full(raw.shape, np.inf, dtype=np.float32)
        found = source != np.iinfo(np.int64).min
        repeated_time = np.broadcast_to(self.arrays['time_ns'][hist, None], source.shape)
        age[found] = (repeated_time[found] - source[found]) / 1e9
        x = (raw - self.center[self.input_indices]) / self.scale[self.input_indices]
        x = np.where(mask, x, 0).astype(np.float32)
        yi = np.ix_(self.target_indices, future)
        y_raw = self.arrays['values'][yi].T.copy()
        y_mask = self.arrays['valid'][yi].T.copy() & self.arrays['observed'][yi].T
        online = self.arrays['valid'][self.load_index, future] & (self.arrays['values'][self.load_index, future] > self.config['operating_floor_mw'])
        y_mask &= online[:, None]
        y = (y_raw - self.center[self.target_indices]) / self.scale[self.target_indices]
        result = {'origin_time_ns': int(self.arrays['time_ns'][origin]),
                  'history_time_ns': self.arrays['time_ns'][hist].copy(),
                  'future_time_ns': self.arrays['time_ns'][future].copy(),
                  'history_values': raw, 'history_normalized': x,
                  'history_valid': mask, 'history_observed': present,
                  'history_age_s': age, 'history_source_time_ns': source,
                  'targets': np.where(y_mask, y_raw, np.nan),
                  'targets_normalized': np.where(y_mask, y, 0).astype(np.float32),
                  'target_mask': y_mask}
        if self.future_mode == 'recorded_replay':
            # These are logged future observations, never promised future inputs.
            for name, channels in [('actions', self.action_indices), ('boundaries', self.boundary_indices)]:
                fi = np.ix_(channels, future)
                result['recorded_future_' + name] = self.arrays['values'][fi].T.copy()
                result['recorded_future_' + name + '_valid'] = self.arrays['valid'][fi].T.copy()
        return result
