"""Saved-only array and chronology audit for the prepared private prototype."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def check(directory: Path) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    public = json.loads((directory / 'public_receipt.json').read_text(encoding='utf-8'))
    archive = directory / 'plant_tenth.npz'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['output_npz_sha256']:
        raise ValueError('Archive SHA mismatch')
    with np.load(archive, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if set(arrays) != set(manifest['arrays']) or set(arrays) != set(public['array_hashes']):
        raise ValueError('Array inventory mismatch')
    for name, array in arrays.items():
        header = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)}, separators=(',', ':'), sort_keys=True).encode('ascii')
        digest = hashlib.sha256(header + b'\n' + array.tobytes(order='C')).hexdigest()
        if digest != manifest['arrays'][name]['array_sha256'] or digest != public['array_hashes'][name]:
            raise ValueError(f'Array identity mismatch: {name}')
    ts = arrays['timestamps_epoch_s']
    assert ts.dtype == np.dtype('int64') and len(ts) == 53077
    assert np.all(np.diff(ts) == 10)
    assert manifest['old_training_rows'] == 530779 and len(ts) == 530779 // 10
    assert manifest['old_validation_measurement_rows_read'] == manifest['old_test_measurement_rows_read'] == 0
    assert manifest['source_prefix_audit']['numeric_records_after_selected_prefix_read'] == 0
    raw = arrays['raw_values']
    np.testing.assert_array_equal(arrays['raw_present_mask'], np.isfinite(raw))
    present_rows = arrays['raw_exported_record_mask']
    assert present_rows.dtype == bool
    assert np.isnan(raw[~present_rows]).all()
    assert int(present_rows.sum()) == manifest['source_prefix_audit']['data_records_read']
    held = np.full(raw.shape, np.nan, dtype=np.float64)
    expected_source = np.full(raw.shape, np.iinfo(np.int64).min, dtype=np.int64)
    # Separate per-column forward state machine; no prep hold helper imported.
    for j in range(raw.shape[1]):
        value, source_time = np.nan, np.iinfo(np.int64).min
        for i in range(len(ts)):
            if np.isfinite(raw[i, j]):
                value, source_time = raw[i, j], ts[i]
            held[i, j], expected_source[i, j] = value, source_time
    np.testing.assert_array_equal(expected_source, arrays['raw_source_timestamp_epoch_s'])
    known = expected_source != np.iinfo(np.int64).min
    ages = np.full(raw.shape, np.inf, dtype=np.float32)
    ages[known] = (np.broadcast_to(ts[:, None], raw.shape)[known] - expected_source[known]).astype(np.float32)
    np.testing.assert_array_equal(ages, arrays['raw_age_s'])
    assert np.all(ages >= 0)
    columns = manifest['source_prefix_audit']['columns']
    for side_index, side in enumerate(['A', 'B']):
        for group, specs in manifest['mapping'][side].items():
            indices = [columns.index(spec[1]) for spec in specs]
            scaled = (held[:, indices] * np.array([spec[2] for spec in specs])).astype(np.float32)
            np.testing.assert_array_equal(scaled, arrays[group][:, side_index])
            np.testing.assert_array_equal(ages[:, indices], arrays[group + '_age_s'][:, side_index])
            np.testing.assert_array_equal(np.isfinite(raw[:, indices]), arrays[group + '_present_mask'][:, side_index])
            np.testing.assert_array_equal(known[:, indices], arrays[group + '_available_mask'][:, side_index])
            np.testing.assert_array_equal(expected_source[:, indices], arrays[group + '_source_timestamp_epoch_s'][:, side_index])
    expected_split = np.concatenate((np.zeros(42461, dtype=np.int8), np.ones(10616, dtype=np.int8)))
    np.testing.assert_array_equal(expected_split, arrays['prototype_split'])
    window_counts = {}
    for side in ['A', 'B']:
        for split, bounds in manifest['split_ranges'].items():
            lo, hi = bounds
            for label in ['basic', 'property_input']:
                starts = arrays[f'{side}_{split}_{label}_window_starts']
                rows = arrays[f'{side}_{label}_row_eligible']
                # Direct convolution of 192-point row masks independently checks membership.
                expected = np.flatnonzero(np.convolve(rows[lo:hi].astype(np.int64), np.ones(192, dtype=np.int64), mode='valid') == 192) + lo
                np.testing.assert_array_equal(starts, expected)
                assert len(starts) == manifest['eligibility'][side]['windows'][f'{split}_{label}']
                # Any retained/held required cell in a selected window must also originate in this split.
                si = 0 if side == 'A' else 1
                sources_in_split = np.ones(len(ts), dtype=bool)
                for group in ['observations', 'positions', 'boundary']:
                    required = slice(None, 6) if group == 'boundary' else slice(None)
                    sources_in_split &= np.all(arrays[group + '_source_timestamp_epoch_s'][:, si, required] >= ts[lo], axis=1)
                valid_source_starts = np.flatnonzero(np.convolve(sources_in_split[lo:hi].astype(np.int64), np.ones(192, dtype=np.int64), mode='valid') == 192) + lo
                assert np.isin(starts, valid_source_starts).all(), 'A held cell originates before its prototype split'
                window_counts[f'{side}_{split}_{label}'] = len(starts)
    return {'status': 'SAVED_ARRAY_CONSISTENCY_VERIFIED', 'array_count': len(arrays),
            'selected_grid_rows': len(ts), 'source_records': int(present_rows.sum()),
            'source_missing_grid_records': int((~present_rows).sum()), 'max_export_cell_age_s': float(np.max(ages[known])),
            'window_counts': window_counts, 'selected_windows_have_no_held_source_before_own_split': True,
            'raw_source_reread': False, 'canonical_arrays_reread': False,
            'model_or_checkpoint_loaded': False, 'scientific_prediction_or_control_validated': False,
            'limitation': 'Verifies retained exported-cell transformations and partitions; not original sensor timing, DCS semantics, upstream causality or physical performance.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check(args.dataset)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as f:
            f.write(text + '\n')
    print(text)
