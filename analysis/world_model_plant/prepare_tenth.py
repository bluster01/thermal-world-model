"""Bounded, prefix-only real plant data preparation; no model imports or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import numpy as np

EXPECTED_CANONICAL_SHA = '24da77960e05e3636cc7b97a60a75e9b4ba470a3abb4f8ba920ddf11c6dad1d0'
EXPECTED_HEADER_SHA = '45d6fdc0aa9ca0c8de3d43096f8d5e68924e638a36326d835514bf66110eadb0'
OLD_TRAIN_ROWS = 530779
EXPECTED_ROWS = 707709
SELECTED_ROWS = OLD_TRAIN_ROWS // 10
NAT = np.iinfo(np.int64).min
OBS_NAMES = ['sh1_inlet_temp', 'sh1_outlet_temp', 'sh2_inlet_temp', 'sh2_outlet_temp', 'final_outlet_temp']
POSITION_NAMES = ['valve1_position', 'valve2_position']
BOUNDARY_NAMES = ['steam_flow', 'coal_command', 'separator_pressure', 'separator_temperature', 'feedwater_temperature', 'outlet_pressure', 'spray_flow_total']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Stable dtype/shape/C-order payload identity, independent of ZIP metadata."""
    if array.dtype.hasobject:
        raise ValueError('Cannot hash object array')
    header = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)}, separators=(',', ':'), sort_keys=True).encode('ascii')
    return hashlib.sha256(header + b'\n' + array.tobytes(order='C')).hexdigest()


def bounded_npy_prefix(path: Path, member: str, rows: int) -> tuple[np.ndarray, dict]:
    """Decode exactly the requested first-axis numeric rows, never np.load entire member."""
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        if not shape or fortran or dtype.hasobject or rows < 0 or rows > shape[0]:
            raise ValueError('Unsupported or out-of-bounds NPY prefix')
        count = rows * int(np.prod(shape[1:], dtype=np.int64))
        payload = f.read(count * dtype.itemsize)
        if len(payload) != count * dtype.itemsize:
            raise ValueError('Truncated numeric prefix')
        array = np.frombuffer(payload, dtype=dtype).reshape((rows, *shape[1:])).copy()
        return array, {'member': member, 'shape': list(shape), 'dtype': dtype.str,
                       'decoded_rows': rows, 'decoded_numeric_bytes': len(payload)}


def parse_timestamp(value: str) -> int:
    dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # computational convention only
    stamp = dt.timestamp()
    if not np.isfinite(stamp) or stamp != int(stamp):
        raise ValueError('Expected exact second timestamp')
    return int(stamp)


def mappings() -> dict:
    result = {}
    for side, word, other in [('A', '左', 'B'), ('B', '右', 'A')]:
        result[side] = {
            'observations': [
                [OBS_NAMES[0], f'选择后{word}侧一过喷水减温器入口', 1.0],
                [OBS_NAMES[1], f'选择后{word}侧一过喷水减温器出口', 1.0],
                [OBS_NAMES[2], f'选择后二级减温器{word}侧入口蒸汽', 1.0],
                [OBS_NAMES[3], f'选择后{word}侧二过喷水减温器出口', 1.0],
                [OBS_NAMES[4], f'选择后{word}侧末级过热器出口汽温', 1.0]],
            'positions': [
                [POSITION_NAMES[0], f'过热器一级减温器{side}侧喷水调节门阀位反馈', 0.01],
                [POSITION_NAMES[1], f'过热器二级减温器{other}侧喷水调节门阀位反馈', 0.01]],
            'boundary': [
                [BOUNDARY_NAMES[0], '主蒸汽流量_60BKAO0312', 1/3.6],
                [BOUNDARY_NAMES[1], '未校正总煤量', 1.0],
                [BOUNDARY_NAMES[2], '选择后分离器最终出口压力', 1.0],
                [BOUNDARY_NAMES[3], '分离器最终出口温度', 1.0],
                [BOUNDARY_NAMES[4], '选择后省煤器出口给水温度', 1.0],
                [BOUNDARY_NAMES[5], f'选择后末级过热器{word}侧出口压力', 1.0],
                [BOUNDARY_NAMES[6], '过热器减温水总流量', 1.0]],
            'command_candidates_unconfirmed': [
                ['stage1_main_output', f'过热器一级减温器{side}喷水调节阀主调输出', 1.0],
                ['stage2_main_output', f'过热器二级减温器{other}喷水调节阀主调输出', 1.0]],
        }
    return result


def read_exact_csv_prefix(path: Path, expected_ts: np.ndarray, columns: list[str], expected_header_sha: str | None = None):
    """Read a sorted timestamp-bounded prefix onto its exact grid; never read future numeric fields.

    Gaps are represented as NaN update rows. The first CSV field is read separately,
    so if the final allowed tick is absent only the next timestamp is inspected.
    Duplicate/off-grid records fail closed rather than silently discarding raw values.
    """
    if not len(expected_ts) or np.any(np.diff(expected_ts) <= 0):
        raise ValueError('Expected timestamps must be nonempty and increasing')
    values = np.full((len(expected_ts), len(columns)), np.nan, dtype=np.float64)
    record_present = np.zeros(len(expected_ts), dtype=bool)
    bad_numeric = np.zeros(len(columns), dtype=np.int64)
    h = hashlib.sha256()
    with path.open('rb') as f:
        header = f.readline()
        h.update(header)
        header_sha = hashlib.sha256(header).hexdigest()
        if expected_header_sha is not None and header_sha != expected_header_sha:
            raise ValueError('CSV header identity mismatch')
        names = next(csv.reader([header.decode('utf-8-sig')], strict=True))
        if len(set(names)) != len(names) or names[0] != 'time':
            raise ValueError('Ambiguous columns or missing first time column')
        indices = [names.index(name) for name in columns]
        timezone_forms = set()
        last_time, records_read, stop_reason, lookahead_time_bytes = None, 0, 'source_eof', 0
        while True:
            first_field = bytearray()
            while True:
                byte = f.read(1)
                if not byte or byte == b',':
                    break
                first_field.extend(byte)
                if byte in {b'\n', b'\r'} or len(first_field) > 128:
                    raise ValueError('Malformed timestamp first field')
            if not byte:
                if first_field:
                    raise ValueError('Truncated timestamp record')
                break
            timestamp_text = next(csv.reader([first_field.decode('utf-8')], strict=True))[0]
            actual = parse_timestamp(timestamp_text)
            if actual > int(expected_ts[-1]):
                stop_reason = 'next_timestamp_exceeds_allowed_cutoff_numeric_fields_not_read'
                lookahead_time_bytes = len(first_field) + 1
                break
            if (last_time is None and actual != int(expected_ts[0])) or (last_time is not None and actual <= last_time):
                raise ValueError('Source start mismatch or duplicate/non-increasing timestamps')
            i = int(np.searchsorted(expected_ts, actual))
            if i >= len(expected_ts) or int(expected_ts[i]) != actual:
                raise ValueError('Source timestamp is not on the allowed canonical grid')
            line = bytes(first_field) + b',' + f.readline()
            h.update(line)
            row = next(csv.reader([line.decode('utf-8')], strict=True))
            if len(row) != len(names):
                raise ValueError(f'Non-flat CSV record at prefix index {i}')
            timezone_forms.add('explicit_offset' if datetime.fromisoformat(row[0].replace('Z', '+00:00')).tzinfo is not None else 'naive_assumed_UTC_for_computation')
            for j, idx in enumerate(indices):
                value = row[idx].strip()
                if not value or value.lower() in {'nan', 'na', 'null', 'none'}:
                    continue
                try:
                    values[i, j] = float(value)
                except ValueError:
                    bad_numeric[j] += 1
            record_present[i] = True
            last_time = actual
            records_read += 1
            if actual == int(expected_ts[-1]):
                stop_reason = 'exact_allowed_final_timestamp_no_lookahead'
                break
        bytes_consumed = f.tell()
    if not records_read:
        raise ValueError('No records in allowed source interval')
    return values, {'header_sha256': header_sha, 'prefix_sha256': h.hexdigest(),
                    'data_records_read': records_read, 'canonical_grid_rows': len(expected_ts),
                    'missing_exported_grid_records': int((~record_present).sum()),
                    'stop_reason': stop_reason, 'timestamp_lookahead_bytes': lookahead_time_bytes,
                    'numeric_records_after_selected_prefix_read': 0,
                    'bytes_consumed_including_header': bytes_consumed, 'source_file_size': path.stat().st_size,
                    'columns': columns, 'csv_column_indices': indices,
                    'non_numeric_nonempty_counts': bad_numeric.tolist(), 'timestamp_forms': sorted(timezone_forms)}, record_present


def causal_hold(raw: np.ndarray, timestamps: np.ndarray):
    if raw.ndim != 2 or not len(timestamps) or raw.shape[0] != len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Invalid causal grid input')
    present = np.isfinite(raw)
    last = np.maximum.accumulate(np.where(present, np.arange(len(raw))[:, None], -1), axis=0)
    available = last >= 0
    held = np.where(available, raw[np.maximum(last, 0), np.arange(raw.shape[1])[None, :]], np.nan)
    source_ts = np.where(available, timestamps[np.maximum(last, 0)], NAT)
    age = np.full(raw.shape, np.inf, dtype=np.float32)
    grid_ts = np.broadcast_to(timestamps[:, None], raw.shape)
    age[available] = (grid_ts[available] - source_ts[available]).astype(np.float32)
    return held, present, available, age, source_ts


def window_starts(n: int, start: int, stop: int, valid: np.ndarray, context: int = 64, horizon: int = 128):
    if not (0 <= start <= stop <= n) or valid.shape != (n,) or context < 1 or horizon < 1:
        raise ValueError('Invalid window partition')
    span = context + horizon
    starts = np.arange(start, max(start, stop - span + 1), dtype=np.int64)
    bad = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
    return starts[(bad[starts + span] - bad[starts]) == 0]


def finite_stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {'count': int(values.size), 'finite_count': int(finite.size),
            'quantiles_0_01_50_99_100': np.quantile(finite, [0, .01, .5, .99, 1]).tolist() if finite.size else None}


def run(canonical: Path, source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing prototype')
    if sha256(canonical) != EXPECTED_CANONICAL_SHA:
        raise ValueError('Canonical identity mismatch')
    timestamps, timestamp_read = bounded_npy_prefix(canonical, 'timestamps.npy', SELECTED_ROWS)
    splits, split_read = bounded_npy_prefix(canonical, 'split.npy', OLD_TRAIN_ROWS + 1)
    if timestamp_read['shape'] != [EXPECTED_ROWS] or split_read['shape'] != [EXPECTED_ROWS]:
        raise ValueError('Unexpected canonical shape')
    if not np.all(splits[:OLD_TRAIN_ROWS] == 0) or splits[OLD_TRAIN_ROWS] != 1:
        raise ValueError('Old train boundary mismatch')
    if timestamps.dtype != np.dtype('int64') or not np.all(np.diff(timestamps) == 10):
        raise ValueError('Not a continuous 10 second prefix')
    mapping = mappings()
    columns = list(dict.fromkeys(spec[1] for side in mapping.values() for group in side.values() for spec in group))
    raw, source_audit, record_present = read_exact_csv_prefix(source, timestamps, columns, EXPECTED_HEADER_SHA)
    held, present, available, age, source_ts = causal_hold(raw, timestamps)
    arrays = {'timestamps_epoch_s': timestamps, 'raw_values': raw,
              'raw_exported_record_mask': record_present,
              'raw_present_mask': present, 'raw_age_s': age, 'raw_source_timestamp_epoch_s': source_ts}
    train_stop = len(timestamps) * 4 // 5
    arrays['prototype_split'] = np.where(np.arange(len(timestamps)) < train_stop, 0, 1).astype(np.int8)
    stats, eligibility = {}, {}
    for group in ['observations', 'positions', 'boundary', 'command_candidates_unconfirmed']:
        block, masks, fresh, ages, sources = [], [], [], [], []
        for side in ['A', 'B']:
            specs = mapping[side][group]
            idx = [columns.index(spec[1]) for spec in specs]
            block.append((held[:, idx] * np.array([spec[2] for spec in specs])).astype(np.float32))
            masks.append(available[:, idx]); fresh.append(present[:, idx])
            ages.append(age[:, idx]); sources.append(source_ts[:, idx])
        arrays[group] = np.stack(block, axis=1)
        arrays[group + '_available_mask'] = np.stack(masks, axis=1)
        arrays[group + '_present_mask'] = np.stack(fresh, axis=1)
        arrays[group + '_age_s'] = np.stack(ages, axis=1)
        arrays[group + '_source_timestamp_epoch_s'] = np.stack(sources, axis=1)
    for side_index, side in enumerate(['A', 'B']):
        for group, specs in mapping[side].items():
            for j, spec in enumerate(specs):
                values = arrays[group][:, side_index, j]
                channel_age = arrays[group + '_age_s'][:, side_index, j]
                stats[f'{side}.{group}.{spec[0]}'] = {
                    **finite_stats(values), 'unit_scale': spec[2],
                    'source_present_fraction': float(arrays[group + '_present_mask'][:, side_index, j].mean()),
                    'max_finite_age_s': float(np.max(channel_age[np.isfinite(channel_age)])) if np.isfinite(channel_age).any() else None}
        obs = arrays['observations'][:, side_index]
        pos = arrays['positions'][:, side_index]
        b = arrays['boundary'][:, side_index]
        basic = np.ones(len(timestamps), dtype=bool)
        for group in ['observations', 'positions', 'boundary']:
            required = slice(None, 6) if group == 'boundary' else slice(None)
            basic &= np.all(np.isfinite(arrays[group][:, side_index, required]), axis=1)
            basic &= np.all(arrays[group + '_age_s'][:, side_index, required] <= 30, axis=1)
        checks = {
            'five_observations_in_Tg': np.all((obs >= 300) & (obs <= 650), axis=1),
            'separator_temperature_in_Tg': (b[:, 3] >= 300) & (b[:, 3] <= 650),
            'feedwater_temperature_in_liquid_grid': (b[:, 4] >= 295) & (b[:, 4] <= 355),
            'separator_pressure_in_P_grid': (b[:, 2] >= 8) & (b[:, 2] <= 30),
            'outlet_pressure_in_P_grid': (b[:, 5] >= 8) & (b[:, 5] <= 30),
            'pressure_order': b[:, 2] >= b[:, 5], 'positive_steam_flow': b[:, 0] > 0,
            'nonnegative_recorded_coal': b[:, 1] >= 0,
            'positions_in_unit_interval': np.all((pos >= 0) & (pos <= 1), axis=1)}
        strict = basic.copy()
        for check in checks.values(): strict &= check
        arrays[f'{side}_basic_row_eligible'] = basic
        arrays[f'{side}_property_input_row_eligible'] = strict
        eligibility[side] = {'basic_eligible_rows': int(basic.sum()), 'property_input_eligible_rows': int(strict.sum()),
                             'failed_or_missing_rows_per_check': {key: int((~value).sum()) for key, value in checks.items()}, 'windows': {}}
        for split, lo, hi in [('train', 0, train_stop), ('prototype_validation', train_stop, len(timestamps))]:
            for label, valid in [('basic', basic), ('property_input', strict)]:
                key = f'{side}_{split}_{label}_window_starts'
                arrays[key] = window_starts(len(timestamps), lo, hi, valid)
                eligibility[side]['windows'][f'{split}_{label}'] = len(arrays[key])
    manifest = {
        'schema_version': 1, 'status': 'PREPARED_OBSERVATIONAL_INPUTS_NOT_TRAINED',
        'dataset_id': 'thermal_plant_contiguous_tenth_v1_20260909',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'canonical_path': str(canonical),
        'canonical_sha256': EXPECTED_CANONICAL_SHA, 'canonical_timestamp_prefix_read': timestamp_read,
        'canonical_split_metadata_prefix_read': split_read,
        'old_training_rows': OLD_TRAIN_ROWS, 'selected_rows': SELECTED_ROWS, 'selection': 'first floor(old_training_rows/10) continuous gridpoints',
        'old_validation_measurement_rows_read': 0, 'old_test_measurement_rows_read': 0,
        'selected_source_path': str(source), 'source_prefix_audit': source_audit,
        'source_full_file_sha256_verified': False, 'dt_seconds': 10, 'side_order': ['A_left', 'B_right'],
        'first_epoch_s': int(timestamps[0]), 'last_epoch_s': int(timestamps[-1]),
        'encoded_first_time': datetime.fromtimestamp(int(timestamps[0]), timezone.utc).isoformat(),
        'encoded_last_time': datetime.fromtimestamp(int(timestamps[-1]), timezone.utc).isoformat(),
        'actual_DCS_timezone_and_arrival_times_certified': False, 'upstream_resampling_causality_certified': False,
        'local_reconstruction': 'per-column causal last finite exported cell; no right interpolation, no clipping, no zero imputation',
        'mask_age_semantics': 'nonempty exported cell event time, not actual sensor refresh or network arrival',
        'split_ranges': {'train': [0, train_stop], 'prototype_validation': [train_stop, len(timestamps)]},
        'window_context': 64, 'window_future': 128, 'window_span': 192, 'input_age_limit_s': 30,
        'command_semantics': 'UNCONFIRMED; command candidates excluded from observational input',
        'future_conditioning': 'recorded actual positions and boundaries; not executable commands or known future forecasts',
        'spray_flow_total_policy': 'archive only; downstream exclude from all history masks and use declared zero future placeholder',
        'coal_command_slot_semantics': 'recorded uncorrected total coal, executable command unconfirmed',
        'mapping': mapping, 'stats': stats, 'eligibility': eligibility,
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'array_sha256': array_sha256(value)} for key, value in arrays.items()},
        'script_sha256': sha256(Path(__file__)),
        'limitations': ['No model inference, fit or control performed.', 'Contiguous early operating block is not representative sampling or cross-device validation.',
                        'Property input checks do not certify composed queries, phase validity, anchors or plant safety.',
                        'Prototype validation is inside the historical training block, not an independent locked test.']}
    output.mkdir(parents=True)
    np.savez_compressed(output / 'plant_tenth.npz', **arrays)
    manifest['output_npz_sha256'] = sha256(output / 'plant_tenth.npz')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    public_receipt = {
        'schema_version': 1, 'dataset_id': manifest['dataset_id'], 'status': manifest['status'],
        'canonical_sha256': EXPECTED_CANONICAL_SHA, 'raw_header_sha256': source_audit['header_sha256'],
        'raw_selected_prefix_sha256': source_audit['prefix_sha256'],
        'selected_timestamps_array_sha256': array_sha256(timestamps),
        'raw_selected_records_read': source_audit['data_records_read'], 'selected_grid_rows': SELECTED_ROWS, 'old_training_rows': OLD_TRAIN_ROWS,
        'old_validation_measurement_rows_read': 0, 'old_test_measurement_rows_read': 0,
        'split_ranges': manifest['split_ranges'], 'dt_seconds': 10,
        'window_context': 64, 'window_future': 128, 'script_sha256': manifest['script_sha256'],
        'array_hash_algorithm': 'sha256(json.dumps(dtype.str+shape,sort_keys=True,separators=(comma,colon)).ASCII + newline + C-order bytes)',
        'array_hashes': {key: value['array_sha256'] for key, value in manifest['arrays'].items()},
        'public_payload_scope': 'Hashes, counts, software identities only; no raw measurements, statistics, timestamps, tag names or machine paths.',
        'command_semantics': 'UNCONFIRMED', 'actual_DCS_timezone_and_arrival_times_certified': False,
        'upstream_resampling_causality_certified': False,
    }
    (output / 'public_receipt.json').write_text(json.dumps(public_receipt, indent=2, allow_nan=False), encoding='utf-8')
    return {'output': str(output), 'selected_rows': SELECTED_ROWS, 'npz_sha256': manifest['output_npz_sha256'], 'eligibility': eligibility}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--source', '--raw', dest='source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.canonical, args.source, args.output), ensure_ascii=False, indent=2))
