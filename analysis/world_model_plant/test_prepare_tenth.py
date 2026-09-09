"""Small authored fixtures only: no real data or model calls."""
import csv
from pathlib import Path

import numpy as np
import pytest

from analysis.world_model_plant.prepare_tenth import (
    NAT, array_sha256, bounded_npy_prefix, causal_hold, mappings, parse_timestamp,
    read_exact_csv_prefix, window_starts,
)


def test_prefix_npy_reads_only_requested_rows(tmp_path):
    path = tmp_path / 'toy.npz'
    data = np.array([[1., 2.], [3., 4.], [999999., 999999.]])
    np.savez_compressed(path, data=data)
    actual, meta = bounded_npy_prefix(path, 'data.npy', 2)
    np.testing.assert_array_equal(actual, [[1., 2.], [3., 4.]])
    assert meta['decoded_rows'] == 2 and meta['decoded_numeric_bytes'] == 32
    assert meta['shape'] == [3, 2]
    with pytest.raises(ValueError): bounded_npy_prefix(path, 'data.npy', 4)


def test_object_and_fortran_rejected(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, obj=np.array([{'secret': 1}], dtype=object), fort=np.asfortranarray(np.ones((3, 2))))
    with pytest.raises(ValueError): bounded_npy_prefix(path, 'obj.npy', 1)
    with pytest.raises(ValueError): bounded_npy_prefix(path, 'fort.npy', 1)


def test_csv_no_read_of_sentinel_future_record(tmp_path):
    path = tmp_path / 'source.csv'
    allowed = b'time,x\n1970-01-01 00:00:00+00:00,2\n1970-01-01 00:00:10+00:00,\n'
    path.write_bytes(allowed + b'LOCKED_DO_NOT_PARSE,999999999\n')
    values, audit, records = read_exact_csv_prefix(path, np.array([0, 10]), ['x'])
    assert values[0, 0] == 2 and np.isnan(values[1, 0])
    assert audit['bytes_consumed_including_header'] == len(allowed)
    assert audit['data_records_read'] == 2
    assert audit['numeric_records_after_selected_prefix_read'] == 0


@pytest.mark.parametrize('bad', [b'1970-01-01 00:00:20+00:00,2\n', b'1970-01-01 00:00:00+00:00,2,3\n'])
def test_csv_rejects_time_or_shape_mismatch(tmp_path, bad):
    path = tmp_path / 'bad.csv'
    path.write_bytes(b'time,x\n' + bad)
    with pytest.raises(ValueError): read_exact_csv_prefix(path, np.array([0]), ['x'])


def test_csv_non_numeric_is_missing_not_zero(tmp_path):
    path = tmp_path / 'source.csv'
    path.write_bytes(b'time,x\n1970-01-01 00:00:00+00:00,BAD\n')
    data, audit, records = read_exact_csv_prefix(path, np.array([0]), ['x'])
    assert np.isnan(data[0, 0]) and audit['non_numeric_nonempty_counts'] == [1]


def test_causal_hold_never_future_fills_and_tracks_source():
    raw = np.array([[np.nan, 5], [2, np.nan], [np.nan, np.nan], [99, 6]], dtype=float)
    held, present, available, age, sources = causal_hold(raw, np.array([0, 10, 20, 30]))
    assert np.isnan(held[0, 0]) and not available[0, 0]
    assert np.isinf(age[0, 0]) and sources[0, 0] == NAT
    np.testing.assert_array_equal(held[1:3, 0], [2, 2])
    np.testing.assert_array_equal(age[1:3, 0], [0, 10])
    assert held[2, 1] == 5 and age[2, 1] == 20
    assert not present[2].any() and available[2].all()
    changed = raw.copy(); changed[3] = [1234, 9876]
    np.testing.assert_array_equal(causal_hold(changed, np.array([0, 10, 20, 30]))[0][:3], held[:3])


def test_missing_source_tick_becomes_missing_update_then_left_hold(tmp_path):
    path = tmp_path / 'gaps.csv'
    path.write_bytes(b'time,x\n1970-01-01 00:00:00+00:00,1\n1970-01-01 00:00:20+00:00,8\nDO_NOT_READ,999\n')
    ts = np.array([0, 10, 20])
    raw, audit, records = read_exact_csv_prefix(path, ts, ['x'])
    np.testing.assert_array_equal(records, [True, False, True])
    assert np.isnan(raw[1, 0]) and audit['data_records_read'] == 2
    held, present, available, age, source = causal_hold(raw, ts)
    np.testing.assert_array_equal(held[:, 0], [1, 1, 8])
    np.testing.assert_array_equal(age[:, 0], [0, 10, 0])


def test_missing_last_tick_reads_only_next_timestamp_not_numeric(tmp_path):
    path = tmp_path / 'gaps.csv'
    prefix = b'time,x\n1970-01-01 00:00:00+00:00,1\n'
    next_ts = b'1970-01-01 00:00:20+00:00,'
    path.write_bytes(prefix + next_ts + b'LOCKED_FIELD_NOT_READ\n')
    raw, audit, records = read_exact_csv_prefix(path, np.array([0, 10]), ['x'])
    assert audit['bytes_consumed_including_header'] == len(prefix + next_ts)
    assert audit['timestamp_lookahead_bytes'] == len(next_ts)
    assert audit['numeric_records_after_selected_prefix_read'] == 0
    assert np.isnan(raw[1, 0])


def test_windows_are_wholly_inside_disjoint_ranges():
    valid = np.ones(25, dtype=bool)
    train = window_starts(25, 0, 20, valid, context=3, horizon=2)
    val = window_starts(25, 20, 25, valid, context=3, horizon=2)
    np.testing.assert_array_equal(train, np.arange(16))
    np.testing.assert_array_equal(val, [20])
    assert train[-1] + 4 < val[0]
    valid[10] = False
    filtered = window_starts(25, 0, 20, valid, context=3, horizon=2)
    assert set(range(6, 11)).isdisjoint(filtered)
    assert len(window_starts(25, 20, 24, valid, context=3, horizon=2)) == 0


def test_actual_tenth_arithmetic_and_windows():
    n = 530779 // 10
    split = n * 4 // 5
    assert n == 53077 and split == 42461
    valid = np.ones(n, dtype=bool)
    assert len(window_starts(n, 0, split, valid)) == 42270
    assert len(window_starts(n, split, n, valid)) == 10425


def test_mapping_has_stage1_same_and_stage2_cross():
    m = mappings()
    assert '一级减温器A侧' in m['A']['positions'][0][1]
    assert '二级减温器B侧' in m['A']['positions'][1][1]
    assert '一级减温器B侧' in m['B']['positions'][0][1]
    assert '二级减温器A侧' in m['B']['positions'][1][1]
    assert m['A']['positions'][0][2] == .01
    assert m['A']['boundary'][0][2] == 1 / 3.6


def test_timestamp_offset_and_naive_convention():
    assert parse_timestamp('1970-01-01 08:00:00+08:00') == 0
    assert parse_timestamp('1970-01-01 00:00:00') == 0
    with pytest.raises(ValueError): parse_timestamp('1970-01-01 00:00:00.5')


def test_array_hash_is_independent_of_memory_layout_and_checks_dtype():
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert array_sha256(a) == array_sha256(np.asfortranarray(a))
    assert array_sha256(a) != array_sha256(a.astype(np.float64))
