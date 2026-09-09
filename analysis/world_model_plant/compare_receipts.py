"""Compare allowlisted public data receipts, without loading plant measurements.

Only pure LF/CRLF conversions of the exact pinned preparer source are accepted.
Container NPZ byte hashes and raw cross-host script-hash equality are not data
identity criteria. CLI failure reports never echo input values or input paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


CRITICAL_FIELDS = (
    'schema_version', 'dataset_id', 'status', 'canonical_sha256', 'raw_header_sha256',
    'raw_selected_prefix_sha256', 'selected_timestamps_array_sha256',
    'raw_selected_records_read', 'selected_grid_rows', 'old_training_rows',
    'old_validation_measurement_rows_read', 'old_test_measurement_rows_read',
    'split_ranges', 'dt_seconds', 'window_context', 'window_future',
    'array_hash_algorithm', 'array_hashes', 'command_semantics',
    'actual_DCS_timezone_and_arrival_times_certified',
    'upstream_resampling_causality_certified',
)
RECEIPT_FIELDS = set(CRITICAL_FIELDS) | {'script_sha256', 'public_payload_scope'}
BASE_ARRAYS = {
    'timestamps_epoch_s', 'raw_values', 'raw_exported_record_mask', 'raw_present_mask',
    'raw_age_s', 'raw_source_timestamp_epoch_s', 'prototype_split',
}
ARRAY_NAMES = BASE_ARRAYS | {
    group + suffix
    for group in ('observations', 'positions', 'boundary', 'command_candidates_unconfirmed')
    for suffix in ('', '_available_mask', '_present_mask', '_age_s', '_source_timestamp_epoch_s')
} | {
    f'{side}_{quality}_row_eligible' for side in ('A', 'B') for quality in ('basic', 'property_input')
} | {
    f'{side}_{split}_{quality}_window_starts' for side in ('A', 'B')
    for split in ('train', 'prototype_validation') for quality in ('basic', 'property_input')
}
DIGEST_FIELDS = ('canonical_sha256', 'raw_header_sha256', 'raw_selected_prefix_sha256', 'selected_timestamps_array_sha256')
IDENTITY_FIELDS = {'schema_version', 'identity_kind', 'expected_receipt', 'preparer_identity'}
SOURCE_FIELDS = {'lf_normalized_sha256', 'accepted_raw_sha256', 'reference_receipt_raw_sha256', 'normalization_rule'}
NORMALIZATION_RULE = 'Accept exactly canonical UTF-8 source with all LF or all CRLF line endings; no other byte changes.'
DATASET_ID = 'thermal_plant_contiguous_tenth_v1_20260909'
PUBLIC_SCOPE = 'Hashes, counts, software identities only; no raw measurements, statistics, timestamps, tag names or machine paths.'


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def strict_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def source_identity(source: bytes, receipt_script_sha: str) -> dict:
    source.decode('utf-8', errors='strict')
    lf = source.replace(b'\r\n', b'\n')
    if b'\r' in lf:
        raise ValueError('Unsupported bare CR in preparer source')
    hashes = {'lf': hashlib.sha256(lf).hexdigest(), 'crlf': hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest()}
    if receipt_script_sha not in set(hashes.values()):
        raise ValueError('Reference receipt does not match either exact source form')
    return {'lf_normalized_sha256': hashes['lf'], 'accepted_raw_sha256': hashes,
            'reference_receipt_raw_sha256': receipt_script_sha, 'normalization_rule': NORMALIZATION_RULE}


def _validate_expected(identity) -> bool:
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        return False
    if type(identity['schema_version']) is not int or identity['schema_version'] != 1 or identity['identity_kind'] != 'EXPECTED_PRIVATE_DATA_IDENTITY_ONLY':
        return False
    expected = identity['expected_receipt']
    source = identity['preparer_identity']
    if not isinstance(expected, dict) or set(expected) != set(CRITICAL_FIELDS):
        return False
    if (expected['dataset_id'] != DATASET_ID or expected['status'] != 'PREPARED_OBSERVATIONAL_INPUTS_NOT_TRAINED'
            or type(expected['schema_version']) is not int or expected['schema_version'] != 1):
        return False
    if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
        return False
    accepted = source['accepted_raw_sha256']
    if not isinstance(accepted, dict) or set(accepted) != {'lf', 'crlf'} or not all(_digest(v) for v in accepted.values()):
        return False
    if source['lf_normalized_sha256'] != accepted['lf'] or source['reference_receipt_raw_sha256'] not in accepted.values() or source['normalization_rule'] != NORMALIZATION_RULE:
        return False
    arrays = expected['array_hashes']
    if not isinstance(arrays, dict) or set(arrays) != ARRAY_NAMES or not all(_digest(v) for v in arrays.values()):
        return False
    if not all(_digest(expected[key]) for key in DIGEST_FIELDS):
        return False
    return (expected['command_semantics'] == 'UNCONFIRMED'
            and expected['actual_DCS_timezone_and_arrival_times_certified'] is False
            and expected['upstream_resampling_causality_certified'] is False)


def compare(receipt, identity, current_preparer_source: bytes) -> dict:
    errors = []
    report = {'schema_version': 1, 'status': 'DATA_RECEIPT_MISMATCH',
              'comparison_scope': 'Published receipt and pinned preparer identities; no plant arrays loaded.',
              'plant_arrays_loaded': False, 'optimizer_started': False,
              'npz_container_sha_used_as_cross_host_criterion': False,
              'mismatch_codes': errors}
    if not _validate_expected(identity):
        errors.append('expected_identity_schema')
        return report
    expected = identity['expected_receipt']
    source = identity['preparer_identity']
    accepted = set(source['accepted_raw_sha256'].values())
    current_raw = hashlib.sha256(current_preparer_source).hexdigest()
    current_lf = hashlib.sha256(current_preparer_source.replace(b'\r\n', b'\n')).hexdigest()
    report['reference_receipt_preparer_raw_sha256'] = source['reference_receipt_raw_sha256']
    report['current_checkout_preparer_raw_sha256'] = current_raw
    report['expected_preparer_lf_normalized_sha256'] = source['lf_normalized_sha256']
    if current_raw not in accepted or current_lf != source['lf_normalized_sha256']:
        errors.append('current_preparer_source_identity')
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        errors.append('receipt_top_level_fields')
        return report
    receipt_raw = receipt['script_sha256']
    if receipt['public_payload_scope'] != PUBLIC_SCOPE:
        errors.append('receipt_public_scope')
    if _digest(receipt_raw):
        report['observed_receipt_preparer_raw_sha256'] = receipt_raw
    if not _digest(receipt_raw) or receipt_raw not in accepted:
        errors.append('receipt_preparer_source_identity')
    for field in CRITICAL_FIELDS:
        if field == 'array_hashes':
            arrays = receipt[field]
            if not isinstance(arrays, dict) or set(arrays) != ARRAY_NAMES:
                errors.append('array_hash_inventory')
            else:
                for name in sorted(ARRAY_NAMES):
                    if not strict_equal(arrays[name], expected[field][name]):
                        errors.append('array_hash.' + name)
        elif not strict_equal(receipt[field], expected[field]):
            # Field names are from our fixed allowlist; untrusted keys/values never enter the output.
            errors.append('receipt_field.' + field)
    if not errors:
        report['status'] = 'DATA_RECEIPT_MATCH'
        report['matched_array_count'] = len(ARRAY_NAMES)
        report['dataset_id'] = DATASET_ID
        report['source_line_endings_compatible'] = True
    return report


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def read_json(path: Path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs, parse_constant=reject_constant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(read_json(args.receipt), read_json(args.expected), Path(__file__).with_name('prepare_tenth.py').read_bytes())
    except (OSError, ValueError, TypeError, UnicodeError):
        report = {'schema_version': 1, 'status': 'DATA_RECEIPT_MISMATCH',
                  'mismatch_codes': ['input_read_or_parse_failure'], 'plant_arrays_loaded': False, 'optimizer_started': False}
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    try:
        with args.output.open('x', encoding='utf-8') as f:
            f.write(content)
    except OSError:
        # Deliberately omit Python exception text, which could expose private paths.
        print(json.dumps({'schema_version': 1, 'status': 'COMPARISON_OUTPUT_FAILED', 'mismatch_codes': ['output_not_new_or_not_writable']}))
        return 2
    print(content, end='')
    return 0 if report['status'] == 'DATA_RECEIPT_MATCH' else 1


if __name__ == '__main__':
    raise SystemExit(main())
