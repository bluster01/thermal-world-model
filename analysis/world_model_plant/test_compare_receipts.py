"""Handwritten receipt/source fixtures: no raw plant data or preparation runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from analysis.world_model_plant.compare_receipts import (
    ARRAY_NAMES, CRITICAL_FIELDS, DATASET_ID, PUBLIC_SCOPE, compare, main, read_json, source_identity,
)


SOURCE_LF = b'# toy preparer fixture\nVALUE = 1\n'


def fixture():
    digest = 'a' * 64
    receipt = {
        'schema_version': 1, 'dataset_id': DATASET_ID,
        'status': 'PREPARED_OBSERVATIONAL_INPUTS_NOT_TRAINED',
        'canonical_sha256': digest, 'raw_header_sha256': digest,
        'raw_selected_prefix_sha256': digest, 'selected_timestamps_array_sha256': digest,
        'raw_selected_records_read': 53071, 'selected_grid_rows': 53077,
        'old_training_rows': 530779, 'old_validation_measurement_rows_read': 0,
        'old_test_measurement_rows_read': 0,
        'split_ranges': {'train': [0, 42461], 'prototype_validation': [42461, 53077]},
        'dt_seconds': 10, 'window_context': 64, 'window_future': 128,
        'script_sha256': hashlib.sha256(SOURCE_LF).hexdigest(),
        'array_hash_algorithm': 'authored fixture algorithm',
        'array_hashes': {name: digest for name in ARRAY_NAMES},
        'public_payload_scope': PUBLIC_SCOPE,
        'command_semantics': 'UNCONFIRMED',
        'actual_DCS_timezone_and_arrival_times_certified': False,
        'upstream_resampling_causality_certified': False,
    }
    identity = {'schema_version': 1, 'identity_kind': 'EXPECTED_PRIVATE_DATA_IDENTITY_ONLY',
                'expected_receipt': {key: deepcopy(receipt[key]) for key in CRITICAL_FIELDS},
                'preparer_identity': source_identity(SOURCE_LF, receipt['script_sha256'])}
    return receipt, identity


@pytest.mark.parametrize('receipt_ending,current_ending', [('lf', 'lf'), ('lf', 'crlf'), ('crlf', 'lf'), ('crlf', 'crlf')])
def test_only_line_ending_changes_match(receipt_ending, current_ending):
    receipt, identity = fixture()
    forms = {'lf': SOURCE_LF, 'crlf': SOURCE_LF.replace(b'\n', b'\r\n')}
    receipt['script_sha256'] = hashlib.sha256(forms[receipt_ending]).hexdigest()
    report = compare(receipt, identity, forms[current_ending])
    assert report['status'] == 'DATA_RECEIPT_MATCH'
    assert report['matched_array_count'] == 39
    assert report['observed_receipt_preparer_raw_sha256'] == receipt['script_sha256']
    assert not report['npz_container_sha_used_as_cross_host_criterion']


def test_preparer_change_or_mixed_line_endings_rejected():
    receipt, identity = fixture()
    for source in [SOURCE_LF.replace(b'VALUE = 1', b'VALUE = 2'), SOURCE_LF.replace(b'\n', b'\r\n', 1)]:
        report = compare(receipt, identity, source)
        assert 'current_preparer_source_identity' in report['mismatch_codes']


def test_receipt_script_hash_not_trusted_by_itself():
    receipt, identity = fixture()
    receipt['script_sha256'] = 'e' * 64
    assert 'receipt_preparer_source_identity' in compare(receipt, identity, SOURCE_LF)['mismatch_codes']
    receipt['script_sha256'] = ['not', 'a', 'hash']
    assert 'receipt_preparer_source_identity' in compare(receipt, identity, SOURCE_LF)['mismatch_codes']


@pytest.mark.parametrize('field', [key for key in CRITICAL_FIELDS if key != 'array_hashes'])
def test_every_critical_field_checked_without_echoing_values(field):
    receipt, identity = fixture()
    receipt[field] = '/private/SECRET_TAG_VALUE'
    report = compare(receipt, identity, SOURCE_LF)
    assert 'receipt_field.' + field in report['mismatch_codes']
    assert 'SECRET_TAG_VALUE' not in json.dumps(report)


def test_numeric_bool_and_split_nesting_types_are_strict():
    receipt, identity = fixture()
    receipt['old_test_measurement_rows_read'] = False
    receipt['split_ranges']['train'][0] = False
    report = compare(receipt, identity, SOURCE_LF)
    assert 'receipt_field.old_test_measurement_rows_read' in report['mismatch_codes']
    assert 'receipt_field.split_ranges' in report['mismatch_codes']


def test_array_missing_extra_and_changed_fail():
    receipt, identity = fixture()
    receipt['array_hashes'].pop('raw_values')
    assert 'array_hash_inventory' in compare(receipt, identity, SOURCE_LF)['mismatch_codes']
    receipt, identity = fixture()
    receipt['array_hashes']['PRIVATE_SOURCE_PATH'] = 'x'
    result = compare(receipt, identity, SOURCE_LF)
    assert 'array_hash_inventory' in result['mismatch_codes']
    assert 'PRIVATE_SOURCE_PATH' not in json.dumps(result)
    receipt, identity = fixture()
    receipt['array_hashes']['positions'] = 'b' * 64
    assert 'array_hash.positions' in compare(receipt, identity, SOURCE_LF)['mismatch_codes']


def test_no_unknown_fields_or_private_values_echoed():
    receipt, identity = fixture()
    receipt['/private/path'] = [12345, 'SECRET']
    result = compare(receipt, identity, SOURCE_LF)
    assert result['mismatch_codes'] == ['receipt_top_level_fields']
    assert '/private' not in json.dumps(result) and 'SECRET' not in json.dumps(result)
    receipt, identity = fixture()
    receipt['public_payload_scope'] = '/private/SECRET'
    result = compare(receipt, identity, SOURCE_LF)
    assert 'receipt_public_scope' in result['mismatch_codes']
    assert '/private' not in json.dumps(result) and 'SECRET' not in json.dumps(result)


def test_expected_must_have_full_allowlist_and_all_arrays():
    receipt, identity = fixture()
    identity['expected_receipt']['array_hashes'].pop('raw_values')
    assert compare(receipt, identity, SOURCE_LF)['mismatch_codes'] == ['expected_identity_schema']
    receipt, identity = fixture()
    identity['expected_receipt']['actual_DCS_timezone_and_arrival_times_certified'] = True
    assert compare(receipt, identity, SOURCE_LF)['mismatch_codes'] == ['expected_identity_schema']


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    path = tmp_path / 'bad.json'
    for content in ['{"status": "A", "status": "B"}', '{"x": NaN}']:
        path.write_text(content, encoding='utf-8')
        with pytest.raises(ValueError): read_json(path)


def test_cli_match_new_output_then_refuses_overwrite(tmp_path, monkeypatch, capsys):
    receipt, identity = fixture()
    source = Path(__file__).with_name('prepare_tenth.py').read_bytes()
    receipt['script_sha256'] = hashlib.sha256(source).hexdigest()
    identity['preparer_identity'] = source_identity(source, receipt['script_sha256'])
    rp, ip, out = tmp_path / 'receipt.json', tmp_path / 'identity.json', tmp_path / 'out.json'
    rp.write_text(json.dumps(receipt)); ip.write_text(json.dumps(identity))
    monkeypatch.setattr(sys, 'argv', ['compare', '--receipt', str(rp), '--expected', str(ip), '--output', str(out)])
    assert main() == 0
    assert json.loads(out.read_text())['status'] == 'DATA_RECEIPT_MATCH'
    original = out.read_bytes()
    assert main() == 2
    assert out.read_bytes() == original
    assert str(tmp_path) not in capsys.readouterr().out


def test_cli_parse_failure_has_no_private_path_or_content(tmp_path, monkeypatch, capsys):
    bad, out = tmp_path / 'PRIVATE_FILENAME.json', tmp_path / 'out.json'
    bad.write_text('{PRIVATE_SECRET_VALUE')
    monkeypatch.setattr(sys, 'argv', ['compare', '--receipt', str(bad), '--expected', str(bad), '--output', str(out)])
    assert main() == 1
    text = capsys.readouterr().out + out.read_text()
    assert 'PRIVATE' not in text and str(tmp_path) not in text
    assert json.loads(out.read_text())['mismatch_codes'] == ['input_read_or_parse_failure']
