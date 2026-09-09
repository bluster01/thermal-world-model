import hashlib
from pathlib import Path

from experiments.world_model_plant.preflight import file_identity, public_receipt


def test_missing_and_mismatched_files_are_not_ready(tmp_path):
    path = tmp_path / 'fixture'
    assert file_identity(None)['matches'] is False
    assert file_identity(path, 'f' * 64)['matches'] is False
    path.write_bytes(b'handwritten fixture')
    assert file_identity(path, 'f' * 64)['matches'] is False
    assert file_identity(path, hashlib.sha256(path.read_bytes()).hexdigest())['matches'] is True
    assert file_identity(path)['matches'] is None


def test_public_receipt_excludes_private_extra_fields_and_never_claims_training():
    private = '/private/plant/source.csv'
    details = dict(task_id='fixture', recorded_utc='fixture', source_sha256='f' * 64,
                   platform='Linux', python_version='fixture', dependencies={
                       'numpy': {'available': True, 'version': 'fixture', 'install_path': private},
                       'private_dependency': {'version': private}},
                   prerequisites_present=True, python_executable=private,
                   private_notes=private, assets={'raw': dict(path=private, exists=True, provided=True, secret=private)})
    receipt = public_receipt(details)
    assert private not in str(receipt)
    assert receipt['training_started'] is False
    assert receipt['plant_arrays_loaded'] is False
    assert receipt['raw_csv_values_read'] is False
    assert receipt['remote_training_completion'] is False


def test_raw_probe_does_not_read_csv_contents(tmp_path, monkeypatch):
    path = tmp_path / 'raw.csv'
    path.write_text('private,raw,values\n', encoding='utf-8')
    def forbidden(*args, **kwargs):
        raise AssertionError('raw contents must not be read')
    monkeypatch.setattr(Path, 'open', forbidden)
    assert file_identity(path)['exists']
