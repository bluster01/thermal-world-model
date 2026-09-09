"""Read-only Linux readiness check for GitHub-mediated prototype handoff.

Creates local details plus an allowlisted public receipt. Does not read plant
arrays, start training, install packages, or contact GitHub.
"""
import argparse
import hashlib
import importlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TASK_ID = 'thermal_world_model_tenth_preflight_v1'
EXPECTED_CANONICAL = '24da77960e05e3636cc7b97a60a75e9b4ba470a3abb4f8ba920ddf11c6dad1d0'
EXPECTED_PROPERTIES = '9fd7a1dba96a5b968661f644fa185ac755266c85853d689d065432a2d41f6e92'


def file_identity(path, expected=None):
    if path is None:
        return {'provided': False, 'exists': False, 'expected_sha256': expected, 'matches': False}
    path = Path(path).resolve()
    result = {'provided': True, 'path': str(path), 'exists': path.is_file(), 'expected_sha256': expected}
    if result['exists']:
        result['bytes'] = path.stat().st_size
        if expected is not None:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            result['sha256'] = digest.hexdigest()
            result['matches'] = result['sha256'] == expected
        else:
            # Raw CSV identity/preparation belongs to the bounded data task.
            result['matches'] = None
    else:
        result['matches'] = False
    return result


def dependency_versions():
    versions = {}
    for name in ('numpy', 'torch', 'pandas', 'pytest'):
        try:
            module = importlib.import_module(name)
            versions[name] = {'available': True, 'version': str(module.__version__)}
            if name == 'torch':
                versions[name]['cuda_available'] = bool(module.cuda.is_available())
                versions[name]['cuda_device_count'] = int(module.cuda.device_count())
                versions[name]['cuda_runtime'] = module.version.cuda
        except Exception as error:
            # Type only: dependency exception messages can contain private paths.
            versions[name] = {'available': False, 'error_type': type(error).__name__}
    return versions


def public_receipt(details):
    """An explicit allowlist: never publish file paths or machine/user identity."""
    return {
        'task_id': details['task_id'], 'task_stage': 'preflight',
        'recorded_utc': details['recorded_utc'], 'source_sha256': details['source_sha256'],
        'platform': details['platform'], 'python_version': details['python_version'],
        'dependencies': {name: {key: record[key] for key in
                         ('available', 'version', 'error_type', 'cuda_available', 'cuda_device_count', 'cuda_runtime')
                         if key in record}
                         for name, record in details['dependencies'].items()
                         if name in ('numpy', 'torch', 'pandas', 'pytest')},
        'assets': {name: {key: record.get(key) for key in ('provided', 'exists', 'expected_sha256', 'sha256', 'matches')}
                   for name, record in details['assets'].items()},
        'prerequisites_present': details['prerequisites_present'],
        'training_started': False, 'plant_arrays_loaded': False,
        'raw_csv_values_read': False, 'remote_training_completion': False,
        'next': 'Wait for the exact frozen training task; this receipt alone does not start a fit.',
    }


def inspect(canonical, raw, properties, output_parent):
    assets = {'canonical': file_identity(canonical, EXPECTED_CANONICAL),
              'raw_merged': file_identity(raw),
              'properties': file_identity(properties, EXPECTED_PROPERTIES)}
    dependencies = dependency_versions()
    result = {
        'task_id': TASK_ID, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'platform': platform.system(), 'python_version': platform.python_version(),
        'python_executable': sys.executable, 'dependencies': dependencies, 'assets': assets,
        'free_output_bytes': shutil.disk_usage(output_parent).free,
        'scope': 'File byte identity and runtime availability only; no plant arrays or raw CSV values read.',
    }
    result['prerequisites_present'] = (
        result['platform'] == 'Linux' and all(record['available'] for record in dependencies.values())
        and assets['canonical']['matches'] is True and assets['properties']['matches'] is True
        and assets['raw_merged']['exists'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--properties', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; preserve previous receipts')
    parent = args.output.resolve().parent
    if not parent.is_dir():
        parser.error('Output parent must exist')
    details = inspect(args.canonical, args.raw, args.properties, parent)
    receipt = public_receipt(details)
    args.output.mkdir()
    for name, value in (('local_details.json', details), ('public_receipt.json', receipt)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if details['prerequisites_present'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
