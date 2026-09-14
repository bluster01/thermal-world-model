"""Metadata only: this module cannot load data or enable held-out scoring."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect_metadata(path):
    path=Path(path)
    if path.suffix.lower()!='.json':raise ValueError('JSON metadata only; no array/CSV access')
    raw=path.read_bytes();data=json.loads(raw)
    blockers=[]
    if data.get('history_variables')!=23 or not data.get('fmts_alignment_verified',False):
        blockers.append('FMTS_SCHEMA_NOT_CONFIRMED')
    if data.get('exposure_status')=='historically_exposed':
        blockers.append('HISTORICAL_EXPOSURE_CONFIRMED')
    elif data.get('exposure_status')!='certified_unused_for_development':
        blockers.append('EXPOSURE_NOT_CERTIFIED')
    if not data.get('final_checkpoint_manifest_sha256'):
        blockers.append('FINAL_MODELS_NOT_FROZEN')
    if not data.get('evaluation_window_manifest_sha256'):
        blockers.append('FMTS_WINDOWS_NOT_FROZEN')
    return dict(status='METADATA_ONLY_NOT_SCORING_AUTHORIZATION',metadata_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=data.get('source_sha256'),extension_start_epoch_s=data.get('partition_start_epoch_s',{}).get('extension'),
        declared_exposure_status=data.get('exposure_status','unverified'),blockers=blockers,
        scoring_enabled=False,ready_for_independent_scoring=False,
        old_test_cannot_be_declared_unseen=True,targets_read=False,
        note='Even satisfied declarations require source/exposure review and separately frozen scoring protocol.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--metadata',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();result=inspect_metadata(a.metadata)
    with Path(a.out).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
