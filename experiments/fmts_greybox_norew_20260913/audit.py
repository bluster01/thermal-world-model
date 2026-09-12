"""Bound supplement provenance, then reuse unchanged v0.2 numerical replay."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from experiments.fmts_mainsteam_20260911.audit import audit as parent_audit
from experiments.fmts_mainsteam_20260911.manifests import sha256
from src.final_wm.contracts import FinalWMProtocolError
from .spec import ARM, specs, serialized_spec
from .run import ROOT


def audit(out,parent,record=None,properties=None,mapping='configs/final_wm/channel_mapping_v2.json',device='cpu'):
    out,parent=Path(out),Path(parent)
    identity=json.loads((out/'identity.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    if identity['protocol']!=serialized_spec():
        raise FinalWMProtocolError('supplement protocol mismatch')
    if identity['frozen_spec_sha256']!=sha256(Path(__file__).with_name('frozen_spec.json')):
        raise FinalWMProtocolError('frozen supplement snapshot changed')
    if identity['registration_sha256']!=sha256(ROOT/'docs/fmts2026/PREREG_GREYBOX_NOREW_20260913.md'):
        raise FinalWMProtocolError('supplement registration changed')
    for field,name in [('parent_identity_sha256','identity.json'),('parent_summary_sha256','summary.json')]:
        if identity[field]!=sha256(parent/name):
            raise FinalWMProtocolError('parent identity/summary changed')
    if sha256(out/'indices.npz')!=sha256(parent/'indices.npz'):
        raise FinalWMProtocolError('parent windows changed')
    for source,digest in identity['supplement_sources'].items():
        if sha256(ROOT/source)!=digest:
            raise FinalWMProtocolError('supplement code changed')
    expected={(ARM,s) for s in ((0,) if identity['smoke'] else (0,1,2))}
    if {(r['arm'],r['seed']) for r in summary['runs']}!=expected:
        raise FinalWMProtocolError('unexpected supplement arm/seed set')
    lookup={s.seed:asdict(s) for s in specs()}
    for r in summary['runs']:
        if r['status']!='complete':
            continue
        payload=torch.load(out/f"{ARM}_seed{r['seed']}"/'best.pt',map_location='cpu',weights_only=False)
        if payload['spec']!=lookup[r['seed']] or payload['protocol']!=serialized_spec():
            raise FinalWMProtocolError('checkpoint scientific spec changed')
    result=parent_audit(out,record,properties,mapping,device)
    result['complete']=result['runs_checked']==len(expected) and not result['failed_runs']
    result['protocol_id']='FMTS-GNR1'
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--parent',default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--record')
    ap.add_argument('--properties')
    ap.add_argument('--mapping',default='configs/final_wm/channel_mapping_v2.json')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--save',help='new audit JSON path; never overwrite')
    a=ap.parse_args()
    result=audit(a.out,a.parent,a.record,a.properties,a.mapping,a.device)
    if a.save:
        with Path(a.save).open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['complete']:
        raise SystemExit(1)
