"""CORE1 saved-artifact audit, optionally full inference replay; never training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.fmts_m7fusion_20260913.audit import close_tree
from experiments.fmts_mainsteam_20260911.data import RichRecord
from experiments.fmts_greybox_norew_20260913.audit import audit as audit_gnr
from src.final_wm.properties import load_grid_properties
from .run import ROOT,PACKAGE,REG,VARIANTS,read,write,npz,sha256,load_source,CoreView,collect,replay_check,metrics,contrasts,aggregate


def audit(out,parent,gnr,record_path=None,properties=None,mapping='configs/final_wm/channel_mapping_v2.json',device='cpu'):
    out,parent,gnr=Path(out),Path(parent),Path(gnr)
    identity=read(out/'identity.json');summary=read(out/'summary.json')
    assert not (out/'failure.json').exists(),'failed execution preserved'
    assert identity['protocol']==read(PACKAGE/'protocol.json')
    assert identity['protocol_sha256']==sha256(PACKAGE/'protocol.json')
    assert identity['registration_sha256']==sha256(REG)
    audit_gnr(gnr,parent)
    for label,folder in [('parent',parent),('gnr',gnr)]:
        for filename in ['identity','summary']:
            assert sha256(folder/f'{filename}.json')==identity[f'{label}_{filename}_sha256']
    pid=read(parent/'identity.json')
    for k in ['record','mapping','properties','indices']:assert identity[k]==pid[k]
    for name,digest in {**pid['sources'],**identity['sources']}.items():
        relative='experiments/fmts_mainsteam_20260911/'+Path(name).name if name.startswith('/') else name
        assert sha256(ROOT/relative)==digest,f'source changed: {name}'
    assert sha256(out/'indices.npz')==identity['indices']==sha256(parent/'indices.npz')
    seeds=[0] if identity['smoke'] else [0,1,2]
    assert summary['seeds']==identity['seeds']==seeds
    expected={(variant,seed) for seed in seeds for variant in [*VARIANTS,'GNR']}
    assert len(summary['runs'])==len(expected)
    assert {(r['variant'],r['seed']) for r in summary['runs']}==expected
    assert summary['training_updates']==0 and not summary['locked_test_evaluated']
    assert identity['training_updates']==0 and identity['locked_test_evaluated'] is False
    record=None
    if record_path:
        assert sha256(record_path)==identity['record'] and sha256(mapping)==identity['mapping']
        assert identity['smoke'] or (properties and sha256(properties)==identity['properties'])
        record=RichRecord(record_path,mapping)
    indices=npz(out/'indices.npz');replayed=traces=0
    for report in summary['runs']:
        assert report['status']=='complete'
        variant,seed=report['variant'],report['seed']
        folder=out/f'{variant}_seed{seed}'
        assert read(folder/'report.json')==report
        source_folder=(gnr/f'greybox_steady_none_norew_seed{seed}' if variant=='GNR'
                       else parent/f'fusion_gru_norew_seed{seed}')
        label='GNR' if variant=='GNR' else 'GRU'
        assert sha256(source_folder/'best.pt')==report['source_checkpoint_sha256']==identity['checkpoint_hashes'][f'{label}_seed{seed}']
        raw={}
        for name in ['prediction','response','trace']:
            assert sha256(folder/f'{name}.npz')==report[name+'_sha256']
            raw[name]=npz(folder/f'{name}.npz')
            bank='validation' if name=='prediction' else 'response'
            assert np.array_equal(raw[name]['starts'],indices[bank])
            assert all(np.isfinite(a).all() for a in raw[name].values())
        recomputed=metrics(raw['prediction'],raw['response'])
        assert close_tree(recomputed,{k:report[k] for k in recomputed})
        if variant in ('C11','GNR'):
            checks={name:replay_check(npz(source_folder/f'{name}.npz'),raw[name]) for name in ['prediction','response']}
            assert close_tree(checks,report['reference_replay'])
        if record is not None:
            source=load_source(source_folder,load_grid_properties(properties) if properties else None,device)
            view=CoreView(source,'C00' if variant=='GNR' else variant)
            for name,bank in [('prediction','validation'),('response','response')]:
                actual,trace=collect(view,record,indices[bank],device,name=='response')
                replay_check(raw[name],actual);replayed+=1
                if name=='response':replay_check(raw['trace'],trace);traces+=1
    assert close_tree(contrasts(out),read(out/'contrasts.json'))
    assert close_tree(aggregate(summary['runs']),summary['seed_summary'])
    return dict(protocol_id='FMTS-CORE1',runs_checked=len(expected),
        array_sets_replayed=replayed,trace_sets_replayed=traces,artifacts_complete=True,
        complete=replayed==2*len(expected) and traces==len(expected),
        locked_test_evaluated=False,training_updates=0,scientific_verdict='AUTHOR_REVIEW_PENDING_NO_PLANT_TRUTH')


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--parent',default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--gnr',default='results/fmts_greybox_norew_20260913/linux_full_gnr1')
    ap.add_argument('--record');ap.add_argument('--properties');ap.add_argument('--device',default='cpu')
    ap.add_argument('--mapping',default='configs/final_wm/channel_mapping_v2.json');ap.add_argument('--save')
    a=ap.parse_args();result=audit(a.out,a.parent,a.gnr,a.record,a.properties,a.mapping,a.device)
    if a.save:
        assert not Path(a.save).exists(),'audit save exists'
        write(a.save,result)
    print(result)
    if not result['complete']:raise SystemExit(1)
