"""FMTS-CORE1: fixed-checkpoint inference only; never train or select a model."""
import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch import nn

from experiments.fmts_mainsteam_20260911.data import RichRecord
from experiments.fmts_mainsteam_20260911.manifests import sha256
from experiments.fmts_mainsteam_20260911.models import RichFusion
from experiments.fmts_mainsteam_20260911.run import block_interval, day_mean
from experiments.fmts_greybox_norew_20260913.run import validate_parent
from experiments.fmts_greybox_norew_20260913.audit import audit as audit_gnr
from src.final_wm.contracts import action_support_from_history
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import TrainSpec

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
REG=ROOT/'docs/fmts2026/PREREG_CORE_ABLATION_20260914.md'
VARIANTS={'C00':(False,False),'C10':(True,False),'C01':(False,True),'C11':(True,True)}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def npz(p):
    with np.load(p,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


class CoreView(nn.Module):
    """Use identical physical parameters; toggle only observer/closure execution."""
    def __init__(self, source, variant):
        super().__init__()
        if variant not in VARIANTS: raise ValueError('unregistered variant')
        self.source=source.eval().requires_grad_(False)
        self.observer_on,self.closure_on=VARIANTS[variant]
        self.variant=variant
        assert source.base.config.transition.rewet_ablate
        assert source.base.config.transition.spray_total_mode=='action'

    def initial(self,h,ext):
        base=self.source.base
        anchor=base._steady_initial_state(h)
        if not self.observer_on:return anchor
        normalized=(ext-self.source.history_mean[-9:])/self.source.history_std[-9:]
        mu,_=base.observer.posterior(h.obs,h.actions,torch.cat([h.boundary,normalized],-1),anchor)
        return anchor+(mu-anchor)*base._correction_mask('hybrid',mu.device,mu.dtype)

    def rollout(self,initial,actions,boundary):
        closure=self.source.base.closure if self.closure_on else None
        return self.source.base.transition.integrate(initial,boundary,actions,closure=closure)

    @torch.no_grad()
    def forward(self,h,ext,actions,boundary):
        return self.rollout(self.initial(h,ext),actions,boundary)[1][:,:,4]


@torch.no_grad()
def collect(view,record,starts,device,response=False):
    """One history-derived initial state per paired counterfactual batch."""
    raw,trace={},{}
    def add(dest,key,value):dest.setdefault(key,[]).append(value.detach().cpu().numpy())
    for idx in torch.as_tensor(starts).split(32):
        h,ext,actions,boundary,target,days=record.batch(idx,1,device)
        initial=view.initial(h,ext)
        add(raw,'starts',idx);add(raw,'days',days)
        if not response:
            _,temp=view.rollout(initial,actions,boundary)
            add(raw,'prediction',temp[:,:,4]);add(raw,'target',target[:,:,4])
            add(raw,'persistence',h.obs[:,-1:,4].expand(-1,18))
            continue
        actions=h.actions[:,-1:].expand(-1,18,-1).clone()
        boundary=h.boundary[:,-1:].expand(-1,18,-1).clone()
        states,temp=view.rollout(initial,actions,boundary)
        assert torch.equal(temp,view.rollout(initial,actions.clone(),boundary.clone())[1]),'zero-action identity'
        add(raw,'base',temp[:,:,4])
        add(trace,'starts',idx);add(trace,'initial',initial);add(trace,'base_states',states)
        support=action_support_from_history(h.actions,margin=.05)
        for valve in (0,1):
            changed=actions.clone();changed[:,:,valve]=(changed[:,:,valve]+.05).clamp(max=1.)
            new_states,new_temp=view.rollout(initial,changed,boundary)
            name=f'valve{valve+1}'
            add(raw,name+'_prediction',new_temp[:,:,4])
            add(raw,name+'_support',support.contains(changed)&support.contains(actions))
            add(raw,name+'_dose',changed[:,0,valve]-actions[:,0,valve])
            add(trace,name+'_states',new_states)
        for label,a in [('base',actions),('valve1',actions.clone()),('valve2',actions.clone())]:
            if label!='base':
                j=int(label[-1])-1;a[:,:,j]=(a[:,:,j]+.05).clamp(max=1.)
            q=view.source.base.transition._spray_rates(boundary[:,0,2],a[:,0,0],a[:,0,1],boundary[:,0,6])
            add(trace,label+'_spray_target',torch.stack(q,-1))
    raw={k:np.concatenate(v) for k,v in raw.items()}
    trace={k:np.concatenate(v) for k,v in trace.items()}
    assert all(np.isfinite(x).all() for x in [*raw.values(),*trace.values()]),'nonfinite array'
    return raw,trace


def replay_check(saved,actual):
    assert set(saved)==set(actual),'replay keys'
    max_abs=0.
    for k,v in saved.items():
        assert v.shape==actual[k].shape,'replay shape'
        if v.dtype.kind in 'biu':assert np.array_equal(v,actual[k]),f'identity {k}'
        else:
            assert np.allclose(v,actual[k],rtol=1e-5,atol=1e-5),f'absolute replay {k}'
            max_abs=max(max_abs,float(np.max(np.abs(v.astype(float)-actual[k]))))
    max_diff=0.
    if 'base' in saved:
        for valve in (1,2):
            k=f'valve{valve}_prediction'
            old=saved[k].astype(float)-saved['base']
            new=actual[k].astype(float)-actual['base']
            assert np.allclose(old,new,rtol=1e-4,atol=1e-4),f'difference replay {k}'
            max_diff=max(max_diff,float(np.max(np.abs(new-old))))
    return dict(max_absolute_array_error=max_abs,max_response_difference_error_c=max_diff)


def metrics(pred,response):
    error=np.abs(pred['prediction'].astype(float)-pred['target'])
    curve=day_mean(error.cumsum(1)/np.arange(1,19),pred['days'])
    out=dict(H18=float(curve[-1]),cumulative_mae=curve.tolist(),response={})
    for v in (1,2):
        key=f'valve{v}'
        delta=response[key+'_prediction'].astype(float)-response['base']
        out['response'][key]=dict(interval=block_interval(delta,response['days']),
            negative_fraction=day_mean(delta<0,response['days']).tolist(),
            unsupported_windows=int((~response[key+'_support'].all(1)).sum()),
            n=len(delta))
    return out


def aggregate(reports):
    result={}
    for variant in [*VARIANTS,'GNR']:
        rr=[r for r in reports if r['variant']==variant]
        curves=np.array([r['cumulative_mae'] for r in rr])
        result[variant]=dict(seeds=[r['seed'] for r in rr],mean=curves.mean(0).tolist(),
            seed_sd=curves.std(0,ddof=1 if len(rr)>1 else 0).tolist(),
            three_seed_complete=len(rr)==3,response={
                v:np.mean([r['response'][v]['interval']['mean'] for r in rr],axis=0).tolist()
                for v in ['valve1','valve2']})
    return result


def load_source(folder,properties,device):
    report=read(folder/'report.json')
    assert sha256(folder/'best.pt')==report['checkpoint_sha256']
    p=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert p['identity_sha256']==sha256(folder.parent/'identity.json')
    model=RichFusion(TrainSpec(**p['spec']),p['normalization']['mean'],p['normalization']['std'],properties)
    model.load_state_dict(p['state_dict'],strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in p['state_dict'].items())
    return model.to(device).eval().requires_grad_(False)


def contrasts(out):
    """Saved-array pairings; no selection. Interactions concern fixed weights."""
    rows=[]
    summary=read(out/'summary.json')
    for seed in summary['seeds']:
        data={v:(npz(out/f'{v}_seed{seed}'/'prediction.npz'),npz(out/f'{v}_seed{seed}'/'response.npz'))
              for v in [*VARIANTS,'GNR']}
        terms=[('C00_minus_GNR',{'C00':1,'GNR':-1}),
               ('observer_closure_off',{'C10':1,'C00':-1}),
               ('observer_closure_on',{'C11':1,'C01':-1}),
               ('closure_observer_off',{'C01':1,'C00':-1}),
               ('closure_observer_on',{'C11':1,'C10':-1}),
               ('interaction',{'C11':1,'C10':-1,'C01':-1,'C00':1})]
        for name,coeff in terms:
            pred,response=data['C11']
            for v in coeff:
                p,r=data[v]
                for k in ['starts','days','target','persistence']:assert np.array_equal(p[k],pred[k])
                for k in ['starts','days','valve1_dose','valve2_dose','valve1_support','valve2_support']:assert np.array_equal(r[k],response[k])
            error=sum(w*np.abs(data[v][0]['prediction'].astype(float)-data[v][0]['target']).mean(1) for v,w in coeff.items())
            row=dict(seed=seed,contrast=name,paired_H18=block_interval(error,pred['days']),response={})
            for valve in (1,2):
                delta=sum(w*(data[v][1][f'valve{valve}_prediction'].astype(float)-data[v][1]['base']) for v,w in coeff.items())
                row['response'][str(valve)]=block_interval(delta,response['days'])
            rows.append(row)
    return rows


def execute(parent,gnr,record_path,properties,out,mapping='configs/final_wm/channel_mapping_v2.json',device='cpu',smoke=False):
    parent,gnr,out=Path(parent),Path(gnr),Path(out)
    assert not out.exists(),'output exists; no overwrite/resume'
    protocol=read(PACKAGE/'protocol.json')
    assert protocol['variants']=={k:list(v) for k,v in VARIANTS.items()}
    assert not protocol['training'] and not protocol['locked_test_scoring']
    pid=validate_parent(parent,record_path,mapping,properties,smoke)
    assert sha256(parent/'indices.npz')==pid['indices'],'parent index bytes changed'
    audit_gnr(gnr,parent)
    gid=read(gnr/'identity.json')
    assert sha256(gnr/'indices.npz')==pid['indices'],'GNR index bytes changed'
    for k in ['record','mapping','properties','indices','normalization_mean','normalization_std']:assert pid[k]==gid[k],k
    record=RichRecord(record_path,mapping)
    indices=npz(parent/'indices.npz')
    assert set(indices)=={'train','validation','response'},'index bank keys changed'
    for key,split in [('validation',1),('response',1)]:
        assert torch.isin(torch.from_numpy(indices[key]),record.candidates(split)).all()
    seeds=[0] if smoke else [0,1,2]
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('* -text\n',encoding='ascii')
    shutil.copyfile(parent/'indices.npz',out/'indices.npz')
    identity=dict(protocol=protocol,protocol_sha256=sha256(PACKAGE/'protocol.json'),registration_sha256=sha256(REG),
        parent_identity_sha256=sha256(parent/'identity.json'),gnr_identity_sha256=sha256(gnr/'identity.json'),
        parent_summary_sha256=sha256(parent/'summary.json'),gnr_summary_sha256=sha256(gnr/'summary.json'),
        record=pid['record'],mapping=pid['mapping'],properties=pid['properties'],indices=pid['indices'],
        sources={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in PACKAGE.glob('*.py')},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=platform.python_version(),torch=torch.__version__,device=device,
        cuda_version=torch.version.cuda,cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        smoke=smoke,seeds=seeds,checkpoint_hashes={},training_updates=0,locked_test_evaluated=False)
    for seed in seeds:
        for label,folder in [('GRU',parent/f'fusion_gru_norew_seed{seed}'),('GNR',gnr/f'greybox_steady_none_norew_seed{seed}')]:
            identity['checkpoint_hashes'][f'{label}_seed{seed}']=sha256(folder/'best.pt')
    write(out/'identity.json',identity)
    reports=[]
    try:
        for seed in seeds:
            for label,folder,variants in [('GRU',parent/f'fusion_gru_norew_seed{seed}',list(VARIANTS)),
                                          ('GNR',gnr/f'greybox_steady_none_norew_seed{seed}',['GNR'])]:
                source=load_source(folder,load_grid_properties(properties) if properties else None,device)
                for variant in variants:
                    print(f'CORE1 inference {variant} seed{seed} (fixed weights, no training)',flush=True)
                    view=CoreView(source,'C00' if variant=='GNR' else variant)
                    pred,_=collect(view,record,indices['validation'],device)
                    response,trace=collect(view,record,indices['response'],device,True)
                    result=metrics(pred,response)
                    result.update(seed=seed,variant=variant,status='complete',source_checkpoint_sha256=sha256(folder/'best.pt'))
                    dest=out/f'{variant}_seed{seed}';dest.mkdir()
                    for name,raw in [('prediction',pred),('response',response),('trace',trace)]:
                        np.savez_compressed(dest/f'{name}.npz',**raw)
                        result[name+'_sha256']=sha256(dest/f'{name}.npz')
                    # Preserve finite arrays even if the reference replay gate fails.
                    if variant in ('C11','GNR'):
                        result['reference_replay']={name:replay_check(npz(folder/f'{name}.npz'),raw)
                                                   for name,raw in [('prediction',pred),('response',response)]}
                    write(dest/'report.json',result);reports.append(result)
                    write(out/'progress.json',reports)
        summary=dict(protocol_id='FMTS-CORE1',seeds=seeds,runs=reports,seed_summary=aggregate(reports),training_updates=0,
            locked_test_evaluated=False,scientific_verdict='DESCRIPTIVE_FIXED_WEIGHT_ABLATION_NOT_PLANT_TRUTH',
            selected_hybrid='UNCHANGED_fusion_gru_norew',independently_trained_greybox='GNR1_RETAINED')
        write(out/'summary.json',summary)
        write(out/'contrasts.json',contrasts(out))
    except Exception as exc:
        write(out/'failure.json',dict(error=str(exc),completed=len(reports),training_updates=0,
            failed_seed=locals().get('seed'),failed_variant=locals().get('variant')))
        raise
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--parent',default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--gnr',default='results/fmts_greybox_norew_20260913/linux_full_gnr1')
    ap.add_argument('--record',required=True);ap.add_argument('--properties',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--mapping',default='configs/final_wm/channel_mapping_v2.json')
    a=ap.parse_args()
    execute(a.parent,a.gnr,a.record,a.properties,a.out,a.mapping,a.device)
