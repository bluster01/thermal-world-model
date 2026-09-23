"""Zero-training diagnostic: use one declared reference's R4 coefficients.

This is a reference-conditioned local simulator. Matching the recorded reference
exactly is a construction identity, NOT evidence of forecasting unknown plans.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .bilateral import HERE,OUT_DEFAULT,PARENT
from .bilateral_models import make_model,SIDE_VALVES
from .bilateral_evaluation import probe_cases
from .review_bilateral import ROOT,read,VALVES,MAIN
from .run import save_json
from .data import sha256


@torch.no_grad()
def coefficients(model,history,reference,boundaries):
    response=model.response if hasattr(model,'response') else model
    taps=[dict(valve=[],stage=[]) for _ in response.chains];handles=[]
    def capture(bucket):
        return lambda module,inputs,output:bucket.append(output.detach().clone())
    try:
        for core,tap in zip(response.chains,taps):
            handles.append(core.valve_rate.register_forward_hook(capture(tap['valve'])))
            handles.append(core.stage_rate.register_forward_hook(capture(tap['stage'])))
        base=response.decompose(history,reference,boundaries)
    finally:
        for handle in handles:handle.remove()
    result=[]
    for core,tap,trace in zip(response.chains,taps,base['traces']):
        tau_v=core.valve_tau*torch.exp(.5*torch.tanh(torch.stack(tap['valve'],1)))
        tau_s=core.log_stage_tau.exp()*torch.exp(.5*torch.tanh(torch.stack(tap['stage'],1).squeeze(-1)))
        result.append(dict(alpha_v=-torch.expm1(-core.dt/tau_v),alpha_s=-torch.expm1(-core.dt/tau_s),gain=trace['gain']))
    return result,base['response']


@torch.no_grad()
def replay(model,history,actions,coef):
    response=model.response if hasattr(model,'response') else model
    parts=[]
    for side,(core,c) in enumerate(zip(response.chains,coef)):
        carrier=history.new_zeros((len(history),2,5));modes=history.new_zeros((len(history),2,4));out=[]
        u=(actions[:,:-1,SIDE_VALVES[side]]-history[:,-1:,tuple(10+i for i in SIDE_VALVES[side])])/core.scale[5:7]
        for t in range(u.shape[1]):
            modes=modes+c['alpha_v'][:,t]*(u[:,t,:,None]-modes)
            effect=(modes*core.valve_mix.softmax(-1)).sum(-1)
            shifted=torch.cat((torch.zeros_like(carrier[:,:,:1]),carrier[:,:,:-1]),-1)
            goal=shifted*(core.reachable-core.entry)+effect[:,:,None]*core.entry
            carrier=(carrier+c['alpha_s'][:,t,None]*(goal-carrier))*core.reachable
            out.append(-(c['gain'][:,t]*carrier[:,core.path_valve,core.path_stage])@core.path_output)
        parts.append(torch.stack(out,1))
    return torch.cat(parts,-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'bilateral33_seed11')
    p.add_argument('--out',type=Path,default=ROOT/'bilateral_review_20260923')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    rows=[];checks=[];saved={}
    for arm in ('R3','P3'):
        folder=args.run/'seed11'/f'{arm}_AB_balanced'
        catalog=read(folder/'response_catalog.json')
        with np.load(folder/'responses.npz') as z:
            positions=z['positions'];returned={c['key']:z[c['key']] for c in catalog if c['shape']=='step'
                and c['onset']==0 and abs(c['dose'])==.03 and c['mode']=='native' and c['boundary_action_protocol']=='recorded'}
        with np.load(OUT_DEFAULT) as z:
            model=make_model(arm,z['mean'],z['scale'],11,PARENT)
            h,u,d=[torch.tensor(z[f'{key}_evaluation'][positions],dtype=torch.float64)
                   for key in ('hist30','future_act','future_bnd')]
        checkpoint=args.run/'fits/seed11'/arm/'best_AB_balanced.pt'
        model.load_state_dict(torch.load(checkpoint,weights_only=True)['model']);model.double().eval()
        coef,base=coefficients(model,h,u,d);reproduced=replay(model,h,u,coef)
        error=float((base-reproduced).abs().max());assert error<1e-10
        checks.append(dict(model=arm,checkpoint_sha256=sha256(checkpoint),reference_reconstruction_max_C=error))
        for case in probe_cases():
            if np.count_nonzero(case['vector'])!=1:continue
            H=case['horizon'];v=int(np.flatnonzero(case['vector'])[0]);delta=torch.tensor(case['delta'],dtype=torch.float64)
            # The declared reference remains fixed for every candidate, not
            # silently regenerated from each candidate plan.
            candidate=u[:,:H+1]+delta[None];valid=((candidate>=-.02)&(candidate<=1)).all((1,2))
            candidate=torch.where(valid[:,None,None],candidate,u[:,:H+1])
            change=(replay(model,h,candidate,coef)-reproduced[:,:H]).numpy()
            good=valid.numpy();sign=np.sign(case['dose'])
            ordered=bool((case['delta'][:,v]*sign>=-1e-12).all())
            maximum=float((change[good]*sign).max(initial=0)) if ordered else None
            if ordered:assert maximum<1e-10
            row=dict(model=arm,valve=VALVES[v],shape=case['shape'],onset=case['onset'],dose_pp=case['dose']*100,
                eligible=int(good.sum()),pointwise_ordered=ordered,max_opposite_direction_C=maximum)
            if case['shape']=='step' and case['onset']==0 and abs(case['dose'])==.03:
                original_case=next(c for c in catalog if c['shape']=='step' and c['onset']==0 and c['dose']==case['dose']
                    and c['vector']==case['vector'] and c['mode']=='native' and c['boundary_action_protocol']=='recorded')
                original=returned[original_case['key']][good,:,MAIN[v]]*sign
                frozen=change[good,:,MAIN[v]]*sign
                row.update(original_wrong_tail_count=int((original[:,-6:].mean(1)>1e-4).sum()),
                    shared_wrong_tail_count=int((frozen[:,-6:].mean(1)>1e-4).sum()),
                    original_mean_signed_tail_C=float(original[:,-6:].mean()),
                    shared_mean_signed_tail_C=float(frozen[:,-6:].mean()))
                saved[f'{arm}_{VALVES[v]}_{"open" if sign>0 else "close"}']=change.astype(np.float32)
            rows.append(row)
    np.savez_compressed(args.out/'shared_coefficient_responses.npz',positions=positions,**saved)
    save_json(args.out/'shared_coefficient_diagnostic.json',dict(checks=checks,cases=rows,
        reference='recorded future valve plan and boundary plan, explicitly shared across every candidate',
        proof='Fixed alpha in (0,1), nonnegative valve mixing, forward carrier routing and positive gains define a nonnegative causal kernel with negative temperature readout; pointwise larger valve plans cannot warm reachable outputs.',
        limits='Reference-conditioned comparison only. Exact reference fit is algebraic and uses its known plan. It does not prove a globally monotone adaptive model, correct physical amplitude, or unchanged MAE on plans different from the declared reference.'))
    print('Reference reconstruction:',checks)
    print('Step tails:',[r for r in rows if 'original_wrong_tail_count' in r])


if __name__=='__main__':main()
