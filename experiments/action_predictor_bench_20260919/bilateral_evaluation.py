"""Common bilateral factual errors and saved, paired four-valve response probes."""
import json
from pathlib import Path

import numpy as np
import torch

from .analyze_nadir import trajectory_metrics
from .bilateral_models import BilateralR4,ProtectedBilateral
from .data import sha256
from .dynamic_factual import classify_load,select_probe_windows
from .run import HERE,save_json


def temperature_metrics(prediction,truth):
    if not np.isfinite(prediction).all(): raise ValueError('Nonfinite predictions; do not score a finite subset')
    e=np.abs(prediction.astype(np.float64)-truth)
    result=dict(windows=len(e),per_window_main_mae_C=e[:,:,4].mean(1).tolist(),
                per_step_channel_mae_C=e.mean(0).tolist())
    for H in (32,128,512):
        if H<=e.shape[1]: result[f'H{H}']=dict(main_mae_C=float(e[:,:H,4].mean()),per_channel_mae_C=e[:,:H].mean((0,1)).tolist())
    if e.shape[1]>=512: result['tail_129_512']=dict(main_mae_C=float(e[:,128:,4].mean()),per_channel_mae_C=e[:,128:].mean((0,1)).tolist())
    for start in range(0,e.shape[1],128): result[f'block_{start+1}_{min(start+128,e.shape[1])}_main_mae_C']=float(e[:,start:start+128,4].mean())
    H=min(128,e.shape[1]); dp=prediction[:,6:H,4]-prediction[:,:H-6,4]; dy=truth[:,6:H,4]-truth[:,:H-6,4]
    result['delta60']=dict(mae_C=float(np.abs(dp-dy).mean()),
        correlation=float(np.corrcoef(dp.ravel(),dy.ravel())[0,1]) if dp.std()>1e-12 and dy.std()>1e-12 else None,
        rms_amplitude_ratio=float(np.sqrt(np.mean(dp**2))/np.sqrt(np.mean(dy**2))) if np.mean(dy**2)>1e-12 else None)
    return result


def support_features(history):
    return np.concatenate((history[:,-1,[24,14]],(history[:,-1,24]-history[:,-7,24])[:,None],
                           history[:,-1,10:14],history[:,-1,10:14]-history[:,-7,10:14]),1)


def load_groups(data):
    path=HERE.parents[1]/'results/action_predictor_bench_20260919/ssm_dynamic_20260922/load_context.npz'
    with np.load(path) as z:
        time_to_row={int(t):i for i,t in enumerate(z['times'])}
        ids=[time_to_row[int(t)] for t in data['evaluation_time']]
        load=z['load_MW'][ids]
    groups=classify_load(load)
    low,high=np.quantile(support_features(data['train'][:,:64]),(.01,.99),axis=0)
    outside=(support_features(data['evaluation'][:,:64])<low)|(support_features(data['evaluation'][:,:64])>high)
    # Actual 60s MW increments; a reversal requires both >=+1 and <=-1MW.
    increments=load[:,70:192]-load[:,64:186]
    masks={str(g):groups==g for g in ('rise','fall','vary','steady')}
    masks.update(strict01=data['evaluation_strict01'],near_closed=data['evaluation_near_closed'],
                 support_inside=~outside.any(1),support_outside=outside.any(1),
                 load_reversal=(increments.max(1)>=1)&(increments.min(1)<=-1))
    return groups,masks,load,dict(source_sha256=sha256(path),train_q01=low.tolist(),train_q99=high.tolist(),
        outside_by_dimension=outside.sum(0).tolist(),features=['MW','steam_flow','MW_delta60',
        'u1A','u1B','u2A','u2B','du1A60','du1B60','du2A60','du2B60'])


def probe_cases(smoke=False):
    cases=[]
    def add(shape,onset,dose,vector,post=128,burn=None):
        end=onset+post if burn is None else max(onset+post,burn+128)
        elapsed=np.arange(end+1)-onset
        profile=(elapsed>=0).astype(float)
        if shape=='ramp': profile*=np.clip(elapsed/16,0,1)
        elif shape.startswith('pulse'): profile*=elapsed<int(shape[5:])
        elif shape=='sine': profile*=np.sin(2*np.pi*elapsed/32)
        elif shape=='smooth': profile*=.6*np.sin(2*np.pi*elapsed/48)+.4*np.sin(2*np.pi*elapsed/19)
        elif shape=='double': profile*=((elapsed<8)|((elapsed>=24)&(elapsed<32)))
        vector=np.asarray(vector)
        cases.append(dict(id=f'{shape}_t{onset}_p{post}_d{dose:+.2f}_v'+''.join(map(str,vector)),
            shape=shape,onset=onset,horizon=end,dose=dose,vector=vector.tolist(),burn=burn,
            delta=profile[:,None]*vector[None]*dose))
    for valve in range(4):
        v=[int(i==valve) for i in range(4)]
        for dose in (-.05,-.03,-.01,.01,.03,.05): add('step',0,dose,v,512)
        for shape in ('ramp','pulse12','sine','smooth','double'):
            for dose in (-.03,.03): add(shape,0,dose,v)
        for onset in (31,32,33,63,64,65,128): add('step',onset,.03,v)
        add('step',8,.03,v,burn=32)
        for duration in (3,6): add(f'pulse{duration}',0,.03,v)
    for pair in ((0,1),(2,3),(0,3),(1,2)):
        for second in (-1,1):
            v=[0]*4; v[pair[0]]=1;v[pair[1]]=second;add('step',0,.03,v)
    if smoke:
        return [cases[0],next(c for c in cases if c['shape']=='ramp'),
                next(c for c in cases if c['onset']==64),cases[-1]]
    return cases


def response_numbers(delta,dose,vector):
    """All observed output paths; absent outputs are never inserted as zeros."""
    out=dict(peak_abs_per_channel_C=np.abs(delta).max(1).mean(0).tolist(),
             terminal_per_channel_C=delta[:,-1].mean(0).tolist(),
             tail60_per_channel_C=delta[:,-6:].mean((0,1)).tolist(),
             signed_area_per_channel_Cs=(delta.sum(1)*10).mean(0).tolist())
    if np.count_nonzero(vector)==1:
        sign=np.sign(dose*np.asarray(vector)[np.flatnonzero(vector)[0]])
        per=trajectory_metrics(delta.transpose(0,2,1).reshape(-1,delta.shape[1]),sign)
        for k,v in per.items():
            x=v.reshape(len(delta),delta.shape[2]);finite=np.isfinite(x)
            out[k]=[float(x[finite[:,j],j].mean()) if finite[:,j].any() else None for j in range(x.shape[1])]
    for lo,hi in ((0,6),(6,18),(18,60),(60,128),(128,512)):
        if lo<delta.shape[1]: out[f'mean_{lo*10}_{min(hi,delta.shape[1])*10}s_C']=delta[:,lo:hi].mean((0,1)).tolist()
    if delta.shape[1]>128: out['prefix_H128_same_cohort']=response_numbers(delta[:,:128],dose,vector)
    return out


@torch.no_grad()
def responses(model,data,args,folder,labels,load):
    from .bilateral import unpack
    grouped=select_probe_windows(labels,max(1,args.response_windows//4))
    positions=np.array(sorted({i for ids in grouped.values() for i in ids}),dtype=int)[:args.response_windows]
    h,u,d,_=unpack(data['evaluation'][positions],args.device)
    h,u,d=h.double(),u.double(),d.double()
    model.double().eval()
    modes=('native',) if isinstance(model,BilateralR4) else ('native','block_joint_refresh' if len(model.output_indices)==10 else 'block_context_fixed')
    curves={};rows=[];catalog=[];eligibility={};base_cache={}
    try:
        for case_index,case in enumerate(probe_cases(args.smoke)):
            H=case['horizon']; onset=case['onset']; delta=torch.as_tensor(case['delta'],device=h.device,dtype=h.dtype)
            main=case['shape']=='step' and onset==0 and abs(case['dose'])==.03 and np.count_nonzero(case['vector'])==1
            protocols=('recorded','held_both','held_boundary','held_actions') if main and not args.smoke else ('recorded',)
            for protocol in protocols:
                original=u[:,:H+1] if protocol not in ('held_both','held_actions') else u[:,:1].expand(-1,H+1,-1)
                boundary=d[:,:H+1] if protocol not in ('held_both','held_boundary') else d[:,:1].expand(-1,H+1,-1)
                candidate=original+delta[None]
                valid=((candidate>=-.02)&(candidate<=1)).all((1,2))
                safe=torch.where(valid[:,None,None],candidate,original)
                affected=set(np.flatnonzero(case['vector']))
                supported=affected.issubset(model.action_indices)
                groups=classify_load(load[positions,onset:])
                for mode in modes:
                    protected=isinstance(model,ProtectedBilateral)
                    cache=('continuous_response' if protected else mode,protocol,H)
                    if cache not in base_cache:
                        base_cache[cache]=model.response.decompose(h,original,boundary)['response'] if protected else model.forecast(h,original,boundary,mode)
                    if not protected or mode==modes[0]:
                        prediction=model.response.decompose(h,safe,boundary)['response'] if protected else model.forecast(h,safe,boundary,mode)
                        change=(prediction-base_cache[cache]).cpu().numpy()
                    key=f'c{case_index}_{mode}_{protocol}'
                    full=change.copy(); full[~valid.cpu().numpy()]=np.nan
                    curves[key]=full.astype(np.float32)
                    eligibility[key]=valid.cpu().numpy()
                    entry={k:v for k,v in case.items() if k!='delta'}
                    entry.update(key=key,mode=mode,boundary_action_protocol=protocol,
                        all_perturbed_inputs_supplied=supported,output_indices=list(model.output_indices),
                        pre_onset_max_C=float(np.abs(change[:,:onset]).max()) if onset else 0.,
                        window_positions=positions.tolist(),valid=valid.cpu().tolist(),
                        near_closed=((original.abs()<.02)|(candidate.abs()<.02)).any((1,2)).cpu().tolist(),
                        load_label_scope='first128 steps after action, not the full85min')
                    catalog.append(entry)
                    for group in ('all','rise','fall','vary','steady'):
                        good=valid.cpu().numpy() & (np.ones(len(h),bool) if group=='all' else groups==group)
                        row=dict(key=key,group=group,eligible=int(good.sum()),applicable=supported)
                        if good.any(): row.update(response_numbers(change[good,onset:],case['dose'],case['vector']))
                        rows.append(row)
        # Paired dose/symmetry diagnostics use intersection cohorts, not separate means.
        comparisons=[]
        for mode in modes:
            for valve in range(4):
                found={round(c['dose'],2):c for c in catalog if c['mode']==mode and c['boundary_action_protocol']=='recorded'
                       and c['shape']=='step' and c['onset']==0 and c['horizon']==512
                       and c['vector']==[int(i==valve) for i in range(4)]}
                for small,large in ((.01,.03),(.03,.05),(-.03,-.01),(-.05,-.03),(-.03,.03)):
                    if small not in found or large not in found: continue
                    a,b=found[small],found[large]; good=eligibility[a['key']]&eligibility[b['key']]
                    row=dict(mode=mode,valve=valve,doses=[small,large],eligible=int(good.sum()),applicable=valve in model.action_indices)
                    if good.any():
                        x,y=curves[a['key']][good],curves[b['key']][good]
                        row.update(increasing_opening_positive_difference_fraction=((y-x)>1e-4).mean((0,1)).tolist(),
                                   pair_sum_mean_C=(x+y).mean((0,1)).tolist())
                    comparisons.append(row)
        np.savez_compressed(folder/'responses.npz',positions=positions,**curves)
        save_json(folder/'response_catalog.json',catalog);save_json(folder/'response_metrics.json',rows)
        save_json(folder/'response_pairs.json',comparisons)
    finally: model.float()
    return dict(cases=len(catalog),windows=len(positions),precision='paired float64; saved differences float32',
        P3_response_modes_identical=isinstance(model,ProtectedBilateral),
        interpretation='conditional model differences, not intervention truth; no sign success assigned to absent inputs/outputs')


@torch.no_grad()
def natural_events(model,args,folder):
    """Descriptive train-event cross-check; matched differences are NOT labels.

    At each real event predict its recorded plan, then hold only the target valve
    at its anchor. Other actions/boundaries are identical between the two calls.
    """
    from .bilateral_events import OUT,VALVES
    arrays={};report={}; model.double().eval()
    mode='block_joint_refresh' if len(model.output_indices)==10 else 'block_context_fixed'
    try:
        with np.load(OUT/'event_responses.npz') as z:
            for valve,name in enumerate(VALVES):
                if f'{name}_history' not in z: continue
                count= min(2,len(z[f'{name}_history'])) if args.smoke else len(z[f'{name}_history'])
                predictions=[];changes=[]
                for start in range(0,count,args.batch_size):
                    sl=slice(start,min(start+args.batch_size,count))
                    h,u,d=[torch.as_tensor(z[f'{name}_{key}'][sl],device=args.device,dtype=torch.float64)
                           for key in ('history','future_act','future_bnd')]
                    held=u.clone(); held[:,:,valve]=h[:,-1,10+valve,None]
                    p=model.forecast(h,u,d,mode)
                    if isinstance(model,ProtectedBilateral):
                        delta=model.response.decompose(h,u,d)['response']-model.response.decompose(h,held,d)['response']
                    else: delta=p-model.forecast(h,held,d,mode)
                    if not torch.isfinite(p).all() or not torch.isfinite(delta).all():
                        raise ValueError('Nonfinite natural-event prediction')
                    predictions.append(p.cpu().numpy());changes.append(delta.cpu().numpy())
                p,delta=np.concatenate(predictions),np.concatenate(changes)
                truth=z[f'{name}_truth'][:count,:,model.output_indices]
                arrays.update({f'{name}_prediction':p.astype(np.float32),f'{name}_response':delta.astype(np.float32),
                    f'{name}_truth':truth,f'{name}_epoch':z[f'{name}_event_epoch'][:count],
                    f'{name}_matched_observational_difference':z[f'{name}_matched_difference'][:count,1:,model.output_indices]})
                value=dict(events=count,action_supplied=valve in model.action_indices,
                    factual_mae_per_channel_C=np.abs(p-truth).mean((0,1)).tolist())
                for label,mask in (('opening',z[f'{name}_dose'][:count]>0),('closing',z[f'{name}_dose'][:count]<0)):
                    value[label]=dict(events=int(mask.sum()))
                    if mask.any() and valve in model.action_indices:
                        value[label]['model_response']=response_numbers(delta[mask],1 if label=='opening' else -1,[1])
                report[name]=value
        np.savez_compressed(folder/'natural_events.npz',output_indices=np.array(model.output_indices),**arrays)
        result=dict(mode=mode,groups=report,source_sha256=sha256(OUT/'event_responses.npz'),
            interpretation='TRAIN observational comparison only: no causal MAE and no tuning to the observed valley; target valve held, other actions and boundaries recorded')
        save_json(folder/'natural_events.json',result)
        return result
    finally: model.float()


@torch.no_grad()
def evaluate_model(model,data,args,folder,arm,seed,include_responses,checkpoint=None):
    from .bilateral import unpack
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model.float().eval()
    labels,masks,load,context=load_groups(data)
    joint=len(model.output_indices)==10
    modes=('native','block_context_fixed','block_joint_refresh') if joint else ('native','block_context_fixed')
    arrays={};scores={}
    y=data['evaluation'][:,64:,:10]
    for mode in modes:
        if isinstance(model,BilateralR4) and arrays: p=arrays['native']
        else:
            parts=[]
            for start in range(0,len(y),args.batch_size):
                h,u,d,_=unpack(data['evaluation'][start:start+args.batch_size],args.device)
                parts.append(model.forecast(h,u,d,mode).cpu().numpy())
            p=np.concatenate(parts)
        arrays[mode]=p;scores[mode]={}
        sides=('A','B') if joint else (('B',) if model.output_indices[0]==5 else ('A',))
        for index,side in enumerate(sides):
            # Context-fixed only defines the A comparison in joint models.
            if joint and mode=='block_context_fixed' and side=='B':
                scores[mode][side]=dict(status='not_applicable',reason='B state is not refreshed in this A-side diagnostic');continue
            off=0 if side=='A' else 5
            pp=p[:,:,index*5:(index+1)*5] if joint else p
            yy=y[:,:,off:off+5]
            value=temperature_metrics(pp,yy)
            value['groups']={}
            for group,good in masks.items():
                if good.any():
                    g=temperature_metrics(pp[good],yy[good]);g.pop('per_step_channel_mae_C')
                    value['groups'][group]=g
                else: value['groups'][group]=dict(windows=0,status='not_applicable')
            scores[mode][side]=value
        if joint and mode!='block_context_fixed':
            for name,sign in (('common_mode',1),('side_difference',-1)):
                scores[mode][name]=temperature_metrics((p[:,:,:5]+sign*p[:,:,5:])/2,(y[:,:,:5]+sign*y[:,:,5:])/2)
        print(f'Evaluated {arm} seed{seed} {mode}',flush=True)
    np.savez_compressed(folder/'predictions.npz',**arrays,truth=y,output_indices=np.array(model.output_indices),times=data['evaluation_time'])
    response=responses(model,data,args,folder,labels,load) if include_responses else dict(status='deployment_selector_only')
    events=natural_events(model,args,folder) if include_responses else dict(status='deployment_selector_only')
    result=dict(status='complete',arm=arm,seed=seed,checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        checkpoint_epoch=int(torch.load(checkpoint,weights_only=True,map_location='cpu')['epoch']) if checkpoint else None,
        factual=scores,response=response,natural_events=events,stratification=context,
        R3_mode_aliases=isinstance(model,BilateralR4),missing_outputs='not applicable, never zero-filled')
    save_json(folder/'result.json',result)
    return result


def summary(output):
    rows=[]
    for path in sorted(Path(output).glob('seed*/*/result.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        for mode,sides in r['factual'].items():
            for side,values in sides.items():
                if side not in ('A','B') or 'H32' not in values: continue
                rows.append(dict(model=path.parent.name,seed=r['seed'],mode=mode,side=side,
                    H32=values['H32']['main_mae_C'],H128=values['H128']['main_mae_C'],
                    H512=values['H512']['main_mae_C'],tail129_512=values['tail_129_512']['main_mae_C']))
    save_json(Path(output)/'summary.json',rows)
    lines=['# Bilateral factual errors (development validation)','',
           'All requested arms retained. R3 mode aliases are identical. Full responses are saved for deployment selectors.',
           '', '| Model | Seed | Mode | Side | H32 | H128 | H512 | Tail129–512 |',
           '|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows: lines.append('| '+' | '.join([r['model'],str(r['seed']),r['mode'],r['side'],*[f'{r[k]:.4f}' for k in ('H32','H128','H512','tail129_512')]])+' |')
    lines+=['','## Training stop status','','| Arm | Epochs | Stop | Parameters |','|---|---:|---|---:|']
    for path in sorted(Path(output).glob('fits/seed*/*/fit.json')):
        r=json.loads(path.read_text());lines.append(f'| {path.parent.parent.name}/{r["model"]} | {r["epochs_run"]} | {r["stop_reason"]} | {r["parameters"]} |')
    (Path(output)/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
