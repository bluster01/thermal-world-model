"""Complete local-authored Linux runner: G2 then seven fully budgeted bilateral arms."""
import argparse
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np
import torch

from .bilateral_data import NAMES, REF_MAP, OUT_DEFAULT
from .bilateral_models import ARMS, make_model, BilateralR4, ProtectedBilateral
from .data import sha256
from .full_baselines import fit
from .run import HERE, save_json

RESULTS = HERE.parents[1]/'results/action_predictor_bench_20260919'
PARENT = RESULTS/'focused33_seed11/seed11/D_short/best.pt'


def load_data(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256(path) != meta['pack_sha256']: raise ValueError('Bilateral pack checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        if z['channel_names'].tolist() != NAMES: raise ValueError('Bilateral channel mapping changed')
        data = {k:z[k].copy() for k in ('mean','scale')}
        if not np.isfinite(data['mean']).all() or not np.isfinite(data['scale']).all() or (data['scale']<=0).any():
            raise ValueError('Invalid train normalization')
        for split,n,H in (('train',20371,128),('selector',128,128),('evaluation',256,512)):
            h,u,d,y = [z[f'{key}_{split}'] for key in ('hist30','future_act','future_bnd','future_temp')]
            if (h.shape,u.shape,d.shape,y.shape) != ((n,64,30),(n,H+1,4),(n,H+1,7),(n,H,10)):
                raise ValueError(f'{split}: require all original one-third origins and full horizons')
            if not all(np.isfinite(a).all() for a in (h,u,d,y)) or not z[f'valid_{split}'].all():
                raise ValueError(f'{split}: nonfinite data; no dropping windows')
            if not np.array_equal(h[:,-1,10:14],u[:,0]) or not np.array_equal(h[:,-1,14:21],d[:,0]):
                raise ValueError(f'{split}: left-endpoint control clock mismatch')
            valves = np.concatenate((h[:,:,10:14],u[:,1:]),1)
            if (valves < -.02).any() or (valves > 1).any(): raise ValueError('Raw feedback outside declared tolerance')
            strict = ((valves>=0)&(valves<=1)).all((1,2))
            if not np.array_equal(strict,z[f'strict01_{split}']): raise ValueError('Incorrect strict mask')
            bank = np.zeros((n,64+H,30),dtype=np.float32)
            bank[:,:64] = h; bank[:,64:,:10] = y
            bank[:,64:,10:14] = u[:,1:]; bank[:,64:,14:21] = d[:,1:]
            data[split] = bank
            for key in ('time','starts'): data[f'{split}_{key}'] = z[f'{split}_{key}'].copy()
            for key in ('strict01','near_closed','u2a_negative'): data[f'{split}_{key}'] = z[f'{key}_{split}'].copy()
    return data,meta


def unpack(bank,device):
    b = torch.as_tensor(bank,device=device)
    return b[:,:64],b[:,63:,10:14],b[:,63:,14:21],b[:,64:,:10]


def objective(model,batch):
    h,u,d,y = unpack(batch,batch.device)
    mode = 'block_context_fixed' if model.arm_name=='G2' else 'native'
    p = model.forecast(h,u,d,mode)
    target = y[:,:,model.output_indices]
    scale = model.scale[list(model.output_indices)]
    weights = p.new_tensor([.125,.125,.125,.125,.5])
    values = ((p-target)/scale).square().reshape(len(p),p.shape[1],-1,5)
    per_side = (values*weights).sum(-1)
    losses = .5*per_side[:,:32].mean((0,1))+.5*per_side[:,32:128].mean((0,1))
    labels = ('A','B') if len(losses)==2 else (('B',) if model.arm_name=='S0_B' else ('A',))
    return dict(loss=losses.sum(),**{f'loss_{side}':loss for side,loss in zip(labels,losses)})


@torch.no_grad()
def select(model,bank,args):
    model.eval(); sums = np.zeros(4)
    joint = len(model.output_indices)==10
    for start in range(0,len(bank),args.batch_size):
        h,u,d,y = unpack(bank[start:start+args.batch_size],args.device)
        native = model.forecast(h,u,d,'native')
        block = model.forecast(h,u,d,'block_context_fixed')
        target = y[:,:,model.output_indices]
        predictions = [native,block]
        if joint: predictions.append(model.forecast(h,u,d,'block_joint_refresh'))
        if not all(torch.isfinite(p).all() for p in predictions): raise ValueError('Nonfinite selector')
        sums[0] += float((native[:,:32,4]-target[:,:32,4]).abs().mean())*len(h)
        sums[1] += float((block[:,:128,4]-target[:,:128,4]).abs().mean())*len(h)
        if joint:
            sums[2] += float((native[:,:32,[4,9]]-target[:,:32,[4,9]]).abs().mean())*len(h)
            sums[3] += float((predictions[2][:,:128,[4,9]]-target[:,:128,[4,9]]).abs().mean())*len(h)
    a,b,c,d = sums/len(bank)
    result = dict(short=float(a),balanced=float(.5*(a+b)),block_H128=float(b))
    if joint: result.update(AB_short=float(c),AB_balanced=float(.5*(c+d)))
    return result


@torch.no_grad()
def diagnostics(model,data,args):
    model.eval()
    h,u,d,_ = unpack(data['train'][:min(4,len(data['train']))],args.device)
    # Fixed TRAIN origins, held reference, eligible +1pp probe. No report-set selection.
    u = h[:,-1:,10:14].expand_as(u).clone()
    valve = model.action_indices[0]
    changed = u.clone(); changed[:,:,valve] += .01
    valid = (changed[:,:,valve]<=1).all(1)
    changed[~valid] = u[~valid]
    delta = model.forecast(h,changed,d,'native')-model.forecast(h,u,d,'native')
    answer = dict(train_origins=int(len(h)),eligible=int(valid.sum()),valve_index=int(valve),
                  peak_C=float(delta[valid,:,4].abs().max(1).values.mean()) if valid.any() else None,
                  area_Cs=float(delta[valid,:,4].sum(1).mean()*10) if valid.any() else None)
    response = model.response if isinstance(model,ProtectedBilateral) else model if isinstance(model,BilateralR4) else None
    if response is not None:
        answer['path_gain_raw'] = [c.path_gain_raw.detach().cpu().tolist() for c in response.chains]
        answer['stage_tau_seconds'] = [c.log_stage_tau.exp().detach().cpu().tolist() for c in response.chains]
    if model.arm_name=='G2':
        answer['terminal_gain_raw'] = model.old.response.core.path_gain_raw[[3,5]].detach().cpu().tolist()
    return answer


def validate_resume(folder,config,resume):
    path = folder/'config.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if not resume: raise ValueError('Output already exists; use --resume')
        for key in config:
            if key in ('git_commit','max_epochs'): continue
            if old.get(key)!=config[key]: raise ValueError(f'Resume mismatch: {key}')
        if config['max_epochs'] < old['max_epochs']: raise ValueError('Cannot reduce saved epoch budget')
        if config['max_epochs'] != old['max_epochs']:
            save_json(folder/f'config_before_{config["max_epochs"]}.json',old)
    save_json(path,config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=OUT_DEFAULT)
    p.add_argument('--parent',type=Path,default=PARENT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--seeds',nargs='+',type=int,default=[11])
    p.add_argument('--device',default='cpu'); p.add_argument('--threads',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--min-epochs',type=int,default=12); p.add_argument('--max-epochs',type=int,default=90)
    p.add_argument('--learning-rate',type=float,default=.001); p.add_argument('--min-lr',type=float,default=.0001)
    p.add_argument('--min-delta',type=float,default=.002)
    p.add_argument('--lr-patience',type=int,default=4); p.add_argument('--stop-patience',type=int,default=6)
    p.add_argument('--response-windows',type=int,default=32)
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if min(args.min_epochs,args.max_epochs,args.batch_size,args.threads,args.response_windows,args.lr_patience,args.stop_patience)<1 or args.max_epochs<args.min_epochs or not 0<args.min_lr<=args.learning_rate or args.min_delta<0:
        p.error('Invalid budget')
    if len(set(args.arms))!=len(args.arms) or len(set(args.seeds))!=len(args.seeds): p.error('Duplicate arms/seeds')
    torch.set_num_threads(args.threads)
    data,metadata=load_data(args.data)
    if args.smoke:
        for split,n in (('train',4),('selector',2),('evaluation',2)):
            data[split]=data[split][:n]
            for suffix in ('time','starts','strict01','near_closed','u2a_negative'): data[f'{split}_{suffix}']=data[f'{split}_{suffix}'][:n]
        args.min_epochs=args.max_epochs=1; args.batch_size=2; args.response_windows=2
    config={k:v for k,v in vars(args).items() if k not in ('data','parent','output','resume')}
    config.update(data_sha256=metadata['pack_sha256'],parent_sha256=sha256(args.parent) if 'G2' in args.arms else None,
        source_sha256={name:sha256(HERE/name) for name in ('bilateral.py','bilateral_models.py','bilateral_evaluation.py',
            'full_baselines.py','r4_base.py','r4_transport.py','focused_models.py','models.py',
            'bilateral_data.py','data.py','run.py','analyze_nadir.py','dynamic_factual.py')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        supervision_horizon=128,training_windows=len(data['train']),torch_version=str(torch.__version__),
        selectors='A short/balanced; joint additionally AB short/balanced; joint LR/stop use AB',
        G2='two terminal gains only, fixed old D block32 background and continuous response',
        R3_modes='all three mode names use the identical uninterrupted R4 forecast; not independent replications',
        future_temperatures='targets only; no future true A or B temperature enters model',
        dataset_status='development validation, not a new blind test')
    from .bilateral_events import OUT as event_folder
    event_meta=json.loads((event_folder/'event_summary.json').read_text(encoding='utf-8'))
    if event_meta['source_sha256']!=metadata['pack_sha256']:
        raise ValueError('Natural-event pack and training pack differ')
    config['event_pack_sha256']=sha256(event_folder/'event_responses.npz')
    config['load_context_sha256']=sha256(RESULTS/'ssm_dynamic_20260922/load_context.npz')
    args.output.mkdir(parents=True,exist_ok=True)
    validate_resume(args.output,config,args.resume)
    save_json(args.output/'data_metadata.json',metadata)
    save_json(args.output/'state.json',dict(status='running',arms=args.arms,seeds=args.seeds,smoke=args.smoke))
    from .bilateral_evaluation import evaluate_model, summary
    failures=[]; costs=[]
    for seed in args.seeds:
        for arm in args.arms:
            try:
                model=make_model(arm,data['mean'],data['scale'],seed,args.parent).to(args.device)
                model.arm_name=arm
                folder=args.output/'fits'/f'seed{seed}'/arm; folder.mkdir(parents=True,exist_ok=True)
                joint=len(model.output_indices)==10
                choices=('short','balanced','AB_short','AB_balanced') if joint else ('short','balanced')
                budget=('AB_short','AB_balanced') if joint else ('short','balanced')
                if arm=='G2':
                    reference=args.output/f'seed{seed}/D_fixed'
                    if not (reference/'result.json').exists(): evaluate_model(model,data,args,reference,'D_fixed',seed,True)
                completed=json.loads((folder/'fit.json').read_text()) if (folder/'fit.json').exists() else None
                if completed and (completed['reached_validation_plateau'] or completed['epochs_run']>=args.max_epochs): cost=completed
                else:
                    cost=fit(arm,seed,data,args,folder,model=model,training_objective=objective,supervision_horizon=128,
                        resume=(folder/'last.pt').exists(),selector_fn=select,checkpoint_choices=choices,
                        budget_choices=budget,epoch_diagnostics=diagnostics)
                    # A resumed fit can replace best weights. Never retain stale forecasts.
                costs.append(cost)
                for choice in choices:
                    checkpoint=folder/f'best_{choice}.pt'
                    target=args.output/f'seed{seed}/{arm}_{choice}'
                    old=json.loads((target/'result.json').read_text()) if (target/'result.json').exists() else None
                    digest=sha256(checkpoint)
                    if old and old.get('checkpoint_sha256')==digest: continue
                    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=args.device)['model'])
                    selected = choice==('AB_balanced' if joint else 'balanced')
                    evaluate_model(model,data,args,target,arm,seed,selected,checkpoint)
                summary(args.output)
            except Exception:
                failures.append(dict(seed=seed,arm=arm))
                (args.output/f'failure_seed{seed}_{arm}.txt').write_text(traceback.format_exc(),encoding='utf-8')
                traceback.print_exc()
                if args.smoke: raise
    summary(args.output)
    save_json(args.output/'state.json',dict(status='completed_with_failures' if failures else 'complete',failures=failures,
        smoke=args.smoke,fits=len(costs),expected_fits=len(args.arms)*len(args.seeds),
        not_confirmed_plateau=[f'seed{c["seed"]}/{c["model"]}' for c in costs if not c['reached_validation_plateau']]))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
