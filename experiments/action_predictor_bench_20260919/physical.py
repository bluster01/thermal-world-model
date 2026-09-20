"""Three matched H128 thermal-state fits; JEPA objectives are auxiliary only."""
import argparse
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np
import torch

from .data import sha256
from .focused import load_data, aligned_responses
from .full_baselines import fit, complete_row
from .physical_models import ARMS, Physical
from .run import HERE, aggregate, save_json


@torch.no_grad()
def state_diagnostics(model, bank, device):
    b = torch.as_tensor(bank[:8, :192], device=device)
    h, u, d = b[:, :64], b[:, 63:-1, 5:7], b[:, 63:-1, 7:13]
    p = model.rollout(h,u,d)
    z = model.normalized_state(p[:, [31,63,127]])
    target = model.future_states(b,(32,64,128))
    return dict(estimated_state_consistency_mse=float((z-target).square().mean()),
        predicted_state_std=z.flatten(0,1).std(0,unbiased=False).cpu().tolist(),
        target_estimate_std=target.flatten(0,1).std(0,unbiased=False).cpu().tolist(),
        metal_temperature_min_C=float(p[:,:,5:].min()), metal_temperature_max_C=float(p[:,:,5:].max()),
        warning='Future metal states are observer estimates, not measured ground truth; diagnostic only')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=HERE/'data/screen_A_33pct_h128.npz')
    parser.add_argument('--aux',type=Path,default=HERE/'data/hist_bypass_A_33pct_v1.npz')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--response-windows',type=int,default=8)
    parser.add_argument('--min-epochs',type=int,default=12)
    parser.add_argument('--max-epochs',type=int,default=60)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--min-lr',type=float,default=.0001)
    parser.add_argument('--min-delta',type=float,default=.002)
    parser.add_argument('--lr-patience',type=int,default=4)
    parser.add_argument('--stop-patience',type=int,default=6)
    parser.add_argument('--latent-weight',type=float,default=.05)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.threads,args.batch_size,args.response_windows,args.min_epochs,args.lr_patience,args.stop_patience)<1 or args.max_epochs<args.min_epochs or not 0<args.min_lr<=args.learning_rate or min(args.min_delta,args.latent_weight)<0:
        parser.error('Invalid training budget')
    torch.set_num_threads(args.threads)
    data,metadata=load_data(args.data,args.aux,train_horizon=128)
    if args.smoke:
        for split,n in [('train',16),('selector',4),('evaluation',4)]: data[split]=data[split][:n]
        data['evaluation_time']=data['evaluation_time'][:4]
        args.min_epochs,args.max_epochs,args.batch_size,args.response_windows=1,1,8,2
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('output','resume')}
    config.update(arms=list(ARMS),seed=11,initialization='same seed from scratch, no inherited SSM/R4 weights',
        source_sha256={p.name:sha256(p) for p in HERE.glob('*.py')},
        data_sha256=sha256(args.data),auxiliary_sha256=sha256(args.aux),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),torch_version=torch.__version__,
        supervision_horizon=128,temperature_loss='.5 first32 + .5 next96, original five-channel weights',
        jepa_horizons=[32,64,128],P1_ema_decay=.99,
        selected_auxiliary_channels=metadata['selected_channels'],
        block_protocol='same continuous physical state as native; two API labels, not independent protocols',
        limitations=['constant effective cp','inlet-flow and coolant-temperature proxies','unobserved metal estimates, not truth',
                     'P2 reduced physical view, not independent privileged state or exact JEPA-x reproduction',
                     'energy balance does not guarantee total transient monotonicity'])
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'config.json'
    if path.exists():
        old=json.loads(path.read_text())
        if not args.resume or any(old.get(k)!=v for k,v in config.items() if k!='git_commit'):
            raise ValueError('Existing output requires --resume and identical code/config/data')
    else: save_json(path,config)
    save_json(args.output/'data_metadata.json',metadata)
    save_json(args.output/'state.json',dict(status='running',smoke=args.smoke))
    np.savez_compressed(args.output/'evaluation_inputs.npz',bank=data['evaluation'],times=data['evaluation_time'])
    costs,failures=[],[]
    for arm in ARMS:
        try:
            torch.manual_seed(11)
            model=Physical(data['mean'],data['scale'],data['aux_mean'],data['aux_scale'],arm).to(args.device)
            folder=args.output/'fits/seed11'/arm; folder.mkdir(parents=True,exist_ok=True)
            cost=json.loads((folder/'fit.json').read_text()) if (folder/'fit.json').exists() else fit(
                arm,11,data,args,folder,model=model,supervision_horizon=128,
                training_objective=lambda m,b:m.training_objective(b,args.latent_weight))
            costs.append(cost)
            for choice in ('short','balanced'):
                name=f'{arm}_{choice}'
                model.load_state_dict(torch.load(folder/f'best_{choice}.pt',weights_only=True,map_location=args.device)['model'])
                model.eval()
                row_cost={k:v for k,v in cost.items() if k not in ('status','model','seed')}
                row_cost.update(fit_id=f'seed11/{arm}',best_epoch=cost['best_epochs'][choice])
                complete_row(name,arm,choice,11,model,row_cost,data,args,[('best.pt',folder/f'best_{choice}.pt')])
                target=args.output/'seed11'/name
                if not (target/'aligned_response_metrics.json').exists():
                    rows,curves=aligned_responses(model,data['evaluation'],args.device,args.response_windows,args.smoke)
                    np.savez_compressed(target/'aligned_responses.npz',**curves)
                    save_json(target/'aligned_response_metrics.json',rows)
                save_json(target/'physical_parameters.json',model.parameter_report())
                save_json(target/'state_diagnostics.json',state_diagnostics(model,data['evaluation'],args.device))
                result=json.loads((target/'result.json').read_text())
                result.update(native_protocol='continuous eight-state thermal network',
                    block_protocol='identical continuous eight-state thermal network; no state reset')
                save_json(target/'result.json',result)
        except Exception:
            failures.append(arm)
            (args.output/f'failure_{arm}.txt').write_text(traceback.format_exc(),encoding='utf-8')
            traceback.print_exc()
            if args.smoke: raise
        aggregate(args.output)
    from .analyze_return import analyze
    analyze(args.output,args.output)
    lines=['# 三臂训练状态','','| Arm | Epochs | Stop | Best short | Best balanced |','|---|---:|---|---:|---:|']
    for c in costs:
        lines.append(f"| {c['model']} | {c['epochs_run']} | {c['stop_reason']} | {c['best_epochs']['short']} | {c['best_epochs']['balanced']} |")
    (args.output/'CONVERGENCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(args.output/'state.json',dict(status='completed_with_failures' if failures else 'complete',failures=failures,
        smoke=args.smoke,fits=len(costs),evaluation_rows_expected=6,
        not_confirmed_plateau=[c['model'] for c in costs if not c['reached_validation_plateau']]))
    if failures: raise SystemExit(1)


if __name__=='__main__':
    main()
