"""Export existing-checkpoint response trajectories; never train or tune."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.final_wm.replay_norew_pairs import isolated_record,sampled_indices,digest,write,ARMS
from src.final_wm.data import sample_windows,SPLIT_VAL
from src.final_wm.evaluation import day_block_mean_ci
from src.final_wm.training import TrainSpec,build_world_model
from src.final_wm.properties import load_grid_properties

@torch.no_grad()
def trace_cell(model,record,seed,valve,horizon):
    sample_seed=80000+1000*valve+horizon+seed
    batch=sample_windows(record,SPLIT_VAL,64,96,1,torch.Generator().manual_seed(sample_seed))
    history=batch.history.__class__(*(x.to('cuda') for x in batch.history))
    boundary=batch.future_boundary[:,:1].to('cuda').repeat(1,horizon,1)
    base_actions=batch.future_actions[:,:1].to('cuda').repeat(1,horizon,1)
    step_actions=base_actions.clone()
    step_actions[:,:,valve]=(step_actions[:,:,valve]+.05).clamp(max=1)
    base=model.counterfactual(history,base_actions,boundary_mode='oracle',true_future_boundary=boundary,allow_extrapolation=True)
    step=model.counterfactual(history,step_actions,boundary_mode='oracle',true_future_boundary=boundary,allow_extrapolation=True)
    # Preserve exact original terminal-summary subtraction/reduction order.
    terminal=(step.temps_mu[:,-10:,4]-base.temps_mu[:,-10:,4]).mean(dim=1).cpu()
    delta=(step.temps_mu-base.temps_mu).cpu()
    ci=day_block_mean_ci(terminal,batch.day_ids,n_boot=1000,seed=sample_seed+1)
    days=torch.unique(batch.day_ids)
    day_means=torch.stack([delta[batch.day_ids==day].mean(dim=0) for day in days])
    gen=torch.Generator().manual_seed(sample_seed+1)
    boot_idx=torch.randint(len(days),(1000,len(days)),generator=gen)
    boot=day_means[boot_idx].mean(dim=1)
    point=day_means.mean(dim=0)
    limits=torch.quantile(boot,torch.tensor([.025,.975]),dim=0)
    indices=sampled_indices(record,64,96,1,sample_seed)
    raw={'base_temps_c':base.temps_mu.cpu(),'step_temps_c':step.temps_mu.cpu(),
         'delta_temps_c':delta,'terminal_delta_per_window_c':terminal,'day_ids':batch.day_ids,
         'first_future_indices':torch.tensor(indices),'base_actions':base_actions.cpu(),'step_actions':step_actions.cpu(),
         'frozen_boundary':boundary.cpu(),'baseline_in_support':base.in_support.cpu(),'in_support':step.in_support.cpu()}
    report={'first_future_indices':indices,'valve_index':valve,'horizon':horizon,
            'mean_delta_c':ci['point'],'ci_lo_c':ci['ci_lo'],'ci_hi_c':ci['ci_hi'],
            'pointwise_curve_mean_c':point.tolist(),'pointwise_curve_ci_lo_c':limits[0].tolist(),
            'pointwise_curve_ci_hi_c':limits[1].tolist(),'n_days':len(days),
            'n_unsupported':int((~step.in_support).sum()),'baseline_n_unsupported':int((~base.in_support).sum())}
    return raw,report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--paired',type=Path,required=True,help='Completed paired replay directory')
    parser.add_argument('--out',type=Path,required=True,help='Must not exist')
    args=parser.parse_args()
    if sys.flags.optimize or not torch.cuda.is_available():
        raise RuntimeError('Use CUDA and do not use python -O')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    manifest=json.loads((args.package/'sideA/manifest.json').read_text())
    for item in manifest['inputs'].values():
        assert digest(args.package/'sideA'/item['package_relative_path'])==item['sha256']
    record,isolation=isolated_record(args.package/'inputs/canonical_sideA_v2.npz')
    write(args.out/'isolation.json',isolation)
    protocol={'training':False,'leakage_training':False,'new_checkpoints':0,'arms':ARMS,'seeds':[0,1,2],
              'windows_per_cell':64,'horizons':[18,60],'valves':[0,1],'delta_v':.05,'history':96,
              'scope':'36 existing paired response cells, export all five temperatures at every predicted step',
              'curves':'Equal-day mean and pointwise 95% day bootstrap intervals, 1000 draws; not simultaneous bands; no support filtering',
              'terminal_replay_tolerance':1e-6,'test_numeric_rows_decoded':0,
              'environment':{'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                             'gpu':torch.cuda.get_device_name(0),'matmul':torch.get_float32_matmul_precision(),
                             'cuda_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32},
              'script_sha256':digest(Path(__file__))}
    write(args.out/'protocol.json',protocol)
    properties=load_grid_properties(args.package/'inputs/iapws_surrogate.npz',device='cuda')
    reports=[]
    for seed in range(3):
        for arm in ARMS:
            run=f't1_{arm}_seed{seed}'
            print('START',run,flush=True)
            ckpt=args.package/'sideA/checkpoints'/f'{run}.pt'
            assert digest(ckpt)==manifest['files'][f'checkpoints/{run}.pt']
            payload=torch.load(ckpt,map_location='cpu',weights_only=True)
            spec=TrainSpec(**payload['spec'])
            assert (spec.arm,spec.seed,spec.initial_state_mode,spec.boundary_mode)==(arm,seed,'hybrid','oracle')
            model=build_world_model(spec,properties).to('cuda').eval()
            model.load_state_dict(payload['state_dict'],strict=True)
            before={k:p.detach().clone() for k,p in model.named_parameters()}
            previous=json.loads((args.paired/f'{run}.json').read_text())
            result={'run':run,'arm':arm,'seed':seed,'cells':{}}
            for valve in (0,1):
                for horizon in (18,60):
                    key=f'valve{valve+1}_H{horizon}'
                    raw,cell=trace_cell(model,record,seed,valve,horizon)
                    old=previous['responses'][key]
                    assert cell['first_future_indices']==old['first_future_indices']
                    for name in ('mean_delta_c','ci_lo_c','ci_hi_c'):
                        assert abs(cell[name]-old[name])<=1e-6,(run,key,name)
                    for field,old_field in [('in_support','in_support_mask'),('baseline_in_support','baseline_in_support_mask')]:
                        assert torch.equal(raw[field],torch.tensor(old[old_field],dtype=torch.bool))
                    filename=f'{run}_{key}_traces.pt'
                    torch.save(raw,args.out/filename)
                    cell['trace_file']=filename
                    cell['sha256']=digest(args.out/filename)
                    result['cells'][key]=cell
                    write(args.out/f'{run}.json',result)
                    print(run,key,'TRACE_OK',flush=True)
            assert all(torch.equal(before[k],p) for k,p in model.named_parameters())
            result['parameters_unchanged']=True
            write(args.out/f'{run}.json',result)
            reports.append(result)
            del model,before
    write(args.out/'summary.json',{'runs':reports,'training':False,'test_numeric_rows_decoded':0,'full_R1_verdict':'NOT_ASSESSED'})
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
