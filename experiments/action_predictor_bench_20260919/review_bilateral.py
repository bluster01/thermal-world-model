"""Review returned arrays and checkpoints without fitting or changing selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .bilateral import HERE, PARENT, OUT_DEFAULT
from .bilateral_models import ARMS, make_model
from .data import sha256
from .dynamic_factual import classify_load
from .run import save_json

ROOT = HERE.parents[1]/'results/action_predictor_bench_20260919'
DEPLOY = ['D_fixed','G2_balanced','S0_A_balanced','S0_B_balanced','S1_balanced',
          'S2_balanced','S3_AB_balanced','R3_AB_balanced','P3_AB_balanced']
VALVES = ['u1A','u1B','u2A','u2B']
LOCAL = [1,6,8,3]
MAIN = [4,9,9,4]
COLORS = ['#777777','#000000','#56B4E9','#CC79A7','#E69F00','#0072B2','#009E73','#D55E00','#882255']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate(run, replay):
    config=read(run/'config.json'); state=read(run/'state.json')
    assert state['fits']==8 and not state['failures'] and not config['smoke']
    assert config['arms']==list(ARMS) and config['training_windows']==20371
    assert config['data_sha256']==sha256(OUT_DEFAULT)
    assert config['parent_sha256']==sha256(PARENT)
    sources={name:sha256(HERE/name)==digest for name,digest in config['source_sha256'].items()}
    assert all(sources.values()), sources
    fit_rows=[]
    for arm in ARMS:
        folder=run/'fits/seed11'/arm; fit=read(folder/'fit.json')
        logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        assert len(logs)==fit['epochs_run'] and fit['updates']==160*len(logs)
        assert fit['training_windows']==20371 and fit['supervision_horizon']==128
        assert fit['reached_validation_plateau'] and fit['final_lr']==.0001
        last=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
        assert last['epoch']==fit['epochs_run'] and last['best_epochs']==fit['best_epochs']
        assert logs[-1]['stale_at_current_lr']>=6
        for choice,epoch in fit['best_epochs'].items():
            p=folder/f'best_{choice}.pt';ck=torch.load(p,weights_only=True,map_location='cpu')
            result=read(run/'seed11'/f'{arm}_{choice}'/'result.json')
            assert ck['epoch']==epoch==result['checkpoint_epoch']
            assert sha256(p)==result['checkpoint_sha256']
        fit_rows.append({k:fit[k] for k in ('model','epochs_run','updates','stop_reason','best_epochs',
                        'train_seconds','parameters','trainable_parameters')})
    # Recompute all 23 rows from returned prediction arrays, not from ranking text.
    checked=0; max_metric_error=0.
    for folder in (run/'seed11').iterdir():
        if not (folder/'result.json').exists(): continue
        result=read(folder/'result.json')
        with np.load(folder/'predictions.npz') as z:
            assert z['truth'].shape==(256,512,10)
            outputs=z['output_indices'].tolist()
            for mode,sides in result['factual'].items():
                assert np.isfinite(z[mode]).all()
                for side,item in sides.items():
                    if side not in ('A','B') or 'H32' not in item: continue
                    channel=4 if side=='A' else 9
                    error=np.abs(z[mode][:,:,outputs.index(channel)].astype(float)-z['truth'][:,:,channel])
                    for H in (32,128,512):
                        max_metric_error=max(max_metric_error,abs(float(error[:,:H].mean())-item[f'H{H}']['main_mae_C']))
                    max_metric_error=max(max_metric_error,abs(float(error[:,128:].mean())-item['tail_129_512']['main_mae_C']))
            checked+=1
    assert checked==23 and max_metric_error<1e-10
    result=dict(returned_commit='187673b',trained_commit=config['git_commit'],source_hashes_match=sources,
        fits=fit_rows,prediction_rows_checked=checked,maximum_recomputed_mae_difference_C=max_metric_error,
        data_sha256=config['data_sha256'],single_seed=True,blind_test=False)
    if replay:
        torch.set_num_threads(1); ids=np.array([0,127,255]); replay_rows=[]
        with np.load(OUT_DEFAULT) as z:
            mean,scale=z['mean'],z['scale']
            h,u,d=[torch.tensor(z[f'{key}_evaluation'][ids]) for key in ('hist30','future_act','future_bnd')]
        for name in DEPLOY:
            arm='G2' if name=='D_fixed' else name.rsplit('_AB_',1)[0] if '_AB_' in name else name.rsplit('_',1)[0]
            model=make_model(arm,mean,scale,11,PARENT).eval()
            if name!='D_fixed':
                choice=name[len(arm)+1:]
                model.load_state_dict(torch.load(run/'fits/seed11'/arm/f'best_{choice}.pt',weights_only=True)['model'])
            with np.load(run/'seed11'/name/'predictions.npz') as z,torch.no_grad():
                for mode in ('native','block_joint_refresh' if len(model.output_indices)==10 else 'block_context_fixed'):
                    difference=np.abs(model.forecast(h,u,d,mode).numpy()-z[mode][ids])
                    replay_rows.append(dict(model=name,mode=mode,max_abs_C=float(difference.max()),mean_abs_C=float(difference.mean())))
        result['replay']=dict(local_torch=str(torch.__version__),returned_torch=config['torch_version'],
            positions=ids.tolist(),rows=replay_rows,interpretation='Cross-version CPU float32 differences disclosed separately from exact artifact hashes')
    return result


def factual(run):
    rows=[]
    for name in DEPLOY:
        result=read(run/'seed11'/name/'result.json')
        for mode,sides in result['factual'].items():
            for side,item in sides.items():
                if side not in ('A','B') or 'H32' not in item: continue
                rows.append(dict(model=name,mode=mode,side=side,epoch=result['checkpoint_epoch'],
                    **{f'H{H}':item[f'H{H}']['main_mae_C'] for H in (32,128,512)},
                    tail=item['tail_129_512']['main_mae_C'],delta60=item['delta60'],
                    groups={g:{'n':v['windows'],**{f'H{H}':v[f'H{H}']['main_mae_C'] for H in (32,128,512)}}
                            for g,v in item['groups'].items() if 'H128' in v}))
    return rows


def analyze_responses(run):
    rows=[]; curves={}; catalogs={}; arrays={}
    with np.load(ROOT/'ssm_dynamic_20260922/load_context.npz') as z:
        with np.load(OUT_DEFAULT) as pack:
            assert np.array_equal(z['times'],pack['evaluation_time'])
        load=z['load_MW']
    for name in DEPLOY:
        folder=run/'seed11'/name; catalogs[name]=read(folder/'response_catalog.json')
        with np.load(folder/'responses.npz') as z: arrays[name]={k:z[k] for k in z.files}
        for case in catalogs[name]:
            if not case['all_perturbed_inputs_supplied'] or np.count_nonzero(case['vector'])!=1: continue
            valve=int(np.flatnonzero(case['vector'])[0]); outputs=case['output_indices']
            if MAIN[valve] not in outputs: continue
            raw=arrays[name][case['key']];pos=np.array(case['window_positions']);valid=np.array(case['valid'])
            assert raw.shape==(len(pos),case['horizon'],len(outputs))
            assert np.isfinite(raw[valid]).all() and np.isnan(raw[~valid]).all()
            if case['onset']: assert np.max(np.abs(raw[valid,:case['onset']]),initial=0)<1e-9
            groups=classify_load(load[pos,case['onset']:])
            # Summaries for every shape and timing, with group-specific counts.
            for group in ('all','rise','fall','vary','steady'):
                good=valid & (np.ones(len(pos),bool) if group=='all' else groups==group)
                row=dict(model=name,key=case['key'],valve=VALVES[valve],mode=case['mode'],
                    protocol=case['boundary_action_protocol'],shape=case['shape'],dose_pp=case['dose']*100,
                    onset=case['onset'],burn=case['burn'],group=group,n=int(good.sum()))
                if good.any():
                    a=raw[good,case['onset']:]; b=a[:,:,outputs.index(MAIN[valve])]
                    sign=np.sign(case['dose']); signed=b*sign; avg=b.mean(0)
                    row.update(early60_C=float(b[:,:6].mean()),mean_curve_min_C=float(avg.min()),
                        mean_individual_min_C=float(b.min(1).mean()),tail60_C=float(b[:,-6:].mean()),
                        desired_peak_C=float(np.maximum(-signed.min(1),0).mean()),
                        reverse_area_Cs=float(np.maximum(signed,0).sum(1).mean()*10),
                        desired_area_Cs=float(np.maximum(-signed,0).sum(1).mean()*10),
                        sign_aligned_tail_C=float(signed[:,-6:].mean()),
                        correct_tail_fraction=float((signed[:,-6:].mean(1)<-1e-4).mean()),
                        local_180_C=float((a[:,:18,outputs.index(LOCAL[valve])]-a[:,:18,outputs.index(LOCAL[valve]-1)]).mean()))
                rows.append(row)
            if case['shape']=='step' and case['onset']==0 and case['dose']==.03 and case['boundary_action_protocol']=='held_both':
                curves[(name,valve,case['mode'])]=(raw[valid],outputs)
    return rows,curves,catalogs,arrays


def event_comparisons(run):
    rows=[]
    with np.load(ROOT/'bilateral_events_reviewed_20260923/event_responses.npz') as source:
        for name in DEPLOY:
            meta=read(run/'seed11'/name/'natural_events.json')
            with np.load(run/'seed11'/name/'natural_events.npz') as z:
                outputs=z['output_indices'].tolist()
                for v,valve in enumerate(VALVES):
                    if MAIN[v] not in outputs or not meta['groups'][valve]['action_supplied']: continue
                    a=z[valve+'_response'];b=z[valve+'_matched_observational_difference']
                    assert np.array_equal(z[valve+'_epoch'],source[valve+'_event_epoch'])
                    local,up,main=[outputs.index(i) for i in (LOCAL[v],LOCAL[v]-1,MAIN[v])]
                    for direction,good in (('opening',source[valve+'_dose']>0),('closing',source[valve+'_dose']<0)):
                        rows.append(dict(model=name,valve=valve,direction=direction,n=int(good.sum()),
                            model_local180_C=float((a[good,:18,local]-a[good,:18,up]).mean()),
                            observed_local180_C=float((b[good,:18,local]-b[good,:18,up]).mean()),
                            model_main180_1280_C=float(a[good,18:,main].mean()),
                            observed_main180_1280_C=float(b[good,18:,main].mean())))
    return rows


def plot(rows,curves,catalogs,arrays,out):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    color=dict(zip(DEPLOY,COLORS))
    fig,axes=plt.subplots(2,4,figsize=(14,6),sharex=True,layout='constrained')
    for v in range(4):
        for name in DEPLOY:
            key=(name,v,'native')
            if key not in curves: continue
            a,outputs=curves[key];t=(np.arange(a.shape[1])+1)/6
            local=a[:,:,outputs.index(LOCAL[v])]-a[:,:,outputs.index(LOCAL[v]-1)]
            main=a[:,:,outputs.index(MAIN[v])]
            for ax,y in zip(axes[:,v],(local.mean(0),main.mean(0))):
                ax.plot(t,y,color=color[name],label=name.replace('_AB_balanced','').replace('_balanced',''),lw=1.5)
                ax.axhline(0,color='#aaaaaa',lw=.6);ax.set_xlabel('Minutes after +3pp step')
        axes[0,v].set_title(f'{VALVES[v]}: local post-minus-pre spray')
        axes[1,v].set_title(f'{VALVES[v]}: own-side main steam')
    axes[0,0].set_ylabel('Predicted difference (°C)');axes[1,0].set_ylabel('Predicted difference (°C)')
    handles={}
    for ax in axes.flat:
        h,l=ax.get_legend_handles_labels(); handles.update(zip(l,h))
    fig.legend(handles.values(),handles.keys(),loc='outside lower center',ncol=9,fontsize=8)
    fig.suptitle('Same 32 origins; native forecast; all other inputs held. Model responses, not intervention truth.')
    fig.savefig(out/'response_paths.png',dpi=180);fig.savefig(out/'response_paths.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for side,ax in zip(('A','B'),axes):
        names=[n for n in DEPLOY if any(r['model']==n and r['side']==side and r['mode']=='native' for r in rows)]
        for j,(label,key) in enumerate((('Native H128','H128'),('Native H512','H512'),('Block H512','block'))):
            values=[]
            for name in names:
                mode='native' if key!='block' else 'block_joint_refresh' if '_AB_' in name else 'block_context_fixed'
                r=next(r for r in rows if r['model']==name and r['mode']==mode and r['side']==side)
                values.append(r['H512' if key=='block' else key])
            ax.bar(np.arange(len(names))+(j-1)*.24,values,.24,label=label,color=['#0072B2','#009E73','#E69F00'][j])
        ax.set_xticks(np.arange(len(names)),[n.replace('_AB_balanced','').replace('_balanced','') for n in names],rotation=35,ha='right')
        ax.set_title(f'Side {side}: main-steam MAE (256 origins)');ax.set_ylabel('MAE (°C)');ax.legend(fontsize=8)
    fig.savefig(out/'factual_accuracy.png',dpi=180);fig.savefig(out/'factual_accuracy.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for name,ax in zip(('S3_AB_balanced','R3_AB_balanced','P3_AB_balanced'),axes):
        matrix=[]
        for v in range(4):
            a,_=curves[(name,v,'native')];matrix.append(a[:,-6:].mean((0,1)))
        matrix=np.array(matrix);im=ax.imshow(matrix,cmap='RdBu_r',vmin=-2.5,vmax=2.5,aspect='auto')
        ax.set_xticks(range(10),[f'{s}{i}' for s in ('A','B') for i in range(1,6)],rotation=45)
        ax.set_yticks(range(4),VALVES);ax.set_title(name.replace('_AB_balanced','')+' tail response (°C)')
        for i in range(4):
            for j in range(10):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=axes,label='Temperature difference (°C)',shrink=.8)
    fig.suptitle('+3pp held step, last60s of H512: upstream and opposite-side responses remain visible')
    fig.savefig(out/'response_routing.png',dpi=180);fig.savefig(out/'response_routing.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,3.5),layout='constrained')
    for v,ax in enumerate(axes):
        for name in ('S3_AB_balanced','R3_AB_balanced','P3_AB_balanced'):
            for mode in ('native','block_joint_refresh'):
                if (name,v,mode) not in curves:continue
                a,outputs=curves[(name,v,mode)];y=a[:,:,outputs.index(MAIN[v])].mean(0)
                ax.plot((np.arange(len(y))+1)/6,y,color=color[name],ls='-' if mode=='native' else '--',label=name[:2]+'/'+mode.split('_')[0])
        ax.axhline(0,color='#aaa',lw=.5);ax.set_title(VALVES[v]);ax.set_xlabel('Minutes');ax.set_ylabel('Own main ΔT (°C)')
    axes[0].legend(fontsize=7);fig.suptitle('Same trained weights: native vs block refresh. R3 aliases; P3 action differences identical.')
    fig.savefig(out/'response_rollout.png',dpi=180);fig.savefig(out/'response_rollout.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'bilateral33_seed11')
    p.add_argument('--out',type=Path,default=ROOT/'bilateral_review_20260923')
    p.add_argument('--replay',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    accepted=validate(args.run,args.replay);save_json(args.out/'acceptance.json',accepted)
    fact=factual(args.run);save_json(args.out/'factual_summary.json',fact)
    responses,curves,catalogs,arrays=analyze_responses(args.run)
    save_json(args.out/'response_summary.json',responses);save_json(args.out/'event_comparison.json',event_comparisons(args.run))
    plot(fact,curves,catalogs,arrays,args.out)
    print(json.dumps(dict(fits=len(accepted['fits']),prediction_rows=accepted['prediction_rows_checked'],
        max_mae_error=accepted['maximum_recomputed_mae_difference_C'],response_rows=len(responses),output=str(args.out))))


if __name__=='__main__':main()
