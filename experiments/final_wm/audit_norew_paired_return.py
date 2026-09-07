"""Audit returned tensors, window identities and support masks; no model calls."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.final_wm.replay_norew_pairs import isolated_record, sampled_indices
from src.final_wm.evaluation import WindowMetrics, paired_difference_ci

SOURCE = ROOT/'results/final_wm/norew_paired_replay_20260907'
ORIGINAL = ROOT/'artifacts/final_wm/v07_full_reissue_v1'
OUT = ROOT/'results/final_wm/norew_paired_audit_20260908'
ARMS = ('physics_only','closure_cons','closure_cons_norew')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    summary=read(SOURCE/'summary.json')
    protocol=read(SOURCE/'protocol.json')
    frozen_code=subprocess.check_output(['git','show','db9447b:experiments/final_wm/replay_norew_pairs.py'],cwd=ROOT)
    assert hashlib.sha256(frozen_code).hexdigest()==protocol['script_sha256']
    assert protocol['arms']==list(ARMS) and protocol['seeds']==[0,1,2]
    assert not protocol['training'] and not protocol['leakage_probe'] and not protocol['full_R1_verdict']
    assert not summary['training_run'] and not summary['test_values_accessed']
    assert (SOURCE/'paired_replay_stderr.log').stat().st_size==0
    assert (SOURCE/'paired_replay_stdout.log').read_text().strip().endswith('COMPLETE')
    record,isolation=isolated_record(ORIGINAL/'inputs/canonical_sideA_v2.npz')
    assert isolation==read(SOURCE/'isolation.json')
    assert len(summary['runs'])==9
    runs={r['run']:r for r in summary['runs']}
    assert len(runs)==9
    original_r1=read(ORIGINAL/'sideA/r1_report.json')
    predictions, responses, metrics, masks, pairs = [], [], {}, {}, []
    max_old_response_difference=0.0
    for seed in range(3):
        for arm in ARMS:
            run=f't1_{arm}_seed{seed}'
            report=read(SOURCE/f'{run}.json')
            assert runs[run]==report and report['parameters_unchanged']
            new=torch.load(SOURCE/f'{run}_prediction.pt',weights_only=True,map_location='cpu')
            old=torch.load(ORIGINAL/f'sideA/metrics/{run}.pt',weights_only=True,map_location='cpu')['metrics']
            for key in ('mae','nll','crps','day_ids'):
                assert torch.equal(new[key],old[key]), (run,key,'tensor mismatch')
            assert list(new['mae'].shape)==[256,18]
            anchors=sampled_indices(record,256,96,18,50000+seed)
            assert anchors==report['prediction_indices']
            assert torch.equal(new['day_ids'],record.timestamps[anchors]//86400)
            metrics[(arm,seed)]=WindowMetrics(**new)
            predictions.append({'arm':arm,'seed':seed,'mae_h18_c':float(new['mae'].mean()),'tensor_identity':'EXACT_ALL_FOUR_FIELDS','windows':256})
            assert set(report['responses'])=={'valve1_H18','valve1_H60','valve2_H18','valve2_H60'}
            for key,response in report['responses'].items():
                valve=response['valve_index']
                horizon=response['rollout_steps']
                assert key==f'valve{valve+1}_H{horizon}'
                indices=sampled_indices(record,64,96,1,80000+1000*valve+horizon+seed)
                assert indices==response['first_future_indices']
                ix=torch.tensor(indices)
                history=record.actions[ix[:,None]+torch.arange(-96,0)[None,:]]
                # Recompute the frozen support box directly from source actions,
                # independently of saved mask/count values; no model inference.
                lo=(history.amin(dim=1)-0.05).clamp(0,1)
                hi=(history.amax(dim=1)+0.05).clamp(0,1)
                base=record.actions[ix,None,:].repeat(1,horizon,1)
                step=base.clone()
                step[:,:,valve]=(step[:,:,valve]+0.05).clamp(max=1)
                for actions,mask_key,count_key,rate_key in (
                    (base,'baseline_in_support_mask','baseline_n_unsupported','baseline_support_rate'),
                    (step,'in_support_mask','n_unsupported','support_rate')):
                    expected=((actions>=lo[:,None,:]) & (actions<=hi[:,None,:])).all(dim=-1)
                    actual=torch.tensor(response[mask_key],dtype=torch.bool)
                    assert torch.equal(expected,actual) and list(actual.shape)==[64,horizon]
                    assert int((~actual).sum())==response[count_key]
                    assert abs(float(actual.float().mean())-response[rate_key])<1e-6
                mask_pair=(response['in_support_mask'],response['baseline_in_support_mask'])
                mask_key=(seed,key)
                if mask_key in masks:
                    assert masks[mask_key]==mask_pair
                masks[mask_key]=mask_pair
                assert response['n_days']==len(torch.unique(record.timestamps[ix]//86400))
                assert response['ci_lo_c']<=response['mean_delta_c']<=response['ci_hi_c']
                if arm=='closure_cons_norew':
                    prev=next(r for r in original_r1['reports'] if r['seed']==seed)['directions'][f'valve{valve+1}'][f'H{horizon}']
                    for field in ('mean_delta_c','ci_lo_c','ci_hi_c'):
                        max_old_response_difference=max(max_old_response_difference,abs(prev[field]-response[field]))
                    for field in ('in_support_mask','baseline_in_support_mask','n_unsupported','baseline_n_unsupported'):
                        assert prev[field]==response[field]
                category='negative' if response['ci_hi_c']<0 else ('positive' if response['ci_lo_c']>0 else 'crosses_zero')
                responses.append({'arm':arm,'seed':seed,'valve':valve+1,'horizon':horizon,
                    'mean_delta_c':response['mean_delta_c'],'ci_lo_c':response['ci_lo_c'],'ci_hi_c':response['ci_hi_c'],
                    'ci_category':category,'unsupported_steps':response['n_unsupported'],
                    'baseline_unsupported_steps':response['baseline_n_unsupported'],
                    'unsupported_windows':int((~torch.tensor(response['in_support_mask'])).any(dim=1).sum()),
                    'n_windows':64,'all_steps_supported':response['n_unsupported']==0 and response['baseline_n_unsupported']==0})
    # CI values are recomputed on CPU, allowing 1e-6 only for arithmetic
    # reduction drift, not changing scientific direction/sign thresholds.
    assert len(summary['paired_mae_equal_day_ci'])==6
    for item in summary['paired_mae_equal_day_ci']:
        result=paired_difference_ci(metrics[(item['base'],item['seed'])],metrics[(item['arm'],item['seed'])],horizon=18,metric='mae',seed=item['seed'])
        for key,value in result._asdict().items():
            assert abs(value-item[key])<1e-6,(key,value,item[key])
        pairs.append({**item,'local_recompute':'MATCH_WITHIN_1E-6'})
    compact=[]
    for arm in ARMS:
        r=[r for r in responses if r['arm']==arm]
        m=[p['mae_h18_c'] for p in predictions if p['arm']==arm]
        compact.append({'arm':arm,'three_seed_window_weighted_mae_c':sum(m)/3,
            'negative_CI':sum(x['ci_category']=='negative' for x in r),
            'positive_CI':sum(x['ci_category']=='positive' for x in r),
            'crosses_zero_CI':sum(x['ci_category']=='crosses_zero' for x in r),
            'unsupported_cells':sum(not x['all_steps_supported'] for x in r)})
    audit={'source_commit':'a34481f','execution_commit':'db9447b','script_hash_verified':True,
        'prediction_tensor_identity_runs':9,'response_cells':36,'support_masks_independently_recomputed':True,
        'indices_reconstructed_and_matched':True,'paired_CIs_independently_recomputed':6,
        'max_norew_response_scalar_difference_vs_original_R1':max_old_response_difference,
        'arms':compact,'model_inference_run_in_this_audit':False,'training_run':False,
        'test_numeric_rows_decoded':isolation['test_numeric_rows_decoded'],
        'response_CI_recomputation_limit':'Per-window temperature deltas were not saved; response CIs are executed frozen-function outputs, not independently bootstrapped from raw deltas here',
        'verdict':'RETURN_AUDITED_NO_FULL_R1_UPGRADE'}
    save_csv(OUT/'prediction.csv',predictions)
    save_csv(OUT/'response.csv',responses)
    save_csv(OUT/'paired_mae_ci.csv',pairs)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
