"""Independent saved-array audit and figure source export. No model runs or training.

Primary estimands follow frozen v0.2: equal day, then equal seed. Bootstrap
resamples 13 day blocks, not windows/seeds, with the registered RNG seed 11000.
Float64 reductions are independent of the runner's float32 reductions.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'results/fmts_mainsteam_20260911/linux_full_v02'
ARMS = ['blackbox_itransformer', 'greybox_steady_none',
        'fusion_gru_norew', 'fusion_token_xattn_norew']


def day_mean(x, days):
    return np.stack([x[days == d].mean(axis=0) for d in np.unique(days)]).mean(axis=0)


def interval(x, days):
    blocks = np.stack([x[days == d].mean(axis=0) for d in np.unique(days)])
    rng = np.random.default_rng(11000)
    boot = blocks[rng.integers(len(blocks), size=(1000, len(blocks)))].mean(axis=1)
    return np.quantile(boot, [.025, .975], axis=0).tolist()


def save_csv(name, rows):
    with (HERE / 'source_data' / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    (HERE / 'source_data').mkdir(parents=True, exist_ok=True)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    identity = json.loads((SOURCE / 'identity.json').read_text())
    assert not identity['smoke'] and not summary['locked_test_evaluated']
    assert identity['protocol']['protocol_version'] == '0.2'
    hashes = {}
    for filename, digest in identity['sources'].items():
        local = ROOT / (filename if not filename.startswith('/') else
                        'experiments/fmts_mainsteam_20260911/' + Path(filename).name)
        assert hashlib.sha256(local.read_bytes()).hexdigest() == digest, filename
    raw, response, records, prediction_rows, response_rows = {}, {}, [], [], []
    indices = np.load(SOURCE / 'indices.npz')
    assert hashlib.sha256((SOURCE/'indices.npz').read_bytes()).hexdigest() == identity['indices']
    for arm in ARMS:
        for seed in range(3):
            key = f'{arm}_seed{seed}'
            folder = SOURCE / key
            report = json.loads((folder/'report.json').read_text())
            assert report['status'] == 'complete' and report['identity_gate']
            for name, field in [('prediction.npz','prediction_sha256'),
                                ('response.npz','response_sha256'), ('best.pt','checkpoint_sha256')]:
                digest = hashlib.sha256((folder/name).read_bytes()).hexdigest()
                assert digest == report[field], (key, name)
                hashes[f'{key}/{name}'] = digest
            with np.load(folder/'prediction.npz') as f:
                a = {k:f[k].copy() for k in f.files}
            with np.load(folder/'response.npz') as f:
                r = {k:f[k].copy() for k in f.files}
            assert np.array_equal(a['starts'], indices['validation'])
            assert np.array_equal(r['starts'], indices['response'])
            assert a['prediction'].shape == (256,18) and r['base'].shape == (64,18)
            for data in (a,r):
                assert all(np.isfinite(v).all() for v in data.values())
            if raw:
                other = next(iter(raw.values()))
                for k in ['starts','days','target','persistence']:
                    assert np.array_equal(a[k],other[k]), (key,k)
                other = next(iter(response.values()))
                for k in ['starts','days','valve1_dose','valve2_dose','valve1_support','valve2_support']:
                    assert np.array_equal(r[k],other[k]), (key,k)
            raw[key], response[key] = a,r
            errors = np.abs(a['prediction'].astype('float64')-a['target'])
            cumulative = np.cumsum(errors,axis=1)/np.arange(1,19)
            curve = day_mean(cumulative,a['days'])
            np.testing.assert_allclose(curve,report['cumulative_mae'],rtol=2e-6,atol=2e-7)
            single = day_mean(errors,a['days'])
            persistence = day_mean(np.abs(a['persistence'].astype('float64')-a['target']).cumsum(1)/np.arange(1,19),a['days'])
            rec = dict(arm=arm,seed=seed,H1=float(curve[0]),H6=float(curve[5]),H18=float(curve[-1]),
                       validation_days=len(np.unique(a['days'])),response_days=len(np.unique(r['days'])),
                       updates=report['updates'],best_step=report['best_step'],stop_reason=report['stop_reason'],
                       active_parameters=report['parameters_with_gradients'])
            for h in range(18):
                prediction_rows.append(dict(arm=arm,seed=seed,horizon=h+1,seconds=10*(h+1),
                    cumulative_mae=curve[h],single_step_mae=single[h],persistence_mae=persistence[h]))
            for valve in (1,2):
                delta = r[f'valve{valve}_prediction'].astype('float64')-r['base']
                mask = r[f'valve{valve}_support'].all(axis=1)
                c = day_mean(delta,r['days'])
                ci = np.asarray(interval(delta,r['days']))
                np.testing.assert_allclose(c,summary['response'][key][f'valve{valve}']['curve'],atol=2e-7,rtol=2e-6)
                dose = r[f'valve{valve}_dose']
                assert np.allclose(dose,.05,atol=2e-8,rtol=0)
                rec.update({f'valve{valve}_H18':float(c[-1]),f'valve{valve}_unsupported':int((~mask).sum()),
                            f'valve{valve}_H18_ci95':ci[:,-1].tolist()})
                for h in range(18):
                    response_rows.append(dict(arm=arm,seed=seed,valve=valve,horizon=h+1,seconds=10*(h+1),
                        response=c[h],day_ci_low=ci[0,h],day_ci_high=ci[1,h],n_windows=len(mask),
                        unsupported=int((~mask).sum()),dose_min=float(dose.min()),dose_max=float(dose.max())))
            records.append(rec)
    pairs = []
    for candidate,baseline in itertools.combinations(ARMS,2):
        for seed in range(3):
            a,b = raw[f'{candidate}_seed{seed}'],raw[f'{baseline}_seed{seed}']
            x = (np.abs(a['prediction'].astype('float64')-a['target']).mean(1)
                 -np.abs(b['prediction'].astype('float64')-b['target']).mean(1))
            lo,hi = interval(x,a['days'])
            pairs.append(dict(candidate=candidate,baseline=baseline,seed=seed,
                              difference=day_mean(x,a['days']),ci95_low=lo,ci95_high=hi,days=13))
    aggregate = {}
    for arm in ARMS:
        group = [r for r in records if r['arm']==arm]
        aggregate[arm] = {field:dict(mean=float(np.mean([r[field] for r in group])),
                                    sample_std=float(np.std([r[field] for r in group],ddof=1)))
                           for field in ['H1','H6','H18','valve1_H18','valve2_H18']}
    get = lambda arm: aggregate[arm]['H18']['mean']
    gru, bb, grey, token = [get(a) for a in [ARMS[2],ARMS[0],ARMS[1],ARMS[3]]]
    wins = sum(rawr['H18'] < next(r['H18'] for r in records if r['arm']==ARMS[2] and r['seed']==rawr['seed'])
               for rawr in records if rawr['arm']==ARMS[3])
    selected = ARMS[3] if wins>=2 and token<=gru else ARMS[2]
    assert selected==summary['selected_hybrid'] and wins==0
    output = dict(status='SAVED_ARRAY_AUDIT_PASSED',protocol='0.2',returned_commit='399a60ce71905ad5236ae2692fe2c6021da8ffd1',
                  executed_commit=identity['git_head'],inputs={k:identity[k] for k in ['record','mapping','properties','indices']},
                  source_hashes_checked=len(identity['sources']),artifact_hashes=hashes,
                  checkpoint_identity_audit='existing audit.py: 12 passed (separately executed)',
                  locally_recomputed_array_sets=24,locally_model_replayed_array_sets=0,
                  linux_claimed_replayed_array_sets=24,linux_replay_evidence='return commit message; no standalone audit.json returned',
                  selected_hybrid=selected,token_wins= wins,records=records,aggregate=aggregate,
                  derived=dict(gru_minus_blackbox=gru-bb,relative_prediction_cost_pct=100*(gru/bb-1),
                               gru_reduction_vs_grey_pct=100*(1-gru/grey),token_cost_vs_gru_pct=100*(token/gru-1)),
                  plant_response_truth=False,locked_test_evaluated=False,
                  uncertainty='Seed min-max figure bands are descriptive, not confidence intervals; paired 1000 day bootstrap is exploratory.',
                  selection_caveat='Validation used for checkpoint selection and prior design; not an unbiased held-out estimate.')
    (HERE/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save_csv('prediction.csv',prediction_rows)
    save_csv('response.csv',response_rows)
    save_csv('paired_H18.csv',pairs)
    save_csv('per_seed.csv',records)
    labels = {'blackbox_itransformer':('Black box','纯黑箱'),
              'greybox_steady_none':('Grey box','纯灰箱'),
              'fusion_gru_norew':('Hybrid GRU','GRU 融合'),
              'fusion_token_xattn_norew':('Hybrid token','Token 融合')}
    for lang,idx in [('en',0),('zh',1)]:
        lines = [f"{labels[r['arm']][idx]} / {r['seed']} & {r['H18']:.4f} & "
                 f"${r['valve1_H18']:+.6f}$ & ${r['valve2_H18']:+.6f}$ & {r['best_step']:,} "
                 + r'\\' for r in records]
        header = ('Method / seed & H18 MAE & Valve 1 H18 & Valve 2 H18 & Best update'
                  if lang=='en' else '方法 / 种子 & H18 MAE & 阀 1 H18 & 阀 2 H18 & 最佳更新')
        table = r'\begin{tabular}{lrrrr}'+'\n'+r'\toprule'+'\n'+header+r'\\'+'\n'+r'\midrule'+'\n'
        table += '\n'.join(lines)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
        (HERE/f'per_seed_rows_{lang}.tex').write_text(table,encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['status','selected_hybrid','derived']},indent=2))
    print('All 24 arrays, all source hashes, all pairing and 36 artifact hashes checked; no model execution.')


if __name__=='__main__':
    main()
