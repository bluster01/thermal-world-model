"""Read-only source/artifact audit and independent saved-array statistics.

No model training, new checkpoint selection, test access, or plant inference.
Run from the repository root; output is separate from immutable Linux bundles.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.fmts_m7fusion_20260913.audit import audit as audit_m7
from experiments.fmts_m7fusion_20260913.audit import close_tree
from experiments.fmts_greybox_norew_20260913.audit import audit as audit_gnr

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
PARENT = ROOT / 'results/fmts_mainsteam_20260911/linux_full_v02'
M7 = ROOT / 'results/fmts_m7fusion_20260913/linux_full_m7r1'
GNR = ROOT / 'results/fmts_greybox_norew_20260913/linux_full_gnr1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as f:
        a = {k: f[k].copy() for k in f.files}
    assert all(np.isfinite(x).all() for x in a.values())
    return a


def blocks(a, days):
    return np.stack([a[days == d].mean(0) for d in np.unique(days)])


def interval(a, days):
    b = blocks(a, days)
    rng = np.random.default_rng(11000)
    samples = b[rng.integers(len(b), size=(1000, len(b)))].mean(1)
    return dict(mean=b.mean(0).tolist(), ci95=np.quantile(samples, [.025, .975], axis=0).tolist(),
                days=len(b), degenerate_single_day=len(b) == 1)


def health(p):
    a = arrays(p)
    f = a['correction_fraction'].astype(float)
    assert np.allclose(np.tanh(a['raw']), f, atol=1e-6)
    assert np.allclose(1-f*f, a['tanh_derivative'], atol=1e-6)
    return dict(n=len(f), mean=f.mean(0).tolist(), std=f.std(0).tolist(),
                saturation_fraction=(np.abs(f) > .99).mean(0).tolist(),
                derivative_mean=a['tanh_derivative'].mean(0).tolist(),
                reversal_abs_change=np.abs(f-a['reversed_history_fraction']).mean(0).tolist())


def main():
    local = dict(M7R1=audit_m7(M7, PARENT), GNR1=audit_gnr(GNR, PARENT))
    assert local['M7R1']['artifacts_complete'] and local['GNR1']['runs_checked'] == 3
    linux = {name: read(p/'audit.json') for name, p in [('M7R1', M7), ('GNR1', GNR)]}
    for name, a in linux.items():
        assert a['complete'] and a['runs_checked'] == 3 and a['array_sets_replayed'] == 6
        assert not a['failed_runs']
    assert linux['M7R1']['health_array_sets_replayed'] == 9
    parent_id = read(PARENT/'identity.json')
    ids, rows, bank = {}, [], {}
    for root in [PARENT, M7, GNR]:
        identity, summary = read(root/'identity.json'), read(root/'summary.json')
        ids[root.name] = {k: identity[k] for k in ['git_head','record','mapping','properties','indices','device']}
        assert not identity['smoke'] and not summary['locked_test_evaluated']
        assert digest(root/'indices.npz') == parent_id['indices']
        for key in ['record','mapping','properties','indices','normalization_mean','normalization_std']:
            assert identity[key] == parent_id[key], key
        for name, sha in {**identity['sources'], **identity.get('supplement_sources', {})}.items():
            relative = ('experiments/fmts_mainsteam_20260911/'+Path(name).name
                        if name.startswith('/') else name)
            assert digest(ROOT/relative) == sha, name
        indices = arrays(root/'indices.npz')
        for report in summary['runs']:
            arm, seed = report['arm'], report['seed']
            folder = root/f'{arm}_seed{seed}'
            assert report['status'] == 'complete' and read(folder/'report.json') == report
            for filename in ['prediction.npz','response.npz','best.pt']:
                field = 'checkpoint_sha256' if filename == 'best.pt' else filename.split('.')[0]+'_sha256'
                assert digest(folder/filename) == report[field]
            payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
            assert payload['identity_sha256'] == digest(root/'identity.json')
            pred, response = arrays(folder/'prediction.npz'), arrays(folder/'response.npz')
            assert np.array_equal(pred['starts'], indices['validation'])
            assert np.array_equal(response['starts'], indices['response'])
            assert pred['prediction'].shape == (256,18) and response['base'].shape == (64,18)
            err = np.abs(pred['prediction'].astype(float)-pred['target'])
            curve = blocks(err.cumsum(1)/np.arange(1,19), pred['days']).mean(0)
            assert np.allclose(curve, report['cumulative_mae'], atol=1e-6)
            ledger = [json.loads(x) for x in (folder/'ledger.jsonl').read_text().splitlines()]
            best = min(ledger, key=lambda x: x['validation_mainsteam_mae'])
            assert best['step'] == report['best_step'] == payload['step']
            assert np.isclose(curve[-1], best['validation_mainsteam_mae'], atol=1e-6)
            if root != PARENT:
                assert [x['step'] for x in ledger] == list(range(200,24001,200))
                assert report['updates'] == 24000 and report['stop_reason'] == 'cap'
                assert payload['state_dict']['base.transition.raw.aW1'].item() == -30
                assert payload['state_dict']['base.transition.raw.aW2'].item() == -30
            row = dict(arm=arm, seed=seed, H18=float(curve[-1]), prediction_curve=curve.tolist(),
                       best_step=best['step'], updates=report['updates'], stop_reason=report['stop_reason'],
                       ledger_sha256=digest(folder/'ledger.jsonl'),
                       best_minus_first_mae=float(curve[-1]-ledger[0]['validation_mainsteam_mae']),
                       last_minus_best_mae=float(ledger[-1]['validation_mainsteam_mae']-curve[-1]),
                       prediction_days=len(np.unique(pred['days'])), response={})
            for v in [1,2]:
                delta = response[f'valve{v}_prediction'].astype(float)-response['base']
                ci = interval(delta, response['days'])
                saved = summary['response'][folder.name][f'valve{v}']
                assert np.allclose(ci['mean'], saved['curve'], atol=1e-6)
                assert np.allclose(ci['ci95'], saved['day_block_interval']['ci95'], atol=1e-6)
                row['response'][str(v)] = dict(curve=ci['mean'], H18_ci95=[x[-1] for x in ci['ci95']],
                    endpoint_negative_windows=int((delta[:,-1]<0).sum()),
                    endpoint_zero_windows=int((delta[:,-1]==0).sum()),
                    all_times_positive_cells=int((delta>0).sum()),
                    unsupported_windows=int((~response[f'valve{v}_support'].all(1)).sum()),
                    dose_min=float(response[f'valve{v}_dose'].min()), dose_max=float(response[f'valve{v}_dose'].max()))
            if (folder/'observer_health.npz').exists():
                row['health'] = health(folder/'observer_health.npz')
                gradients = np.array([x['observer_gradient_l2_before_clip'] for x in ledger])
                assert np.isfinite(gradients).all()
                row['logged_observer_gradients'] = dict(min=float(gradients.min()), max=float(gradients.max()),
                    zero_count=int((gradients == 0).sum()), above_global_clip_threshold=int((gradients>10).sum()))
            rows.append(row)
            bank[(arm,seed)] = (pred,response,err.mean(1))
    pairs = []
    for root, candidate, field in [(M7,'fusion_m7var_norew','paired_H18_mae'),
                                  (GNR,'greybox_steady_none_norew','paired_cumulative_H18_mae')]:
        for saved in read(root/'comparison.json')['pairs']:
            seed, reference = saved['seed'], saved['reference']
            new, nr, ne = bank[(candidate,seed)]
            old, oldr, oe = bank[(reference,seed)]
            for key in ['starts','days','target','persistence']:
                assert np.array_equal(new[key], old[key]), key
            for key in ['starts','days','valve1_support','valve2_support','valve1_dose','valve2_dose']:
                assert np.array_equal(nr[key], oldr[key]), key
            ci = interval(ne-oe, new['days'])
            assert close_tree(ci, saved[field])
            if root == GNR:
                for v in [1,2]:
                    delta = (nr[f'valve{v}_prediction'].astype(float)-nr['base']
                             -oldr[f'valve{v}_prediction'].astype(float)+oldr['base'])
                    assert close_tree(interval(delta,nr['days']),saved['response'][str(v)]['new_minus_reference'])
            pairs.append(dict(candidate=candidate,reference=reference,seed=seed,paired_H18=ci))
    stats = {}
    for arm in sorted({x['arm'] for x in rows}):
        rr = [x for x in rows if x['arm']==arm]
        vv = np.array([x['H18'] for x in rr])
        stats[arm] = dict(H18_mean=float(vv.mean()),seed_sd=float(vv.std(ddof=1)),
            H18_seeds=vv.tolist(),valve_H18_mean=[float(np.mean([x['response'][str(v)]['curve'][-1] for x in rr])) for v in [1,2]])
    changes = [x['paired_H18']['mean'] for x in pairs if x['candidate']=='fusion_m7var_norew' and x['reference']=='fusion_gru_norew']
    eligible = sum(x<0 for x in changes)>=2 and np.mean(changes)<=0
    assert bool(eligible) == read(M7/'summary.json')['mae_advance_eligible']
    parent_health = {p.stem:health(p) for p in sorted((M7/'parent_observer_health').glob('*.npz'))}
    result = dict(audit_date='2026-09-14', result_commit='15757e8', identities=ids,
        scope='Independent local source/artifact and saved-array recomputation; NO local plant inference replay',
        linux_replay_records=linux, local_registered_artifact_audits=local,
        caveat='GNR1 local complete flag is artifact completeness, not replay; local replay counters remain zero.',
        arm_summary=stats, runs=rows, pairs=pairs, parent_observer_health=parent_health,
        m7_paired_wins_vs_gru=sum(x<0 for x in changes), m7_mae_eligible=bool(eligible),
        retained_hybrid='fusion_gru_norew', plant_response_truth=False, locked_test_evaluated=False,
        additional_training_authorized=False)
    (DEST/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(arm_summary=stats,m7_wins=sum(x<0 for x in changes),audit_saved=str(DEST/'audit.json')),indent=2))


if __name__ == '__main__':
    main()
