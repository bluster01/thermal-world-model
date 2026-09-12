"""Read-only M7R1 provenance, metric, paired comparison and numerical replay audit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from experiments.fmts_mainsteam_20260911.data import RichRecord
from experiments.fmts_mainsteam_20260911.manifests import sha256
from experiments.fmts_mainsteam_20260911.run import evaluate, day_mean
from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.properties import load_grid_properties
from .spec import ARM, PROTOCOL_ID, serialized_spec, specs
from .models import M7Fusion
from .health import observer_health, summarize_health
from .run import ROOT, REGISTRATION, FROZEN, compare, load_parent_model


def require(ok, message):
    if not ok:
        raise FinalWMProtocolError(message)


def close_tree(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(close_tree(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a)==len(b) and all(close_tree(x,y) for x,y in zip(a,b))
    if isinstance(a, (int,float)) and not isinstance(a, bool):
        return bool(np.isclose(a,b,rtol=1e-5,atol=1e-5))
    return a == b


def check_arrays(expected, actual):
    require(set(expected)==set(actual), 'replay array keys differ')
    for key in expected:
        require(np.shape(expected[key])==np.shape(actual[key]), f'replay shape mismatch: {key}')
        if expected[key].dtype.kind in 'biu':
            require(np.array_equal(expected[key], actual[key]), f'replay identity mismatch: {key}')
        else:
            require(np.allclose(expected[key], actual[key], rtol=1e-5,atol=1e-5), f'replay mismatch: {key}')


def audit(out, parent, record_path=None, properties=None, mapping='configs/final_wm/channel_mapping_v2.json', device='cpu'):
    out, parent = Path(out), Path(parent)
    identity = json.loads((out/'identity.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    parent_id = json.loads((parent/'identity.json').read_text())
    require(identity['protocol']==serialized_spec()==json.loads(FROZEN.read_text()), 'protocol mismatch')
    require(identity['frozen_spec_sha256']==sha256(FROZEN), 'frozen snapshot mismatch')
    require(identity['registration_sha256']==sha256(REGISTRATION), 'registration mismatch')
    require(not summary['locked_test_evaluated'], 'test access')
    require(identity['smoke']==parent_id['smoke'], 'smoke status mismatch')
    require(summary['status']==('SMOKE' if identity['smoke'] else 'VALIDATION_EXPLORATORY'), 'result status mismatch')
    for field, name in [('parent_identity_sha256','identity.json'),('parent_summary_sha256','summary.json')]:
        require(identity[field]==sha256(parent/name), 'parent provenance mismatch')
    for key in ['record','mapping','properties','indices']:
        require(identity[key]==parent_id[key], f'parent input identity mismatch: {key}')
    require(sha256(out/'indices.npz')==identity['indices']==sha256(parent/'indices.npz'), 'window identity mismatch')
    for source, digest in {**identity['sources'], **identity['supplement_sources']}.items():
        path = ROOT/(source if not source.startswith('/') else 'experiments/fmts_mainsteam_20260911/'+Path(source).name)
        require(sha256(path)==digest, f'source fingerprint mismatch: {source}')
    seeds = (0,) if identity['smoke'] else (0,1,2)
    require(len(summary['runs'])==len(seeds) and {(r['arm'],r['seed']) for r in summary['runs']}=={(ARM,s) for s in seeds}, 'arm/seed mismatch')
    record = None
    if record_path:
        require(sha256(record_path)==identity['record'] and sha256(mapping)==identity['mapping'], 'replay input mismatch')
        require(identity['smoke'] or (properties and sha256(properties)==identity['properties']), 'same IAPWS grid required')
        record = RichRecord(record_path, mapping)
    with np.load(out/'indices.npz') as f:
        indices = {k:f[k].copy() for k in f.files}
    mean, std = torch.tensor(identity['normalization_mean']), torch.tensor(identity['normalization_std'])
    if record is not None:
        actual_mean, actual_std = record.normalization()
        require(torch.allclose(mean,actual_mean,rtol=1e-6,atol=1e-6) and torch.allclose(std,actual_std,rtol=1e-6,atol=1e-6), 'normalization recomputation mismatch')
    checked = replayed = health_replayed = 0
    for report in summary['runs']:
        if report['status']!='complete':
            continue
        folder = out/f"{ARM}_seed{report['seed']}"
        require(json.loads((folder/'report.json').read_text())==report, 'report/summary mismatch')
        for name, field in [('best.pt','checkpoint_sha256'),('prediction.npz','prediction_sha256'),
                            ('response.npz','response_sha256'),('observer_health.npz','observer_health_sha256'),('ledger.jsonl','ledger_sha256')]:
            require(sha256(folder/name)==report[field], f'artifact hash mismatch: {name}')
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        spec = specs((report['seed'],))[0]
        require(payload['spec']==asdict(spec) and payload['protocol']==serialized_spec(), 'checkpoint protocol mismatch')
        require(payload['identity_sha256']==sha256(out/'identity.json'), 'checkpoint identity mismatch')
        for key, val in [('mean',mean),('std',std)]:
            require(torch.allclose(payload['normalization'][key], val, rtol=1e-6,atol=1e-6), 'checkpoint normalization mismatch')
        model = M7Fusion(spec, mean, std, load_grid_properties(properties) if properties else None).to(device)
        require(payload['model_config']==model.model_metadata(), 'effective M7 model metadata mismatch')
        if record is not None:
            model.load_state_dict(payload['state_dict'], strict=True)
        ledger = [json.loads(line) for line in (folder/'ledger.jsonl').read_text().splitlines()]
        best = min(ledger, key=lambda row: row['validation_mainsteam_mae'])
        require(best['step']==payload['step']==report['best_step'], 'not MAE-selected checkpoint')
        for name, bank in [('prediction','validation'),('response','response'),('observer_health','validation')]:
            with np.load(folder/f'{name}.npz') as f:
                saved = {k:f[k] for k in f.files}
            require(np.array_equal(saved['starts'],indices[bank]), 'unpaired start indices')
            require(all(np.isfinite(v).all() for v in saved.values()), 'nonfinite saved arrays')
            if name=='prediction':
                e = np.abs(saved['prediction'].astype(float)-saved['target'])
                curve = day_mean(e.cumsum(1)/np.arange(1,19), saved['days'])
                require(np.allclose(curve,report['cumulative_mae'],atol=1e-6), 'MAE recomputation mismatch')
                require(np.isclose(curve[-1],report['H18'],atol=1e-6) and np.isclose(curve[-1],best['validation_mainsteam_mae'],atol=1e-6), 'selected MAE mismatch')
            if name=='observer_health':
                require(close_tree(summarize_health(saved),report['observer_health']), 'health summary mismatch')
            if record is not None:
                starts = torch.from_numpy(indices[bank])
                actual = (observer_health(model,record,starts,device) if name=='observer_health'
                          else evaluate(model,record,starts,device,response=name=='response'))
                check_arrays(saved, actual)
                health_replayed += int(name=='observer_health')
                replayed += int(name!='observer_health')
        checked += 1
    parent_health = out/'parent_observer_health'
    require(sha256(parent_health/'summary.json')==summary['parent_health_summary_sha256'], 'parent health summary hash mismatch')
    health_summary = json.loads((parent_health/'summary.json').read_text())
    expected_names = {f'{a}_seed{s}' for a in ('fusion_gru_norew','fusion_token_xattn_norew') for s in seeds}
    require(set(health_summary)==expected_names, 'parent health arm/seed mismatch')
    for name, entry in health_summary.items():
        require(sha256(parent_health/f'{name}.npz')==entry['sha256'], 'parent health artifact mismatch')
        require(sha256(parent/name/'best.pt')==entry['checkpoint_sha256'], 'parent checkpoint changed')
        with np.load(parent_health/f'{name}.npz') as f:
            saved = {k:f[k] for k in f.files}
        require(np.array_equal(saved['starts'],indices['validation']), 'parent health windows mismatch')
        require(close_tree(summarize_health(saved),entry['summary']), 'parent health summary mismatch')
        if record is not None:
            arm, seed = name.rsplit('_seed',1)
            model = load_parent_model(parent,arm,int(seed),mean,std,device)
            check_arrays(saved,observer_health(model,record,torch.from_numpy(indices['validation']),device))
            health_replayed += 1
    require(sha256(out/'comparison.json')==summary['comparison_sha256'], 'comparison hash mismatch')
    comparison = compare(parent,out,summary['runs'])
    require(close_tree(comparison,json.loads((out/'comparison.json').read_text())), 'paired comparison mismatch')
    changes = [p['paired_H18_mae']['mean'] for p in comparison['pairs'] if p['reference']=='fusion_gru_norew']
    eligible = bool(len(changes)==3 and sum(x<0 for x in changes)>=2 and np.mean(changes)<=0) if not identity['smoke'] else None
    require(eligible==summary['mae_advance_eligible'], 'advance rule mismatch')
    failed = [r for r in summary['runs'] if r['status']!='complete']
    return dict(protocol_id=PROTOCOL_ID,runs_checked=checked,array_sets_replayed=replayed,
                health_array_sets_replayed=health_replayed,failed_runs=failed,
                artifacts_complete=checked==len(seeds) and not failed,
                complete=checked==len(seeds) and not failed and replayed==2*len(seeds) and health_replayed==3*len(seeds),
                status=summary['status'],locked_test_evaluated=False,
                scientific_verdict='PENDING_HUMAN_HEALTH_AND_RESPONSE_REVIEW')


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--parent',default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--record')
    ap.add_argument('--properties')
    ap.add_argument('--mapping',default='configs/final_wm/channel_mapping_v2.json')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--save')
    a=ap.parse_args()
    result=audit(a.out,a.parent,a.record,a.properties,a.mapping,a.device)
    if a.save:
        with Path(a.save).open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['complete']:
        raise SystemExit(1)
