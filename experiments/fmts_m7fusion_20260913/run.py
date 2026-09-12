"""Frozen M7R1 runner: only one new arm; fixed parent windows and read-only diagnostics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import torch
from experiments.fmts_mainsteam_20260911.data import RichRecord
from experiments.fmts_mainsteam_20260911.models import RichFusion
from experiments.fmts_mainsteam_20260911.run import evaluate, summarize, write_json, day_mean, block_interval
from experiments.fmts_mainsteam_20260911.manifests import sha256
from experiments.fmts_mainsteam_20260911.spec import structured_specs
from experiments.fmts_greybox_norew_20260913.run import validate_parent
from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.properties import load_grid_properties
from .models import M7Fusion
from .health import observer_health, summarize_health
from .spec import ARM, PROTOCOL_ID, REFERENCE_ARMS, serialized_spec, specs

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT/'docs/fmts2026/PREREG_M7_FUSION_20260913.md'
FROZEN = Path(__file__).with_name('frozen_spec.json')


def extra_sources():
    return [*sorted(Path(__file__).parent.glob('*.py')),
            ROOT/'src/world_model.py', ROOT/'src/config.py',
            ROOT/'experiments/fmts_greybox_norew_20260913/run.py']


def load_parent_model(parent, arm, seed, mean, std, device):
    folder = parent/f'{arm}_seed{seed}'
    report = json.loads((folder/'report.json').read_text())
    if sha256(folder/'best.pt') != report['checkpoint_sha256']:
        raise FinalWMProtocolError('parent checkpoint hash mismatch')
    payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    if payload['identity_sha256'] != sha256(parent/'identity.json'):
        raise FinalWMProtocolError('parent checkpoint identity mismatch')
    for key, value in [('mean',mean),('std',std)]:
        if not torch.allclose(payload['normalization'][key], value.cpu(), rtol=1e-6, atol=1e-6):
            raise FinalWMProtocolError('parent observer normalization mismatch')
    spec = next(s for s in structured_specs((seed,)) if s.arm == arm)
    # Only observer forward is used; analytic property placeholder is not used.
    model = RichFusion(spec, mean, std).to(device)
    state = {k: v for k, v in payload['state_dict'].items() if k.startswith('base.observer.')}
    model.base.observer.load_state_dict({k.removeprefix('base.observer.'): v for k,v in state.items()})
    return model


def train_one(spec, record, indices, mean, std, properties, device, out, smoke):
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    model = M7Fusion(spec, mean, std, properties).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    generator = torch.Generator().manual_seed(20000+spec.seed)
    batch, every, cap, patience = spec.batch_size, spec.batches_per_epoch, spec.epochs*spec.batches_per_epoch, spec.patience
    if smoke:
        batch, every, cap, patience = 2, 1, 2, 2
    folder = out/f'{ARM}_seed{spec.seed}'
    folder.mkdir()
    best, stale, best_step = float('inf'), 0, -1
    active = set()
    started = time.time()
    with (folder/'ledger.jsonl').open('w', encoding='utf-8') as ledger:
        for step in range(1, cap+1):
            model.train()
            selection = torch.randint(len(indices['train']), (batch,), generator=generator)
            h, ext, a, b, y, _ = record.batch(indices['train'][selection], 0, device)
            prediction = model(h, ext, a, b)
            loss = model.base.observation_nll(prediction.temps_mu, prediction.temps_sigma, y)
            if not torch.isfinite(loss):
                raise FinalWMProtocolError('nonfinite training loss')
            opt.zero_grad()
            loss.backward()
            active.update(n for n,p in model.named_parameters() if p.grad is not None)
            gradient_norm = None
            if step % every == 0:
                gradient_norm = float(torch.stack([p.grad.detach().norm().square()
                    for p in model.base.observer.parameters() if p.grad is not None]).sum().sqrt())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            opt.step()
            if step % every == 0:
                raw = evaluate(model, record, indices['validation'], device)
                score = float(day_mean(np.abs(raw['prediction']-raw['target']).mean(1), raw['days']))
                health = summarize_health(observer_health(model, record, indices['validation'][:32], device))
                entry = dict(step=step, loss=float(loss.detach()), validation_mainsteam_mae=score,
                             observer_gradient_l2_before_clip=gradient_norm, observer_health=health)
                ledger.write(json.dumps(entry, allow_nan=False)+'\n')
                ledger.flush()
                if score < best:
                    best, best_step, stale = score, step, 0
                    torch.save(dict(state_dict=model.state_dict(), arm=ARM, seed=spec.seed,
                                    spec=asdict(spec), model_config=model.model_metadata(),
                                    protocol=serialized_spec(), step=step,
                                    identity_sha256=sha256(out/'identity.json'),
                                    normalization=dict(mean=mean, std=std)), folder/'best.pt')
                else:
                    stale += 1
                if stale >= patience:
                    break
    payload = torch.load(folder/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    raw = evaluate(model, record, indices['validation'], device)
    probe = evaluate(model, record, indices['response'], device, response=True)
    health = observer_health(model, record, indices['validation'], device)
    for name, arrays in [('prediction', raw), ('response', probe), ('observer_health', health)]:
        np.savez_compressed(folder/f'{name}.npz', **arrays)
    errors = np.abs(raw['prediction']-raw['target'])
    curve = day_mean(errors.cumsum(1)/np.arange(1,19), raw['days'])
    persistence = np.abs(raw['persistence']-raw['target'])
    report = dict(arm=ARM, seed=spec.seed, status='complete', updates=step, best_step=best_step,
                  stop_reason='patience' if stale >= patience else 'cap', seconds=time.time()-started,
                  parameters_instantiated=sum(p.numel() for p in model.parameters()),
                  parameters_with_gradients=sum(p.numel() for n,p in model.named_parameters() if n in active),
                  parameter_names_with_gradients=sorted(active), cumulative_mae=curve.tolist(), H18=float(curve[-1]),
                  single_step_mae=day_mean(errors, raw['days']).tolist(), identity_gate=True,
                  persistence_cumulative_mae=day_mean(persistence.cumsum(1)/np.arange(1,19), raw['days']).tolist(),
                  checkpoint_sha256=sha256(folder/'best.pt'), prediction_sha256=sha256(folder/'prediction.npz'),
                  response_sha256=sha256(folder/'response.npz'), observer_health_sha256=sha256(folder/'observer_health.npz'),
                  ledger_sha256=sha256(folder/'ledger.jsonl'),
                  observer_health=summarize_health(health))
    write_json(folder/'report.json', report)
    return report


def compare(parent, out, reports):
    pairs = []
    for report in reports:
        if report['status'] != 'complete':
            continue
        seed = report['seed']
        for arm in REFERENCE_ARMS:
            with np.load(out/f'{ARM}_seed{seed}'/'prediction.npz') as a, np.load(parent/f'{arm}_seed{seed}'/'prediction.npz') as b:
                for key in ['starts','days','target','persistence']:
                    if not np.array_equal(a[key], b[key]):
                        raise FinalWMProtocolError(f'prediction pairing mismatch: {key}')
                diff = np.abs(a['prediction'].astype(float)-a['target']).mean(1)-np.abs(b['prediction'].astype(float)-b['target']).mean(1)
                metric = block_interval(diff, a['days'])
            with np.load(out/f'{ARM}_seed{seed}'/'response.npz') as a, np.load(parent/f'{arm}_seed{seed}'/'response.npz') as b:
                for key in ['starts','days','valve1_support','valve2_support','valve1_dose','valve2_dose']:
                    if not np.array_equal(a[key], b[key]):
                        raise FinalWMProtocolError(f'response pairing mismatch: {key}')
                response = {str(v): block_interval((a[f'valve{v}_prediction'].astype(float)-a['base'])-
                    (b[f'valve{v}_prediction'].astype(float)-b['base']), a['days']) for v in (1,2)}
            pairs.append(dict(candidate=ARM, reference=arm, seed=seed, paired_H18_mae=metric,
                              response_difference=response))
    return dict(pairs=pairs, response_used_for_selection=False, plant_response_truth=False)


def execute(parent, record_path, mapping, properties, out, device='cpu', smoke=False):
    parent, out = Path(parent), Path(out)
    if out.exists():
        raise FinalWMProtocolError('output exists; preserve it, no overwrite or implicit resume')
    if json.loads(FROZEN.read_text()) != serialized_spec():
        raise FinalWMProtocolError('frozen M7 spec changed')
    parent_id = validate_parent(parent, record_path, mapping, properties, smoke)
    record = RichRecord(record_path, mapping)
    with np.load(parent/'indices.npz') as f:
        indices = {k: torch.from_numpy(f[k].copy()) for k in f.files}
    for name, split in [('train',0),('validation',1),('response',1)]:
        if not torch.isin(indices[name], record.candidates(split)).all():
            raise FinalWMProtocolError('parent manifest has ineligible windows')
    mean, std = record.normalization()
    for name, value in [('mean',mean),('std',std)]:
        if not torch.allclose(value, torch.tensor(parent_id[f'normalization_{name}']), rtol=1e-6, atol=1e-6):
            raise FinalWMProtocolError('parent normalization mismatch')
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('* -text\n', encoding='ascii')
    shutil.copyfile(parent/'indices.npz', out/'indices.npz')
    identity = dict(parent_id)
    identity['parent_protocol'] = identity.pop('protocol')
    identity.update(protocol=serialized_spec(), frozen_spec_sha256=sha256(FROZEN),
                    registration_sha256=sha256(REGISTRATION), parent_identity_sha256=sha256(parent/'identity.json'),
                    parent_summary_sha256=sha256(parent/'summary.json'),
                    supplement_sources={p.relative_to(ROOT).as_posix():sha256(p) for p in extra_sources()},
                    git_head=subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
                    python=platform.python_version(), torch=torch.__version__, device=device,
                    cuda_version=torch.version.cuda, matmul_precision=torch.get_float32_matmul_precision(),
                    cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    write_json(out/'identity.json', identity)
    seeds = (0,) if smoke else (0,1,2)
    diagnostic_dir = out/'parent_observer_health'
    diagnostic_dir.mkdir()
    health_reports = {}
    for arm in REFERENCE_ARMS[:2]:
        for seed in seeds:
            name = f'{arm}_seed{seed}'
            model = load_parent_model(parent, arm, seed, mean, std, device)
            raw = observer_health(model, record, indices['validation'], device)
            np.savez_compressed(diagnostic_dir/f'{name}.npz', **raw)
            health_reports[name] = dict(summary=summarize_health(raw), sha256=sha256(diagnostic_dir/f'{name}.npz'),
                                       checkpoint_sha256=sha256(parent/name/'best.pt'))
            del model
    write_json(diagnostic_dir/'summary.json', health_reports)
    reports = []
    for spec in specs(seeds):
        print(f'Starting {ARM} seed{spec.seed}', flush=True)
        try:
            props = load_grid_properties(properties) if properties else None
            report = train_one(spec, record, indices, mean, std, props, device, out, smoke)
        except Exception as exc:
            report = dict(arm=ARM, seed=spec.seed, status='failed', error=str(exc))
            write_json(out/f'{ARM}_seed{spec.seed}_failure.json', report)
        reports.append(report)
        write_json(out/'progress.json', reports)
    summarize(out, reports, smoke)
    comparison = compare(parent, out, reports)
    write_json(out/'comparison.json', comparison)
    summary = json.loads((out/'summary.json').read_text())
    changes = [p['paired_H18_mae']['mean'] for p in comparison['pairs'] if p['reference']=='fusion_gru_norew']
    eligible = bool(len(changes)==3 and sum(x<0 for x in changes)>=2 and np.mean(changes)<=0) if not smoke else None
    summary.update(protocol_id=PROTOCOL_ID, selected_hybrid='PENDING_AUDIT_AND_HEALTH_REVIEW',
                   mae_advance_eligible=eligible, parent_selected_hybrid='fusion_gru_norew',
                   comparison_sha256=sha256(out/'comparison.json'),
                   parent_health_summary_sha256=sha256(diagnostic_dir/'summary.json'),
                   supplement_only=True, scientific_verdict='PENDING_NOT_PLANT_FIDELITY')
    write_json(out/'summary.json', summary)
    if any(r['status']!='complete' for r in reports):
        raise FinalWMProtocolError('failed M7 runs; preserve failure artifacts')
    return reports


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--parent', default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--record', required=True)
    ap.add_argument('--mapping', default='configs/final_wm/channel_mapping_v2.json')
    ap.add_argument('--properties')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    execute(a.parent, a.record, a.mapping, a.properties, a.out, a.device, a.smoke)
