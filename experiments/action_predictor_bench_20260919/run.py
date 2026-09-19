"""One sequential Linux entry point. No full fitting is launched on import."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from .data import load_pack, unpack, TRAIN_H
from .evaluation import evaluate_forecasts, evaluate_responses, forecast, gradient_probe, response_comparisons
from .models import TRAINED, Anchored, build

HERE = Path(__file__).resolve().parent


def save_json(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    path = Path(path)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def loss_fn(model, prediction, target):
    weights = prediction.new_tensor([.125, .125, .125, .125, .5])
    return ((((prediction-target) / model.scale[:5]).square()) * weights).sum(-1).mean()


@torch.no_grad()
def selector_score(model, bank, device, batch_size):
    total = 0.
    for i in range(0, len(bank), batch_size):
        h, u, d, y = unpack(bank[i:i+batch_size], device)
        p = forecast(model, h, u, d)
        if not torch.isfinite(p).all(): return float('inf')
        total += float((p[:, :, 4]-y[:, :, 4]).abs().sum())
    return total / (len(bank) * TRAIN_H)


def train(name, seed, data, args, folder):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = build(name, data['mean'], data['scale']).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    started, updates, best = time.perf_counter(), 0, float('inf')
    best_epoch = 0
    with (folder / 'training.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, args.epochs+1):
            # Exact same epoch order for every architecture at a given seed.
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(data['train']))
            model.train()
            accumulated = 0.
            for start in range(0, len(order), args.batch_size):
                positions = order[start:start+args.batch_size]
                h, u, d, y = unpack(data['train'][positions], args.device)
                p = model(h, u[:, :-1], d[:, :-1])
                loss = loss_fn(model, p, y)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                accumulated += float(loss.detach()) * len(positions)
            model.eval()
            score = selector_score(model, data['selector'], args.device, args.batch_size)
            row = dict(epoch=epoch, updates=updates, train_loss=accumulated/len(order),
                       selector_main_mae_C=score, seconds=time.perf_counter()-started)
            if score < best:
                best, best_epoch = score, epoch
                torch.save({'model': model.state_dict(), 'name': name, 'seed': seed, 'epoch': epoch}, folder / 'best.pt')
            log.write(json.dumps(row)+'\n')
            log.flush()
            print(json.dumps({'model': name, 'seed': seed, **row}), flush=True)
    if not np.isfinite(best): raise ValueError('No finite checkpoint')
    torch.save({'model': model.state_dict(), 'name': name, 'seed': seed, 'epoch': args.epochs}, folder / 'last.pt')
    model.load_state_dict(torch.load(folder / 'best.pt', map_location=args.device, weights_only=True)['model'])
    return model.eval(), dict(parameters=sum(p.numel() for p in model.parameters()),
                             train_seconds=time.perf_counter()-started, updates=updates, best_epoch=best_epoch,
                             selector_main_mae_C=best, training_windows=len(data['train']))


def score_model(model, data, args, folder):
    started = time.perf_counter()
    scores, arrays = evaluate_forecasts(model, data['evaluation'], args.device, args.batch_size)
    elapsed = time.perf_counter()-started
    np.savez_compressed(folder / 'forecasts.npz', **arrays)
    rows, curves = evaluate_responses(model, data['evaluation'], args.device,
                                     windows=args.response_windows, limit=4 if args.smoke else None)
    np.savez_compressed(folder / 'responses.npz', **curves)
    save_json(folder / 'response_metrics.json', rows)
    save_json(folder / 'response_comparisons.json', response_comparisons(curves))
    # Float64 finite differences avoid subtraction at ~550C masking tiny gains.
    model.double()
    gradient = gradient_probe(model, data['evaluation'], args.device)
    model.float()
    numerical_failure = any(v.get('finite_window_fraction', 1) < 1 for v in scores.values() if isinstance(v, dict))
    return {'forecast': scores, 'gradient': gradient, 'numerical_failure': numerical_failure, 'forecast_seconds': elapsed,
            'total_evaluation_seconds': time.perf_counter()-started, 'response_scenarios_per_mode': len(curves['case_ids']),
            'response_modes': [k for k in curves if k in ('native', 'block')],
            'native_protocol': 'hybrid: nominal direct blocks + uninterrupted R4 difference' if isinstance(model, Anchored) else 'uninterrupted model state'}


def aggregate(output):
    entries = []
    for path in sorted(output.glob('seed*/**/result.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if r.get('status') != 'complete':
            entries.append({'model': path.parent.name, 'seed': int(path.parents[1].name[4:]), 'status': r.get('status')})
            continue
        f = r['forecast']['block_recorded']
        native = r['forecast'].get('native_recorded', {})
        entry = dict(model=r['name'], seed=r['seed'], status=r['status'], parameters=r['parameters'],
                     train_seconds=r['train_seconds'], evaluation_seconds=r['total_evaluation_seconds'],
                     numerical_failure=r.get('numerical_failure', False))
        for horizon in (32, 128, 512):
            entry[f'block_H{horizon}_mae_C'] = f.get(f'H{horizon}', {}).get('main_mae_C')
            entry[f'native_H{horizon}_mae_C'] = native.get(f'H{horizon}', {}).get('main_mae_C')
        entry['block_tail_33_128_mae_C'] = f.get('tail_33_128', {}).get('main_mae_C')
        entry['block_tail_129_512_mae_C'] = f.get('tail_129_512', {}).get('main_mae_C')
        entry['native_tail_129_512_mae_C'] = native.get('tail_129_512', {}).get('main_mae_C')
        entry['delta60_mae_C'] = f.get('delta_60s', {}).get('mae_C')
        entry['delta60_correlation'] = f.get('delta_60s', {}).get('correlation')
        rows = json.loads((path.parent / 'response_metrics.json').read_text(encoding='utf-8'))
        block = [x for x in rows if x['mode'] == 'block' and x['status'] == 'ok']
        entry['max_pre_onset_response_C'] = max((x['pre_onset_max_abs_C'] for x in block), default=None)
        entry['max_unreachable_response_C'] = max((x.get('unreachable_max_abs_C', 0) for x in block), default=None)
        entry['mean_near_zero_fraction'] = float(np.mean([x['near_zero_fraction'] for x in block])) if block else None
        entries.append(entry)
    save_json(output / 'summary.json', entries)
    if entries:
        columns = list(dict.fromkeys(k for e in entries for k in e))
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, columns); w.writeheader(); w.writerows(entries)
    grouped = {}
    for name in sorted({e['model'] for e in entries}):
        group = [e for e in entries if e['model'] == name and e['status'] == 'complete']
        stats = {'successful_seeds': len(group)}
        if group:
            for key in group[0]:
                values = [e.get(key) for e in group]
                if key != 'seed' and all(isinstance(v, (int, float)) for v in values):
                    stats[key] = {'mean': float(np.mean(values)), 'sample_std': float(np.std(values, ddof=1)) if len(values)>1 else None}
        grouped[name] = stats
    save_json(output / 'seed_summary.json', grouped)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data/screen_A_10pct.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=TRAINED, default=TRAINED)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11])
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--response-windows', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip finished models; restart incomplete model fits from seed')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.response_windows < 1:
        parser.error('Positive budgets required')
    torch.set_num_threads(args.threads)
    data, metadata = load_pack(args.data)
    if args.smoke:
        data['train'], data['selector'], data['evaluation'] = data['train'][:16], data['selector'][:8], data['evaluation'][:4]
        data['evaluation_time'] = data['evaluation_time'][:4]
        args.epochs, args.batch_size, args.response_windows = 1, 8, 2
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('resume', 'data', 'output')}
    config.update(data_sha256=metadata['pack_sha256'], source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch_version=torch.__version__, conditioning='recorded future controls; held-boundary sensitivity separately',
                  selection='best selector H32 main MAE, fixed full epochs; separate chronological reporting windows')
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        # Git commit may change due solely to packaging a checked version.
        keys = set(config)-{'git_commit'}
        if not args.resume or any(previous.get(k) != config[k] for k in keys):
            raise ValueError('Existing output requires --resume with identical config/data/code')
    else:
        save_json(config_path, config)
        save_json(args.output / 'data_metadata.json', metadata)
        np.savez_compressed(args.output / 'evaluation_inputs.npz', bank=data['evaluation'], times=data['evaluation_time'])
    failures = []
    save_json(args.output / 'state.json', {'status': 'running', 'smoke': args.smoke})
    for seed in args.seeds:
        names = ['persistence', *args.models]
        if 'direct' in args.models and 'r4' in args.models: names.append('anchored_hold')
        for name in names:
            folder = args.output / f'seed{seed}' / name
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / 'result.json'
            if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'complete':
                print(f'Skipping completed seed{seed}/{name}', flush=True); continue
            try:
                if name == 'persistence':
                    model = build(name, data['mean'], data['scale']).to(args.device).eval()
                    training = dict(parameters=0, train_seconds=0., updates=0)
                elif name == 'anchored_hold':
                    parts, costs = [], []
                    for component in ('direct', 'r4'):
                        parent = folder.parent / component
                        cost = json.loads((parent / 'result.json').read_text())
                        if cost['status'] != 'complete': raise ValueError('Anchored parent fit failed')
                        m = build(component, data['mean'], data['scale']).to(args.device)
                        m.load_state_dict(torch.load(parent / 'best.pt', map_location=args.device, weights_only=True)['model'])
                        parts.append(m.eval()); costs.append(cost)
                    model = Anchored(*parts).to(args.device).eval()
                    training = dict(parameters=sum(c['parameters'] for c in costs), train_seconds=sum(c['train_seconds'] for c in costs),
                                    updates=0, new_fit=False, component_costs_included=True)
                else:
                    model, training = train(name, seed, data, args, folder)
                result = dict(name=name, seed=seed, status='complete', **training, **score_model(model, data, args, folder))
                save_json(result_path, result)
            except Exception as error:
                save_json(result_path, {'name': name, 'seed': seed, 'status': 'failed', 'error': repr(error)})
                (folder / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                failures.append(f'seed{seed}/{name}')
                traceback.print_exc()
                if args.smoke: raise
            aggregate(args.output)
    save_json(args.output / 'state.json', {'status': 'completed_with_failures' if failures else 'complete', 'failures': failures, 'smoke': args.smoke})
    from .report import report
    report(args.output)
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    main()
