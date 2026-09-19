"""Round2: two zero-fit combinations and four matched-budget continuation arms."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from .data import load_pack, unpack, sha256
from .models import build
from .round2_models import ProtectedReference, response_matching_loss
from .run import HERE, aggregate, loss_fn, save_json, score_model, selector_score

ARMS = ['r4_protected', 'ssm_protected', 'ssm_continue', 'ssm_soft_response', 'ssm_r4_joint', 'ssm_r4_teacher']
NO_FIT = {'r4_protected', 'ssm_protected'}


def parents(args, data):
    models, costs = {}, {}
    for name in ('ssm', 'r4_mlp'):
        folder = args.parents / f'seed{args.seed}' / name
        costs[name] = json.loads((folder / 'result.json').read_text())
        if costs[name]['status'] != 'complete': raise ValueError(f'Incomplete parent {name}')
        model = build(name, data['mean'], data['scale']).to(args.device)
        model.load_state_dict(torch.load(folder / 'best.pt', weights_only=True, map_location=args.device)['model'])
        models[name] = model.eval()
    return models, costs


def initialize(name, source):
    if name in ('ssm_continue', 'ssm_soft_response'):
        return copy.deepcopy(source['ssm'])
    predictor = source['r4_mlp' if name == 'r4_protected' else 'ssm']
    return ProtectedReference(copy.deepcopy(predictor), copy.deepcopy(source['r4_mlp']))


def fit(name, model, source, data, args, folder):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = selector_score(model, data['selector'], args.device, args.batch_size)
    best_epoch, updates = 0, 0
    torch.save({'model': model.state_dict(), 'name': name, 'epoch': 0}, folder / 'best.pt')
    start_time = time.perf_counter()
    with (folder / 'training.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, args.epochs+1):
            # Same order in every arm; starts after original six training epochs.
            order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch+6])).permutation(len(data['train']))
            total, extra_total = 0., 0.
            model.train()
            for start in range(0, len(order), args.batch_size):
                ids = order[start:start+args.batch_size]
                h, u, d, y = unpack(data['train'][ids], args.device)
                u, d = u[:, :-1], d[:, :-1]
                prediction = model(h, u, d)
                factual = loss_fn(model, prediction, y)
                extra = factual.new_zeros(())
                if name == 'ssm_soft_response':
                    extra = response_matching_loss(model, source['r4_mlp'], h, d, updates)
                if name == 'ssm_r4_teacher':
                    with torch.no_grad(): teacher_prediction = source['ssm'](h, u, d)
                    extra = .25 * loss_fn(model, prediction, teacher_prediction)
                loss = factual + extra
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(factual.detach()) * len(ids)
                extra_total += float(extra.detach()) * len(ids)
                updates += 1
            model.eval()
            score = selector_score(model, data['selector'], args.device, args.batch_size)
            row = dict(epoch=epoch, updates=updates, train_loss=total/len(order),
                       extra_loss=extra_total/len(order), selector_main_mae_C=score,
                       seconds=time.perf_counter()-start_time)
            if score < best:
                best, best_epoch = score, epoch
                torch.save({'model': model.state_dict(), 'name': name, 'epoch': epoch}, folder / 'best.pt')
            log.write(json.dumps(row)+'\n'); log.flush()
            print(json.dumps({'arm': name, **row}), flush=True)
    torch.save({'model': model.state_dict(), 'name': name, 'epoch': args.epochs}, folder / 'last.pt')
    model.load_state_dict(torch.load(folder / 'best.pt', weights_only=True, map_location=args.device)['model'])
    return dict(new_train_seconds=time.perf_counter()-start_time, updates=updates,
                best_epoch=best_epoch, selector_main_mae_C=best)


def comparison(args):
    """Direct baseline deltas, with protocols and parent costs visible."""
    if args.smoke:
        (args.output / 'COMPARISON.md').write_text('Smoke only: tiny evaluation windows; do not compare with full parent scores.\n', encoding='utf-8')
        return
    rows = json.loads((args.output / 'summary.json').read_text())
    original = json.loads((args.parents / 'summary.json').read_text())
    selected = [r for r in original if r['seed'] == args.seed and r['model'] in ('ssm', 'r4_mlp')]
    baseline = next(r for r in selected if r['model'] == 'ssm')
    lines = ['# Round2 comparison', '', 'Main-temperature MAE (C). Same data, horizons and selector as round1.',
             'Protected block = nominal reference block32 + continuous R4 carrier. Native = both uninterrupted.',
             'Original SSM/R4 parents: 6 epochs. Continued arms: up to 6 additional epochs; epoch0 can be retained.',
             'Single seed discovery; no blanket accuracy-preservation claim.', '',
             '| Arm | H32 | Block H128 | Block H512 | Native H512 | H32 change vs original SSM |',
             '|---|---:|---:|---:|---:|---:|']
    for r in selected + rows:
        if r['status'] != 'complete': continue
        keys = ('block_H32_mae_C', 'block_H128_mae_C', 'block_H512_mae_C', 'native_H512_mae_C')
        cells = [f'{r[k]:.4f}' for k in keys]
        delta = (r['block_H32_mae_C']/baseline['block_H32_mae_C']-1)*100
        lines.append('| ' + ' | '.join([r['model'], *cells, f'{delta:+.2f}%']) + ' |')
    lines += ['', 'Read DIAGNOSIS.md for per-mode response direction/topology and action-active strata.',
              'Use ssm_continue to separate extra optimization from response constraints; report failures too.']
    (args.output / 'COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', type=Path, default=Path('results/action_predictor_bench_20260919/screen33_seed11'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=HERE / 'data/screen_A_33pct.npz')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--response-windows', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.response_windows, args.threads) < 1 or args.learning_rate <= 0:
        parser.error('Positive budgets and learning rate required')
    torch.set_num_threads(args.threads)
    data, metadata = load_pack(args.data)
    parent_config = json.loads((args.parents / 'config.json').read_text())
    if parent_config['data_sha256'] != metadata['pack_sha256'] or parent_config['smoke']:
        raise ValueError('Round2 requires full-screen parents on the identical pack')
    if args.smoke:
        data['train'], data['selector'], data['evaluation'] = data['train'][:16], data['selector'][:8], data['evaluation'][:4]
        data['evaluation_time'] = data['evaluation_time'][:4]
        args.epochs, args.batch_size, args.response_windows = 1, 8, 2
    source, costs = parents(args, data)
    for model in source.values(): model.requires_grad_(False)
    config = {k: v for k, v in vars(args).items() if k not in ('parents', 'output', 'data', 'resume')}
    config.update(data_sha256=metadata['pack_sha256'],
                  parent_checkpoint_sha256={name: sha256(args.parents / f'seed{args.seed}' / name / 'best.pt') for name in source},
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                  parent_git_commit=parent_config['git_commit'], torch_version=torch.__version__,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  selection='best selector H32 MAE including initialized epoch0; never reporting windows',
                  response_loss='SSM: match frozen R4 responses, normalized by 0.1*training std, weight1',
                  prediction_teacher_weight=.25)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'config.json'
    if path.exists():
        old = json.loads(path.read_text())
        if not args.resume or any(old.get(k) != v for k, v in config.items() if k != 'git_commit'):
            raise ValueError('Existing output needs --resume and identical config/code/parents')
    else:
        save_json(path, config)
    save_json(args.output / 'data_metadata.json', metadata)
    np.savez_compressed(args.output / 'evaluation_inputs.npz', bank=data['evaluation'], times=data['evaluation_time'])
    save_json(args.output / 'state.json', dict(status='running', smoke=args.smoke))
    failures = []
    for name in args.arms:
        folder = args.output / f'seed{args.seed}' / name
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / 'result.json'
        if result_path.exists() and json.loads(result_path.read_text())['status'] == 'complete': continue
        try:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            model = initialize(name, source).to(args.device).eval()
            model.requires_grad_(True)
            training = dict(new_train_seconds=0., updates=0, best_epoch=0)
            if name not in NO_FIT: training = fit(name, model, source, data, args, folder)
            else: torch.save({'model': model.state_dict(), 'name': name, 'epoch': 0}, folder / 'best.pt')
            used = ['r4_mlp'] if name == 'r4_protected' else (['ssm'] if name == 'ssm_continue' else ['ssm', 'r4_mlp'])
            result = dict(name=name, seed=args.seed, status='complete', new_fit=name not in NO_FIT,
                parameters=sum(p.numel() for p in model.parameters()), **training,
                parent_costs={k: costs[k]['train_seconds'] for k in used},
                parent_updates=sum(costs[k]['updates'] for k in used),
                train_seconds=training['new_train_seconds']+sum(costs[k]['train_seconds'] for k in used),
                **score_model(model.eval(), data, args, folder))
            result['block_protocol'] = getattr(model, 'block_protocol', 'full model block32 reencoding')
            result['native_protocol'] = getattr(model, 'native_protocol', 'uninterrupted model state')
            save_json(result_path, result)
        except Exception as error:
            failures.append(name)
            save_json(result_path, dict(name=name, seed=args.seed, status='failed', error=repr(error)))
            (folder / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
            traceback.print_exc()
            if args.smoke: raise
        aggregate(args.output)
    from .report import report
    from .analyze_return import analyze
    report(args.output)
    if any(r['status'] == 'complete' for r in json.loads((args.output / 'summary.json').read_text())):
        analyze(args.output, args.output)
        comparison(args)
    save_json(args.output / 'state.json', dict(status='completed_with_failures' if failures else 'complete', failures=failures, smoke=args.smoke))
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    main()
