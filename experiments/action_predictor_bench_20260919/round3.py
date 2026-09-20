"""3x2 quick screen: SSM/hold/planned reference crossed with H32/H128 learning."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from .data import load_pack, unpack, sha256
from .evaluation import forecast
from .round2 import parents
from .round3_models import NominalPolicy, PlannedReference
from .run import HERE, aggregate, loss_fn, save_json, score_model

ARMS = [f'{kind}_h{h}' for kind in ('ssm', 'hold', 'planned') for h in (32, 128)]


def write_comparison(args):
    rows = json.loads((args.output/'summary.json').read_text())
    references = json.loads((args.parents/'summary.json').read_text())
    parents_to_show = [r for r in references if r['seed'] == args.seed and r['model'] in ('ssm', 'r4_mlp')]
    lines = ['# Horizon and nominal-plan comparison', '',
             'Six temperature fits; short/balanced rows reuse the same training trajectory.',
             'Protected block refreshes the reference only; the response stays continuous.',
             'H128 cells supervise steps33–128; H512 remains beyond training horizon.',
             'Compare within each selector rule. Neither rule selects on reporting windows.', '',
             '| Model | H32 MAE | Block H128 | Block H512 | Native H512 |',
             '|---|---:|---:|---:|---:|']
    if args.smoke:
        lines.insert(2, 'SMOKE ONLY: tiny windows; parent scores omitted because their reporting set differs.')
        parents_to_show = []
    for row in parents_to_show + rows:
        if row['status'] != 'complete': continue
        keys = ('block_H32_mae_C', 'block_H128_mae_C', 'block_H512_mae_C', 'native_H512_mae_C')
        lines.append('| '+' | '.join([row['model'], *[f'{row[k]:.4f}' for k in keys]])+' |')
    lines += ['', 'Policy quality: policy/result.json (valve MAE in percentage points versus hold).',
              'Response quality: DIAGNOSIS.md and per-row response metrics; all110 scenarios retained.',
              'Temperature fit cost is shared by two selectors; do not sum duplicated row costs.']
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def horizon_loss(model, prediction, truth):
    short = loss_fn(model, prediction[:, :32], truth[:, :32])
    if prediction.shape[1] == 32: return short
    return .5*short + .5*loss_fn(model, prediction[:, 32:], truth[:, 32:])


@torch.no_grad()
def selector_metrics(model, bank, args):
    totals = np.zeros(3)
    for start in range(0, len(bank), args.batch_size):
        h, u, d, y = unpack(bank[start:start+args.batch_size], args.device)
        block = forecast(model, h, u, d, mode='block')
        native = forecast(model, h, u, d, mode='native')
        for j, (p, end) in enumerate(((block, 32), (block, 128), (native, 128))):
            totals[j] += float((p[:, :end, 4]-y[:, :end, 4]).abs().mean())*len(h)
    values = totals/len(bank)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite selector forecast')
    return dict(short=float(values[0]), balanced=float(values @ np.array([.5, .25, .25])),
                block_H128=float(values[1]), native_H128=float(values[2]))


@torch.no_grad()
def policy_metrics(policy, bank, args):
    sums = np.zeros(4)
    for start in range(0, len(bank), args.batch_size):
        h, u, d, _ = unpack(bank[start:start+args.batch_size], args.device)
        p = policy(h, d)
        held = h[:, -1:, 5:7].expand_as(u)
        for j, end in enumerate((33, 129)):
            sums[2*j] += float((p[:, 1:end]-u[:, 1:end]).abs().mean())*len(h)*100
            sums[2*j+1] += float((held[:, 1:end]-u[:, 1:end]).abs().mean())*len(h)*100
    return dict(zip(('H32_mae_pp', 'hold_H32_mae_pp', 'H128_mae_pp', 'hold_H128_mae_pp'), (sums/len(bank)).tolist()))


def fit_policy(data, args, folder):
    torch.manual_seed(args.seed)
    policy = NominalPolicy(data['mean'], data['scale']).to(args.device)
    folder.mkdir(parents=True, exist_ok=True)
    initial = policy_metrics(policy, data['selector'], args)
    best, best_epoch = initial['H128_mae_pp'], 0
    torch.save({'model': policy.state_dict(), 'epoch': 0}, folder/'best.pt')
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    started = time.perf_counter()
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, args.policy_epochs+1):
            policy.train()
            order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch])).permutation(len(data['train']))
            total = 0.
            for start in range(0, len(order), args.batch_size):
                ids = order[start:start+args.batch_size]
                h, u, d, _ = unpack(data['train'][ids], args.device)
                prediction = policy(h, d)
                loss = ((prediction[:, 1:]-u[:, 1:])/policy.scale[5:7]).square().mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite policy loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); total += float(loss.detach())*len(ids)
            scores = policy_metrics(policy, data['selector'], args)
            if scores['H128_mae_pp'] < best:
                best, best_epoch = scores['H128_mae_pp'], epoch
                torch.save({'model': policy.state_dict(), 'epoch': epoch}, folder/'best.pt')
            row = dict(epoch=epoch, train_loss=total/len(order), **scores)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps({'policy': row}), flush=True)
    policy.load_state_dict(torch.load(folder/'best.pt', weights_only=True, map_location=args.device)['model'])
    policy.requires_grad_(False).eval()
    # Reporting action MAE is computed only after selection and never feeds back.
    result = dict(train_seconds=time.perf_counter()-started, best_epoch=best_epoch, initial_selector=initial,
                  selector=policy_metrics(policy, data['selector'], args),
                  reporting=policy_metrics(policy, data['evaluation'][:, :192], args))
    save_json(folder/'result.json', result)
    return policy, result


def make_model(arm, source, policy):
    predictor = copy.deepcopy(source['ssm']).requires_grad_(True)
    if arm.startswith('ssm_'): return predictor
    response = copy.deepcopy(source['r4_mlp']).requires_grad_(True)
    nominal = copy.deepcopy(policy) if arm.startswith('planned_') else None
    return PlannedReference(predictor, response, nominal)


def fit(arm, model, source, data, args, folder):
    horizon = int(arm.rsplit('h', 1)[1])
    scores = selector_metrics(model, data['selector'], args)
    best, epochs = {k: scores[k] for k in ('short', 'balanced')}, dict(short=0, balanced=0)
    for choice in best: torch.save({'model': model.state_dict(), 'epoch': 0}, folder/f'best_{choice}.pt')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    started, updates = time.perf_counter(), 0
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, args.epochs+1):
            model.train()
            order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch+6])).permutation(len(data['train']))
            total, extra_total = 0., 0.
            for start in range(0, len(order), args.batch_size):
                ids = order[start:start+args.batch_size]
                h, u, d, y = unpack(data['train'][ids, :64+horizon], args.device)
                p = model(h, u[:, :-1], d[:, :-1])
                factual = horizon_loss(model, p, y)
                extra = factual.new_zeros(())
                if not arm.startswith('ssm_'):
                    # Short-horizon teacher only: do not teach long extrapolation errors.
                    with torch.no_grad(): teacher = source['ssm'](h, u[:, :32], d[:, :32])
                    extra = .25*loss_fn(model, p[:, :32], teacher)
                loss = factual+extra
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                total += float(factual.detach())*len(ids); extra_total += float(extra.detach())*len(ids)
            scores = selector_metrics(model, data['selector'], args)
            for choice in best:
                if scores[choice] < best[choice]:
                    best[choice], epochs[choice] = scores[choice], epoch
                    torch.save({'model': model.state_dict(), 'epoch': epoch}, folder/f'best_{choice}.pt')
            row = dict(epoch=epoch, updates=updates, train_loss=total/len(order), extra_loss=extra_total/len(order),
                       selector_main_mae_C=scores['short'], selector=scores, seconds=time.perf_counter()-started)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps({'arm': arm, **row}), flush=True)
    torch.save({'model': model.state_dict(), 'epoch': args.epochs}, folder/'last.pt')
    result = dict(new_train_seconds=time.perf_counter()-started, updates=updates, best_epochs=epochs,
                  best_selector=best, train_horizon=horizon)
    save_json(folder/'fit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', type=Path, default=Path('results/action_predictor_bench_20260919/screen33_seed11'))
    parser.add_argument('--data', type=Path, default=HERE/'data/screen_A_33pct_h128.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--policy-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--response-windows', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.policy_epochs, args.batch_size, args.threads, args.response_windows) < 1 or args.learning_rate <= 0:
        parser.error('Positive budgets required')
    torch.set_num_threads(args.threads)
    data, metadata = load_pack(args.data)
    previous = json.loads((args.parents/'config.json').read_text())
    if previous['data_sha256'] != metadata.get('contains_previous_pack_sha256') or previous['smoke'] or metadata['train_horizon'] != 128:
        raise ValueError('Need H128 extension of the exact full parent pack')
    if args.smoke:
        data['train'], data['selector'], data['evaluation'] = data['train'][:16], data['selector'][:8], data['evaluation'][:4]
        data['evaluation_time'] = data['evaluation_time'][:4]
        args.epochs, args.policy_epochs, args.batch_size, args.response_windows = 1, 1, 8, 2
    source, costs = parents(args, data)
    for model in source.values(): model.requires_grad_(False)
    config = {k: v for k, v in vars(args).items() if k not in ('parents', 'output', 'data', 'resume')}
    config.update(data_sha256=metadata['pack_sha256'],
        parent_checkpoint_sha256={n: sha256(args.parents/f'seed{args.seed}'/n/'best.pt') for n in source},
        source_sha256={p.name: sha256(p) for p in HERE.glob('*.py')},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), torch_version=torch.__version__,
        selector='short=H32 block; balanced=.5 H32+.25 block H128+.25 native H128; evaluate both, no reporting selection',
        training='H32 original loss; H128 .5 first32 + .5 steps33:128; protected arms add .25 short prediction teacher',
        policy='causal GRU24, configured policy_epochs at lr.001, selector H128 action MAE, then frozen')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'config.json'
    if path.exists():
        old = json.loads(path.read_text())
        if not args.resume or any(old.get(k) != v for k, v in config.items() if k != 'git_commit'):
            raise ValueError('Resume needs identical config/code/data/parents')
    else: save_json(path, config)
    save_json(args.output/'data_metadata.json', metadata)
    np.savez_compressed(args.output/'evaluation_inputs.npz', bank=data['evaluation'], times=data['evaluation_time'])
    save_json(args.output/'state.json', dict(status='running', smoke=args.smoke))
    # Policy stage is cheap; resume reuses its completed frozen checkpoint.
    pf = args.output/'policy'
    if (pf/'result.json').exists():
        policy = NominalPolicy(data['mean'], data['scale']).to(args.device)
        policy.load_state_dict(torch.load(pf/'best.pt', weights_only=True, map_location=args.device)['model'])
        policy.requires_grad_(False); policy_cost = json.loads((pf/'result.json').read_text())
    else: policy, policy_cost = fit_policy(data, args, pf)
    failures = []
    for arm in args.arms:
        folder = args.output/'fits'/arm
        folder.mkdir(parents=True, exist_ok=True)
        try:
            torch.manual_seed(args.seed)
            model = make_model(arm, source, policy).to(args.device)
            training = json.loads((folder/'fit.json').read_text()) if (folder/'fit.json').exists() else fit(arm, model, source, data, args, folder)
            inherited = costs['ssm']['train_seconds'] + (0 if arm.startswith('ssm_') else costs['r4_mlp']['train_seconds'])
            policy_seconds = policy_cost['train_seconds'] if arm.startswith('planned_') else 0.
            for choice in ('short', 'balanced'):
                name = f'{arm}_{choice}'
                out = args.output/f'seed{args.seed}'/name
                out.mkdir(parents=True, exist_ok=True)
                if (out/'result.json').exists() and json.loads((out/'result.json').read_text())['status'] == 'complete': continue
                checkpoint = folder/f'best_{choice}.pt'
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=args.device)['model'])
                result = dict(name=name, arm=arm, seed=args.seed, status='complete', selection=choice, fit_id=arm,
                    parameters=sum(p.numel() for p in model.parameters()), best_epoch=training['best_epochs'][choice],
                    **training, train_seconds=inherited+policy_seconds+training['new_train_seconds'],
                    parent_train_seconds=inherited, policy_train_seconds=policy_seconds,
                    **score_model(model.eval(), data, args, out))
                result['block_protocol'] = getattr(model, 'block_protocol', 'full model block32')
                result['native_protocol'] = getattr(model, 'native_protocol', 'uninterrupted model state')
                save_json(out/'result.json', result)
                shutil.copy2(checkpoint, out/'best.pt'); shutil.copy2(folder/'training.jsonl', out/'training.jsonl')
            # A successful retry supersedes its prior failure row; keep the trace.
            (args.output/f'seed{args.seed}'/f'{arm}_failed'/'result.json').unlink(missing_ok=True)
        except Exception as error:
            failures.append(arm)
            out = args.output/f'seed{args.seed}'/f'{arm}_failed'; out.mkdir(parents=True, exist_ok=True)
            save_json(out/'result.json', dict(name=arm, seed=args.seed, status='failed', error=repr(error)))
            (out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8'); traceback.print_exc()
            if args.smoke: raise
        aggregate(args.output)
    from .report import report
    from .analyze_return import analyze
    report(args.output); analyze(args.output, args.output); write_comparison(args)
    save_json(args.output/'state.json', dict(status='completed_with_failures' if failures else 'complete', failures=failures, smoke=args.smoke))
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    main()
