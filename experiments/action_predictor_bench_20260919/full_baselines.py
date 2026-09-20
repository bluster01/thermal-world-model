"""All nine original families: longer fitting, plateau accounting, two selectors."""
import argparse
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
from .models import TRAINED, Anchored, build
from .run import HERE, aggregate, loss_fn, save_json, score_model

CHOICES = ('short', 'balanced')


@torch.no_grad()
def select(model, bank, args):
    """Same information/metric for every family, including fixed H32 heads."""
    totals = np.zeros(2)
    model.eval()
    for start in range(0, len(bank), args.batch_size):
        h, u, d, y = unpack(bank[start:start+args.batch_size], args.device)
        p = forecast(model, h, u, d, mode='block')
        if not torch.isfinite(p).all(): raise ValueError('Nonfinite selector block forecast')
        totals[0] += float((p[:, :32, 4]-y[:, :32, 4]).abs().mean())*len(h)
        totals[1] += float((p[:, :128, 4]-y[:, :128, 4]).abs().mean())*len(h)
    short, long = totals/len(bank)
    return dict(short=float(short), balanced=float(.5*short+.5*long), block_H128=float(long))


def advance_budget(scores, milestones, stale, lr, epoch, args):
    """Reduce LR only when neither selector improves materially; floor then stop."""
    improved = [key for key in CHOICES if scores[key] < milestones[key]-args.min_delta]
    milestones = {**milestones, **{key: scores[key] for key in improved}}
    stale = 0 if improved else stale+1
    if lr > args.min_lr*(1+1e-8) and stale >= args.lr_patience:
        lr, stale = max(args.min_lr, lr*.5), 0
    stop = epoch >= args.min_epochs and lr <= args.min_lr*(1+1e-8) and stale >= args.stop_patience
    return milestones, stale, lr, stop


def fit(name, seed, data, args, folder, model=None, training_objective=None, supervision_horizon=32):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = (build(name, data['mean'], data['scale']) if model is None else model).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial = select(model, data['selector'], args)
    save_json(folder/'initial_selector.json', initial)
    best = {key: initial[key] for key in CHOICES}
    milestones, best_epochs = best.copy(), dict(short=0, balanced=0)
    for choice in CHOICES: torch.save({'model': model.state_dict(), 'epoch': 0}, folder/f'best_{choice}.pt')
    started, updates, stale, reason = time.perf_counter(), 0, 0, 'budget_limit'
    with (folder/'training.jsonl').open('w') as log:
        for epoch in range(1, args.max_epochs+1):
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(data['train']))
            model.train(); total = 0.
            lr_used = optimizer.param_groups[0]['lr']
            for start in range(0, len(order), args.batch_size):
                ids = order[start:start+args.batch_size]
                # Long data bank supplies selector labels, NOT H128 supervision.
                if training_objective is None:
                    h, u, d, y = unpack(data['train'][ids, :64+supervision_horizon], args.device)
                    p = model(h, u[:, :-1], d[:, :-1])
                    loss = loss_fn(model, p, y)
                else:
                    batch = torch.as_tensor(data['train'][ids, :64+supervision_horizon], device=args.device)
                    loss = training_objective(model, batch)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                if hasattr(model, 'after_optimizer_step'):
                    model.after_optimizer_step()
                total += float(loss.detach())*len(ids)
            scores = select(model, data['selector'], args)
            for choice in CHOICES:
                if scores[choice] < best[choice]:
                    best[choice], best_epochs[choice] = scores[choice], epoch
                    torch.save({'model': model.state_dict(), 'epoch': epoch}, folder/f'best_{choice}.pt')
            milestones, stale, lr_next, stop = advance_budget(scores, milestones, stale, lr_used, epoch, args)
            for group in optimizer.param_groups: group['lr'] = lr_next
            if stop: reason = 'validation_plateau'
            row = dict(epoch=epoch, updates=updates, train_loss=total/len(order), selector=scores,
                selector_main_mae_C=scores['short'], lr_used=lr_used, lr_next=lr_next,
                stale_at_current_lr=stale, seconds=time.perf_counter()-started)
            log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps({'model': name, 'seed': seed, **row}), flush=True)
            # Preserve optimizer and stopping state for any later budget extension.
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                updates=updates, milestones=milestones, stale=stale, best=best, best_epochs=best_epochs), folder/'last.pt')
            if stop: break
    result = dict(status='complete', model=name, seed=seed, epochs_run=epoch, updates=updates,
        stop_reason='smoke_budget' if args.smoke else reason,
        reached_validation_plateau=stop and not args.smoke, best_epochs=best_epochs,
        best_selector=best, final_selector=scores, final_lr=optimizer.param_groups[0]['lr'],
        train_seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model.parameters()),
        training_windows=len(data['train']), supervision_horizon=supervision_horizon)
    save_json(folder/'fit.json', result)
    return result


def load_model(name, folder, choice, data, args):
    model = build(name, data['mean'], data['scale']).to(args.device)
    model.load_state_dict(torch.load(folder/f'best_{choice}.pt', weights_only=True, map_location=args.device)['model'])
    return model.eval()


def complete_row(name, family, choice, seed, model, cost, data, args, checkpoints=()):
    folder = args.output/f'seed{seed}'/name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'result.json'
    if path.exists() and json.loads(path.read_text())['status'] == 'complete': return
    result = dict(name=name, family=family, selection=choice, seed=seed, status='complete',
        **cost, **score_model(model, data, args, folder))
    for label, source in checkpoints: shutil.copy2(source, folder/label)
    save_json(path, result)


def tables(output):
    rows = json.loads((output/'summary.json').read_text())
    fits = [json.loads(p.read_text()) for p in sorted((output/'fits').glob('seed*/*/fit.json'))]
    lines = ['# 全基线训练状态', '', 'validation_plateau=满足预设验证停滞规则；budget_limit=预算耗尽，未确认停滞。', '',
             '| Seed | 模型 | 实际轮数 | 停止原因 | 最终学习率 | 短期最优轮 | 综合最优轮 |',
             '|---|---|---:|---|---:|---:|---:|']
    for r in fits:
        lines.append(f"| {r['seed']} | {r['model']} | {r['epochs_run']} | {r['stop_reason']} | {r['final_lr']:.6g} | {r['best_epochs']['short']} | {r['best_epochs']['balanced']} |")
    seen = {(r['seed'], r['model']) for r in fits}
    for r in rows:
        family = r['model'].rsplit('_', 1)[0]
        if r['status'] != 'complete' and family in TRAINED and (r['seed'], family) not in seen:
            lines.append(f"| {r['seed']} | {family} | — | 训练未完成/失败 | — | — | — |")
            seen.add((r['seed'], family))
    (output/'CONVERGENCE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    lines = ['# 全部基线：两种选模规则分别展示', '',
             '9个训练家族＋persistence＋anchored_hold。两种选择复用同一次训练，不是重复拟合。',
             '所有家族的balanced均为0.5×selector H32＋0.5×selector block H128；不使用native指标选模。',
             'native列的—表示固定输出头不支持该模式；anchored_hold的native是Direct分块＋R4响应连续的混合模式。',
             '是否观察到训练停滞请看CONVERGENCE.md，不把完整运行自动称为充分收敛。']
    for choice in CHOICES:
        lines += ['', f'## {choice}', '', '| Seed | 模型 | H32 | block H128 | block H512 | native H512 |', '|---|---|---:|---:|---:|---:|']
        for r in rows:
            if r['model'] != 'persistence' and not r['model'].endswith('_'+choice): continue
            if r['status'] != 'complete':
                lines.append(f"| {r['seed']} | {r['model']}（失败） | — | — | — | — |")
                continue
            values = [r.get(k) for k in ('block_H32_mae_C', 'block_H128_mae_C', 'block_H512_mae_C', 'native_H512_mae_C')]
            lines.append('| '+' | '.join([str(r['seed']), r['model'], *[f'{v:.4f}' if v is not None else '—' for v in values]])+' |')
    (output/'ALL_BASELINES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE/'data/screen_A_33pct_h128.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11])
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--min-epochs', type=int, default=12)
    parser.add_argument('--lr-patience', type=int, default=4)
    parser.add_argument('--stop-patience', type=int, default=6)
    parser.add_argument('--min-delta', type=float, default=.002)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--min-lr', type=float, default=.0001)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--response-windows', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.min_epochs, args.max_epochs, args.lr_patience, args.stop_patience, args.batch_size, args.response_windows, args.threads) < 1:
        parser.error('Positive budgets required')
    if args.max_epochs < args.min_epochs or not 0 < args.min_lr <= args.learning_rate or args.min_delta < 0:
        parser.error('Invalid epoch bounds, learning rates or min-delta')
    torch.set_num_threads(args.threads)
    data, metadata = load_pack(args.data)
    if data['selector'].shape[1] != 192 or metadata['counts']['train'] != 20371:
        raise ValueError('Use the complete one-third pack with H128 selector labels')
    # Reduce resident training memory after validating the long bank.
    data['train'] = data['train'][:, :96].copy()
    if args.smoke:
        data['train'], data['selector'], data['evaluation'] = data['train'][:16], data['selector'][:8], data['evaluation'][:4]
        data['evaluation_time'] = data['evaluation_time'][:4]
        args.max_epochs, args.min_epochs, args.batch_size, args.response_windows = 1, 1, 8, 2
    config = {k: v for k, v in vars(args).items() if k not in ('data', 'output', 'resume')}
    config.update(models=TRAINED, supervision_horizon=32, selector_horizon=128,
        data_sha256=metadata['pack_sha256'], source_sha256={p.name: sha256(p) for p in HERE.glob('*.py')},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), torch_version=torch.__version__,
        selection='short=H32; balanced=.5 H32+.5 block H128; common across all nine families',
        initialization='from scratch at each seed, original epoch orders, no SSM-only continuation')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'config.json'
    if path.exists():
        old = json.loads(path.read_text())
        if not args.resume or any(old.get(k) != v for k, v in config.items() if k != 'git_commit'):
            raise ValueError('Existing output needs --resume and identical config/code/data')
    else: save_json(path, config)
    save_json(args.output/'data_metadata.json', metadata)
    np.savez_compressed(args.output/'evaluation_inputs.npz', bank=data['evaluation'], times=data['evaluation_time'])
    save_json(args.output/'state.json', dict(status='running', smoke=args.smoke))
    failures = []
    for seed in args.seeds:
        for family in ['persistence', *TRAINED, 'anchored_hold']:
            choices = ('shared',) if family == 'persistence' else CHOICES
            try:
                fit_dir = args.output/'fits'/f'seed{seed}'/family
                if family in TRAINED:
                    fit_dir.mkdir(parents=True, exist_ok=True)
                    cost = json.loads((fit_dir/'fit.json').read_text()) if (fit_dir/'fit.json').exists() else fit(family, seed, data, args, fit_dir)
                for choice in choices:
                    name = family if family == 'persistence' else f'{family}_{choice}'
                    if family == 'persistence':
                        model = build(family, data['mean'], data['scale']).to(args.device).eval()
                        cost = dict(parameters=0, train_seconds=0., updates=0, new_fit=False)
                        checkpoints = []
                    elif family == 'anchored_hold':
                        parts, parent_costs = [], []
                        for parent in ('direct', 'r4'):
                            parent_dir = args.output/'fits'/f'seed{seed}'/parent
                            parent_costs.append(json.loads((parent_dir/'fit.json').read_text()))
                            parts.append(load_model(parent, parent_dir, choice, data, args))
                        model = Anchored(*parts).to(args.device).eval()
                        cost = dict(parameters=sum(c['parameters'] for c in parent_costs),
                            train_seconds=sum(c['train_seconds'] for c in parent_costs), updates=0, new_fit=False,
                            parent_stop_reasons={c['model']: c['stop_reason'] for c in parent_costs})
                        checkpoints = []
                    else:
                        model = load_model(family, fit_dir, choice, data, args)
                        checkpoints = [('best.pt', fit_dir/f'best_{choice}.pt'), ('training.jsonl', fit_dir/'training.jsonl')]
                    # Remove metadata keys that belong to the row rather than the fit.
                    row_cost = {k: v for k, v in cost.items() if k not in ('status', 'model', 'seed')}
                    row_cost.update(fit_id=f'seed{seed}/{family}', best_epoch=cost.get('best_epochs', {}).get(choice))
                    complete_row(name, family, choice, seed, model, row_cost, data, args, checkpoints)
            except Exception as error:
                failures.append(f'seed{seed}/{family}')
                for choice in choices:
                    name = family if family == 'persistence' else f'{family}_{choice}'
                    out = args.output/f'seed{seed}'/name; out.mkdir(parents=True, exist_ok=True)
                    path = out/'result.json'
                    if path.exists() and json.loads(path.read_text()).get('status') == 'complete': continue
                    save_json(path, dict(name=name, seed=seed, status='failed', error=repr(error)))
                    (out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                traceback.print_exc()
                if args.smoke: raise
            aggregate(args.output); tables(args.output)
    from .report import report
    from .analyze_return import analyze
    report(args.output); analyze(args.output, args.output)
    fits = [json.loads(p.read_text()) for p in (args.output/'fits').glob('seed*/*/fit.json')]
    pending = [f"seed{r['seed']}/{r['model']}" for r in fits if not r['reached_validation_plateau']]
    save_json(args.output/'state.json', dict(status='completed_with_failures' if failures else 'complete',
        failures=failures, smoke=args.smoke, training_families=TRAINED, not_confirmed_plateau=pending,
        note='Completion is not a convergence claim; inspect CONVERGENCE.md for every model'))
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    main()
