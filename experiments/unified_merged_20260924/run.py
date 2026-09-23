"""Linux execution of the frozen three-task baseline/mechanism matrix.

Writes all values, checkpoints, detailed metrics and paths to the specified
private output. public_status.json contains only execution progress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.unified_data_20260923.dataset import IndustrialDataset
from experiments.unified_experiments_20260923.models import build_model

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_release():
    manifest = read_json(HERE / 'release_manifest.json')
    root = HERE.parents[1]
    for relative, expected in manifest['source_sha256_lf'].items():
        content = (root / relative).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f'Released source changed: {relative}')
    return sha(HERE / 'release_manifest.json')


def linux_device():
    if platform.system() != 'Linux':
        raise RuntimeError('Scientific execution is assigned to remote Linux')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this formal matrix; record prerequisite missing')
    return torch.device('cuda')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def move(batch, device):
    needed = ('history_normalized', 'history_valid', 'history_observed', 'history_age_s',
              'targets_normalized', 'target_mask')
    return {key: batch[key].to(device, non_blocking=True) for key in needed}


def macro_mse(prediction, target, mask):
    counts = mask.sum((0, 1))
    good = counts > 0
    if not good.any():
        return prediction.sum() * 0
    squared = torch.where(mask, (prediction - target).square(), 0).sum((0, 1))
    return (squared[good] / counts[good]).mean()


def objective(prediction, batch, spec):
    target, mask = batch['targets_normalized'], batch['target_mask']
    base = macro_mse(prediction, target, mask)
    changes = []
    for lag in spec['increment_lags']:
        both = mask[:, lag:] & mask[:, :-lag]
        if both.any():
            changes.append(macro_mse(prediction[:, lag:] - prediction[:, :-lag],
                                     target[:, lag:] - target[:, :-lag], both))
    return base + (spec['increment_weight'] * torch.stack(changes).mean() if changes else 0)


def anchor(batch, indices):
    values = batch['history_normalized'][:, :, indices]
    mask = batch['history_valid'][:, :, indices]
    steps = torch.arange(values.shape[1], device=values.device)[None, :, None]
    last = torch.where(mask, steps, -1).amax(1)
    selected = values.gather(1, last.clamp_min(0)[:, None, :]).squeeze(1)
    return torch.where(last >= 0, selected, 0)


def loader(dataset, batch_size, workers, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=seed is not None,
                      generator=generator, num_workers=workers, pin_memory=True,
                      persistent_workers=False, drop_last=False)


def construct(arm, dataset, task, protocol, device):
    return build_model(arm, dataset.input_names, dataset.target_names, task,
                       hidden=protocol['hidden'], task_spec=dataset.task,
                       target_center=dataset.center[dataset.target_indices].tolist(),
                       target_scale=dataset.scale[dataset.target_indices].tolist(),
                       history_length=protocol['history_steps']).to(device)


@torch.no_grad()
def evaluate(model, dataset, protocol, device, workers, output=None):
    model.eval()
    underlying = dataset.dataset if isinstance(dataset, Subset) else dataset
    o = len(underlying.target_names)
    horizon = underlying.profile['forecast_steps']
    indices = [underlying.input_names.index(name) for name in underlying.target_names]
    scales = underlying.scale[underlying.target_indices].astype(np.float64)
    total_count = np.zeros(o, dtype=np.int64)
    total_sse = np.zeros(o); total_abs = np.zeros(o); persistence_abs = np.zeros(o)
    endpoints = {h: {'count': np.zeros(o, dtype=np.int64), 'abs': np.zeros(o), 'sse': np.zeros(o)}
                 for h in protocol['validation']['endpoints'] if h <= horizon}
    delta_count = np.zeros(o, dtype=np.int64); delta_abs = np.zeros(o)
    preview = []
    origin_statistics = []
    for cpu in loader(dataset, protocol['validation']['batch_size'], workers):
        batch = move(cpu, device)
        prediction = model(batch, horizon=horizon)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation prediction')
        pred = prediction.cpu().numpy().astype(np.float64)
        truth = batch['targets_normalized'].cpu().numpy().astype(np.float64)
        mask = batch['target_mask'].cpu().numpy()
        error = pred - truth
        base = anchor(batch, indices).cpu().numpy().astype(np.float64)
        total_count += mask.sum((0, 1))
        total_sse += np.where(mask, error ** 2, 0).sum((0, 1))
        total_abs += np.where(mask, np.abs(error) * scales, 0).sum((0, 1))
        persistence_abs += np.where(mask, np.abs(base[:, None] - truth) * scales, 0).sum((0, 1))
        both = mask[:, 1:] & mask[:, :-1]
        delta_count += both.sum((0, 1))
        delta_abs += np.where(both, np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1)) * scales, 0).sum((0, 1))
        for h, sums in endpoints.items():
            sums['count'] += mask[:, h - 1].sum(0)
            sums['abs'] += np.where(mask[:, h - 1], np.abs(error[:, h - 1]) * scales, 0).sum(0)
            sums['sse'] += np.where(mask[:, h - 1], error[:, h - 1] ** 2 * scales ** 2, 0).sum(0)
        if output is not None:
            origin_statistics.append({
                'origin_time_ns': cpu['origin_time_ns'].numpy(),
                'label_count': mask.sum(1),
                'normalized_sse': np.where(mask, error ** 2, 0).sum(1),
                'physical_absolute_error': np.where(mask, np.abs(error) * scales, 0).sum(1),
                'persistence_absolute_error': np.where(mask, np.abs(base[:, None] - truth) * scales, 0).sum(1),
                'increment_count': both.sum(1),
                'increment_absolute_error': np.where(both, np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1)) * scales, 0).sum(1)})
        if output is not None and len(preview) < 3:
            preview.append({'origin': cpu['origin_time_ns'].numpy(), 'prediction_normalized': pred.astype(np.float32),
                            'truth_normalized': truth.astype(np.float32), 'mask': mask})
    if not (total_count > 0).all():
        raise RuntimeError('At least one target has no eligible evaluation labels')
    per_target = {}
    for j, name in enumerate(underlying.target_names):
        per_target[name] = {'count': int(total_count[j]), 'MAE': float(total_abs[j] / total_count[j]),
                            'RMSE': float(math.sqrt(total_sse[j] / total_count[j]) * scales[j]),
                            'persistence_MAE': float(persistence_abs[j] / total_count[j]),
                            'increment_MAE': float(delta_abs[j] / delta_count[j]) if delta_count[j] else None,
                            'increment_count': int(delta_count[j]),
                            'endpoints': {str(h): {'count': int(s['count'][j]),
                                'MAE': float(s['abs'][j] / s['count'][j]) if s['count'][j] else None,
                                'RMSE': float(math.sqrt(s['sse'][j] / s['count'][j])) if s['count'][j] else None}
                                for h, s in endpoints.items()}}
    report = {'target_macro_normalized_mse': float((total_sse / total_count).mean()),
              'origins': len(dataset), 'horizon_steps': horizon, 'targets': per_target}
    if output is not None:
        arrays = {f'preview_{key}_{i}': value for i, item in enumerate(preview) for key, value in item.items()}
        arrays.update({key: np.concatenate([item[key] for item in origin_statistics])
                       for key in origin_statistics[0]})
        arrays['target_names'] = np.asarray(underlying.target_names)
        temporary = output.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(output.with_suffix('.npz'))
        report.update(artifact_sha256=sha(output.with_suffix('.npz')),
                      artifact_scope='all-origin error sums/counts; first three batches of trajectories only',
                      complete=True)
        # The JSON is the final commit marker, written only after the NPZ exists.
        write_json(output.with_suffix('.json'), report)
    return report


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, wait, identity, best_checkpoint=None):
    value = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best, 'wait': wait,
             'identity': identity, 'best_checkpoint': best_checkpoint,
             'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def fit_one(task, arm, seed, data, out, protocol, device, workers, common, resume):
    path = out / 'runs' / f'{task}__{arm}__s{seed}'
    status_path = path / 'status.json'
    if status_path.exists() and read_json(status_path).get('status') in {'EARLY_STOPPED', 'BUDGET_EXHAUSTED'}:
        if read_json(status_path)['identity']['common'] != common:
            raise RuntimeError('Completed fit belongs to another release/data identity')
        return
    existed = path.exists()
    if existed and not resume:
        raise RuntimeError('Fit directory exists; explicit --resume required')
    path.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    training = IndustrialDataset(data, task, 'train', protocol['primary_profile'])
    validation = IndustrialDataset(data, task, 'validation', protocol['primary_profile'])
    count = min(len(validation), protocol['validation']['max_selection_origins'])
    selection = np.linspace(0, len(validation) - 1, count, dtype=np.int64)
    selected = Subset(validation, selection.tolist())
    np.save(path / 'validation_selection_indices.npy', selection)
    if not len(training) or not count:
        raise RuntimeError('Empty train/validation partition')
    model = construct(arm, training, task, protocol, device)
    spec = protocol['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=spec['plateau_factor'],
                                                         patience=spec['plateau_patience'], min_lr=spec['minimum_learning_rate'])
    identity = {'task': task, 'arm': arm, 'seed': seed, 'common': common,
                'input_names': training.input_names, 'target_names': training.target_names,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'model_class': type(model).__name__,
                'training_origins': len(training), 'selection_origins': len(selected)}
    epoch_start, best, wait = 0, math.inf, 0
    best_checkpoint = None
    if existed:
        if not (path / 'last.pt').exists():
            raise RuntimeError('No complete-epoch checkpoint; preserve failed pre-epoch run')
        saved = torch.load(path / 'last.pt', map_location=device, weights_only=False)
        if saved['identity'] != identity:
            raise RuntimeError('Cannot resume changed model/data/protocol')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        epoch_start, best, wait = saved['epoch'] + 1, saved['best'], saved['wait']
        best_checkpoint = saved['best_checkpoint']
        if not best_checkpoint or not (path / best_checkpoint).is_file():
            raise RuntimeError('Resume is missing its referenced best checkpoint')
        random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng']['cuda']])
    state = {'status': 'RUNNING', 'identity': identity, 'started_utc': now(), 'epoch': epoch_start, 'pid': os.getpid()}
    write_json(status_path, state)
    final_status = 'BUDGET_EXHAUSTED'
    already_stopped = epoch_start >= spec['min_epochs'] and wait >= spec['early_stop_patience']
    if already_stopped:
        final_status = 'EARLY_STOPPED'
    for epoch in range(epoch_start, epoch_start if already_stopped else spec['max_epochs']):
        model.train(); total = 0.; batches = 0; began = time.monotonic()
        for cpu in loader(training, spec['batch_size'], workers, seed=seed * 100000 + epoch):
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch, horizon=protocol['training_horizon'])
            loss = objective(prediction, batch, spec)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()); batches += 1
        score = evaluate(model, selected, protocol, device, workers)['target_macro_normalized_mse']
        scheduler.step(score)
        improved = score < best - spec['minimum_improvement']
        best = score if improved else best
        wait = 0 if improved else wait + 1
        if improved:
            best_checkpoint = f'best_epoch_{epoch + 1:03d}.pt'
            save_checkpoint(path / best_checkpoint, model, optimizer, scheduler, epoch, best, wait,
                            identity, best_checkpoint)
        # A complete epoch references an immutable best checkpoint. An interrupted
        # save leaves either the previous epoch or this fully committed epoch.
        save_checkpoint(path / 'last.pt', model, optimizer, scheduler, epoch, best, wait,
                        identity, best_checkpoint)
        row = {'epoch': epoch + 1, 'train_loss': total / max(batches, 1), 'validation_normalized_mse': score,
               'best': best, 'improved': improved, 'learning_rate': optimizer.param_groups[0]['lr'],
               'batches': batches, 'seconds': time.monotonic() - began, 'utc': now()}
        with (path / 'ledger.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        state.update(epoch=epoch + 1, updated_utc=now(), best_validation=best)
        write_json(status_path, state)
        public_status(out, protocol, 'RUNNING', path.name)
        print(json.dumps({'run': path.name, **row}), flush=True)
        if epoch + 1 >= spec['min_epochs'] and wait >= spec['early_stop_patience']:
            final_status = 'EARLY_STOPPED'; break
    shutil.copyfile(path / best_checkpoint, path / 'best.pt.tmp')
    (path / 'best.pt.tmp').replace(path / 'best.pt')
    state.update(status=final_status, finished_utc=now(), best_checkpoint=best_checkpoint,
                 best_epoch=int(best_checkpoint.removeprefix('best_epoch_').removesuffix('.pt')),
                 best_validation=best, checkpoint_sha256=sha(path / 'best.pt'))
    write_json(status_path, state)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()


def public_status(out, protocol, status, current=None):
    states = [read_json(p) for p in (out / 'runs').glob('*/status.json')]
    complete = sum(s['status'] in {'EARLY_STOPPED', 'BUDGET_EXHAUSTED'} for s in states)
    value = {'experiment_id': protocol['experiment_id'], 'status': status,
             'updated_utc': now(), 'fits_completed': complete, 'fits_planned': protocol['fit_count'],
             'current_run': current, 'private_outputs_retained_on_worker': True}
    write_json(out / 'public_status.json', value)


def preflight(data, out, protocol, device):
    from .data_contract import verify_pack
    identity = verify_pack(Path(data), read_json(HERE / 'expected_data.json'))
    write_json(out / 'data_identity.json', identity)
    if identity.get('status') != 'PASS':
        raise RuntimeError('Data identity check did not pass')
    # Only Linux and real training windows; tiny optimizer checks precede the matrix.
    checks = []
    for task in protocol['tasks']:
        ds = IndustrialDataset(data, task, 'train', protocol['primary_profile'])
        batch = move(next(iter(loader(Subset(ds, list(range(min(4, len(ds))))), 4, 0))), device)
        for arm in protocol['arms']:
            seed_all(723)
            model = construct(arm, ds, task, protocol, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            initial, final = None, None
            for step in range(8):
                optimizer.zero_grad(set_to_none=True)
                value = model(batch, horizon=protocol['training_horizon'])
                loss = objective(value, batch, protocol['training'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Preflight nonfinite {task}/{arm}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                if initial is None: initial = float(loss.detach())
            with torch.no_grad():
                final = float(objective(model(batch, horizon=protocol['training_horizon']), batch, protocol['training']))
                longer = model(batch, horizon=protocol['long_horizon'])
                if not torch.isfinite(longer).all():
                    raise FloatingPointError('Nonfinite long-horizon preflight')
            if final >= initial:
                raise RuntimeError(f'Tiny-batch loss did not decrease for {task}/{arm}')
            checks.append({'task': task, 'arm': arm, 'initial_loss': initial, 'final_loss': final,
                           'parameters': sum(p.numel() for p in model.parameters())})
            del model, optimizer
    result = {'status': 'PASS', 'checks': checks, 'utc': now(),
              'platform': platform.platform(), 'python': platform.python_version(),
              'torch': torch.__version__, 'numpy': np.__version__,
              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(device)}
    write_json(out / 'preflight.json', result)
    return result


def final_evaluation(data, out, protocol, device, workers):
    for seed in protocol['seeds']:
        for task in protocol['tasks']:
            for arm in protocol['arms']:
                path = out / 'runs' / f'{task}__{arm}__s{seed}'
                state = read_json(path / 'status.json')
                if state['status'] not in {'EARLY_STOPPED', 'BUDGET_EXHAUSTED'}:
                    raise RuntimeError('Evaluation requires the complete frozen fit matrix')
    # Every fit is complete before any historical-test score is computed.
    for seed in protocol['seeds']:
        for task in protocol['tasks']:
            for arm in protocol['arms']:
                path = out / 'runs' / f'{task}__{arm}__s{seed}'
                receipt = read_json(path / 'status.json')
                if sha(path / 'best.pt') != receipt['checkpoint_sha256']:
                    raise RuntimeError('Frozen best checkpoint hash changed')
                base = IndustrialDataset(data, task, 'validation', protocol['primary_profile'])
                model = construct(arm, base, task, protocol, device)
                state = torch.load(path / 'best.pt', map_location=device, weights_only=False)
                model.load_state_dict(state['model'])
                for split in ['validation', 'historical_test']:
                    for profile in [protocol['primary_profile'], protocol['long_profile']]:
                        output = path / f'evaluation_{split}_{profile}'
                        if output.with_suffix('.json').exists() and output.with_suffix('.npz').exists():
                            previous = read_json(output.with_suffix('.json'))
                            if previous.get('complete') and previous.get('artifact_sha256') == sha(output.with_suffix('.npz')):
                                continue
                        ds = IndustrialDataset(data, task, split, profile)
                        evaluate(model, ds, protocol, device, workers, output=output)
                del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight', 'run'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    device = linux_device()
    from .data_contract import require_private_output, verify_pack
    args.data, args.out = args.data.resolve(), args.out.resolve()
    require_private_output(args.out)
    require_private_output(args.data)
    if args.workers < 0:
        raise ValueError('workers must be nonnegative')
    protocol = read_json(HERE / 'protocol.json')
    release = verify_release()
    torch.set_num_threads(protocol['execution']['cpu_threads'])
    if args.out.exists() and not args.resume:
        raise RuntimeError('Output exists; use a new private path or explicit --resume')
    args.out.mkdir(parents=True, exist_ok=True)
    # Linux flock is released by the OS on exit, including crashes. The empty
    # lock file can remain; only the kernel lock owns this execution directory.
    import fcntl
    execution_lock = (args.out / '.execution.lock').open('a')
    try:
        fcntl.flock(execution_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another process already owns this output directory') from None
    if shutil.disk_usage(args.out).free < 10 * 1024 ** 3:
        raise RuntimeError('At least 10 GiB of free private output space is required')
    common = {'release_sha256': release, 'protocol_sha256': sha(HERE / 'protocol.json'),
              'expected_data_sha256': sha(HERE / 'expected_data.json'),
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'numpy': np.__version__, 'cuda': torch.version.cuda,
                          'device': torch.cuda.get_device_name(device),
                          'visible_devices': torch.cuda.device_count()}}
    saved = args.out / 'run_identity.json'
    if saved.exists() and read_json(saved)['common'] != common:
        raise RuntimeError('Output belongs to a different released experiment')
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    write_json(saved, {'common': common, 'source_commit': source_commit, 'pid': os.getpid(),
                       'updated_utc': now(), 'private_data': str(args.data.resolve())})
    write_json(args.out / 'protocol.json', protocol)
    active_fit = None
    try:
        # Recheck actual arrays on every launch, including --resume.
        identity = verify_pack(args.data, read_json(HERE / 'expected_data.json'))
        write_json(args.out / 'data_identity.json', identity)
        if not (args.out / 'preflight.json').exists():
            public_status(args.out, protocol, 'PREFLIGHT_RUNNING')
            preflight(args.data, args.out, protocol, device)
        elif read_json(args.out / 'preflight.json').get('status') != 'PASS':
            raise RuntimeError('Existing preflight is not a PASS')
        if args.command == 'preflight':
            public_status(args.out, protocol, 'PREFLIGHT_PASSED'); return
        for seed in protocol['seeds']:
            for task in protocol['tasks']:
                for arm in protocol['arms']:
                    name = f'{task}__{arm}__s{seed}'
                    active_fit = args.out / 'runs' / name / 'status.json'
                    public_status(args.out, protocol, 'RUNNING', name)
                    fit_one(task, arm, seed, args.data, args.out, protocol, device,
                            args.workers, common, args.resume)
                    active_fit = None
        public_status(args.out, protocol, 'EVALUATING_FROZEN_MATRIX')
        final_evaluation(args.data, args.out, protocol, device, args.workers)
        from experiments.unified_experiments_20260923.summarize import summarize
        summarize(args.out, protocol)
        public_status(args.out, protocol, 'COMPLETED_PRIVATE_RESULTS_READY')
    except BaseException:
        if active_fit is not None and active_fit.exists():
            state = read_json(active_fit)
            if state['status'] == 'RUNNING':
                state.update(status='FAILED', failed_utc=now())
                write_json(active_fit, state)
        (args.out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        public_status(args.out, protocol, 'FAILED_PRIVATE_LOG_AVAILABLE')
        raise
    finally:
        execution_lock.close()


if __name__ == '__main__':
    main()
