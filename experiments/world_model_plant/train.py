"""Linux-only frozen-task runner for the private observational prototype.

No automatic retry/resume or new sample draw after an unsupported window.
This CLI requires a complete release manifest; a preflight receipt is not one.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import PlantDataset, file_sha256, text_sha256, keys_from_records
from .model import PlantModelConfig, PROPERTY_SHA256, VARIANTS, build_model, checkpoint_payload, model_identity, state_tensor_sha256
from .objective import oracle_conditioned_objective

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainingConfig:
    requested_batches: int = 250
    batch_size: int = 4
    train_horizon: int = 32
    validation_horizon: int = 128
    validation_interval: int = 50
    anchors_per_side: int = 16
    sample_seed: int = 72001
    learning_rate: float = .0003
    gradient_clip: float = 1.
    free_weight: float = 1.
    prior_weight: float = 1.

    def __post_init__(self):
        for name in ('requested_batches', 'batch_size', 'train_horizon', 'validation_horizon', 'validation_interval', 'anchors_per_side', 'sample_seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'sample_seed' else 1):
                raise ValueError(f'Invalid training option: {name}')
        if self.batch_size % 2 or not 1 <= self.train_horizon <= 128 or self.validation_horizon != 128 or self.anchors_per_side > 16:
            raise ValueError('Require balanced even batches, train H<=128, validation H128, and <=16 anchors per side')
        for name in ('learning_rate', 'gradient_clip', 'free_weight', 'prior_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid training scalar: {name}')
        if self.learning_rate == 0 or self.gradient_clip == 0 or self.free_weight + self.prior_weight == 0:
            raise ValueError('Learning rate, clipping, and some objective weight must be positive')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def verify_release(root, release_path, config_path):
    """Bind all scientific source namespaces and the exact proposed task config."""
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    release = json.loads(Path(release_path).read_text(encoding='utf-8'))
    if release.get('kind') != 'plant_oracle_training_release_v1' or release.get('scientific_training_authorized') is not True:
        raise ValueError('A frozen scientific training release is required; preflight is insufficient')
    required = set()
    for folder in ('src/final_wm', 'src/world_model_vnext', 'src/world_model_streaming', 'src/world_model_actuated',
                   'src/world_model_evidence', 'experiments/world_model_plant'):
        sources = list((root / folder).glob('*.py'))
        if not sources:
            raise ValueError('A required scientific source namespace is missing')
        required.update(p.relative_to(root).as_posix() for p in sources)
    config_rel = config_path.relative_to(root).as_posix()
    required.add(config_rel)
    if not required <= set(release['files']):
        raise ValueError('Release omits required source/config files')
    for name, digest in release['files'].items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or hashlib.sha256(candidate.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError(f'Release source identity mismatch: {name}')
    if release.get('properties_sha256') != PROPERTY_SHA256:
        raise ValueError('Release property identity mismatch')
    return release


def supported_training_loss(result, batch, model, config):
    """Exclude unsupported rows without replacing draws or hiding denominators."""
    unsupported = result.domain_diagnostics['unsupported_rows']
    if len(unsupported) != len(batch.keys):
        raise ValueError('Per-window property support accounting is incomplete')
    support = torch.tensor([not v for v in unsupported], device=batch.observed_right_mask.device)
    mask = batch.observed_right_mask & support[:, None, None]
    count = int(mask.sum())
    if not count:
        return None, dict(supported_windows=int(support.sum()), unsupported_windows=int((~support).sum()),
                          supported_observed_scalars=0)
    losses = []
    for prediction in (result.free_predictions, result.prior_predictions):
        labels = torch.where(mask, batch.observations_right, prediction)
        loss = ((prediction - labels) / model.backend.observation_scale).square().sum() / count
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite supported training loss')
        losses.append(loss)
    total = config.free_weight * losses[0] + config.prior_weight * losses[1]
    if not bool(torch.isfinite(total)):
        raise ValueError('Nonfinite weighted supported training loss')
    return total, dict(supported_windows=int(support.sum()), unsupported_windows=int((~support).sum()),
                      supported_observed_scalars=count)


def validation_selection(records, requested_windows):
    """A conditional subset score never becomes a complete validation selection."""
    supported = [r for r in records if r['state'] == 'supported']
    incomplete = len(records) != requested_windows or len(supported) != requested_windows
    finite_score = all(math.isfinite(r['free_normalized_mse']) for r in supported)
    value = sum(r['free_normalized_mse'] for r in supported) / len(supported) if supported and finite_score else None
    return dict(requested_windows=requested_windows, completed_records=len(records),
                supported_windows=len(supported), unsupported_windows=sum(r['state'] == 'unsupported' for r in records),
                failed_windows=sum(r['state'] == 'failed' for r in records),
                conditional_supported_free_normalized_mse=value,
                selection_complete=not incomplete and finite_score,
                selection_value=value if not incomplete and finite_score else None,
                reduction='equal_weight_requested_window_only_if_all_supported; overlapping_windows_are_not_independent_episodes')


def evaluate_anchors(model, dataset, anchors, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    records, batches = [], []
    model.eval()
    for offset in range(0, len(anchors), config.batch_size):
        keys = keys_from_records(anchors[offset:offset + config.batch_size])
        start = time.perf_counter()
        try:
            batch = dataset.batch(keys, horizon=config.validation_horizon,
                                  device=model.backend.state_loc.device, dtype=model.backend.state_loc.dtype)
            with torch.no_grad():
                result = oracle_conditioned_objective(model, batch, for_training=False,
                            free_weight=config.free_weight, prior_weight=config.prior_weight)
            for i, key in enumerate(keys):
                mask = batch.observed_right_mask[i]
                count = int(mask.sum())
                if not count:
                    records.append(dict(**key.as_dict(), state='failed', reason='no_fresh_labels'))
                    continue
                pred, target = result.free_predictions[i][mask], batch.observations_right[i][mask]
                scale = model.backend.observation_scale.expand_as(batch.observations_right[i])[mask]
                records.append(dict(**key.as_dict(),
                    state='unsupported' if result.domain_diagnostics['unsupported_rows'][i] else 'supported',
                    observed_scalars=count, free_mae_degC=float((pred - target).abs().mean()),
                    free_normalized_mse=float(((pred - target) / scale).square().mean())))
            np.savez_compressed(directory / f'batch_{offset:04d}_private_arrays.npz',
                                free_predictions=result.free_predictions.cpu().numpy(),
                                prior_predictions=result.prior_predictions.cpu().numpy(),
                                observations=batch.observations_right.cpu().numpy(), mask=batch.observed_right_mask.cpu().numpy())
            batches.append(dict(offset=offset, keys=[key.as_dict() for key in keys],
                                anchor=result.anchor_diagnostics, domain=result.domain_diagnostics,
                                elapsed_seconds=time.perf_counter() - start))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            records += [dict(**key.as_dict(), state='failed', reason=type(exc).__name__) for key in keys]
            batches.append(dict(offset=offset, error=str(exc), exception_type=type(exc).__name__,
                                domain=model.backend.transition.properties.summary(), elapsed_seconds=time.perf_counter() - start))
    report = dict(records=records, batches=batches, summary=validation_selection(records, len(anchors)))
    write_json(directory / 'private_results.json', report)
    return report['summary']


def run(*, dataset_path, properties_path, output, config_path, release_path, variant, seed, device):
    # The user assigned scientific fitting to Linux; tests call pure helpers.
    if platform.system() != 'Linux':
        raise RuntimeError('Scientific prototype fitting must execute on Linux')
    release = verify_release(ROOT, release_path, config_path)
    torch.set_num_threads(1)
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    task = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if variant not in task['variants'] or variant not in VARIANTS or seed not in task['seeds']:
        raise ValueError('Variant/seed was not frozen in this task')
    settings = TrainingConfig(**task['training'])
    configuration = PlantModelConfig(variant=variant, seed=seed, **task['model'])
    expected_dataset = task['dataset']
    if expected_dataset != release['dataset']:
        raise ValueError('Task/release dataset binding mismatch')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device is unavailable')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output already exists; automatic resume/retry is forbidden')
    output.mkdir(parents=True)
    counters = dict(requested_batches=0, optimizer_steps=0, requested_windows=0, supported_windows=0,
                    unsupported_windows=0, skipped_batches=0, nonfinite_batches=0, failed_batches=0,
                    failed_unclassified_windows=0, nonfinite_batch_member_windows=0)
    status = dict(schema_version=1, task_id=task['task_id'], variant=variant, seed=seed,
                  stage='initializing', state='running', requested_batch_budget=settings.requested_batches,
                  best_present=False, selection_incomplete=True, device=device,
                  started_utc=datetime.now(timezone.utc).isoformat(), counters=counters)
    write_json(output / 'public_status.json', status)
    model, pending_keys, pending_classified = None, (), False
    try:
        dataset = PlantDataset.load(dataset_path, expected_identity_path=ROOT / task['expected_data_identity'])
        if dataset.identity != expected_dataset:
            raise ValueError('Loaded dataset metadata differs from frozen identity')
        plan = dataset.plan(updates=settings.requested_batches, batch_size=settings.batch_size,
                            sample_seed=settings.sample_seed, anchors_per_side=settings.anchors_per_side)
        write_json(output / 'private_input_plan.json', plan)
        plan_hash = text_sha256(output / 'private_input_plan.json')
        # The caller freezes this deterministic plan hash before any arm fits.
        if plan_hash != task['input_plan_sha256']:
            raise ValueError('Deterministic input plan differs from the frozen task')
        model = build_model(configuration, properties_path, device=device)
        torch.use_deterministic_algorithms(True)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        write_json(output / 'private_run_identity.json', dict(model=model_identity(model), dataset=dataset.identity,
                    local_container_identity=dataset.local_container_identity,
                    input_plan_sha256=plan_hash, release_sha256=file_sha256(release_path), training=asdict(settings),
                    initial_state_tensor_sha256=state_tensor_sha256(model.state_dict()),
                    runtime=dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                                 numpy=np.__version__, device=device, cuda_version=torch.version.cuda,
                                 cuda_device=torch.cuda.get_device_name() if device == 'cuda' else None,
                                 cpu_threads=torch.get_num_threads())))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
        best = math.inf

        def validate(step):
            nonlocal best
            status['stage'] = 'validation'
            write_json(output / 'public_status.json', status)
            summaries = {}
            for split in ('train', 'prototype_validation'):
                summaries[split] = evaluate_anchors(model, dataset, plan['diagnostic_anchors'][split], settings,
                                                    output / f'validation_{step:06d}' / split)
            selected = summaries['prototype_validation']
            status['selection_incomplete'] = not selected['selection_complete']
            if selected['selection_complete'] and selected['selection_value'] < best:
                best = selected['selection_value']
                payload = checkpoint_payload(model, dataset_identity=dataset.identity, plan_sha256=plan_hash,
                                             step=step, selection_value=best)
                payload['optimizer_steps'] = counters['optimizer_steps']
                torch.save(payload, output / 'best_private.pt')
                status['best_present'] = True
                status['best_requested_batch'] = step
                status['best_optimizer_steps'] = counters['optimizer_steps']
                status['best_validation_free_normalized_mse'] = best
            status['latest_validation'] = selected
            append_jsonl(output / 'public_validation_log.jsonl', dict(requested_batch=step,
                         optimizer_steps=counters['optimizer_steps'], summaries=summaries))
            append_jsonl(output / 'private_validation_log.jsonl', dict(requested_batch=step,
                         optimizer_steps=counters['optimizer_steps'], summaries=summaries))
            return summaries

        validate(0)
        for step, records in enumerate(plan['training'], 1):
            status['stage'] = 'training'
            counters['requested_batches'] += 1
            counters['requested_windows'] += len(records)
            write_json(output / 'public_status.json', status)
            started = time.perf_counter()
            keys = keys_from_records(records)
            pending_keys, pending_classified = keys, False
            batch = dataset.batch(keys, horizon=settings.train_horizon,
                                  device=model.backend.state_loc.device, dtype=model.backend.state_loc.dtype)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = oracle_conditioned_objective(model, batch, for_training=True,
                        free_weight=settings.free_weight, prior_weight=settings.prior_weight)
            loss, counts = supported_training_loss(result, batch, model, settings)
            counters['supported_windows'] += counts['supported_windows']
            counters['unsupported_windows'] += counts['unsupported_windows']
            pending_classified = True
            grad_norm = None
            if loss is None:
                counters['skipped_batches'] += 1
            else:
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise FloatingPointError('Nonfinite training gradient')
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip, error_if_nonfinite=True))
                optimizer.step()
                counters['optimizer_steps'] += 1
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('Nonfinite parameter after optimizer update')
            append_jsonl(output / 'private_training_log.jsonl', dict(requested_batch=step,
                         optimizer_steps=counters['optimizer_steps'], keys=records,
                         supported_loss=None if loss is None else float(loss.detach()),
                         all_requested_finite_loss=float(result.total.detach()), gradient_norm=grad_norm, counts=counts,
                         anchor=result.anchor_diagnostics, domain=result.domain_diagnostics,
                         elapsed_seconds=time.perf_counter() - started))
            pending_keys = ()
            if step % settings.validation_interval == 0 or step == settings.requested_batches:
                validate(step)
            write_json(output / 'public_status.json', status)
        status['state'] = 'complete_with_incomplete_selection' if status['selection_incomplete'] else 'complete'
        status['stage'] = 'requested_schedule_exhausted'
        status['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else None
        torch.save(dict(identity=model_identity(model), dataset_identity=dataset.identity, plan_sha256=plan_hash,
                        counters=dict(counters), state_dict={n: t.detach().cpu() for n, t in model.state_dict().items()},
                        optimizer_state=optimizer.state_dict()), output / 'last_private.pt')
    except Exception as exc:
        status['state'], status['stage'] = 'failed', 'terminal_no_automatic_retry'
        status['exception_type'] = type(exc).__name__
        nonfinite = 'nonfinite' in str(exc).lower() or 'not finite' in str(exc).lower()
        counters['failed_batches'] += int(bool(pending_keys))
        counters['nonfinite_batches'] += int(nonfinite and bool(pending_keys))
        counters['nonfinite_batch_member_windows'] += len(pending_keys) if nonfinite else 0
        counters['failed_unclassified_windows'] += len(pending_keys) if not pending_classified else 0
        write_json(output / 'private_failure.json', dict(error=str(exc), exception_type=type(exc).__name__,
                    requested_batch=counters['requested_batches'], counters=dict(counters),
                    domain=model.backend.transition.properties.summary() if model is not None else None))
        if model is not None:
            torch.save(dict(identity=model_identity(model), counters=dict(counters),
                            state_dict={n: t.detach().cpu() for n, t in model.state_dict().items()}), output / 'failure_state_private.pt')
        write_json(output / 'public_status.json', status)
        raise
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'public_status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'properties', 'output', 'config', 'release'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    print(json.dumps(run(dataset_path=args.dataset, properties_path=args.properties, output=args.output,
                         config_path=args.config, release_path=args.release, variant=args.variant, seed=args.seed,
                         device=args.device), indent=2, allow_nan=False))
