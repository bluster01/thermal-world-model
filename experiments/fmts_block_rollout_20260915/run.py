"""Fixed H18 checkpoints, complete target-block feedback, no training."""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from experiments.fmts_core_20260914.run import load_source, npz, read, replay_check, write
from experiments.fmts_greybox_norew_20260913.audit import audit as audit_gnr
from experiments.fmts_greybox_norew_20260913.run import validate_parent
from experiments.fmts_mainsteam_20260911.data import RichRecord
from experiments.fmts_mainsteam_20260911.manifests import sha256
from experiments.fmts_mainsteam_20260911.models import RichBlackbox
from experiments.fmts_mainsteam_20260911.run import day_mean
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
REG = ROOT / 'docs/fmts2026/PREREG_BLOCK_ROLLOUT_20260915.md'
P = read(PACKAGE / 'protocol.json')
REASONS = ('out_of_bounds', 'non_validation', 'invalid_base', 'time_gap', 'invalid_history_extension')


def private_output_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT) or any((p / '.git').exists() for p in (path, *path.parents)):
        raise ValueError('private traces must be outside the repository')
    return path


def eligibility(record, starts):
    """Fixed-origin quality filter, independent of predictions or error values."""
    starts = np.asarray(starts, dtype=np.int64)
    result = dict(starts=starts, rows=np.arange(len(starts), dtype=np.int64))
    result.update({k: np.zeros(len(starts), dtype=bool) for k in REASONS})
    for row, s in enumerate(starts):
        lo, hi = int(s) - 96, int(s) + 126
        if lo < 0 or hi > record.n:
            result['out_of_bounds'][row] = True
            continue
        result['non_validation'][row] = not bool((record.split[lo:hi] == 1).all())
        result['invalid_base'][row] = not bool(record.base_valid[lo:hi].all())
        result['time_gap'][row] = not bool((torch.diff(record.timestamps[lo:hi]) == 10).all())
        result['invalid_history_extension'][row] = not bool(record.ext_valid[lo:int(s)+108].all())
    result['eligible'] = ~np.stack([result[k] for k in REASONS]).any(0)
    return result


@torch.no_grad()
def rollout_batch(model, record, starts, device):
    """Only main temperature is synthetic. Each model call reinitializes itself."""
    starts = torch.as_tensor(starts, dtype=torch.long, device='cpu')
    idx = starts[:, None] + torch.arange(-96, 0)
    obs = record.obs[idx].to(device).clone()
    pieces = []
    for offset in range(0, 120, 18):
        past = starts[:, None] + offset + torch.arange(-96, 0)
        future = starts[:, None] + offset + torch.arange(18)
        history = HistoryWindow(obs, record.actions[past].to(device), record.boundary[past].to(device))
        out = model(history, record.boundary_ext[past].to(device),
                    record.actions[future].to(device), record.boundary[future].to(device))
        pred = out if isinstance(out, torch.Tensor) else out.temps_mu[:, :, 4]
        assert pred.shape == (len(starts), 18) and bool(torch.isfinite(pred).all()), 'invalid prediction block'
        pieces.append(pred[:, :min(18, 120-offset)])
        if offset + 18 < 120:
            next_obs = record.obs[future].to(device).clone()
            next_obs[:, :, 4] = pred
            obs = torch.cat([obs[:, 18:], next_obs], dim=1)
    return torch.cat(pieces, dim=1)


def history_export(record, starts):
    idx = torch.as_tensor(starts, dtype=torch.long)
    past = idx[:, None] + torch.arange(-96, 0)
    future = idx[:, None] + torch.arange(18)
    return dict(starts=np.asarray(starts), rows=np.arange(len(starts)),
                days=(record.timestamps[idx] // 86400).numpy(),
                history=record.obs[past, 4].numpy(), history_times=record.timestamps[past].numpy(),
                target=record.obs[future, 4].numpy(), target_times=record.timestamps[future].numpy())


def observed_rollout(record, selected):
    rows = selected['rows'][selected['eligible']]
    starts = selected['starts'][selected['eligible']]
    idx = torch.as_tensor(starts)
    past = idx[:, None] + torch.arange(-96, 0)
    future = idx[:, None] + torch.arange(120)
    history = record.obs[past, 4].numpy()
    return dict(rows=rows, starts=starts, days=(record.timestamps[idx] // 86400).numpy(),
                history=history, history_times=record.timestamps[past].numpy(),
                target=record.obs[future, 4].numpy(), target_times=record.timestamps[future].numpy(),
                persistence=np.repeat(history[:, -1:], 120, axis=1))


def metrics(prediction, target, days):
    assert prediction.shape == target.shape and prediction.shape[1] == 120
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    error = np.abs(prediction.astype(float) - target)
    curve = day_mean(error.cumsum(1) / np.arange(1, 121), days)
    return dict(cumulative_mae=curve.tolist(), **{f'H{h}': float(curve[h-1]) for h in (18, 36, 60, 120)})


def aggregate(rows):
    expected = {(a, s) for a in P['arms'] for s in P['seeds']}
    assert len(rows) == len(expected) and {(r['arm'], r['seed']) for r in rows} == expected, 'incomplete cells'
    result = {}
    for arm in P['arms']:
        curves = np.array([r['metrics']['cumulative_mae'] for r in sorted(rows, key=lambda r: r['seed']) if r['arm'] == arm])
        result[arm] = dict(seeds=P['seeds'], mean=curves.mean(0).tolist(), seed_sd=curves.std(0, ddof=1).tolist())
    return result


def source_hashes():
    files = [REG, PACKAGE / 'protocol.json']
    for directory in ('src/final_wm', 'experiments/fmts_mainsteam_20260911',
                      'experiments/fmts_greybox_norew_20260913', 'experiments/fmts_core_20260914',
                      'experiments/fmts_block_rollout_20260915'):
        files.extend(sorted((ROOT / directory).glob('*.py')))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def source_folder(parent, gnr, arm, seed):
    return Path(gnr if arm == P['arms'][2] else parent) / f'{arm}_seed{seed}'


def validate_sources(parent, gnr, record, mapping, properties):
    for prefix, folder in [('parent', Path(parent)), ('gnr', Path(gnr))]:
        for name in ('identity', 'summary'):
            assert sha256(folder / f'{name}.json') == P[f'{prefix}_{name}'], f'{prefix} {name} fingerprint'
        assert sha256(folder / 'indices.npz') == P['indices'], 'fixed windows changed'
    assert sha256(record) == P['record'] and sha256(properties) == P['properties'], 'real input fingerprint'
    validate_parent(parent, record, mapping, properties, smoke=False)
    assert audit_gnr(gnr, parent)['complete'], 'GNR artifact provenance'
    for arm in P['arms']:
        for seed in P['seeds']:
            folder = source_folder(parent, gnr, arm, seed)
            frozen = next(r for r in read(folder.parent / 'summary.json')['runs'] if (r['arm'], r['seed']) == (arm, seed))
            assert read(folder / 'report.json') == frozen, 'source report differs from pinned summary'
            assert sha256(folder / 'best.pt') == frozen['checkpoint_sha256'], 'source checkpoint'


def load_model(folder, arm, properties, device):
    folder = Path(folder)
    if arm != 'blackbox_itransformer':
        return load_source(folder, properties, device)
    assert sha256(folder / 'best.pt') == read(folder / 'report.json')['checkpoint_sha256']
    payload = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    assert payload['identity_sha256'] == sha256(folder.parent / 'identity.json')
    model = RichBlackbox(payload['normalization']['mean'], payload['normalization']['std'])
    model.load_state_dict(payload['state_dict'], strict=True)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in payload['state_dict'].items())
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def original_gate(model, record, starts, folder, device):
    parts = []
    for idx in torch.as_tensor(starts).split(32):
        h, ext, actions, boundary, target, days = record.batch(idx, 1, device)
        out = model(h, ext, actions, boundary)
        pred = out if isinstance(out, torch.Tensor) else out.temps_mu[:, :, 4]
        parts.append(dict(starts=idx.numpy(), days=days.numpy(), prediction=pred.cpu().numpy(),
                          target=target[:, :, 4].cpu().numpy(), persistence=h.obs[:, -1:, 4].expand(-1, 18).cpu().numpy()))
    actual = {k: np.concatenate([r[k] for r in parts]) for k in parts[0]}
    saved = npz(folder / 'prediction.npz')
    for k in ('starts', 'days', 'target', 'persistence'):
        assert np.array_equal(saved[k], actual[k]), f'H18 exact {k}'
    return replay_check(saved, actual)


def collect(model, record, starts, device):
    return np.concatenate([rollout_batch(model, record, idx, device).cpu().numpy()
                           for idx in torch.as_tensor(starts).split(32)])


def prepare(record_path, mapping, properties_path, parent, gnr):
    validate_sources(parent, gnr, record_path, mapping, properties_path)
    record = RichRecord(record_path, mapping)
    starts = npz(Path(parent) / 'indices.npz')['validation']
    assert len(starts) == 256 and starts[P['fixed_case_rows']].tolist() == P['fixed_case_starts'], 'origin manifest'
    allowed = set(record.candidates(1).tolist())
    assert all(int(s) in allowed for s in starts), 'original H18 window no longer eligible'
    selected = eligibility(record, starts)
    return record, starts, selected


def execute(args):
    out = private_output_path(args.private_out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        record, starts, selected = prepare(args.record, args.mapping, args.properties, args.parent, args.gnr)
        identity = dict(protocol=P, sources=source_hashes(), mapping=sha256(args.mapping),
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                        platform=platform.platform(), device=args.device, cuda_version=torch.version.cuda,
                        cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
                        matmul_precision=torch.get_float32_matmul_precision(),
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',
                        training_updates=0, locked_test_evaluated=False,
                        checkpoints={f'{a}_seed{s}': sha256(source_folder(args.parent, args.gnr, a, s) / 'best.pt')
                                     for a in P['arms'] for s in P['seeds']})
        write(out / 'identity.json', identity)
        write(out / 'progress.json', dict(status='running', completed_cells=0))
        np.savez_compressed(out / 'eligibility.npz', **selected)
        np.savez_compressed(out / 'history_h18.npz', **history_export(record, starts))
        assert selected['eligible'].any(), 'no eligible long windows; do not resample'
        observed = observed_rollout(record, selected)
        np.savez_compressed(out / 'observed.npz', **observed)
        properties = load_grid_properties(args.properties)
        for arm in P['arms']:
            for seed in P['seeds']:
                folder = source_folder(args.parent, args.gnr, arm, seed)
                model = load_model(folder, arm, properties, args.device)
                gate = original_gate(model, record, starts, folder, args.device)
                prediction = collect(model, record, observed['starts'], args.device)
                frozen = npz(folder / 'prediction.npz')['prediction'][observed['rows']]
                replay_check(dict(prediction=frozen), dict(prediction=prediction[:, :18]))
                name = f'{arm}_seed{seed}'
                np.savez_compressed(out / f'{name}.npz', prediction=prediction)
                report = dict(arm=arm, seed=seed, original_h18_gate=gate, status='complete',
                              prediction_sha256=sha256(out / f'{name}.npz'),
                              metrics=metrics(prediction, observed['target'], observed['days']))
                write(out / f'{name}.json', report)
                rows.append(report)
                write(out / 'progress.json', dict(status='running', completed_cells=len(rows)))
                print(f'{name}: complete ({len(rows)}/9)', flush=True)
                del model
        summary = dict(protocol_id=P['id'], status='complete', training_updates=0, locked_test_evaluated=False,
                       original_windows=len(starts), eligible_windows=int(selected['eligible'].sum()),
                       days=len(np.unique(observed['days'])),
                       exclusion_reason_counts={k: int(selected[k].sum()) for k in REASONS},
                       fixed_cases_eligible=selected['eligible'][P['fixed_case_rows']].tolist(),
                       runs=rows, aggregate=aggregate(rows),
                       persistence=metrics(observed['persistence'], observed['target'], observed['days']))
        write(out / 'summary.json', summary)
        artifacts = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.suffix in ('.npz', '.json') and p.name != 'progress.json'}
        write(out / 'manifest.json', artifacts)
        write(out / 'progress.json', dict(status='complete', completed_cells=9, audited=False))
        return summary
    except Exception as error:
        write(out / 'failure.json', dict(status='failed', completed_cells=len(rows), error=str(error),
                                       training_updates=0, locked_test_evaluated=False))
        raise


def parser():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--private-out', required=True, help='new directory OUTSIDE Git checkout')
    ap.add_argument('--record', required=True)
    ap.add_argument('--properties', required=True)
    ap.add_argument('--mapping', default='configs/final_wm/channel_mapping_v2.json')
    ap.add_argument('--parent', default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--gnr', default='results/fmts_greybox_norew_20260913/linux_full_gnr1')
    ap.add_argument('--device', default='cpu')
    return ap


if __name__ == '__main__':
    print(json.dumps(execute(parser().parse_args()), indent=2))
