"""Linux entry: matched four-arm continuation with starting-history auxiliary data."""
import argparse
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np
import torch

from .data import load_pack, sha256, unpack
from .evaluation import forecast, scenarios, response_summary
from .focused_models import ARMS, Focused
from .full_baselines import fit, complete_row
from .run import HERE, aggregate, save_json


def load_data(base_path, aux_path, train_horizon=32):
    data, meta = load_pack(base_path)
    aux, aux_meta = load_pack(aux_path)
    if len(data['train']) != 20371 or data['selector'].shape[1] != 192:
        raise ValueError('Use the full one-third H128-selector pack')
    names = aux['channel_names'].tolist()
    if 'sp2_b_set' not in names or 'sp2_a_set' in names:
        raise ValueError('v1 B stage2 setpoint required for crossed A-side action')
    selected = [i for i, name in enumerate(names) if name != 'fuel_ctl']
    for split in ('train', 'selector', 'evaluation'):
        for suffix in ('starts', 'time'):
            if not np.array_equal(data[f'{split}_{suffix}'], aux[f'{split}_{suffix}']):
                raise ValueError(f'{split} {suffix} mismatch')
        values = aux[f'hist12_{split}'][:, :, selected]
        if values.shape != (len(data[split]), 64, len(selected)) or not np.isfinite(values).all() or not aux[f'valid_{split}'].all():
            raise ValueError(f'{split} auxiliary coverage invalid; no silent row dropping')
        bank = data[split][:, :64+train_horizon] if split == 'train' else data[split]
        extended = np.zeros((*bank.shape[:2], 13+len(selected)), dtype=np.float32)
        extended[:, :, :13] = bank
        extended[:, :64, 13:] = values
        data[split] = extended
    data['aux_mean'], data['aux_scale'] = aux['mean'][selected], aux['scale'][selected]
    if not np.isfinite(data['aux_scale']).all() or (data['aux_scale'] <= 0).any():
        raise ValueError('Invalid auxiliary scale')
    return data, dict(base=meta, auxiliary=aux_meta, selected_channels=[names[i] for i in selected])


def make_model(arm, data, parents, device):
    # Identical adapter initialization for B/D, fixed parent training seed11.
    torch.manual_seed(11)
    model = Focused(data['mean'], data['scale'], data['aux_mean'], data['aux_scale'], arm)
    model.background.load_state_dict(torch.load(parents/'ssm/best_short.pt', weights_only=True, map_location='cpu')['model'])
    if model.response is not None:
        model.response.load_state_dict(torch.load(parents/'r4/best_short.pt', weights_only=True, map_location='cpu')['model'])
    return model.to(device)


@torch.no_grad()
def aligned_responses(model, bank, device, windows, smoke=False):
    """Same 110 shapes, each observed for 128 steps AFTER its onset."""
    positions = np.linspace(0, len(bank)-1, min(windows, len(bank)), dtype=int)
    history = unpack(bank[positions], device)[0].double()
    cases = scenarios(length=272)[:4 if smoke else 110]
    model.double().eval()
    rows, curves = [], {}
    try:
        for mode in ('block', 'native'):
            parts = []
            for case in cases:
                horizon = case['onset']+128
                nominal = history[:, -1:, 5:7].expand(-1, horizon+1, -1)
                boundary = history[:, -1:, 7:13].expand(-1, horizon+1, -1)
                candidate = nominal + torch.as_tensor(case['delta'][:horizon+1], device=device, dtype=history.dtype)[None]
                valid = ((candidate >= 0) & (candidate <= 1)).all((1, 2))
                safe = torch.where(valid[:, None, None], candidate, nominal)
                paired = forecast(model, history.repeat(2, 1, 1), torch.cat((safe, nominal)), boundary.repeat(2, 1, 1), mode=mode)
                delta = (paired[:len(history)]-paired[len(history):]).cpu().numpy()
                row = response_summary(delta, case, valid.cpu().numpy(), burn=case['onset'])
                row.update(mode=mode, observation_steps_after_onset=128, boundary_protocol='held at original history endpoint')
                rows.append(row)
                post = delta[:, case['onset']:].copy()
                post[~valid.cpu().numpy()] = np.nan
                parts.append(post)
            curves[mode] = np.stack(parts)
    finally:
        model.float()
    return rows, dict(**curves, window_positions=positions, case_ids=np.asarray([c['id'] for c in cases]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=HERE/'data/screen_A_33pct_h128.npz')
    p.add_argument('--aux', type=Path, default=HERE/'data/hist_bypass_A_33pct_v1.npz')
    p.add_argument('--parents', type=Path, default=HERE.parents[1]/'results/action_predictor_bench_20260919/full33_seed11/fits/seed11')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--response-windows', type=int, default=8)
    p.add_argument('--min-epochs', type=int, default=12)
    p.add_argument('--max-epochs', type=int, default=60)
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--min-lr', type=float, default=.0001)
    p.add_argument('--min-delta', type=float, default=.002)
    p.add_argument('--lr-patience', type=int, default=4)
    p.add_argument('--stop-patience', type=int, default=6)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if min(args.threads, args.batch_size, args.response_windows, args.min_epochs, args.lr_patience, args.stop_patience) < 1 or args.max_epochs < args.min_epochs or not 0 < args.min_lr <= args.learning_rate or args.min_delta < 0:
        p.error('Invalid budget')
    torch.set_num_threads(args.threads)
    data, metadata = load_data(args.data, args.aux)
    if args.smoke:
        for split, count in [('train', 16), ('selector', 4), ('evaluation', 4)]:
            data[split] = data[split][:count]
        data['evaluation_time'] = data['evaluation_time'][:4]
        args.min_epochs, args.max_epochs, args.batch_size, args.response_windows = 1, 1, 8, 2
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('resume', 'output')}
    config.update(arms=list(ARMS), seed=11, selected_auxiliary_channels=metadata['selected_channels'],
        data_sha256=sha256(args.data), auxiliary_sha256=sha256(args.aux),
        parent_sha256={n: sha256(args.parents/n/'best_short.pt') for n in ('ssm', 'r4')},
        source_sha256={f.name: sha256(f) for f in HERE.glob('*.py')},
        torch_version=torch.__version__, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        supervision_horizon=32, response_frozen=True, auxiliary_future='never read; initial summary retained across blocks')
    args.output.mkdir(parents=True, exist_ok=True)
    cp = args.output/'config.json'
    if cp.exists():
        old = json.loads(cp.read_text())
        if not args.resume or any(old.get(k) != v for k,v in config.items() if k != 'git_commit'):
            raise ValueError('Existing output requires --resume and identical config/code/data')
    else:
        save_json(cp, config)
    save_json(args.output/'data_metadata.json', metadata)
    np.savez_compressed(args.output/'evaluation_inputs.npz', bank=data['evaluation'], times=data['evaluation_time'])
    save_json(args.output/'state.json', dict(status='running', smoke=args.smoke))
    failures, costs = [], []
    for arm in ARMS:
        try:
            folder = args.output/'fits/seed11'/arm
            folder.mkdir(parents=True, exist_ok=True)
            model = make_model(arm, data, args.parents, args.device)
            cost = json.loads((folder/'fit.json').read_text()) if (folder/'fit.json').exists() else fit(arm, 11, data, args, folder, model=model)
            costs.append(cost)
            for choice in ('short', 'balanced'):
                name = f'{arm}_{choice}'
                model.load_state_dict(torch.load(folder/f'best_{choice}.pt', weights_only=True, map_location=args.device)['model'])
                model.eval()
                row_cost = {k: v for k,v in cost.items() if k not in ('status','model','seed')}
                row_cost.update(fit_id=f'seed11/{arm}', best_epoch=cost['best_epochs'][choice])
                complete_row(name, arm, choice, 11, model, row_cost, data, args, [('best.pt', folder/f'best_{choice}.pt')])
                target = args.output/'seed11'/name
                if not (target/'aligned_response_metrics.json').exists():
                    rows, curves = aligned_responses(model, data['evaluation'], args.device, args.response_windows, args.smoke)
                    np.savez_compressed(target/'aligned_responses.npz', **curves)
                    save_json(target/'aligned_response_metrics.json', rows)
        except Exception:
            failures.append(arm)
            (args.output/f'failure_{arm}.txt').write_text(traceback.format_exc(), encoding='utf-8')
            traceback.print_exc()
            if args.smoke:
                raise
        aggregate(args.output)
    from .analyze_return import analyze
    analyze(args.output, args.output)
    lines = ['# 四臂训练状态', '', '| Arm | Epochs | Stop | Best short | Best balanced |', '|---|---:|---|---:|---:|']
    for c in costs:
        lines.append(f"| {c['model']} | {c['epochs_run']} | {c['stop_reason']} | {c['best_epochs']['short']} | {c['best_epochs']['balanced']} |")
    (args.output/'CONVERGENCE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(args.output/'state.json', dict(status='completed_with_failures' if failures else 'complete', failures=failures,
        smoke=args.smoke, fits=len(costs), evaluation_rows_expected=8,
        not_confirmed_plateau=[c['model'] for c in costs if not c['reached_validation_plateau']]))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
