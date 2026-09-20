"""Linux-only fixed-weight diagnostics. No training, model selection or test scoring."""
import argparse
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from experiments.fmts_block_rollout_20260915.run import (
    load_model, private_output_path, source_folder, validate_sources,
)
from experiments.fmts_core_20260914.run import npz, read, replay_check, write
from experiments.fmts_mainsteam_20260911.data import RichRecord
from experiments.fmts_mainsteam_20260911.manifests import sha256
from experiments.fmts_mainsteam_20260911.run import evaluate
from src.final_wm.contracts import action_support_from_history
from src.final_wm.properties import load_grid_properties
from .design import P, PACKAGE, scenarios, perturb, opening_bins, opening_coverage, scenario_summary, factual_summary

ROOT = PACKAGE.parents[1]
REG = ROOT / 'docs/fmts2026/PREREG_VALVE_DIAGNOSTIC_20260920.md'


def canonical_hash(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_sha(path):
    """Source text identity is LF-normalized for Windows/Linux checkouts."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def source_hashes():
    files = [REG, PACKAGE / 'protocol.json', ROOT / P['source_contract']]
    for folder in ('src/final_wm', 'experiments/fmts_mainsteam_20260911',
                   'experiments/fmts_greybox_norew_20260913', 'experiments/fmts_core_20260914',
                   'experiments/fmts_block_rollout_20260915', 'experiments/fmts_valve_diagnostic_20260920'):
        files.extend(sorted((ROOT / folder).glob('*.py')))
    return {p.relative_to(ROOT).as_posix(): source_sha(p) for p in files}


def weight_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def context_export(record, starts, thresholds):
    starts = torch.as_tensor(starts, dtype=torch.long)
    past = starts[:, None] + torch.arange(-96, 0)
    future = starts[:, None] + torch.arange(18)
    context = dict(starts=starts.numpy(), days=(record.timestamps[starts] // 86400).numpy(),
                   history_obs=record.obs[past].numpy(), history_actions=record.actions[past].numpy(),
                   history_boundary=record.boundary[past].numpy(), history_extension=record.boundary_ext[past].numpy(),
                   history_times=record.timestamps[past].numpy(), future_times=record.timestamps[future].numpy(),
                   future_actions=record.actions[future].numpy(), future_boundary=record.boundary[future].numpy(),
                   target=record.obs[future].numpy())
    context['opening_bin'] = opening_bins(context['history_actions'][:, -1], thresholds)
    return context


@torch.no_grad()
def collect_probes(model, record, starts, device):
    """Paired runs use original futures; only history probes re-infer from altered history."""
    ss = scenarios()
    collected = {s['id']: {} for s in ss}
    for idx in torch.as_tensor(starts).split(P['batch_size']):
        h, ext, _, _, _, _ = record.batch(idx, 1, device)
        actions = h.actions[:, -1:].expand(-1, 18, -1).clone()
        boundary = h.boundary[:, -1:].expand(-1, 18, -1).clone()

        def predict(history, a, b):
            out = model(history, ext, a, b)
            pred = out if isinstance(out, torch.Tensor) else out.temps_mu[:, :, 4]
            if pred.shape != (len(idx), 18) or not torch.isfinite(pred).all():
                raise ValueError('nonfinite or wrongly shaped prediction')
            return pred

        base = predict(h, actions, boundary)
        if not torch.equal(base, predict(h, actions.clone(), boundary.clone())):
            raise ValueError('zero-change identity failed')
        support = action_support_from_history(h.actions, P['support_margin'])
        for scenario in ss:
            hh, a, b, da, dw = perturb(h, actions, boundary, scenario)
            pred = predict(hh, a, b)
            if scenario['family'] == 'history':
                supported = support.contains(hh.actions) & support.contains(h.actions)
            else:
                supported = support.contains(a) & support.contains(actions)
            values = dict(starts=idx, base=base, prediction=pred, valve_dose=da, spray_dose=dw, support=supported)
            for key, value in values.items():
                collected[scenario['id']].setdefault(key, []).append(value.detach().cpu().numpy())
    return {name: {k: np.concatenate(v) for k, v in raw.items()} for name, raw in collected.items()}


def summarize_joint(raws):
    """Factorial non-additivity under artificial, explicitly uncalibrated joint changes."""
    ss = scenarios()
    result = {}
    for joint in [s for s in ss if s['family'] == 'joint']:
        valve = next(s for s in ss if s['family'] == 'future' and s['valve'] == joint['valve']
                     and s['onset'] == 0 and s['shape'] == 'step' and s['dose'] == joint['dose'])
        water = next(s for s in ss if s['family'] == 'spray' and s['spray_dose'] == joint['spray_dose'])
        j, v, w = [raws[s['id']] for s in (joint, valve, water)]
        interaction = j['prediction'].astype(float)-v['prediction'].astype(float)-w['prediction'].astype(float)+j['base'].astype(float)
        result[joint['id']] = interaction
    return result


def aggregate(reports):
    expected = {(a, s) for a in P['arms'] for s in P['seeds']}
    if len(reports) != len(expected) or {(r['arm'], r['seed']) for r in reports} != expected:
        raise ValueError('incomplete or duplicate checkpoint cells')
    result = {}
    for arm in P['arms']:
        rows = sorted([r for r in reports if r['arm'] == arm], key=lambda r: r['seed'])
        cells = {}
        for scenario in scenarios():
            values = [r['scenarios'][scenario['id']] for r in rows]
            entry = dict(n_seeds=len(rows), n_windows=values[0]['n_windows'], n_days=values[0]['n_days'])
            for key in ('signed_curve_c', 'absolute_curve_c', 'elapsed60_signed_c',
                        'elapsed60_absolute_c', 'pre_action_max_abs_c', 'pre_action_mean_abs_c'):
                if values[0][key] is None:
                    entry[key] = None
                else:
                    a = np.asarray([v[key] for v in values])
                    entry[key] = dict(mean=a.mean(0).tolist(), seed_sd=a.std(0, ddof=1).tolist())
            cells[scenario['id']] = entry
        result[arm] = cells
    return result


def execute(args):
    if platform.system() != P['formal_execution_platform']:
        raise RuntimeError('formal real-record inference is Linux-only; use local fixture tests')
    out = private_output_path(args.private_out)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    try:
        source_contract = read(ROOT / P['source_contract'])
        assert canonical_hash(source_contract) == P['source_contract_canonical_sha256'], 'source contract changed'
        assert P['arms'] == source_contract['arms'] and P['seeds'] == source_contract['seeds']
        assert not P['training'] and not P['locked_test_evaluated'] and not P['plant_response_truth']
        validate_sources(args.parent, args.gnr, args.record, args.mapping, args.properties)
        record = RichRecord(args.record, args.mapping)
        indices = npz(Path(args.parent) / 'indices.npz')
        assert len(indices['response']) == 64 and len(indices['validation']) == 256
        allowed = set(record.candidates(1).tolist())
        assert all(int(s) in allowed for key in ('response', 'validation') for s in indices[key]), 'window eligibility changed'
        training_last = torch.as_tensor(indices['train']) - 1
        assert bool((record.split[training_last] == 0).all()), 'opening bins must use training only'
        training_openings = record.actions[training_last].numpy()
        assert np.isfinite(training_openings).all()
        thresholds = np.quantile(training_openings, P['opening_quantiles'], axis=0).T
        contexts = {key: context_export(record, indices[key], thresholds) for key in ('response', 'validation')}
        for key, context in contexts.items():
            np.savez_compressed(out / f'{key}_inputs.npz', **context)
        identity = dict(protocol=P, scenarios=scenarios(), opening_thresholds=thresholds.tolist(),
                        source_contract=source_contract, sources=source_hashes(), source_hash_format='LF-normalized text',
                        record_sha256=sha256(args.record), mapping_sha256=sha256(args.mapping),
                        properties_sha256=sha256(args.properties), indices_sha256=sha256(Path(args.parent)/'indices.npz'),
                        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                        device=args.device, cuda_version=torch.version.cuda,
                        float32_matmul_precision=torch.get_float32_matmul_precision(),
                        cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
                        training_updates=0, locked_test_evaluated=False, plant_response_truth=False)
        write(out / 'identity.json', identity)
        for arm in P['arms']:
            for seed in P['seeds']:
                cell = f'{arm}_seed{seed}'
                print(f'VD1 {cell}: original replay then 78 diagnostics; no training', flush=True)
                dest = out / cell
                dest.mkdir()
                folder = source_folder(args.parent, args.gnr, arm, seed)
                model = load_model(folder, arm, load_grid_properties(args.properties), args.device)
                model.eval().requires_grad_(False)
                before = weight_hash(model)
                factual = evaluate(model, record, torch.as_tensor(indices['validation']), args.device)
                original_response = evaluate(model, record, torch.as_tensor(indices['response']), args.device, response=True)
                for name, raw in [('prediction', factual), ('response', original_response)]:
                    np.savez_compressed(dest / f'original_{name}.npz', **raw)
                gates = {name: replay_check(npz(folder/f'{name}.npz'), raw)
                         for name, raw in [('prediction', factual), ('response', original_response)]}
                raws = collect_probes(model, record, indices['response'], args.device)
                after = weight_hash(model)
                assert before == after, 'checkpoint weights mutated'
                summary = {}
                for scenario in scenarios():
                    raw = raws[scenario['id']]
                    np.savez_compressed(dest / f'{scenario["id"]}.npz', **raw)
                    summary[scenario['id']] = scenario_summary(raw, contexts['response'], scenario)
                    # Exact same-hardware/reference protocol reproduction on matching probes.
                    if scenario['family'] == 'future' and scenario['onset'] == 0 and scenario['shape'] == 'step' and scenario['dose'] == .05:
                        assert np.array_equal(raw['base'], original_response['base']), 'new baseline differs'
                        assert np.array_equal(raw['prediction'], original_response[f'valve{scenario["valve"]+1}_prediction']), 'matching probe differs'
                from .design import summarize_effect
                joint = {sid: summarize_effect(delta, contexts['response']['days'], next(s for s in scenarios() if s['id'] == sid))
                         for sid, delta in summarize_joint(raws).items()}
                report = dict(arm=arm, seed=seed, checkpoint_sha256=sha256(folder/'best.pt'),
                              original_replay=gates, weight_hash_before=before, weight_hash_after=after,
                              scenarios=summary, artificial_joint_interaction=joint,
                              factual=factual_summary(factual, contexts['validation']), training_updates=0)
                write(dest / 'report.json', report)
                reports.append(report)
                write(out / 'progress.json', [dict(arm=r['arm'], seed=r['seed']) for r in reports])
                del model
        result = dict(id=P['id'], cells_completed=len(reports), runs=reports, aggregate=aggregate(reports),
                      opening_thresholds=thresholds.tolist(), opening_coverage={k: opening_coverage(c) for k, c in contexts.items()},
                      training_updates=0, locked_test_evaluated=False, plant_response_truth=False,
                      scientific_verdict='EXPLORATORY_INPUT_SENSITIVITY_NOT_PLANT_RESPONSE_VALIDATION')
        write(out / 'summary.json', result)
        files = sorted(p for p in out.rglob('*') if p.is_file())
        write(out / 'manifest.json', {p.relative_to(out).as_posix(): sha256(p) for p in files})
        return result
    except Exception as exc:
        write(out / 'failure.json', dict(error=str(exc), cells_completed=len(reports), training_updates=0))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--parent', default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--gnr', default='results/fmts_greybox_norew_20260913/linux_full_gnr1')
    ap.add_argument('--record', required=True)
    ap.add_argument('--properties', required=True)
    ap.add_argument('--mapping', default='configs/final_wm/channel_mapping_v2.json')
    ap.add_argument('--private-out', required=True)
    ap.add_argument('--device', default='cuda')
    execute(ap.parse_args())
