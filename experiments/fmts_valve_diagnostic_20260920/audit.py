"""Recompute VD1 statistics from saved traces; no new model inference."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.final_wm.model import HistoryWindow
from src.final_wm.contracts import action_support_from_history
from .design import P, scenarios, perturb, opening_bins, opening_coverage, scenario_summary, factual_summary, summarize_effect
from .run import (ROOT, read, write, npz, sha256, canonical_hash, source_folder,
                  replay_check, summarize_joint, aggregate, source_sha)


def audit(out, parent=None, gnr=None, verify_local_sources=True):
    out = Path(out)
    if (out / 'failure.json').exists():
        raise ValueError('run has a failure receipt')
    parent = Path(parent) if parent else ROOT / 'results/fmts_mainsteam_20260911/linux_full_v02'
    gnr = Path(gnr) if gnr else ROOT / 'results/fmts_greybox_norew_20260913/linux_full_gnr1'
    manifest = read(out / 'manifest.json')
    for name, fingerprint in manifest.items():
        path = (out / name).resolve()
        assert path.is_relative_to(out.resolve()) and path.is_file(), 'unsafe/missing manifest path'
        assert sha256(path) == fingerprint, f'changed artifact: {name}'
    identity, summary = read(out / 'identity.json'), read(out / 'summary.json')
    assert identity['protocol'] == P and identity['scenarios'] == scenarios(), 'protocol mismatch'
    assert canonical_hash(identity['source_contract']) == P['source_contract_canonical_sha256'], 'parent contract mismatch'
    for key in ('record', 'properties', 'indices'):
        assert identity[f'{key}_sha256'] == identity['source_contract'][key], f'{key} mismatch'
    if verify_local_sources:
        for name, fingerprint in identity['sources'].items():
            path = (ROOT / name).resolve()
            assert path.is_relative_to(ROOT) and source_sha(path) == fingerprint, f'local source differs: {name}'
    contexts = {key: npz(out / f'{key}_inputs.npz') for key in ('response', 'validation')}
    thresholds = np.asarray(identity['opening_thresholds'])
    assert summary['opening_thresholds'] == identity['opening_thresholds']
    for key, context in contexts.items():
        assert len(context['starts']) == P['response_windows' if key == 'response' else 'prediction_windows']
        for a in context.values():
            assert np.isfinite(a).all(), 'nonfinite context'
        assert np.array_equal(context['opening_bin'], opening_bins(context['history_actions'][:, -1], thresholds))
    assert summary['opening_coverage'] == {k: opening_coverage(c) for k, c in contexts.items()}
    ctx = contexts['response']
    history = HistoryWindow(*(torch.from_numpy(ctx[key]) for key in ('history_obs', 'history_actions', 'history_boundary')))
    actions = history.actions[:, -1:].expand(-1, 18, -1).clone()
    boundaries = history.boundary[:, -1:].expand(-1, 18, -1).clone()
    support = action_support_from_history(history.actions, P['support_margin'])
    required = {'identity.json', 'response_inputs.npz', 'validation_inputs.npz', 'progress.json', 'summary.json'}
    reports = []
    for arm in P['arms']:
        for seed in P['seeds']:
            label = f'{arm}_seed{seed}'
            cell, source = out / label, source_folder(parent, gnr, arm, seed)
            report = read(cell / 'report.json')
            assert report['arm'] == arm and report['seed'] == seed
            assert report['checkpoint_sha256'] == sha256(source / 'best.pt'), 'checkpoint changed'
            assert report['weight_hash_before'] == report['weight_hash_after'], 'weights changed'
            required.update(f'{label}/{name}' for name in ('report.json', 'original_prediction.npz', 'original_response.npz'))
            factual, original = npz(cell / 'original_prediction.npz'), npz(cell / 'original_response.npz')
            assert np.array_equal(factual['starts'], contexts['validation']['starts'])
            assert np.array_equal(factual['days'], contexts['validation']['days'])
            assert np.array_equal(factual['target'], contexts['validation']['target'][:, :, 4])
            assert np.array_equal(original['starts'], ctx['starts']) and np.array_equal(original['days'], ctx['days'])
            for name, raw in [('prediction', factual), ('response', original)]:
                assert report['original_replay'][name] == replay_check(npz(source / f'{name}.npz'), raw)
            assert report['factual'] == factual_summary(factual, contexts['validation'])
            raws = {}
            for scenario in scenarios():
                sid = scenario['id']
                required.add(f'{label}/{sid}.npz')
                raw = npz(cell / f'{sid}.npz')
                assert np.array_equal(raw['starts'], ctx['starts'])
                assert np.array_equal(raw['base'], original['base'])
                assert all(np.isfinite(a).all() for a in raw.values())
                hh, aa, _, da, dw = perturb(history, actions, boundaries, scenario)
                assert np.array_equal(raw['valve_dose'], da.numpy()) and np.array_equal(raw['spray_dose'], dw.numpy()), 'dose mismatch'
                expected_support = (support.contains(hh.actions) & support.contains(history.actions) if scenario['family'] == 'history'
                                    else support.contains(aa) & support.contains(actions))
                assert np.array_equal(raw['support'], expected_support.numpy()), 'support mismatch'
                assert report['scenarios'][sid] == scenario_summary(raw, ctx, scenario), 'derived statistics mismatch'
                if scenario['family'] == 'future' and scenario['onset'] == 0 and scenario['shape'] == 'step' and scenario['dose'] == .05:
                    assert np.array_equal(raw['prediction'], original[f'valve{scenario["valve"]+1}_prediction'])
                raws[sid] = raw
            joint = {sid: summarize_effect(delta, ctx['days'], next(s for s in scenarios() if s['id'] == sid))
                     for sid, delta in summarize_joint(raws).items()}
            assert report['artificial_joint_interaction'] == joint
            reports.append(report)
    assert required == set(manifest), 'manifest incomplete/unexpected files'
    assert summary['runs'] == reports and summary['aggregate'] == aggregate(reports), 'aggregate differs'
    assert summary['cells_completed'] == 9 and summary['training_updates'] == 0
    assert identity['training_updates'] == 0 and not identity['locked_test_evaluated']
    assert not summary['locked_test_evaluated'] and not summary['plant_response_truth']
    return dict(id=P['id'], complete_artifact_audit=True, cells_checked=9,
                scenarios_checked=9*P['scenarios_per_checkpoint'], original_replays_checked=18,
                new_probe_full_inference_replay=False, training_updates=0,
                locked_test_evaluated=False, plant_response_truth=False,
                manifest_sha256=sha256(out / 'manifest.json'), identity_sha256=sha256(out / 'identity.json'),
                scientific_verdict='PENDING_AUTHOR_INTERPRETATION')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--private-out', required=True)
    ap.add_argument('--parent')
    ap.add_argument('--gnr')
    ap.add_argument('--public-receipt', required=True)
    args = ap.parse_args()
    result = audit(args.private_out, args.parent, args.gnr)
    receipt = Path(args.public_receipt)
    receipt.mkdir(parents=True, exist_ok=False)
    write(receipt / 'audit.json', result)
    write(receipt / 'summary.json', read(Path(args.private_out) / 'summary.json'))
