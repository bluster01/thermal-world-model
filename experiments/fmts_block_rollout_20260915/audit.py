"""Check private artifacts; optionally replay against the exact Linux record."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import run as R


def audit(out, parent, gnr, record=None, properties=None,
          mapping='configs/final_wm/channel_mapping_v2.json', device='cpu'):
    out = Path(out)
    manifest = R.read(out / 'manifest.json')
    names = [f'{a}_seed{s}' for a in R.P['arms'] for s in R.P['seeds']]
    expected = {'identity.json', 'summary.json', 'eligibility.npz', 'history_h18.npz', 'observed.npz'}
    expected |= {n + suffix for n in names for suffix in ('.json', '.npz')}
    assert set(manifest) == expected, 'manifest incomplete or unexpected'
    assert not (out / 'failure.json').exists(), 'failure recorded; not a complete run'
    for name, digest in manifest.items():
        assert R.sha256(out / name) == digest, f'artifact hash: {name}'
    identity = R.read(out / 'identity.json')
    assert identity['protocol'] == R.P, 'protocol drift'
    assert identity['sources'] == R.source_hashes(), 'wrapper/dependency source drift'
    assert identity['mapping'] == R.sha256(mapping), 'mapping drift'
    assert identity['training_updates'] == 0 and identity['locked_test_evaluated'] is False
    assert set(identity['checkpoints']) == set(names), 'checkpoint set'
    summary = R.read(out / 'summary.json')
    assert summary['status'] == 'complete' and summary['protocol_id'] == R.P['id']
    assert summary['training_updates'] == 0 and summary['locked_test_evaluated'] is False
    selected = R.npz(out / 'eligibility.npz')
    history = R.npz(out / 'history_h18.npz')
    observed = R.npz(out / 'observed.npz')
    starts = R.npz(Path(parent) / 'indices.npz')['validation']
    for prefix, folder in [('parent', Path(parent)), ('gnr', Path(gnr))]:
        for name in ('identity', 'summary'):
            assert R.sha256(folder / f'{name}.json') == R.P[f'{prefix}_{name}'], 'source identity'
        assert R.sha256(folder / 'indices.npz') == R.P['indices'], 'source indices'
    assert len(starts) == 256 and starts[R.P['fixed_case_rows']].tolist() == R.P['fixed_case_starts']
    assert set(selected) == {'rows', 'starts', 'eligible', *R.REASONS}
    assert np.array_equal(selected['starts'], starts) and np.array_equal(selected['rows'], np.arange(256))
    assert all(selected[k].dtype.kind == 'b' and selected[k].shape == (256,) for k in (*R.REASONS, 'eligible'))
    assert np.array_equal(selected['eligible'], ~np.stack([selected[k] for k in R.REASONS]).any(0))
    mask = selected['eligible']
    n = int(mask.sum())
    assert n > 0, 'empty common cohort'
    assert np.array_equal(observed['rows'], selected['rows'][mask])
    assert np.array_equal(observed['starts'], starts[mask])
    assert observed['target'].shape == observed['persistence'].shape == (n, 120)
    assert history['history'].shape == (256, 96) and history['target'].shape == (256, 18)
    assert observed['history'].shape == (n, 96)
    assert np.array_equal(history['rows'], np.arange(256)) and np.array_equal(history['starts'], starts)
    for k in ('history', 'history_times', 'days'):
        assert np.array_equal(observed[k], history[k][mask]), f'history pairing: {k}'
    for k in ('target', 'target_times'):
        assert np.array_equal(observed[k][:, :18], history[k][mask]), f'target pairing: {k}'
    assert np.array_equal(observed['persistence'], np.repeat(observed['history'][:, -1:], 120, axis=1))
    for arrays in (history, observed):
        assert all(np.isfinite(v).all() for v in arrays.values())
        timeline = np.concatenate([arrays['history_times'], arrays['target_times']], axis=1)
        assert (np.diff(timeline, axis=1) == 10).all(), 'trace time alignment'
        assert np.array_equal(arrays['days'], arrays['target_times'][:, 0] // 86400)
    assert summary['original_windows'] == 256 and summary['eligible_windows'] == n
    assert summary['days'] == len(np.unique(observed['days']))
    assert summary['exclusion_reason_counts'] == {k: int(selected[k].sum()) for k in R.REASONS}
    assert summary['fixed_cases_eligible'] == mask[R.P['fixed_case_rows']].tolist()
    assert summary['persistence'] == R.metrics(observed['persistence'], observed['target'], observed['days'])
    if record is not None:
        assert properties is not None, 'exact IAPWS required for real replay'
        actual_record, actual_starts, actual_selected = R.prepare(record, mapping, properties, parent, gnr)
        for saved, actual in [(selected, actual_selected), (history, R.history_export(actual_record, actual_starts)),
                              (observed, R.observed_rollout(actual_record, actual_selected))]:
            assert set(saved) == set(actual)
            assert all(np.array_equal(saved[k], actual[k]) for k in saved), 'exact record/cohort export mismatch'
        grid = R.load_grid_properties(properties)
    reports, replayed = [], 0
    for arm in R.P['arms']:
        for seed in R.P['seeds']:
            name = f'{arm}_seed{seed}'
            folder = R.source_folder(parent, gnr, arm, seed)
            assert R.sha256(folder / 'best.pt') == identity['checkpoints'][name]
            source_report = next(r for r in R.read(folder.parent / 'summary.json')['runs'] if (r['arm'], r['seed']) == (arm, seed))
            assert identity['checkpoints'][name] == source_report['checkpoint_sha256'], 'pinned checkpoint'
            assert R.sha256(folder / 'prediction.npz') == source_report['prediction_sha256'], 'pinned H18 arrays'
            pred = R.npz(out / f'{name}.npz')
            assert set(pred) == {'prediction'} and pred['prediction'].shape == (n, 120)
            saved = R.npz(folder / 'prediction.npz')
            for k in ('starts', 'days', 'target'):
                assert np.array_equal(saved[k], history[k]), f'original evidence {k}'
            R.replay_check(dict(prediction=saved['prediction'][mask]), dict(prediction=pred['prediction'][:, :18]))
            report = R.read(out / f'{name}.json')
            assert (report['arm'], report['seed'], report['status']) == (arm, seed, 'complete')
            assert report['prediction_sha256'] == manifest[f'{name}.npz']
            assert report['metrics'] == R.metrics(pred['prediction'], observed['target'], observed['days'])
            assert 'original_h18_gate' in report
            reports.append(report)
            if record is not None:
                model = R.load_model(folder, arm, grid, device)
                R.original_gate(model, actual_record, starts, folder, device)
                actual = R.collect(model, actual_record, observed['starts'], device)
                R.replay_check(pred, dict(prediction=actual))
                replayed += 1
                del model
    assert summary['runs'] == reports and summary['aggregate'] == R.aggregate(reports)
    return dict(protocol_id=R.P['id'], artifact_checks_passed=True, cells_checked=len(reports),
                real_record_replayed=record is not None, cells_replayed=replayed, complete=replayed == 9,
                training_updates=0, locked_test_evaluated=False,
                manifest_sha256=R.sha256(out / 'manifest.json'), identity_sha256=manifest['identity.json'],
                summary_sha256=manifest['summary.json'])


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--private-out', required=True)
    ap.add_argument('--parent', default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--gnr', default='results/fmts_greybox_norew_20260913/linux_full_gnr1')
    ap.add_argument('--record')
    ap.add_argument('--properties')
    ap.add_argument('--mapping', default='configs/final_wm/channel_mapping_v2.json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--save', required=True, help='new private audit JSON file')
    ap.add_argument('--public-receipt', help='new directory: aggregate-only return, no plant time series')
    args = ap.parse_args()
    R.private_output_path(args.save)
    if Path(args.save).exists() or (args.public_receipt and Path(args.public_receipt).exists()):
        raise FileExistsError('never overwrite audit or receipt')
    try:
        result = audit(args.private_out, args.parent, args.gnr, args.record, args.properties, args.mapping, args.device)
    except Exception as error:
        # Detailed failure paths/messages stay private; public receipt has only a status.
        result = dict(protocol_id=R.P['id'], complete=False, artifact_checks_passed=False,
                      status='failed', error=str(error), training_updates=0, locked_test_evaluated=False)
    with Path(args.save).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    if args.public_receipt:
        public = Path(args.public_receipt)
        public.mkdir(parents=True, exist_ok=False)
        (public / '.gitattributes').write_text('* -text\n', encoding='utf-8')
        if result['complete']:
            summary = R.read(Path(args.private_out) / 'summary.json')
            R.write(public / 'receipt.json', dict(audit=result, summary=summary,
                    private_audit_sha256=R.sha256(args.save), private_traces_included=False))
        else:
            R.write(public / 'receipt.json', dict(protocol_id=R.P['id'], complete=False,
                    private_audit_sha256=R.sha256(args.save), private_traces_included=False))
    print(json.dumps(result, indent=2))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
