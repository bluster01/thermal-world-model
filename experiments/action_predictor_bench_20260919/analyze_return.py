"""Compact cross-model diagnosis from returned arrays; no fitting or selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from .run import save_json


def analyze(root, destination):
    root, destination = Path(root), Path(destination)
    bank = np.load(root / 'evaluation_inputs.npz')['bank']
    truth = bank[:, 64:, 4]
    # A descriptive reporting stratum, never used to select checkpoints.
    change = np.abs(bank[:, 64:96, 5:7] - bank[:, 63:64, 5:7]).mean((1, 2))
    threshold = np.quantile(change, .75)
    active = change >= threshold
    rows = []
    for folder in sorted(root.glob('seed*/*')):
        if not (folder / 'result.json').exists(): continue
        result = json.loads((folder / 'result.json').read_text())
        if result['status'] != 'complete': continue
        probes = json.loads((folder / 'response_metrics.json').read_text())
        with np.load(folder / 'forecasts.npz') as predictions:
            for mode in ('block', 'native'):
                key = mode + '_recorded'
                if key not in predictions: continue
                error = np.abs(predictions[key][:, :, 4].astype(float) - truth)
                finite = bool(np.isfinite(predictions[key]).all())
                if not finite: error[:] = np.nan  # keep the failed mode, never select finite windows
                rr = [x for x in probes if x['mode'] == mode and x['status'] == 'ok']
                signed = [x for x in rr if 'opposite_sign_fraction' in x]
                record = dict(model=result['name'], seed=result['seed'], mode=mode, forecast_is_finite=finite,
                    H32=float(error[:, :32].mean()), H128=float(error[:, :128].mean()),
                    H512=float(error.mean()), active_H32=float(error[active, :32].mean()),
                    quiet_H32=float(error[~active, :32].mean()),
                    upstream_max=max((x.get('unreachable_max_abs_C', 0) for x in rr), default=None),
                    pre_onset_max=max((x['pre_onset_max_abs_C'] for x in rr), default=None),
                    # Equal scenario weights; not a fraction pooled over all cells.
                    signed_scenario_mean_wrong_fraction=float(np.mean([x['opposite_sign_fraction'] for x in signed])) if signed else None,
                    step3pp_at0=[{k: x[k] for k in ('valves', 'terminal_main_C', 'peak_abs_main_C')}
                                 for x in rr if x['shape'] == 'step' and x['dose'] == .03
                                 and x['relative_onset'] == 0 and sum(abs(v) for v in x['valves']) == 1])
                rows.append(record)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / 'diagnosis.json', dict(active_threshold_fraction=float(threshold),
              active_windows=int(active.sum()), reporting_windows=len(bank), rows=rows))
    lines = ['# Returned baseline diagnosis', '',
             'One optimization seed; exploratory validation results. Main-temperature MAE in C.', '',
             '| Model | Mode | H32 | H128 | H512 | Active H32 | Quiet H32 | Upstream max | Wrong-sign scenario mean |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append('| ' + ' | '.join([x['model'], x['mode'], *[f'{x[k]:.4f}' if x[k] is not None and np.isfinite(x[k]) else '—' for k in
            ('H32', 'H128', 'H512', 'active_H32', 'quiet_H32', 'upstream_max', 'signed_scenario_mean_wrong_fraction')]]) + ' |')
    lines += ['', f'Active = top quartile of mean absolute recorded valve displacement over H32 ({active.sum()} windows).',
              'Wrong-sign mean includes only single-valve step/pulse/ramp/double-pulse cases with the metric defined.',
              'Response amplitude is a model sensitivity, not intervention accuracy. Zero response alone cannot qualify.',
              'Native and block are different simulation protocols; protected models explicitly retain carrier state in block mode.']
    (destination / 'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    analyze(args.root, args.destination)
