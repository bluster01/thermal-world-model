"""Compact, common-axis plots; no model selection in reporting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report(output):
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    entries = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    bank = np.load(output / 'evaluation_inputs.npz')['bank']
    truth = bank[:, 64:, 4]
    time = np.arange(1, truth.shape[1]+1) * 10 / 60
    selected = np.linspace(0, len(bank)-1, min(6, len(bank)), dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = sorted({r['model'] for r in entries})
    colors = dict(zip(names, plt.cm.tab20(np.linspace(0, 1, len(names)))))
    learning, la = plt.subplots(1, 2, figsize=(12, 4))
    for r in entries:
        if r['status'] != 'complete': continue
        folder = output / f"seed{r['seed']}" / r['model']
        with np.load(folder / 'forecasts.npz') as data:
            pred = data['block_recorded'][:, :, 4]
            label = f"{r['model']} s{r['seed']}"
            error = np.abs(pred.astype(float)-truth)
            axes[0].plot(time, error.mean(0), label=label, lw=1, color=colors[r['model']])
            axes[1].plot(time, error.mean(0).cumsum()/np.arange(1, len(time)+1), label=label, lw=1, color=colors[r['model']])
            detail, axs = plt.subplots(3, 2, figsize=(11, 9), squeeze=False)
            for ax, index in zip(axs.flat, selected):
                ax.plot(time, truth[index], c='black', lw=1.5, label='Observed')
                ax.plot(time, pred[index], lw=1.2, label='Block32 recorded boundary')
                if 'native_recorded' in data:
                    ax.plot(time, data['native_recorded'][index, :, 4], '--', lw=1, label='Native uninterrupted')
                ax.axvline(32*10/60, color='gray', ls=':', lw=.8)
                ax.set(title=f'Fixed window {index}', xlabel='Minutes', ylabel='Main temperature (C)')
            axs.flat[0].legend(fontsize=7)
            detail.suptitle(label)
            detail.tight_layout()
            detail.savefig(figures / f"{r['model']}_seed{r['seed']}_trajectories.png", dpi=140)
            plt.close(detail)
        response = json.loads((folder / 'response_metrics.json').read_text(encoding='utf-8'))
        rf, ra = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
        patterns = [('step', -24, [1, 0]), ('step', 0, [1, 0]), ('step', 64, [1, 0]), ('step', 112, [0, 1]),
                    ('pulse', 0, [1, 0]), ('ramp', 0, [1, 0]), ('sine', 0, [1, 0]), ('step', 0, [1, 1])]
        for ax, (shape, onset, valves) in zip(ra.flat, patterns):
            for row in response:
                if row['shape'] == shape and row['relative_onset'] == onset and row['valves'] == valves and row['dose'] == .03 and row['status'] == 'ok':
                    ax.plot(np.arange(1, 129)*10/60, row['mean_curve_main_C'],
                            '-' if row['mode'] == 'block' else '--', label=row['mode'])
            ax.axhline(0, color='gray', lw=.7)
            ax.axvline(onset*10/60, color='gray', ls=':', lw=.8)
            ax.set(title=f'{shape}, onset {onset*10}s, valves {valves}', xlabel='Scored minutes', ylabel='Delta main T (C)')
        if ra.flat[0].get_legend_handles_labels()[0]: ra.flat[0].legend(fontsize=8)
        rf.suptitle(label + ' / same 3pp perturbation, no curve smoothing')
        rf.tight_layout()
        rf.savefig(figures / f"{r['model']}_seed{r['seed']}_responses.png", dpi=140)
        plt.close(rf)
        log = folder / 'training.jsonl'
        if log.exists():
            history = [json.loads(line) for line in log.read_text().splitlines() if line]
            for ax, key in zip(la, ('train_loss', 'selector_main_mae_C')):
                ax.plot([x['updates'] for x in history], [x[key] for x in history], label=label, color=colors[r['model']])
    for ax, title in zip(axes, ('Per-step main MAE', 'Cumulative main MAE')):
        ax.axvline(32*10/60, color='black', ls=':', lw=.8)
        ax.set(title=title, xlabel='Minutes', ylabel='MAE (C)')
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / 'comparison_mae.png', dpi=160)
    plt.close(fig)
    for ax, title in zip(la, ('Weighted normalized training MSE', 'Selector H32 main MAE (C)')):
        ax.set(title=title, xlabel='Optimizer updates')
        ax.grid(alpha=.15)
    if la[1].get_legend_handles_labels()[0]: la[1].legend(fontsize=6, ncol=2)
    learning.tight_layout()
    learning.savefig(figures / 'training_curves.png', dpi=160)
    plt.close(learning)
    lines = ['# Quick benchmark results', '', 'Common block32 protocol; native metrics are separate. Missing native entries mean fixed-horizon head, not failure.', '',
             '| Model | Seed | H32 MAE | Block H128 MAE | Block H512 MAE | Native H512 MAE |',
             '|---|---:|---:|---:|---:|---:|']
    def fmt(x): return f'{x:.4f}' if isinstance(x, (float, int)) else '—'
    for row in entries:
        lines.append('| ' + ' | '.join([row['model'], str(row['seed']), *[fmt(row.get(k)) for k in
                     ('block_H32_mae_C', 'block_H128_mae_C', 'block_H512_mae_C', 'native_H512_mae_C')]]) + ' |')
    lines += ['', 'Read `summary.csv` for costs, tails and dynamic metrics; `seed_summary.json` for optimization-seed spread.',
              'Response strength is not a ranking score. Compare input shapes/doses with `response_metrics.json` and raw `responses.npz`.',
              'Pre-window probes contain 320s model-generated prehistory; early/mid/late events are measured from the scored window.',
              'Recorded-boundary forecasts and held-boundary stress are different information settings. No test/extension data are used.',
              'Anchored native is a hybrid: direct nominal blocks plus uninterrupted R4 action differences. Its block variant resets both components. These modes must not be conflated.',
              'Anchored combination preserves the predictor at the fixed hold-last nominal plan; it is evaluated against actual outcomes under recorded actions, not credited with automatic accuracy preservation.',
              '', '![MAE comparison](figures/comparison_mae.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    report(p.parse_args().output)
