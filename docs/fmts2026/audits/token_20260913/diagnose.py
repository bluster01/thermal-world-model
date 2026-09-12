"""Read-only checkpoint diagnostic; synthetic histories, no training or test access.

Run from repository root: python docs/fmts2026/audits/token_20260913/diagnose.py
Writes only diagnostic.json beside this script. Synthetic saturation is NOT an
estimate of saturation prevalence on plant validation histories.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.final_wm.contracts import ObserverConfig, StateLayout
from src.final_wm.observer import ProbabilisticObserver


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def daymean(x, days):
    return np.stack([x[days == d].mean(0) for d in np.unique(days)]).mean(0)


def main():
    torch.set_num_threads(1)
    root = ROOT / 'results/fmts_mainsteam_20260911/linux_full_v02'
    identity = json.loads((root / 'identity.json').read_text())
    source = ROOT / 'src/final_wm/observer.py'
    assert sha(source) == identity['sources']['src/final_wm/observer.py']
    result = dict(training=False, locked_test_access=False,
                  source_sha256=sha(source), torch_version=torch.__version__,
                  histories='32 synthetic histories per suite, seed 873; NOT plant windows',
                  active_state_indices=[3, 4, 5, 6], runs=[])
    for arm in ('fusion_gru_norew', 'fusion_token_xattn_norew'):
        for seed in (0, 1, 2):
            directory = root / f'{arm}_seed{seed}'
            report = json.loads((directory / 'report.json').read_text())
            assert sha(directory / 'best.pt') == report['checkpoint_sha256']
            assert sha(directory / 'prediction.npz') == report['prediction_sha256']
            ckpt = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
            observer = ProbabilisticObserver(
                ObserverConfig(**ckpt['model_config']['observer']), StateLayout(0)).eval()
            observer.load_state_dict({k.removeprefix('base.observer.'): v
                for k, v in ckpt['state_dict'].items() if k.startswith('base.observer.')})
            log = [json.loads(line) for line in (directory / 'ledger.jsonl').read_text().splitlines()]
            minimum = min(log, key=lambda row: row['validation_mainsteam_mae'])
            assert minimum['step'] == ckpt['step'] == report['best_step']
            with np.load(directory / 'prediction.npz') as raw:
                mae = float(daymean(np.abs(raw['prediction']-raw['target']).mean(1), raw['days']))
                persistence = float(daymean(np.abs(raw['persistence']-raw['target']).mean(1), raw['days']))
                assert abs(mae - report['H18']) < 1e-6
            row = dict(arm=arm, seed=seed, checkpoint_sha256=sha(directory / 'best.pt'),
                       H18=mae, persistence_H18=persistence, best_step=ckpt['step'],
                       tail_mae=[x['validation_mainsteam_mae'] for x in log[-5:]], suites={})
            m, s = ckpt['normalization']['mean'], ckpt['normalization']['std']
            generator = torch.Generator().manual_seed(873)
            iid = torch.randn(32, 96, 23, generator=generator)*.2
            offset = torch.randn(32, 1, 23, generator=generator)*.2
            slope = torch.randn(32, 1, 23, generator=generator)*.1
            smooth = offset + slope*torch.linspace(-1, 1, 96)[None, :, None]
            for name, perturbation in [('iid_0p2_train_std', iid), ('smooth_offsets_slopes', smooth)]:
                h = m[None, None, :] + perturbation*s[None, None, :]
                b = torch.cat([h[:, :, 7:14], (h[:, :, 14:]-m[-9:])/s[-9:]], -1)
                captures = []
                hook = observer.mu_head.register_forward_hook(lambda module, args, out: captures.append(out))
                with torch.no_grad():
                    mu, _ = observer.posterior(h[:, :, :5], h[:, :, 5:7], b, torch.zeros(32, 11))
                hook.remove()
                raw = captures[0].reshape(32, 11)[:, 3:7]
                fraction = mu[:, 3:7]/(.1*observer.state_scale[3:7])
                row['suites'][name] = dict(
                    correction_bound_fraction_mean=fraction.mean(0).tolist(),
                    correction_bound_fraction_std=fraction.std(0).tolist(),
                    raw_min=raw.min(0).values.tolist(), raw_max=raw.max(0).values.tolist(),
                    tanh_local_derivative_mean=(1-torch.tanh(raw).square()).mean(0).tolist(),
                    saturation_fraction_abs_gt_0p99=float((fraction.abs()>.99).float().mean()))
            result['runs'].append(row)
    path = Path(__file__).with_name('diagnostic.json')
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for row in result['runs']:
        suite = row['suites']['smooth_offsets_slopes']
        print(row['arm'], row['seed'], 'MAE', round(row['H18'], 6),
              'smooth saturation', suite['saturation_fraction_abs_gt_0p99'],
              'history std', suite['correction_bound_fraction_std'])


if __name__ == '__main__':
    main()
