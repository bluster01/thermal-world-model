"""Read-only comparison of returned GNR1 and selected GRU physical constants.

No training, rollout, raw plant input or held-out target access. The explicit
release checkout supplies the exact model definitions, not the dirty main tree.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.release.resolve()
    sys.path.insert(0, str(root))
    import torch
    from experiments.fmts_mainsteam_20260911.models import RichFusion
    from src.final_wm.training import TrainSpec

    pairs = []
    configs = []
    for seed in (0, 1, 2):
        entries = {}
        for name, folder, arm in [
            ('gru', 'results/fmts_mainsteam_20260911/linux_full_v02', 'fusion_gru_norew'),
            ('gnr', 'results/fmts_greybox_norew_20260913/linux_full_gnr1', 'greybox_steady_none_norew'),
        ]:
            run = root/folder/f'{arm}_seed{seed}'
            report = json.loads((run/'report.json').read_text())
            digest = sha(run/'best.pt')
            assert digest == report['checkpoint_sha256']
            p = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
            assert p['identity_sha256'] == sha(root/folder/'identity.json')
            model = RichFusion(TrainSpec(**p['spec']), p['normalization']['mean'], p['normalization']['std'])
            raw = {k.removeprefix('base.transition.raw.'):v for k,v in p['state_dict'].items()
                   if k.startswith('base.transition.raw.')}
            model.base.transition.raw.load_state_dict(raw, strict=True)
            values = {k:float(model.base.transition.val(k).detach()) for k in raw}
            assert raw['aW1'].item() == raw['aW2'].item() == -30.0
            entries[name] = dict(checkpoint=str(run/'best.pt'),sha256=digest,best_step=p['step'],
                transition_config=asdict(model.base.config.transition),
                observation_config=asdict(model.base.config.observation),physical_values=values)
        assert entries['gru']['transition_config'] == entries['gnr']['transition_config']
        assert entries['gru']['observation_config'] == entries['gnr']['observation_config']
        pairs.append(dict(seed=seed,models=entries,gnr_over_gru={
            k:entries['gnr']['physical_values'][k]/v if v else None
            for k,v in entries['gru']['physical_values'].items()}))
        configs.append(entries['gru']['transition_config'])
    result = dict(status='READ_ONLY_CHECKPOINT_PARAMETER_AUDIT',date='2026-09-14',
        release_root=str(root),source_hashes={p:sha(root/p) for p in [
            'src/final_wm/transition.py','src/final_wm/training.py',
            'experiments/fmts_mainsteam_20260911/models.py']},
        same_transition_config=True,same_observation_config=True,pairs=pairs,
        no_retraining=True,no_real_record_rollout=True,no_heldout_access=True,
        interpretation='Parameter divergence is observed; causal contribution and plant fidelity are untested.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for p in pairs:
        print('seed',p['seed'],{k:round(p['gnr_over_gru'][k],3)
              for k in ['UA1','UA2','Cm1','Cm2','tauB','tau_mix2','th1','th2']})
    print('Saved',args.out)


if __name__ == '__main__':
    main()
