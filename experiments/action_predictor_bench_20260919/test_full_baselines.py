import json
from argparse import Namespace
import numpy as np
import torch

from .full_baselines import advance_budget, select
from .models import TRAINED, build
from .analyze_return import analyze
from .test_bench import MEAN, SCALE, bank


def settings(**kwargs):
    return Namespace(**dict(dict(min_delta=.002, min_lr=.00025, lr_patience=2,
        stop_patience=3, min_epochs=1, batch_size=2, device='cpu'), **kwargs))


def test_lr_floor_gets_full_patience_before_stopping():
    args = settings()
    score = dict(short=1., balanced=2.)
    milestone, stale, lr = score.copy(), 0, .001
    transitions = []
    for epoch in range(1, 8):
        milestone, stale, lr, stop = advance_budget(score, milestone, stale, lr, epoch, args)
        transitions.append((lr, stop))
    assert transitions[1] == (.0005, False)
    assert transitions[3] == (.00025, False)
    assert all(not stop for _, stop in transitions[:6])
    assert transitions[6] == (.00025, True)
    assert not advance_budget(score, milestone, 10, lr, 7, settings(min_epochs=12))[-1]


def test_either_selector_can_keep_training_alive():
    args = settings()
    for scores in (dict(short=.9, balanced=2.1), dict(short=1.1, balanced=1.9)):
        _, stale, lr, stop = advance_budget(scores, dict(short=1., balanced=2.), 2, .001, 20, args)
        assert stale == 0 and lr == .001 and not stop
    milestones, _, _, _ = advance_budget(dict(short=.999, balanced=2.), dict(short=1., balanced=2.), 0, .001, 1, args)
    assert milestones['short'] == 1.  # tiny changes accumulate against last significant milestone


def test_every_original_family_has_the_same_selector_interface():
    assert set(TRAINED) == {'direct_no_action', 'direct', 'gru', 'ssm', 'attention_concat', 'ait', 'r4', 'r4_mlp', 'r4_directref'}
    for name in TRAINED:
        result = select(build(name, MEAN, SCALE), bank(horizon=128), settings())
        assert np.isfinite(list(result.values())).all()
        assert abs(result['balanced']-.5*(result['short']+result['block_H128'])) < 1e-10


def test_failed_long_forecast_remains_visible_in_diagnosis(tmp_path):
    folder = tmp_path/'seed11'/'unstable'
    folder.mkdir(parents=True)
    np.savez(tmp_path/'evaluation_inputs.npz', bank=bank(horizon=512))
    np.savez(folder/'forecasts.npz', native_recorded=np.full((2, 512, 5), np.nan))
    (folder/'result.json').write_text(json.dumps(dict(status='complete', name='unstable', seed=11)))
    (folder/'response_metrics.json').write_text(json.dumps([dict(mode='native', status='nonfinite_response')]))
    analyze(tmp_path, tmp_path)
    row = json.loads((tmp_path/'diagnosis.json').read_text())['rows'][0]
    assert not row['forecast_is_finite'] and row['H512'] is None
    assert row['signed_scenario_mean_wrong_fraction'] is None
