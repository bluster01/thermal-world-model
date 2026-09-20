"""Deterministic engineering fixtures only; never plant scientific results."""
import numpy as np
import pytest
import torch

from experiments.fmts_valve_diagnostic_20260920.design import (
    scenarios, perturb, summarize_effect, opening_bins, factual_metrics,
)
from src.final_wm.model import HistoryWindow


@pytest.fixture
def inputs():
    torch.set_num_threads(1)
    h = HistoryWindow(torch.zeros(2, 96, 5), torch.full((2, 96, 2), .4),
                      torch.ones(2, 96, 7))
    return h, torch.zeros(2, 96, 9), h.actions[:, -1:].expand(-1, 18, -1).clone(), h.boundary[:, -1:].expand(-1, 18, -1).clone()


def select(**kwargs):
    return next(s for s in scenarios() if all(s[k] == v for k, v in kwargs.items()))


def test_exact_scenario_matrix():
    ss = scenarios()
    assert len(ss) == len({s['id'] for s in ss}) == 78
    assert {k: sum(s['family'] == k for s in ss) for k in ['history', 'future', 'spray', 'joint']} == dict(history=24, future=48, spray=2, joint=4)


def test_history_last_does_not_change_future_or_other_inputs(inputs):
    h, ext, a, b = inputs
    s = select(family='history', valve=0, lag=0, dose=.05)
    hh, aa, bb, da, dw = perturb(h, a, b, s)
    assert torch.equal(a, aa) and torch.equal(b, bb)
    assert torch.equal(hh.obs, h.obs) and torch.equal(hh.boundary, h.boundary)
    torch.testing.assert_close(hh.actions[:, -3:, 0], torch.full((2, 3), .45))
    assert torch.equal(h.actions, torch.full((2, 96, 2), .4))
    assert da.shape == (2, 96) and torch.count_nonzero(da) == 6
    assert not dw.any()


@pytest.mark.parametrize('onset', [0, 6, 12])
@pytest.mark.parametrize('shape', ['pulse', 'step'])
def test_future_timing_and_prefix(inputs, onset, shape):
    h, _, a, b = inputs
    s = select(family='future', valve=1, onset=onset, shape=shape, dose=-.02)
    hh, aa, bb, da, dw = perturb(h, a, b, s)
    assert torch.equal(hh.actions, h.actions) and torch.equal(bb, b)
    assert torch.equal(aa[:, :onset], a[:, :onset])
    assert torch.equal(aa[:, :, 0], a[:, :, 0])
    assert (da != 0).sum() == 2 * (3 if shape == 'pulse' else 18-onset)
    assert not dw.any()


def test_clipping_reports_actual_not_nominal_dose(inputs):
    h, _, a, b = inputs
    a[0, :, 0] = .99
    s = select(family='future', valve=0, onset=0, shape='step', dose=.05)
    _, aa, _, da, _ = perturb(h, a, b, s)
    assert aa.max() <= 1.
    torch.testing.assert_close(da[0], torch.full((18,), .01))
    torch.testing.assert_close(da[1], torch.full((18,), .05))


def test_spray_only_and_joint_factorial_are_explicit(inputs):
    h, _, a, b = inputs
    s = select(family='spray', spray_dose=-2.)
    _, aa, bb, da, dw = perturb(h, a, b, s)
    assert torch.equal(aa, a) and not da.any()
    assert torch.equal(bb[:, :, :6], b[:, :, :6])
    assert torch.equal(dw, torch.full((2, 18), -1.))
    s = select(family='joint', valve=1, dose=.05)
    _, aa, bb, da, dw = perturb(h, a, b, s)
    torch.testing.assert_close(da, torch.full((2, 18), .05))
    assert torch.equal(dw, torch.full((2, 18), 2.))


def test_signed_cancellation_and_anticipation_are_not_hidden():
    delta = np.tile(np.array([1., -1.])[:, None], (1, 18))
    s = select(family='future', valve=0, onset=12, shape='pulse', dose=.05)
    m = summarize_effect(delta, np.array([1, 1]), s)
    assert m['signed_curve_c'] == [0.] * 18
    assert m['absolute_curve_c'] == [1.] * 18
    assert m['pre_action_max_abs_c'] == 1.
    assert m['elapsed60_signed_c'] == 0.
    delta[:, :12] = 0
    assert summarize_effect(delta, np.array([1, 1]), s)['pre_action_max_abs_c'] == 0.


def test_equal_day_not_equal_window_and_empty_strata():
    delta = np.repeat(np.array([1., 3., 9.])[:, None], 18, axis=1)
    s = scenarios()[0]
    assert summarize_effect(delta, np.array([0, 0, 1]), s)['signed_curve_c'][-1] == 5.5
    assert summarize_effect(delta[:0], np.array([], dtype=int), s) == {'n_windows': 0, 'n_days': 0}


def test_opening_bins_fixed_thresholds_and_ties():
    bins = opening_bins(np.array([[.1, .1], [.3, .3], [.8, .8]]), np.array([[.2, .6], [.2, .6]]))
    assert bins.tolist() == [[0, 0], [1, 1], [2, 2]]


def test_factual_increment_error_not_counterfactual_truth():
    target = np.tile(np.arange(1, 19), (2, 1)).astype(float)
    pred = target + 2
    m = factual_metrics(pred, target, np.zeros(2), np.array([0, 1]))
    assert m['level_mae_c'] == 2.
    assert m['increment_mae_c'] == pytest.approx(2/18)
    assert m['increment_correlation'] is None  # constant observed increments


class ToyRecord:
    """Only for verifying runner plumbing, deliberately not a plant simulator."""
    def __init__(self):
        self.n = 600
        self.obs = torch.tensor([420., 410., 500., 490., 540.]).repeat(self.n, 1)
        self.actions = torch.tensor([.4, .5]).repeat(self.n, 1)
        self.boundary = torch.tensor([400., 180., 24., 400., 310., 22., 20.]).repeat(self.n, 1)
        self.boundary_ext = torch.zeros(self.n, 9)
        self.timestamps = torch.arange(self.n, dtype=torch.int64) * 10
        self.split = torch.ones(self.n, dtype=torch.int64)
        self.split[:130] = 0

    def candidates(self, split):
        assert split == 1
        return torch.arange(250, 550)

    def batch(self, starts, split, device):
        assert split == 1 and device == 'cpu'
        past = starts[:, None] + torch.arange(-96, 0)
        future = starts[:, None] + torch.arange(18)
        return (HistoryWindow(self.obs[past], self.actions[past], self.boundary[past]),
                self.boundary_ext[past], self.actions[future], self.boundary[future],
                self.obs[future], self.timestamps[starts] // 86400)


class ToyModel(torch.nn.Module):
    def __init__(self, anticipate=False):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.))
        self.anticipate = anticipate

    def forward(self, h, ext, actions, boundary):
        assert not torch.is_grad_enabled()
        effect = (actions.sum(-1) + boundary[:, :, 6] / 10).cumsum(1)
        if self.anticipate:
            effect = effect + actions.sum((1, 2))[:, None]
        return h.obs[:, -1:, 4] + effect * self.scale + h.actions.mean((1, 2))[:, None]


def test_collector_detects_anticipation_and_keeps_weights():
    from experiments.fmts_valve_diagnostic_20260920.run import collect_probes, weight_hash
    record = ToyRecord()
    scenario = select(family='future', onset=12, valve=0, shape='pulse', dose=.05)
    for anticipates in (False, True):
        model = ToyModel(anticipates).eval()
        before = weight_hash(model)
        raw = collect_probes(model, record, np.array([250, 250]), 'cpu')[scenario['id']]
        prefix = (raw['prediction']-raw['base'])[:, :12]
        assert bool(np.any(prefix)) == anticipates
        assert weight_hash(model) == before
        assert raw['starts'].tolist() == [250, 250]


def test_formal_windows_execution_is_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.fmts_valve_diagnostic_20260920 import run
    monkeypatch.setattr(run.platform, 'system', lambda: 'Windows')
    with pytest.raises(RuntimeError, match='Linux-only'):
        run.execute(SimpleNamespace(private_out=str(tmp_path/'private')))
    assert not (tmp_path/'private').exists()


def test_source_hash_cross_platform_newlines(tmp_path):
    from experiments.fmts_valve_diagnostic_20260920.run import source_sha
    a, b = tmp_path/'lf.py', tmp_path/'crlf.py'
    a.write_bytes(b'x = 1\ny = 2\n')
    b.write_bytes(b'x = 1\r\ny = 2\r\n')
    assert source_sha(a) == source_sha(b)


def test_clipped_dose_accounting_visible_in_summary(inputs):
    from experiments.fmts_valve_diagnostic_20260920.design import scenario_summary
    h, _, a, b = inputs
    a[0, :, 0] = 1.
    s = select(family='future', valve=0, onset=0, shape='pulse', dose=.05)
    _, _, _, da, dw = perturb(h, a, b, s)
    raw = dict(prediction=np.zeros((2, 18)), base=np.zeros((2, 18)), valve_dose=da.numpy(),
               spray_dose=dw.numpy(), support=np.ones((2, 18), dtype=bool))
    m = scenario_summary(raw, dict(days=np.zeros(2), opening_bin=np.zeros((2, 2), dtype=int)), s)
    assert m['valve_dose_accounting']['zero_dose_windows'] == 1
    assert m['valve_dose_accounting']['clipped_windows'] == 1
    assert m['opening_strata']['high'] == {'n_windows': 0, 'n_days': 0}


def test_fixture_end_to_end_nine_cells_and_artifact_audit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.fmts_valve_diagnostic_20260920 import run as R, design as D, audit as A
    record = ToyRecord()
    parent, gnr = tmp_path/'parent', tmp_path/'gnr'
    parent.mkdir(); gnr.mkdir()
    indices = dict(train=np.arange(100, 120), validation=np.tile(np.arange(250, 378), 2), response=np.arange(250, 314))
    np.savez_compressed(parent/'indices.npz', **indices)
    record_file, prop_file, map_file = [tmp_path/name for name in ('record.stub', 'properties.stub', 'mapping.stub')]
    for f in (record_file, prop_file, map_file):
        f.write_bytes(b'ENGINEERING_FIXTURE_NOT_PLANT_DATA')
    source = R.read(R.ROOT/R.P['source_contract'])
    source.update(record=R.sha256(record_file), properties=R.sha256(prop_file), indices=R.sha256(parent/'indices.npz'))
    source_path = tmp_path/'source_contract.json'
    R.write(source_path, source)
    protocol = dict(R.P, source_contract=str(source_path), source_contract_canonical_sha256=R.canonical_hash(source))
    for module in (R, D, A):
        monkeypatch.setattr(module, 'P', protocol)
    monkeypatch.setattr(R.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(R, 'validate_sources', lambda *args: None)
    monkeypatch.setattr(R, 'RichRecord', lambda *args: record)
    monkeypatch.setattr(R, 'load_grid_properties', lambda *args: None)
    monkeypatch.setattr(R, 'load_model', lambda *args: ToyModel().eval())
    monkeypatch.setattr(R, 'source_hashes', lambda: {})
    for arm in R.P['arms']:
        for seed in R.P['seeds']:
            folder = R.source_folder(parent, gnr, arm, seed)
            folder.mkdir()
            (folder/'best.pt').write_bytes(b'NOT_A_CHECKPOINT_TEST_FIXTURE')
            for name, key, response in [('prediction', 'validation', False), ('response', 'response', True)]:
                raw = R.evaluate(ToyModel(), record, torch.as_tensor(indices[key]), 'cpu', response=response)
                np.savez_compressed(folder/f'{name}.npz', **raw)
    args = SimpleNamespace(parent=parent, gnr=gnr, record=record_file, properties=prop_file,
                           mapping=map_file, private_out=tmp_path/'private', device='cpu')
    result = R.execute(args)
    assert result['cells_completed'] == 9 and result['training_updates'] == 0
    receipt = A.audit(args.private_out, parent, gnr, verify_local_sources=False)
    assert receipt['scenarios_checked'] == 702 and receipt['original_replays_checked'] == 18
    assert receipt['new_probe_full_inference_replay'] is False
    with pytest.raises(FileExistsError):
        R.execute(args)
    with pytest.raises(ValueError, match='incomplete'):
        R.aggregate(result['runs'][:-1])
    scenario_path = args.private_out/f'{R.P["arms"][0]}_seed0'/'s000.npz'
    raw = R.npz(scenario_path)
    raw['prediction'][0, 0] += 1
    np.savez_compressed(scenario_path, **raw)
    with pytest.raises(AssertionError, match='changed artifact'):
        A.audit(args.private_out, parent, gnr, verify_local_sources=False)
