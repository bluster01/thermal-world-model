"""Behavioral tests for scientific interfaces, not duplicate implementation tests."""
import numpy as np
import pytest
import torch
from torch import nn

from .data import CONTEXT, TRAIN_H, stratified_sample, unpack
from .evaluation import forecast, metrics, response_summary, scenarios, evaluate_responses, response_comparisons, gradient_probe
from .models import TRAINED, Anchored, R4, R4_PLAN, build
from .r4_transport import TransportWorldModel
from .run import loss_fn

torch.set_num_threads(1)
MEAN = np.array([450, 440, 520, 510, 550, .5, .5, 1500, 200, 20, 450, 320, 18], dtype=np.float32)
SCALE = np.array([10]*5 + [.1, .1] + [100, 20, 1, 10, 10, 1], dtype=np.float32)


def bank(count=2, horizon=TRAIN_H):
    rng = np.random.default_rng(4)
    x = MEAN + .03 * rng.standard_normal((count, CONTEXT+horizon, 13)).astype(np.float32)*SCALE
    return x


@pytest.mark.parametrize('name', TRAINED)
def test_finite_forward_backward_and_checkpoint(name, tmp_path):
    torch.manual_seed(11)
    m = build(name, MEAN, SCALE)
    h, u, d, y = unpack(bank())
    p = forecast(m, h, u, d)
    assert p.shape == y.shape and torch.isfinite(p).all()
    loss = loss_fn(m, p, y)
    loss.backward()
    grads = [v.grad for v in m.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    restored = build(name, MEAN, SCALE)
    restored.load_state_dict(torch.load(tmp_path / 'm.pt', weights_only=True))
    assert torch.equal(forecast(restored, h, u, d), p)


def test_left_right_alignment_and_future_temperature_isolation():
    x = bank(horizon=128)
    h, u, d, y = unpack(x)
    assert torch.equal(u[:, 0], h[:, -1, 5:7])
    assert torch.equal(d[:, 0], h[:, -1, 7:])
    assert np.array_equal(y.numpy(), x[:, CONTEXT:, :5])
    poisoned = x.copy(); poisoned[:, CONTEXT:, :5] += 1000
    hp, up, dp, _ = unpack(poisoned)
    for a, b in zip((h, u, d), (hp, up, dp)): assert torch.equal(a, b)


def test_rollout_feedback_uses_predictions_and_right_endpoint_controls():
    class Recorder(nn.Module):
        native_long = False
        def __init__(self): super().__init__(); self.calls = []
        def forward(self, h, u, d):
            self.calls.append(h.clone())
            return h[:, -1:, :5] + torch.arange(1, u.shape[1]+1)[None, :, None]
    h, u, d, _ = unpack(bank(horizon=65))
    u[:, :, 0] = torch.arange(66)
    d[:, :, 0] = torch.arange(66)+100
    m = Recorder(); p = forecast(m, h, u, d)
    assert p.shape[1] == 65
    assert torch.equal(m.calls[1][:, -32:, :5], p[:, :32])
    assert torch.equal(m.calls[1][:, -32:, 5:7], u[:, 1:33])
    assert torch.equal(m.calls[1][:, -32:, 7:], d[:, 1:33])
    assert torch.equal(m.calls[2][:, -32:, :5], p[:, 32:64])


def test_r4_snapshot_parity_and_zero_adapter():
    torch.manual_seed(11); original = TransportWorldModel(MEAN, SCALE, R4_PLAN)
    torch.manual_seed(11); wrapped = R4(MEAN, SCALE)
    torch.manual_seed(11); adapter = R4(MEAN, SCALE, 'r4_mlp')
    h, u, d, _ = unpack(bank())
    p = original(h, u[:, :-1], d[:, :-1])['prediction']
    assert torch.equal(p, wrapped(h, u[:, :-1], d[:, :-1]))
    assert torch.equal(p, adapter(h, u[:, :-1], d[:, :-1]))


@pytest.mark.parametrize('name', ['gru', 'ssm', 'attention_concat', 'ait', 'r4', 'r4_mlp'])
def test_causal_action_prefix(name):
    m = build(name, MEAN, SCALE)
    h, u, d, _ = unpack(bank())
    changed = u.clone(); changed[:, 24:, 0] += .03
    with torch.no_grad():
        a, b = forecast(m, h, u, d, mode='native'), forecast(m, h, changed, d, mode='native')
    assert torch.equal(a[:, :24], b[:, :24])


def test_unrestricted_direct_can_expose_future_action_leakage():
    m = build('direct', MEAN, SCALE)
    h, u, d, _ = unpack(bank())
    changed = u.clone(); changed[:, 24:, 0] += .1
    a, b = forecast(m, h, u, d), forecast(m, h, changed, d)
    assert not torch.equal(a[:, :24], b[:, :24])


def test_r4_hold_identity_and_reachable_paths():
    m = build('r4', MEAN, SCALE)
    h, _, d, _ = unpack(bank(horizon=128))
    u = h[:, -1:, 5:7].expand(-1, 129, -1).clone()
    base = m.core(h, u[:, :-1], d[:, :-1])
    assert torch.count_nonzero(base['action_response_C']) == 0
    changed = u.clone(); changed[:, :, 1] += .03
    stepped = m.core(h, changed[:, :-1], d[:, :-1])
    assert torch.equal(base['reference_prediction'], stepped['reference_prediction'])
    assert torch.count_nonzero(stepped['action_response_C'][:, :, :3]) == 0
    assert (stepped['action_response_C'] <= 0).all()
    assert stepped['action_response_C'][:, -1, 4].abs().max() > 1e-4


def test_anchor_is_fixed_across_blocks_and_preserves_nominal_prediction():
    predictor, response = build('direct', MEAN, SCALE), build('r4', MEAN, SCALE)
    m = Anchored(predictor, response)
    h, u, d, _ = unpack(bank(horizon=65))
    nominal = h[:, -1:, 5:7].expand_as(u)
    assert torch.equal(forecast(m, h, nominal, d), forecast(predictor, h, nominal, d))
    changed = nominal.clone(); changed[:, 20:, 0] += .03
    expected = forecast(predictor, h, nominal, d) + (forecast(response, h, changed, d) - forecast(response, h, nominal, d))
    assert torch.equal(forecast(m, h, changed, d), expected)
    base_native = forecast(m, h, nominal, d, mode='native')
    changed_native = forecast(m, h, changed, d, mode='native')
    assert torch.equal(base_native[:, :, 0], changed_native[:, :, 0])


def test_metric_horizon_and_tail_math():
    y = np.zeros((2, 512, 5)); p = y.copy(); p[:, 32:, 4] = 2
    r = metrics(p, y, np.zeros((2, 5)))
    assert r['H32']['main_mae_C'] == 0
    assert r['H128']['main_mae_C'] == 1.5
    assert r['tail_129_512']['main_mae_C'] == 2
    p[0, 0, 0] = np.nan
    assert 'failure' in metrics(p, y, np.zeros((2, 5)))


def test_response_suite_covers_shapes_timing_doses_and_invalid_plans():
    cases = scenarios()
    assert len({c['id'] for c in cases}) == len(cases)
    assert {'step', 'pulse', 'ramp', 'sine', 'smooth', 'double_pulse'} <= {c['shape'] for c in cases}
    assert {-24, 0, 64, 112, 31, 32, 33} <= {c['relative_onset'] for c in cases}
    assert {-.06, -.03, -.01, .01, .03, .06} <= {c['dose'] for c in cases}
    response = np.zeros((2, 160, 5))
    row = response_summary(response, cases[0], np.array([False, False]))
    assert row['status'] == 'no_eligible_windows' and row['excluded_out_of_bounds'] == 2


def test_stratified_subset_is_spread_and_deterministic():
    x = np.arange(1000)
    a = stratified_sample(x, 100, 11)
    assert np.array_equal(a, stratified_sample(x, 100, 11))
    assert np.array_equal(a // 10, np.arange(100))


def test_attention_fusion_and_interleaving_have_same_parameter_count():
    a, b = build('attention_concat', MEAN, SCALE), build('ait', MEAN, SCALE)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_all_response_shapes_complete_with_pairwise_statistics():
    m = build('r4', MEAN, SCALE).eval()
    rows, curves = evaluate_responses(m, bank(horizon=512), 'cpu', windows=2)
    assert curves['native'].shape == (110, 2, 160, 5)
    assert len(rows) == 220
    assert all(x['status'] == 'ok' for x in rows)
    pairs = response_comparisons(curves)
    assert {p['type'] for p in pairs} == {'dose_and_symmetry', 'superposition'}
    assert all(x['pre_onset_max_abs_C'] < 1e-9 for x in rows if x['mode'] == 'native')
    assert all(x.get('unreachable_max_abs_C', 0) < 1e-9 for x in rows if x['mode'] == 'native')
    assert m.mean.dtype == torch.float32


def test_float64_gradient_probe_and_no_action_control():
    data = bank()
    m = build('r4', MEAN, SCALE).double()
    r = gradient_probe(m, data, 'cpu')
    assert r['absolute_difference'] < 1e-3
    assert gradient_probe(build('direct_no_action', MEAN, SCALE).double(), data, 'cpu')['gradient_abs_max_C_per_fraction'] == 0
