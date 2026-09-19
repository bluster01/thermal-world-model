import numpy as np
import pytest
import torch

from .data import expanded_sample, unpack
from .evaluation import forecast, evaluate_responses, gradient_probe
from .models import build
from .round2_models import ProtectedReference, response_matching_loss
from .test_bench import MEAN, SCALE, bank


def test_nested_sample_retains_original_windows_and_is_deterministic():
    pool = np.arange(1000) * 80
    old = pool[::10]
    a = expanded_sample(pool, old, 333, 11)
    assert len(a) == len(np.unique(a)) == 333
    assert np.isin(old, a).all() and np.isin(a, pool).all()
    assert np.array_equal(a, expanded_sample(pool, old, 333, 11))
    with pytest.raises(ValueError): expanded_sample(pool, np.array([-1]), 333, 11)
    with pytest.raises(ValueError): expanded_sample(pool, old, 30, 11)


@pytest.mark.parametrize('predictor_name', ['ssm', 'r4_mlp'])
def test_protected_nominal_identity_prefix_topology_and_backward(predictor_name):
    model = ProtectedReference(build(predictor_name, MEAN, SCALE), build('r4_mlp', MEAN, SCALE)).double()
    h, u, d, _ = unpack(bank(horizon=128))
    h, u, d = h.double(), u.double(), d.double()
    hold = h[:, -1:, 5:7].expand_as(u)
    changed = hold.clone(); changed[:, 63:, 1] += .03
    for mode in ('block', 'native'):
        base = forecast(model, h, hold, d, mode=mode)
        assert torch.equal(base, forecast(model.predictor, h, hold, d, mode=mode))
        step = forecast(model, h, changed, d, mode=mode)
        assert torch.equal(step[:, :63], base[:, :63])
        assert torch.equal(step[:, :, :3], base[:, :, :3])
        assert (step[:, :, 3:] <= base[:, :, 3:]).all()
    y = model(h, changed[:, :32], d[:, :32])
    y.mean().backward()
    assert any(p.grad is not None for p in model.predictor.parameters())
    # Input changed at63 is outside H32; use a nonzero in-window action for gradient FD.
    assert gradient_probe(model, bank(), 'cpu')['absolute_difference'] < 1e-6


def test_protected_all110_shapes_preserve_response_across_reference_refresh():
    model = ProtectedReference(build('ssm', MEAN, SCALE), build('r4_mlp', MEAN, SCALE))
    rows, curves = evaluate_responses(model, bank(horizon=512), 'cpu', windows=2)
    assert len(rows) == 220 and all(r['status'] == 'ok' for r in rows)
    assert max(r['pre_onset_max_abs_C'] for r in rows) < 1e-9
    assert max(r.get('unreachable_max_abs_C', 0) for r in rows) < 1e-9
    assert max(r.get('opposite_sign_fraction', 0) for r in rows) == 0
    assert np.max(np.abs(curves['block']-curves['native'])) < 1e-9
    assert np.max(np.abs(curves['block'][..., 4])) > 1e-4


def test_soft_loss_fits_student_without_updating_response_teacher():
    model, teacher = build('ssm', MEAN, SCALE), build('r4_mlp', MEAN, SCALE)
    teacher.requires_grad_(False)
    h, _, d, _ = unpack(bank())
    loss = response_matching_loss(model, teacher, h, d[:, :-1], 16)
    loss.backward()
    assert torch.isfinite(loss) and loss.requires_grad
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in teacher.parameters())
