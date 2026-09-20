import copy
import numpy as np
import torch

from .data import unpack
from .evaluation import forecast, evaluate_responses
from .models import build
from .round3 import horizon_loss
from .round3_models import NominalPolicy, PlannedReference
from .test_bench import MEAN, SCALE, bank


def test_policy_hold_initialization_bounds_prefix_and_gradients():
    h, u, d, _ = unpack(bank(horizon=128))
    policy = NominalPolicy(MEAN, SCALE).double()
    h, u, d = h.double(), u.double(), d.double()
    h[0, -1, 5:7] = h.new_tensor([0., 1.])
    held = h[:, -1:, 5:7].expand_as(u)
    assert torch.equal(policy(h, d), held)
    with torch.no_grad(): policy.readout.weight.normal_(std=.05)
    p = policy(h, d)
    assert torch.equal(p[:, 0], h[:, -1, 5:7])
    assert (p >= 0).all() and (p <= 1).all()
    changed = d.clone(); changed[:, 64:] += 100
    assert torch.equal(policy(h, changed)[:, :65], p[:, :65])
    assert torch.equal(policy(h, d[:, :33]), p[:, :33])
    (p-u).square().mean().backward()
    assert any(x.grad is not None and x.grad.abs().max() > 0 for x in policy.parameters())


def test_planned_nominal_identity_full_horizon_and_frozen_policy():
    policy = NominalPolicy(MEAN, SCALE)
    with torch.no_grad(): policy.readout.weight.normal_(std=.03)
    model = PlannedReference(build('ssm', MEAN, SCALE), build('r4_mlp', MEAN, SCALE), policy).double()
    h, u, d, _ = unpack(bank(horizon=128))
    h, d = h.double(), d.double()
    nominal = model.nominal(h, d)
    for mode in ('block', 'native'):
        expected = forecast(model.predictor, h, nominal, d, mode=mode)
        assert torch.equal(forecast(model, h, nominal, d, mode=mode), expected)
    assert torch.equal(model(h, nominal[:, :32], d[:, :32]), forecast(model, h, nominal[:, :33], d[:, :33]))
    prediction = model(h, u[:, :32].double(), d[:, :32])
    prediction.mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.policy.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.response.parameters())


def test_nominal_policy_cannot_change_any_of_the110_paired_responses():
    policy = NominalPolicy(MEAN, SCALE)
    with torch.no_grad(): policy.readout.bias.copy_(torch.tensor([.15, -.15]))
    planned = PlannedReference(build('ssm', MEAN, SCALE), build('r4_mlp', MEAN, SCALE), policy)
    hold = copy.deepcopy(planned); hold.policy = None
    data = bank(horizon=512)
    rows, plans = evaluate_responses(planned, data, 'cpu', windows=2)
    _, baseline = evaluate_responses(hold, data, 'cpu', windows=2)
    assert len(rows) == 220
    for mode in ('native', 'block'):
        assert np.max(np.abs(plans[mode]-baseline[mode])) < 1e-9
    assert max(r['pre_onset_max_abs_C'] for r in rows) < 1e-9
    assert max(r.get('unreachable_max_abs_C', 0) for r in rows) < 1e-9
    assert max(r.get('opposite_sign_fraction', 0) for r in rows) == 0


def test_long_loss_reaches_beyond_step32_without_diluting_short_term():
    model = build('ssm', MEAN, SCALE)
    truth = torch.zeros(1, 128, 5)
    p = truth.clone(); p[:, 32:, 4] = model.scale[4]; p.requires_grad_(True)
    loss = horizon_loss(model, p, truth)
    assert abs(float(loss)-.25) < 1e-6
    loss.backward()
    assert p.grad[:, 32:, 4].abs().min() > 0
    assert horizon_loss(model, p[:, :32], truth[:, :32]) == 0
