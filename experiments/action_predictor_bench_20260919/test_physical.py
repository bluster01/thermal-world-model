import numpy as np
import pytest
import torch

from .physical_models import Physical
from .evaluation import forecast


def fixture(arm='P0', horizon=128):
    torch.manual_seed(11)
    mean = np.array([450, 440, 490, 480, 540, .3, .3, 700, 400, 20, 420, 280, 18], dtype=np.float32)
    scale = np.array([10]*5+[.1, .1, 100, 100, 2, 10, 10, 2], dtype=np.float32)
    m = Physical(mean, scale, np.zeros(11), np.ones(11), arm).double()
    bank = torch.zeros(2, 64+horizon, 24, dtype=torch.float64)
    bank[:, :, :13] = torch.tensor(mean)
    bank[:, :, :5] += torch.arange(64+horizon, dtype=torch.float64)[None, :, None]*.001
    h = bank[:, :64]
    u, d = bank[:, 63:, 5:7], bank[:, 63:, 7:13]
    return m, bank, h, u, d


def test_implicit_step_conserves_discrete_network_energy():
    m, _, h, u, d = fixture(horizon=3)
    c = m.coefficients(h, u[:, :-1], d[:, :-1])
    p = m.rollout(h, u[:, :-1], d[:, :-1], coefficients=c)
    initial = m.observe(h)[0]
    old = torch.cat((initial[:, None], p[:, :-1]), 1)
    capacity, loss, source = c[2:5]
    residual = capacity*(p-old)/10 + torch.einsum('btij,btj->bti', loss, p)-source
    assert residual.abs().max() < 1e-7
    # Internal advective transfers and steam-metal exchange cancel in the sum.
    water = c[5]
    expected = d[:, :-1, 0]*d[:, :-1, 3]+water.sum(-1)*d[:, :-1, 4] + source[:, :, 5:].sum(-1) -(d[:, :-1, 0]+water.sum(-1))*p[:, :, 4]
    torch.testing.assert_close((capacity*(p-old)/10).sum(-1), expected, rtol=1e-10, atol=1e-7)


def test_water_mapping_monotone_and_never_changes_heat_scenario():
    m, _, h, u, d = fixture(horizon=8)
    c1 = m.coefficients(h, u[:, :-1], d[:, :-1])
    c2 = m.coefficients(h, u[:, :-1]+.03, d[:, :-1])
    assert (c2[5] > c1[5]).all()
    torch.testing.assert_close(c1[4][:, :, 5:], c2[4][:, :, 5:], rtol=0, atol=0)


def test_cold_water_cools_uniform_network_without_early_or_upstream_effect():
    m, _, h, u, d = fixture(horizon=64)
    h = h.clone(); h[:, :, :5] = 450
    d = d.clone(); d[:, :, 3] = 450; d[:, :, 4] = 100
    with torch.no_grad():
        m.raw_heat.fill_(-60)
        m.initial_wall_logit.fill_(-60)
    u = torch.zeros_like(u)
    changed = u.clone(); changed[:, 16:, 1] = .3
    with torch.no_grad():
        delta = forecast(m,h,changed,d)-forecast(m,h,u,d)
    assert delta[:, :16].abs().max()<1e-9
    assert delta[:, :, :3].abs().max()<1e-9
    assert delta[:, :, 3:].max()<1e-9
    assert delta[:, -1, 4].min()<-.001


@pytest.mark.parametrize('arm', ['P0', 'P1', 'P2'])
def test_training_gradients_and_target_future_is_not_prediction_input(arm):
    m, bank, h, u, d = fixture(arm)
    prediction = forecast(m,h,u,d)
    loss = m.training_objective(bank)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.raw_heat.grad is not None and torch.isfinite(m.raw_heat.grad).all()
    assert m.encoder[-1].weight.grad.abs().max()>0
    if arm=='P1':
        assert all(p.grad is None for p in m.teacher.parameters())
    if arm=='P2':
        assert m.physical_encoder[-1].weight.grad.abs().max()>0
    modified = bank.clone(); modified[:,64:,:5] += 5
    torch.testing.assert_close(prediction, forecast(m,modified[:,:64],u,d), rtol=0,atol=0)
    assert not torch.allclose(m.future_states(bank,(32,64,128)),m.future_states(modified,(32,64,128)))


def test_both_forecast_interfaces_preserve_same_physical_state():
    m, _, h, u, d = fixture(horizon=64)
    torch.testing.assert_close(forecast(m,h,u,d,mode='block'),forecast(m,h,u,d,mode='native'),rtol=0,atol=0)


def test_training_mean_initialization_has_nominal_steady_heat_balance():
    m, _, h, u, d = fixture(horizon=32)
    h=h.clone(); h[:,:,:13]=m.mean
    u=m.mean[5:7].expand(2,33,2)
    d=m.mean[7:13].expand(2,33,6)
    with torch.no_grad(): p=forecast(m,h,u,d)
    torch.testing.assert_close(p,m.mean[:5].expand_as(p),rtol=0,atol=1e-4)
