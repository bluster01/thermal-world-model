import numpy as np
import pytest
import torch

from .data import unpack
from .evaluation import forecast, scenarios
from .focused_models import Focused


def sample(arm):
    torch.manual_seed(11)
    model = Focused(np.zeros(13), np.ones(13), np.zeros(11), np.ones(11), arm).double()
    history = torch.randn(2, 64, 24, dtype=torch.float64)*.1
    history[:, :, 5:7] = .4
    u = history[:, -1:, 5:7].expand(-1, 97, -1).clone()
    d = history[:, -1:, 7:13].expand(-1, 97, -1)
    return model, history, u, d


@pytest.mark.parametrize('mode', ['block', 'native'])
def test_zero_adapter_preserves_original_forecast(mode):
    model, h, u, d = sample('B')
    with torch.no_grad():
        actual = forecast(model, h, u, d, mode=mode)
        expected = forecast(model.background, h[:, :, :13], u, d, mode=mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['block', 'native'])
def test_protected_candidate_difference_is_only_continuous_response(mode):
    model, h, u, d = sample('D')
    torch.nn.init.normal_(model.adapter[-1].weight, std=.1)
    candidate = u.clone()
    candidate[:, 40:, 0] += .03
    with torch.no_grad():
        delta = forecast(model, h, candidate, d, mode=mode)-forecast(model, h, u, d, mode=mode)
        raw = h[:, :, :13]
        expected = model.response.core(raw, candidate[:, :-1], d[:, :-1])['action_response_C']-model.response.core(raw, u[:, :-1], d[:, :-1])['action_response_C']
    torch.testing.assert_close(delta, expected, rtol=0, atol=1e-10)
    assert delta[:, :40].abs().max() < 1e-10
    assert delta[:, :, 0].abs().max() < 1e-10
    assert delta[:, :, 4].abs().max() > 1e-8


def test_auxiliary_future_cannot_enter_controls_or_labels():
    bank = np.zeros((2, 96, 24), dtype=np.float32)
    first = unpack(bank)
    bank[:, 64:, 13:] = np.nan
    second = unpack(bank)
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y)
    assert second[2].shape[-1] == 6


def test_background_and_adapter_train_but_response_is_frozen():
    model, h, u, d = sample('D')
    model(h, u[:, :32], d[:, :32]).square().mean().backward()
    assert model.adapter[-1].weight.grad.abs().max() > 0
    assert any(p.grad is not None for p in model.background.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.response.parameters())


def test_long_scenarios_preserve_prefix_and_cover_every_onset():
    old, new = scenarios(), scenarios(length=272)
    assert len(old) == len(new) == 110
    for a, b in zip(old, new):
        assert a['id'] == b['id']
        np.testing.assert_array_equal(a['delta'], b['delta'][:161])
        assert len(b['delta']) >= b['onset']+128+1
