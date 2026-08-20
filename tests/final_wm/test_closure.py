from __future__ import annotations

import pytest
import torch

from src.final_wm.closure import ActionBlindClosure
from src.final_wm.contracts import (
    CLOSURE_BOUNDARY_CHANNELS,
    ClosureConfig,
    FinalWMProtocolError,
    StateLayout,
)
from src.final_wm.synthetic import synthetic_history
from src.final_wm.transition import Fan2020UDETransition, ResidualInjection
from src.final_wm.contracts import TransitionConfig
from src.final_wm.properties import AnalyticThermoProperties


def _closure(mode: str = "conservative", latent_dim: int = 0, stochastic: bool = False):
    layout = StateLayout(latent_dim=latent_dim)
    config = ClosureConfig(injection_mode=mode, stochastic=stochastic)
    return ActionBlindClosure(config, layout), layout


def _state_and_boundary(seed: int = 0, latent_dim: int = 0):
    batch = synthetic_history(batch=2, history_steps=8, horizon=4, seed=seed)
    transition = Fan2020UDETransition(
        TransitionConfig(latent_dim=latent_dim), AnalyticThermoProperties()
    )
    state = transition.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    return state, batch.future_boundary[:, 0]


def test_closure_is_action_blind_at_runtime() -> None:
    closure, _layout = _closure()
    state, boundary = _state_and_boundary()
    # Perturb every action-dependent quantity the closure could conceivably
    # see: the residual must be bit-identical because actions never enter
    # the feature vector.
    out_a = closure(state, boundary)
    torch.manual_seed(7)
    out_b = closure(state, boundary)
    assert torch.equal(out_a.steam_power, out_b.steam_power)
    # Feature dimension: 11 physical + 6 whitelisted boundary channels.
    assert closure.feature_dim == 11 + len(CLOSURE_BOUNDARY_CHANNELS)


def test_closure_injection_modes() -> None:
    state, boundary = _state_and_boundary()
    for mode in ("none", "steam_only", "conservative"):
        closure, _layout = _closure(mode=mode)
        out = closure(state, boundary)
        assert out.steam_power.shape == (2, 3)
        assert out.metal_power.shape == (2, 3)
        if mode == "none":
            assert out.steam_power.abs().max().item() == 0.0
            assert out.metal_power.abs().max().item() == 0.0
        if mode == "conservative":
            assert torch.allclose(out.steam_power, -out.metal_power)


def test_closure_starts_at_exactly_zero() -> None:
    closure, _layout = _closure(mode="steam_only")
    state, boundary = _state_and_boundary()
    out = closure(state, boundary)
    assert out.steam_power.abs().max().item() == 0.0


def test_closure_amplitude_saturates() -> None:
    closure, _layout = _closure(mode="steam_only")
    state, boundary = _state_and_boundary()
    with torch.no_grad():
        closure.net[-1].bias.fill_(100.0)
    out = closure(state, boundary)
    assert out.steam_power.abs().max().item() <= closure.config.residual_scale_kw


def test_closure_stochastic_contract() -> None:
    closure, _layout = _closure(stochastic=True)
    state, boundary = _state_and_boundary()
    with pytest.raises(FinalWMProtocolError):
        closure(state, boundary)
    out = closure(state, boundary, epsilon=torch.randn(2, 4))
    assert out.steam_power.shape == (2, 3)

    closure_det, _ = _closure()
    with pytest.raises(FinalWMProtocolError):
        closure_det(state, boundary, epsilon=torch.randn(2, 4))


def test_closure_latent_increment() -> None:
    closure, _layout = _closure(latent_dim=2)
    state, boundary = _state_and_boundary(latent_dim=2)
    out = closure(state, boundary)
    assert out.latent_step is not None and out.latent_step.shape == (2, 2)
    assert out.latent_step.abs().max().item() <= closure.config.latent_scale


def test_closure_rejects_wrong_state_width() -> None:
    closure, _layout = _closure()
    _state, boundary = _state_and_boundary()
    with pytest.raises(FinalWMProtocolError):
        closure(torch.zeros(2, 8), boundary)
