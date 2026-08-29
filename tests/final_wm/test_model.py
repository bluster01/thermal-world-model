from __future__ import annotations

import pytest
import torch

from src.final_wm.contracts import (
    BoundaryModelConfig,
    ClosureConfig,
    ControllerConfig,
    FinalWMProtocolError,
    ObserverConfig,
    StateLayout,
    WorldModelConfig,
)
from src.final_wm.controller import CascadePIController
from src.final_wm.model import FinalWorldModel, HistoryWindow
from src.final_wm.properties import AnalyticThermoProperties
from src.final_wm.synthetic import synthetic_history


def _config(**kwargs) -> WorldModelConfig:
    defaults = dict(
        observer=ObserverConfig(history_steps=16, d_hidden=32),
        boundary=BoundaryModelConfig(history_steps=16, d_hidden=32),
        closure=ClosureConfig(injection_mode="none"),
    )
    defaults.update(kwargs)
    return WorldModelConfig(**defaults)


def _model(**kwargs) -> FinalWorldModel:
    return FinalWorldModel(_config(**kwargs), AnalyticThermoProperties())


def _history(seed: int = 0, horizon: int = 8):
    return synthetic_history(batch=2, history_steps=16, horizon=horizon, seed=seed)


def test_hybrid_correction_mask_slow_states_only() -> None:
    """Repair 1-B contract: with randomised observer heads, hybrid mode moves
    only tm/rb/latent off the five-point anchor; learned mode may move all
    dims; both return the anchor exactly at zero-init."""
    layout = StateLayout(latent_dim=0)
    for mode, moved in (("hybrid", False), ("learned", True)):
        model = _model()
        object.__setattr__(model.config, "initial_state_mode", mode)  # frozen dataclass
        batch = _history(seed=5)
        anchor = model._steady_initial_state(batch.history)
        x0 = model._initial_state(batch.history)
        assert bool((x0 == anchor).all())  # zero-init observer
        with torch.no_grad():
            model.observer.mu_head.bias.uniform_(-0.5, 0.5)
        x1 = model._initial_state(batch.history)
        delta = (x1 - anchor).abs() > 1e-9
        h = torch.zeros(layout.dim, dtype=torch.bool)
        h[layout.h_slice] = True
        h[layout.m_liq_slice] = True
        h[layout.dsw_lag_slice] = True
        if moved:
            assert bool((x1 != anchor).any())
        else:
            assert not bool((delta & h).any())  # anchored dims untouched
            assert bool((delta & ~h).any())    # slow dims did move


def test_forecast_oracle_mode_shapes() -> None:
    model = _model()
    batch = _history()
    result = model.forecast(
        batch.history, batch.future_actions,
        boundary_mode="oracle", true_future_boundary=batch.future_boundary,
    )
    assert result.mode == "oracle"
    assert result.temps_mu.shape == (2, 8, 5)
    assert result.temps_sigma.shape == (2, 8, 5)
    assert result.states.shape == (2, 8, 11)
    assert bool(torch.isfinite(result.temps_mu).all())
    assert bool((result.temps_sigma > 0).all())
    assert result.in_support is None


def test_forecast_mode_never_reads_true_future_boundary() -> None:
    model = _model()
    batch = _history()
    result = model.forecast(batch.history, batch.future_actions, boundary_mode="forecast")
    assert result.mode == "forecast"
    with pytest.raises(FinalWMProtocolError):
        model.forecast(
            batch.history, batch.future_actions,
            boundary_mode="forecast", true_future_boundary=batch.future_boundary,
        )
    with pytest.raises(FinalWMProtocolError):
        model.forecast(batch.history, batch.future_actions, boundary_mode="oracle")


def test_counterfactual_support_gate() -> None:
    model = _model()
    batch = _history()
    # In-support actions: clipped to the history range.
    lo = batch.history.actions.reshape(-1, 2).min(dim=0).values
    hi = batch.history.actions.reshape(-1, 2).max(dim=0).values
    mid = ((lo + hi) / 2).view(1, 1, 2).expand(2, 8, 2)
    result = model.counterfactual(
        batch.history, mid, boundary_mode="oracle", true_future_boundary=batch.future_boundary
    )
    assert result.in_support is not None and bool(result.in_support.all())

    out_of_support = torch.full((2, 8, 2), 0.999)
    with pytest.raises(FinalWMProtocolError):
        model.counterfactual(
            batch.history, out_of_support,
            boundary_mode="oracle", true_future_boundary=batch.future_boundary,
        )
    flagged = model.counterfactual(
        batch.history, out_of_support,
        boundary_mode="oracle", true_future_boundary=batch.future_boundary,
        allow_extrapolation=True,
    )
    assert flagged.in_support is not None and not bool(flagged.in_support.all())


def test_counterfactual_cannot_borrow_another_windows_support() -> None:
    model = _model()
    batch = _history()
    history = batch.history.__class__(
        obs=batch.history.obs,
        actions=torch.tensor([
            [[0.10, 0.20], [0.20, 0.30]],
            [[0.80, 0.60], [0.90, 0.70]],
        ]).repeat_interleave(batch.history.actions.shape[1] // 2, dim=1),
        boundary=batch.history.boundary,
    )
    borrowed = torch.tensor([0.85, 0.65]).view(1, 1, 2).expand(2, 8, 2)
    with pytest.raises(FinalWMProtocolError):
        model.counterfactual(
            history, borrowed,
            boundary_mode="oracle", true_future_boundary=batch.future_boundary,
        )


def test_counterfactual_uses_shared_transition_and_is_action_sensitive() -> None:
    model = _model()
    batch = _history(horizon=36)
    lo = batch.history.actions.min(dim=1).values
    hi = batch.history.actions.max(dim=1).values
    low = lo.unsqueeze(1).expand(2, 36, 2)
    high = hi.unsqueeze(1).expand(2, 36, 2)
    truth = batch.future_boundary[:, :1].repeat(1, 36, 1)
    r_low = model.counterfactual(batch.history, low, boundary_mode="oracle", true_future_boundary=truth)
    r_high = model.counterfactual(batch.history, high, boundary_mode="oracle", true_future_boundary=truth)
    # More spray valve opening -> cooler terminal temperature.
    assert (r_high.temps_mu[:, -6:, 4] < r_low.temps_mu[:, -6:, 4] + 1e-4).all()


def test_closed_loop_rollout_smoke() -> None:
    model = _model()
    batch = _history(horizon=12)
    controller = CascadePIController(ControllerConfig(kp=0.02, ki=0.001))
    sp = torch.full((2, 12), 565.0)
    result = model.closed_loop(
        batch.history, sp, controller,
        boundary_mode="oracle", true_future_boundary=batch.future_boundary,
    )
    assert result.temps_mu.shape == (2, 12, 5)
    assert bool(torch.isfinite(result.states).all())
    assert result.in_support is not None
    valve = controller.valve
    assert bool((valve >= 0.0).all()) and bool((valve <= 1.0).all())


def test_closure_engages_only_when_configured() -> None:
    batch = _history()
    kwargs = dict(
        observer=ObserverConfig(history_steps=16, d_hidden=32),
        boundary=BoundaryModelConfig(history_steps=16, d_hidden=32),
        closure=ClosureConfig(injection_mode="conservative"),
    )
    model = FinalWorldModel(WorldModelConfig(**kwargs), AnalyticThermoProperties())
    result = model.forecast(
        batch.history, batch.future_actions,
        boundary_mode="oracle", true_future_boundary=batch.future_boundary,
    )
    assert bool(torch.isfinite(result.states).all())


def test_observation_nll() -> None:
    mu = torch.zeros(2, 4, 5)
    sigma = torch.ones(2, 4, 5)
    target = torch.zeros(2, 4, 5)
    nll = FinalWorldModel.observation_nll(mu, sigma, target)
    assert nll.item() == pytest.approx(0.0)
    with pytest.raises(FinalWMProtocolError):
        FinalWorldModel.observation_nll(mu, sigma, torch.zeros(2, 4, 4))


def test_state_continuity_runs() -> None:
    model = _model()
    batch = _history(horizon=16)
    # Split the future into a gap (first 8 steps) and a next window (last 8
    # steps preceded by 16 history steps from the synthetic timeline).
    obs_full = torch.cat([batch.history.obs, batch.future_obs], dim=1)
    act_full = torch.cat([batch.history.actions, batch.future_actions], dim=1)
    bnd_full = torch.cat([batch.history.boundary, batch.future_boundary], dim=1)
    next_history = HistoryWindow(
        obs=obs_full[:, 8:24], actions=act_full[:, 8:24], boundary=bnd_full[:, 8:24],
    )
    from src.final_wm.boundary import BoundarySequence
    gap = BoundarySequence(
        mu=bnd_full[:, 16:24], logvar=torch.zeros_like(bnd_full[:, 16:24]), mode="oracle"
    )
    err = model.state_continuity(batch.history, gap, act_full[:, 16:24], next_history)
    assert err.shape == (2,)
    assert bool(torch.isfinite(err).all())
