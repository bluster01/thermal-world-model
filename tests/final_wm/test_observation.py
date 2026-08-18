from __future__ import annotations

import pytest
import torch

from src.final_wm.contracts import ObservationConfig, StateLayout, TransitionConfig
from src.final_wm.observation import ObservationModel
from src.final_wm.properties import AnalyticThermoProperties
from src.final_wm.synthetic import synthetic_history
from src.final_wm.transition import Fan2020UDETransition


def _observation(heteroscedastic: bool = True) -> tuple[ObservationModel, StateLayout]:
    layout = StateLayout()
    return ObservationModel(layout, ObservationConfig(heteroscedastic=heteroscedastic)), layout


def test_sigma_positive_and_bounded() -> None:
    obs_model, layout = _observation()
    state = torch.randn(3, layout.dim) * torch.tensor([500.0] * 3 + [150.0] * 3 + [150.0] + [60.0] * 2) \
        + torch.tensor([3000.0] * 3 + [550.0] * 3 + [250.0] + [60.0] * 2)
    sigma = obs_model.sigma(state)
    assert sigma.shape == (3, 5)
    assert bool((sigma >= obs_model.config.min_sigma_c).all())
    assert bool((sigma <= obs_model.config.max_sigma_c).all())


def test_sigma_init_value() -> None:
    obs_model, layout = _observation(heteroscedastic=False)
    state = torch.zeros(2, layout.dim)
    sigma = obs_model.sigma(state)
    assert sigma.mean().item() == pytest.approx(obs_model.config.init_sigma_c, rel=0.05)


def test_distribution_matches_transition_output() -> None:
    obs_model, layout = _observation()
    transition = Fan2020UDETransition(TransitionConfig(), AnalyticThermoProperties())
    batch = synthetic_history(batch=2, history_steps=8, horizon=4, seed=0)
    state = transition.initial_steady_state(
        batch.future_boundary[:, 0], batch.future_actions[:, 0], batch.history.obs[:, -1]
    )
    mean = transition.output_temperatures(state, batch.future_boundary[:, 0], batch.future_actions[:, 0])
    mu, sigma = obs_model.distribution(mean, state)
    assert torch.equal(mu, mean)
    assert sigma.shape == mean.shape


def test_distribution_validates_shapes() -> None:
    obs_model, layout = _observation()
    with pytest.raises(Exception):
        obs_model.distribution(torch.zeros(2, 4), torch.zeros(2, layout.dim))
    with pytest.raises(Exception):
        obs_model.sigma(torch.zeros(2, layout.dim + 1))
