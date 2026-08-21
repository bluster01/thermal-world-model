from __future__ import annotations

import pytest
import torch

from src.final_wm.contracts import FinalWMProtocolError, ObserverConfig, StateLayout
from src.final_wm.observer import ProbabilisticObserver
from src.final_wm.synthetic import synthetic_history


def _observer(history_steps: int = 16, latent_dim: int = 0) -> ProbabilisticObserver:
    layout = StateLayout(latent_dim=latent_dim)
    return ProbabilisticObserver(ObserverConfig(history_steps=history_steps, d_hidden=32, latent_dim=latent_dim), layout)


def test_posterior_shapes_and_validity() -> None:
    observer = _observer()
    batch = synthetic_history(batch=4, history_steps=16, horizon=4, seed=1)
    anchor = torch.zeros(4, 11)
    mu, sigma = observer.posterior(batch.history.obs, batch.history.actions, batch.history.boundary, anchor)
    assert mu.shape == (4, 11) and sigma.shape == (4, 11)
    assert bool((sigma > 0).all())
    assert bool(torch.isfinite(mu).all()) and bool(torch.isfinite(sigma).all())
    # Repair 1-B: zero-initialised heads return the anchor exactly.
    assert bool((mu == anchor).all())
    with torch.no_grad():
        observer.mu_head.bias.add_(0.01)
    mu2, _ = observer.posterior(batch.history.obs, batch.history.actions, batch.history.boundary, anchor)
    assert not bool((mu2 == anchor).all())


def test_posterior_bounded_correction_around_anchor() -> None:
    observer = _observer()
    batch = synthetic_history(batch=4, history_steps=16, horizon=4, seed=2)
    anchor = torch.randn(4, 11) * observer.state_scale + observer.state_loc
    with torch.no_grad():
        observer.mu_head.bias.uniform_(-0.5, 0.5)  # arbitrary non-zero heads
    mu, _sigma = observer.posterior(batch.history.obs, batch.history.actions, batch.history.boundary, anchor)
    # Repair 1-B: correction bounded by 0.1 x state scale around the anchor.
    assert bool(((mu - anchor).abs() <= 0.1 * observer.state_scale + 1e-5).all())


def test_observer_with_latent_block() -> None:
    observer = _observer(latent_dim=3)
    batch = synthetic_history(batch=2, history_steps=16, horizon=4, seed=3)
    anchor = torch.zeros(2, 14)
    mu, sigma = observer.posterior(batch.history.obs, batch.history.actions, batch.history.boundary, anchor)
    assert mu.shape == (2, 14)
    assert bool(((mu - anchor).abs() <= 0.1 * observer.state_scale + 1e-6).all())


def test_observer_history_length_is_contractual() -> None:
    observer = _observer(history_steps=16)
    batch = synthetic_history(batch=2, history_steps=16, horizon=4, seed=4)
    anchor = torch.zeros(2, 11)
    with pytest.raises(FinalWMProtocolError):
        observer.posterior(batch.history.obs[:, :8], batch.history.actions, batch.history.boundary, anchor)
    with pytest.raises(FinalWMProtocolError):
        observer.posterior(batch.history.obs, batch.history.actions[:, :, :1], batch.history.boundary, anchor)
    with pytest.raises(FinalWMProtocolError):
        observer.posterior(batch.history.obs, batch.history.actions, batch.history.boundary, anchor[:, :8])


def test_state_continuity_error() -> None:
    observer = _observer()
    a = torch.zeros(3, 11)
    b = torch.zeros(3, 11)
    assert observer.state_continuity_error(a, b).abs().max().item() == 0.0
    b[:, 0] = 500.0  # one normalized unit in h1
    err = observer.state_continuity_error(a, b)
    assert err.tolist() == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(FinalWMProtocolError):
        observer.state_continuity_error(a[:, :8], b)


def test_sample_uses_reparameterization() -> None:
    observer = _observer()
    mu = torch.zeros(2, 11)
    sigma = torch.ones(2, 11)
    torch.manual_seed(0)
    sample = observer.sample(mu, sigma)
    assert sample.shape == (2, 11)
    assert sample.requires_grad is False or sample.grad_fn is not None  # reparam path exists
