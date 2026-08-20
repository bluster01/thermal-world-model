"""End-to-end micro-smoke for the assembled final world model.

Local verification only: same-type synthetic teacher, tiny horizons, CPU.
These tests prove interface closure and trainability; they carry no
scientific content and do not authorize any long training run.
"""

from __future__ import annotations

import torch

from src.final_wm.contracts import (
    BoundaryModelConfig,
    ClosureConfig,
    ObserverConfig,
    TransitionConfig,
    WorldModelConfig,
)
from src.final_wm.model import FinalWorldModel
from src.final_wm.properties import AnalyticThermoProperties
from src.final_wm.synthetic import synthetic_history, teacher_rollout_obs


def _smoke_model(closure_mode: str = "none", latent_dim: int = 0) -> FinalWorldModel:
    config = WorldModelConfig(
        transition=TransitionConfig(latent_dim=latent_dim),
        observer=ObserverConfig(history_steps=16, d_hidden=32, latent_dim=latent_dim),
        boundary=BoundaryModelConfig(history_steps=16, d_hidden=32),
        closure=ClosureConfig(injection_mode=closure_mode),
    )
    return FinalWorldModel(config, AnalyticThermoProperties())


def _smoke_batch(seed: int = 0, horizon: int = 12):
    return synthetic_history(batch=8, history_steps=16, horizon=horizon, seed=seed)


def test_observer_training_reduces_nll_on_teacher_data() -> None:
    torch.manual_seed(0)
    model = _smoke_model()
    batch = _smoke_batch()
    target = teacher_rollout_obs(model.transition, batch, seed=0)
    params = list(model.observer.parameters()) + list(model.observation.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)

    losses = []
    for _ in range(60):
        result = model.forecast(
            batch.history, batch.future_actions,
            boundary_mode="oracle", true_future_boundary=batch.future_boundary,
        )
        loss = model.observation_nll(result.temps_mu, result.temps_sigma, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0], f"NLL did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}"
    assert all(torch.isfinite(torch.tensor(losses)))


def test_closure_receives_gradient_through_shared_transition() -> None:
    torch.manual_seed(1)
    model = _smoke_model(closure_mode="conservative")
    batch = _smoke_batch(seed=1)
    target = teacher_rollout_obs(model.transition, batch, seed=1)
    result = model.forecast(
        batch.history, batch.future_actions,
        boundary_mode="oracle", true_future_boundary=batch.future_boundary,
    )
    loss = ((result.temps_mu - target) ** 2).mean()
    loss.backward()
    grad = model.closure.net[-1].weight.grad
    assert grad is not None and bool((grad != 0).any())


def test_full_stack_with_latent_and_closure_smoke() -> None:
    torch.manual_seed(2)
    model = _smoke_model(closure_mode="conservative", latent_dim=2)
    batch = _smoke_batch(seed=2)
    result = model.forecast(
        batch.history, batch.future_actions,
        boundary_mode="oracle", true_future_boundary=batch.future_boundary,
    )
    assert result.states.shape == (8, 12, 13)
    assert bool(torch.isfinite(result.states).all())


def test_determinism_with_fixed_seed() -> None:
    def _run() -> float:
        torch.manual_seed(3)
        model = _smoke_model()
        batch = _smoke_batch(seed=3)
        target = teacher_rollout_obs(model.transition, batch, seed=3)
        params = list(model.observer.parameters()) + list(model.observation.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        loss = None
        for _ in range(5):
            result = model.forecast(
                batch.history, batch.future_actions,
                boundary_mode="oracle", true_future_boundary=batch.future_boundary,
            )
            loss = model.observation_nll(result.temps_mu, result.temps_sigma, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss)

    assert _run() == _run()


def test_forecast_mode_full_pipeline_runs_without_true_future() -> None:
    torch.manual_seed(4)
    model = _smoke_model()
    batch = _smoke_batch(seed=4)
    result = model.forecast(batch.history, batch.future_actions, boundary_mode="forecast")
    assert result.mode == "forecast"
    assert result.temps_mu.shape == (8, 12, 5)
    assert bool(torch.isfinite(result.temps_mu).all())
