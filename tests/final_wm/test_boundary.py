from __future__ import annotations

import pytest
import torch

from src.final_wm.boundary import BoundaryModel, BoundarySequence, require_mode
from src.final_wm.contracts import BoundaryModelConfig, FinalWMProtocolError
from src.final_wm.synthetic import synthetic_history


def _model(horizon: int = 6, scenario_dim: int = 0) -> BoundaryModel:
    return BoundaryModel(BoundaryModelConfig(history_steps=16, horizon=horizon, d_hidden=32, scenario_dim=scenario_dim))


def test_forecast_shapes_and_mode_label() -> None:
    model = _model()
    batch = synthetic_history(batch=3, history_steps=16, horizon=6, seed=0)
    seq = model.forecast(batch.history.boundary, batch.history.actions)
    assert seq.mode == "forecast"
    assert seq.mu.shape == (3, 6, 7)
    assert seq.logvar.shape == (3, 6, 7)
    assert bool(torch.isfinite(seq.mu).all()) and bool(torch.isfinite(seq.logvar).all())


def test_forecast_at_init_stays_near_normalization_center() -> None:
    model = _model()
    batch = synthetic_history(batch=2, history_steps=16, horizon=4, seed=1)
    seq = model.forecast(batch.history.boundary, batch.history.actions)
    loc = model.loc
    assert (seq.mu - loc).abs().max().item() < model.scale.max().item()


def test_oracle_passthrough_labels_mode() -> None:
    model = _model()
    batch = synthetic_history(batch=2, history_steps=16, horizon=4, seed=2)
    seq = model.oracle(batch.future_boundary)
    assert seq.mode == "oracle"
    assert torch.equal(seq.mu, batch.future_boundary)
    assert bool((seq.logvar < -10.0).all())


def test_oracle_rejects_bad_shape() -> None:
    model = _model()
    with pytest.raises(FinalWMProtocolError):
        model.oracle(torch.zeros(2, 4, 6))


def test_require_mode_is_fail_closed() -> None:
    batch = synthetic_history(batch=1, history_steps=16, horizon=2, seed=3)
    seq = BoundarySequence(mu=batch.future_boundary, logvar=torch.zeros_like(batch.future_boundary), mode="oracle")
    require_mode(seq, "oracle")
    with pytest.raises(FinalWMProtocolError):
        require_mode(seq, "forecast")


def test_scenario_contract() -> None:
    model = _model(scenario_dim=3)
    batch = synthetic_history(batch=2, history_steps=16, horizon=4, seed=5)
    with pytest.raises(FinalWMProtocolError):
        model.forecast(batch.history.boundary, batch.history.actions)
    seq = model.forecast(batch.history.boundary, batch.history.actions, scenario=torch.zeros(2, 3), horizon=4)
    assert seq.mu.shape == (2, 4, 7)
    plain = _model()
    with pytest.raises(FinalWMProtocolError):
        plain.forecast(batch.history.boundary, batch.history.actions, scenario=torch.zeros(2, 3))


def test_forecast_history_length_is_contractual() -> None:
    model = _model()
    batch = synthetic_history(batch=2, history_steps=16, horizon=4, seed=6)
    with pytest.raises(FinalWMProtocolError):
        model.forecast(batch.history.boundary[:, :8], batch.history.actions[:, :8])


def test_sample_respects_logvar_scale() -> None:
    model = _model()
    batch = synthetic_history(batch=2, history_steps=16, horizon=4, seed=7)
    seq = model.forecast(batch.history.boundary, batch.history.actions)
    torch.manual_seed(0)
    sample = seq.sample()
    assert sample.shape == seq.mu.shape
    assert bool(torch.isfinite(sample).all())
