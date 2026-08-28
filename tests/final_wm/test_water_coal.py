from __future__ import annotations

import pytest
import torch

from src.final_wm.contracts import (
    BoundaryModelConfig,
    ClosureConfig,
    FinalWMProtocolError,
    ObserverConfig,
    WorldModelConfig,
)
from src.final_wm.model import FinalWorldModel, HistoryWindow
from src.final_wm.properties import AnalyticThermoProperties
from src.final_wm.synthetic import synthetic_history
from src.final_wm.water_coal import (
    WaterCoalReference,
    fit_water_coal_reference,
    promote_water_coal_model,
)


def _reference() -> WaterCoalReference:
    return WaterCoalReference(
        coefficients=(3.6, 0.1, -0.02),
        load_center=350.0,
        load_scale=100.0,
        residual_scale=0.15,
        n_train=100,
    )


def _model() -> FinalWorldModel:
    cfg = WorldModelConfig(
        observer=ObserverConfig(history_steps=16, d_hidden=32),
        boundary=BoundaryModelConfig(history_steps=16, d_hidden=32),
        closure=ClosureConfig(injection_mode="conservative"),
        boundary_mode="oracle",
        initial_state_mode="hybrid",
    )
    return FinalWorldModel(cfg, AnalyticThermoProperties())


def _with_a5_context(boundary: torch.Tensor) -> torch.Tensor:
    load = 350.0 + 20.0 * torch.sin(torch.arange(boundary.shape[-2]) / 5.0)
    load = load.to(boundary).view(1, -1, 1).expand(boundary.shape[0], -1, -1)
    ratio = 3.6 + 0.08 * torch.cos(torch.arange(boundary.shape[-2]) / 4.0)
    ratio = ratio.to(boundary).view(1, -1, 1).expand(boundary.shape[0], -1, -1)
    return torch.cat([boundary, ratio, load], dim=-1)


def test_reference_fit_reads_train_only() -> None:
    load = torch.linspace(200.0, 600.0, 100)
    x = (load - 400.0) / 100.0
    ratio = 3.5 + 0.2 * x - 0.03 * x.square() + 0.01 * torch.sin(x)
    split = torch.cat([torch.zeros(60, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    valid = torch.ones(100, dtype=torch.bool)

    ref_a = fit_water_coal_reference(load, ratio, split, valid)
    ratio[60:] = 99.0
    ref_b = fit_water_coal_reference(load, ratio, split, valid)

    assert ref_a.coefficients == pytest.approx(ref_b.coefficients, abs=1e-12)
    assert ref_a.load_center == pytest.approx(ref_b.load_center, abs=1e-12)
    assert ref_a.load_scale == pytest.approx(ref_b.load_scale, abs=1e-12)
    assert ref_a.residual_scale == pytest.approx(ref_b.residual_scale, abs=1e-12)
    assert ref_a.n_train == 60
    assert ref_a.residual_scale > 0.0


def test_zero_weight_is_exact_nested_identity_and_nonzero_activates() -> None:
    torch.manual_seed(4)
    base = _model().eval()
    a5 = _model().eval()
    a5.load_state_dict(base.state_dict())
    promote_water_coal_model(a5, _reference())

    batch = synthetic_history(batch=2, history_steps=16, horizon=8, seed=5)
    history_a5 = HistoryWindow(
        obs=batch.history.obs,
        actions=batch.history.actions,
        boundary=_with_a5_context(batch.history.boundary),
    )
    future_a5 = _with_a5_context(batch.future_boundary)
    with torch.no_grad():
        out_base = base.forecast(
            batch.history, batch.future_actions,
            boundary_mode="oracle", true_future_boundary=batch.future_boundary,
        )
        out_zero = a5.forecast(
            history_a5, batch.future_actions,
            boundary_mode="oracle", true_future_boundary=future_a5,
        )
    assert torch.equal(out_base.temps_mu, out_zero.temps_mu)

    a5.transition.w_raw.data.fill_(0.5)
    total = a5.transition.water_coal_total_power(future_a5[:, 0])
    assert total.abs().max().item() <= 30_000.0
    with torch.no_grad():
        out_active = a5.forecast(
            history_a5, batch.future_actions,
            boundary_mode="oracle", true_future_boundary=future_a5,
        )
    assert not torch.equal(out_zero.temps_mu, out_active.temps_mu)


def test_a5_is_oracle_only_and_requires_7_plus_2_boundary() -> None:
    model = _model()
    promote_water_coal_model(model, _reference())
    batch = synthetic_history(batch=1, history_steps=16, horizon=3, seed=8)
    with pytest.raises(FinalWMProtocolError, match="9 channels"):
        model.forecast(
            batch.history, batch.future_actions,
            boundary_mode="oracle", true_future_boundary=batch.future_boundary,
        )
    history = HistoryWindow(
        batch.history.obs, batch.history.actions, _with_a5_context(batch.history.boundary)
    )
    with pytest.raises(FinalWMProtocolError, match="oracle-only"):
        model.forecast(history, batch.future_actions, boundary_mode="forecast")
