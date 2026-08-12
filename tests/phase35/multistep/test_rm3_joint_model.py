from __future__ import annotations

import pytest
import torch

from src.phase35.multistep.rm3_joint_model import (
    JointLatentPhysicalInterfaces,
    RM3JointConfig,
    inject_oof_residuals,
    oracle_forecast_consistency_loss,
)


def _inputs(batch: int = 3, horizon: int = 12):
    torch.manual_seed(4)
    return {
        "history_context": torch.randn(batch, 10),
        "boundary_tin": 510 + torch.randn(batch, horizon, 2),
        "explicit_local_effect": 0.2 * torch.randn(batch, horizon, 2),
        "baseline_valve": 40 + torch.randn(batch, 2),
        "baseline_tin": 510 + torch.randn(batch, 2),
        "baseline_local": 15 + torch.randn(batch, 2),
        "baseline_terminal": 540 + torch.randn(batch, 2),
    }


def test_rm3_joint_latent_is_finite_stable_and_shared() -> None:
    model = JointLatentPhysicalInterfaces(RM3JointConfig(context_dim=10, latent_dim=16, horizon=12))
    output = model(**_inputs())
    assert output["latent_state"].shape == (3, 12, 16)
    assert output["terminal_prediction"].shape == (3, 12, 2)
    assert torch.isfinite(output["terminal_prediction"]).all()
    assert output["stable_poles"].max().item() < 1.0


def test_terminal_bypass_is_exactly_invariant_to_future_action_effect() -> None:
    model = JointLatentPhysicalInterfaces(RM3JointConfig(context_dim=10, latent_dim=16, horizon=12))
    inputs = _inputs()
    original = model(**inputs)
    changed = model(**{**inputs, "explicit_local_effect": inputs["explicit_local_effect"] + 2.0})
    assert torch.equal(original["terminal_bypass"], changed["terminal_bypass"])
    assert not torch.allclose(
        original["terminal_physical_prediction"], changed["terminal_physical_prediction"]
    )


def test_future_effect_perturbation_preserves_prefix_causality() -> None:
    model = JointLatentPhysicalInterfaces(RM3JointConfig(context_dim=10, latent_dim=16, horizon=12))
    inputs = _inputs()
    changed_effect = inputs["explicit_local_effect"].clone()
    changed_effect[:, 7:] += 4.0
    original = model(**inputs)["terminal_prediction"]
    changed = model(**{**inputs, "explicit_local_effect": changed_effect})["terminal_prediction"]
    assert torch.allclose(original[:, :7], changed[:, :7], atol=1e-7, rtol=0.0)


def test_oracle_forecast_distillation_stops_oracle_gradient() -> None:
    forecast_value = torch.zeros(2, 3, 2, requires_grad=True)
    oracle_value = torch.ones(2, 3, 2, requires_grad=True)
    forecast = {key: forecast_value for key in ("local_drop_prediction", "terminal_physical_prediction")}
    oracle = {key: oracle_value for key in ("local_drop_prediction", "terminal_physical_prediction")}
    oracle_forecast_consistency_loss(forecast, oracle).backward()
    assert forecast_value.grad is not None
    assert oracle_value.grad is None


def test_oof_residual_injection_uses_empirical_rows_exactly() -> None:
    prediction = torch.zeros(2, 3, 2)
    bank = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    injected = inject_oof_residuals(prediction, bank, torch.tensor([3, 1]))
    assert torch.equal(injected[0], bank[3])
    assert torch.equal(injected[1], bank[1])
