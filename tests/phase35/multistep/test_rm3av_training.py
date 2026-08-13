from __future__ import annotations

import numpy as np
import pytest
import torch

from src.phase35.multistep.gatec_data import paired_history_feature_names
from src.phase35.multistep.rm3av_model import RM3AVModelConfig, build_rm3av_model
from src.phase35.multistep.rm3av_training import (
    fit_oof_action_projection,
    fit_oof_action_outcome_audit,
    fit_oof_r_residuals,
    rm3av_multitask_loss,
    valve_dynamics_loss,
)
from src.phase35.schema import Phase35ProtocolError


FEATURES = paired_history_feature_names()


def _model(candidate_id: str):
    return build_rm3av_model(
        RM3AVModelConfig(
            candidate_id=candidate_id,
            window=6,
            horizon=60,
            n_features=len(FEATURES),
            d_model=8,
            latent_dim=4,
            dropout=0.0,
        ),
        FEATURES,
    )


def _batch(batch: int = 4):
    generator = torch.Generator().manual_seed(404)
    history = torch.randn(batch, 6, len(FEATURES), generator=generator)
    future_sp = 540.0 + torch.randn(batch, 60, 2, generator=generator)
    target = {
        key: torch.randn(batch, 60, 2, generator=generator)
        for key in ("valve", "tin", "local", "terminal")
    }
    return history, future_sp, target


def test_oof_nuisance_residuals_are_group_disjoint_and_finite() -> None:
    rng = np.random.default_rng(8)
    n = 24
    history = rng.normal(size=(n, 6, len(FEATURES))).astype(np.float32)
    action = rng.normal(size=(n, 60, 2)).astype(np.float32)
    outcome = rng.normal(size=(n, 60, 2)).astype(np.float32)
    groups = np.repeat(np.arange(6), 4)
    result = fit_oof_r_residuals(history, action, outcome, groups, ridge=1e-3)
    assert result.action_residual.shape == action.shape
    assert result.outcome_residual.shape == outcome.shape
    assert np.isfinite(result.action_residual).all()
    assert set(result.fold_records[0]["train_groups"]).isdisjoint(
        result.fold_records[0]["held_out_groups"]
    )
    assert sorted(np.concatenate([record["indices"] for record in result.fold_records])) == list(range(n))


def test_oof_action_audit_conditions_on_history_and_future_sp_without_group_overlap() -> None:
    rng = np.random.default_rng(81)
    n = 80
    history = rng.normal(size=(n, 6, len(FEATURES))).astype(np.float32)
    future_sp = rng.normal(size=(n, 60, 2)).astype(np.float32)
    action = (0.8 * future_sp + 0.1 * rng.normal(size=future_sp.shape)).astype(np.float32)
    outcome = (0.4 * action + rng.normal(size=action.shape)).astype(np.float32)
    groups = np.repeat(np.arange(10), 8)
    result = fit_oof_action_outcome_audit(
        history, future_sp, action, outcome, groups, ridge=1e-3
    )
    assert result.action_innovation.shape == action.shape
    assert result.outcome_innovation.shape == outcome.shape
    assert all(record["group_overlap_count"] == 0 for record in result.fold_records)
    assert min(value for value in result.action_r2_by_side if value is not None) > 0.5


def test_action_projection_is_train_fitted_and_shields_only_free_context() -> None:
    rng = np.random.default_rng(9)
    history = rng.normal(size=(18, 6, len(FEATURES))).astype(np.float32)
    action = rng.normal(size=(18, 60, 2)).astype(np.float32)
    groups = np.repeat(np.arange(6), 3)
    fitted = fit_oof_action_projection(history, action, groups, ridge=1e-3)
    assert fitted.projector.shape == (len(FEATURES), len(FEATURES))
    assert np.allclose(fitted.projector, fitted.projector.T, atol=1e-5)

    model = _model("C09").eval()
    torch_history, future_sp, _ = _batch(batch=2)
    before = model(torch_history, future_sp)
    model.set_action_shield(torch.from_numpy(fitted.projector))
    after = model(torch_history, future_sp)
    assert torch.equal(before["local_effect"], after["local_effect"])
    assert not torch.equal(
        before["residual_local_prediction"], after["residual_local_prediction"]
    )
    assert after["action_shield_fitted"] is True


def test_action_shield_refuses_validation_fit_or_wrong_shape() -> None:
    model = _model("C09")
    with pytest.raises(Phase35ProtocolError):
        model.set_action_shield(torch.eye(len(FEATURES) + 1))


def test_logged_action_auxiliary_is_real_loss_and_reaches_response_parameters() -> None:
    model = _model("C10")
    history, future_sp, targets = _batch()
    output = model(history, future_sp, logged_future_valve=targets["valve"])
    losses = rm3av_multitask_loss(
        output,
        targets,
        candidate_id="C10",
        target_scales={key: 1.0 for key in targets},
    )
    assert "logged_action_auxiliary" in losses
    losses["total"].backward()
    gradients = [
        parameter.grad
        for parameter in model.base.model.local_response.parameters()
        if parameter.requires_grad
    ]
    assert any(value is not None and torch.count_nonzero(value) for value in gradients)


def test_r_loss_requires_oof_residuals_and_has_declared_component() -> None:
    model = _model("C12")
    history, future_sp, targets = _batch()
    action_residual = torch.randn_like(targets["valve"])
    output = model(
        history,
        future_sp,
        logged_future_valve=targets["valve"],
        oof_action_residual=action_residual,
    )
    with pytest.raises(Phase35ProtocolError, match="OOF residuals"):
        rm3av_multitask_loss(
            output,
            targets,
            candidate_id="C12",
            target_scales={key: 1.0 for key in targets},
        )
    losses = rm3av_multitask_loss(
        output,
        targets,
        candidate_id="C12",
        target_scales={key: 1.0 for key in targets},
        action_residual=action_residual,
        outcome_residual=torch.randn_like(targets["local"]),
    )
    assert losses["oof_r_loss"].item() >= 0.0


def test_oof_r_model_requires_declared_future_sp_when_it_was_fit_with_sp() -> None:
    rng = np.random.default_rng(912)
    n = 40
    history = rng.normal(size=(n, 6, len(FEATURES))).astype(np.float32)
    future_sp = rng.normal(size=(n, 60, 2)).astype(np.float32)
    action_change = (0.6 * future_sp).astype(np.float32)
    outcome = (0.3 * action_change).astype(np.float32)
    groups = np.repeat(np.arange(10), 4)
    from src.phase35.multistep.rm3av_training import fit_oof_r_model

    model, fitted = fit_oof_r_model(
        history, action_change, outcome, groups, ridge=1e-3, future_sp=future_sp
    )
    assert model.uses_future_sp is True
    assert fitted.action_residual.shape == action_change.shape
    with pytest.raises(Phase35ProtocolError, match="future SP"):
        model.residualize(history, action_change, outcome, groups)


def test_valve_dynamics_loss_distinguishes_delta_and_multiscale_roughness() -> None:
    target = torch.zeros(2, 60, 2)
    prediction = target.clone()
    exact = valve_dynamics_loss(prediction, target)
    assert exact["delta"].item() == 0.0
    assert exact["roughness"].item() == 0.0
    prediction[:, 1::2] = 1.0
    changed = valve_dynamics_loss(prediction, target)
    assert changed["delta"].item() > 0.0
    assert changed["roughness"].item() > 0.0


def test_two_window_rollout_carries_explicit_and_downstream_state() -> None:
    model = _model("C31").eval()
    history, future_sp, _ = _batch(batch=2)
    second_history = history + 0.1
    second_sp = future_sp + 0.2
    rollout = model.forward_two_window(history, future_sp, second_history, second_sp)
    assert torch.equal(
        rollout["second"]["continuation_initial_latent_state"],
        rollout["first"]["latent_state"][:, -1],
    )
    assert torch.equal(
        rollout["second"]["continuation_initial_local_state"],
        rollout["first"]["local_state"][:, -1],
    )
    assert rollout["second"]["terminal_prediction"].shape == (2, 60, 2)
