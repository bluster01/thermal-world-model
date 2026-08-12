from __future__ import annotations

import numpy as np
import pytest
import torch

from src.phase35.multistep.rm3_orthogonal import (
    generate_rm3_confounded_synthetic,
    oof_nuisance_residuals,
    orthogonal_mimo_moment,
    orthogonal_r_loss,
    orthogonal_trajectory_moments,
    synthetic_expanding_splits,
    validate_expanding_splits,
)


def test_trajectory_moments_keep_every_prefix_step() -> None:
    rng = np.random.default_rng(53)
    action = rng.normal(size=(200, 6, 2))
    truth = np.stack([np.eye(2) * (step + 1) / 10 for step in range(6)])
    outcome = np.einsum("nhi,hij->nhj", action, truth)
    audits = orthogonal_trajectory_moments(
        action, outcome, ridge_alpha=1e-8, epsilon=1e-10,
        maximum_condition_number=1e4, minimum_differential_to_common_energy=0.01,
    )
    assert len(audits) == 6
    assert np.allclose(np.stack([item.matrix for item in audits]), truth, atol=1e-6)
from src.phase35.schema import Phase35ProtocolError


def _recover(collinear: bool = False):
    x, action, outcome, truth = generate_rm3_confounded_synthetic(
        seed=13, n_rows=1800, collinear_actions=collinear
    )
    splits = synthetic_expanding_splits(len(x))
    residual = oof_nuisance_residuals(
        x, action, outcome, splits, ridge_alpha=1e-3, epsilon=1e-10
    )
    audit = orthogonal_mimo_moment(
        residual.action,
        residual.outcome,
        ridge_alpha=1e-6,
        epsilon=1e-10,
        maximum_condition_number=1000.0,
        minimum_differential_to_common_energy=0.05,
    )
    return residual, audit, truth


def test_rm3_oof_orthogonal_moment_recovers_confounded_known_truth() -> None:
    residual, audit, truth = _recover()
    assert residual.evaluated.sum() == 900
    assert np.max(np.abs(audit.matrix - truth)) < 0.03
    assert audit.independent_channels_supported is True


def test_rm3_collinear_innovation_refuses_independent_channels() -> None:
    _, audit, _ = _recover(collinear=True)
    assert audit.independent_channels_supported is False
    assert audit.differential_to_common_energy_ratio < 0.05


def test_rm3_shuffled_action_destroys_known_truth_recovery() -> None:
    residual, audit, truth = _recover()
    rng = np.random.default_rng(8)
    shuffled = residual.action.copy()
    chosen = np.flatnonzero(residual.evaluated)
    shuffled[chosen] = shuffled[rng.permutation(chosen)]
    placebo = orthogonal_mimo_moment(
        shuffled,
        residual.outcome,
        ridge_alpha=1e-6,
        epsilon=1e-10,
        maximum_condition_number=1000.0,
        minimum_differential_to_common_energy=0.05,
    )
    assert np.linalg.norm(placebo.matrix - truth) > np.linalg.norm(audit.matrix - truth) * 10


def test_rm3_expanding_folds_reject_contemporaneous_training() -> None:
    with pytest.raises(Phase35ProtocolError, match="strictly precede"):
        validate_expanding_splits(((np.arange(10), np.arange(9, 20)),), 20)


def test_rm3_r_loss_is_zero_at_the_orthogonal_target() -> None:
    target = torch.tensor([[1.0, -2.0], [0.5, 0.2]])
    assert orthogonal_r_loss(target.clone(), target).item() == pytest.approx(0.0)
