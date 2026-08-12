"""OOF nuisance residuals and orthogonal MIMO response moments for RM3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..ms3r import crossfit_residual_matrix
from ..schema import Phase35ProtocolError


@dataclass(frozen=True)
class RM3OOFResiduals:
    action: np.ndarray
    outcome: np.ndarray
    fold_id: np.ndarray
    evaluated: np.ndarray


@dataclass(frozen=True)
class RM3MomentAudit:
    matrix: np.ndarray
    action_gram: np.ndarray
    condition_number: float
    common_energy: float
    differential_energy: float
    differential_to_common_energy_ratio: float
    independent_channels_supported: bool


def validate_expanding_splits(
    splits: Sequence[tuple[np.ndarray, np.ndarray]], n_rows: int
) -> None:
    seen = np.zeros(n_rows, dtype=bool)
    last_evaluation_stop = -1
    for train, evaluate in splits:
        train = np.asarray(train, dtype=np.int64)
        evaluate = np.asarray(evaluate, dtype=np.int64)
        if train.ndim != 1 or evaluate.ndim != 1 or not len(train) or not len(evaluate):
            raise Phase35ProtocolError("RM3 OOF folds must be non-empty one-dimensional arrays")
        if min(train.min(), evaluate.min()) < 0 or max(train.max(), evaluate.max()) >= n_rows:
            raise Phase35ProtocolError("RM3 OOF fold index is out of bounds")
        if train.max() >= evaluate.min():
            raise Phase35ProtocolError("RM3 nuisance training must strictly precede evaluation")
        if evaluate.min() <= last_evaluation_stop or seen[evaluate].any():
            raise Phase35ProtocolError("RM3 OOF evaluation folds must be ordered and disjoint")
        seen[evaluate] = True
        last_evaluation_stop = int(evaluate.max())


def oof_nuisance_residuals(
    predictors: np.ndarray,
    action: np.ndarray,
    outcome: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    ridge_alpha: float,
    epsilon: float,
) -> RM3OOFResiduals:
    predictors = np.asarray(predictors, dtype=float)
    action = np.asarray(action, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if predictors.ndim != 2 or action.ndim != 2 or outcome.ndim != 2:
        raise Phase35ProtocolError("RM3 nuisance arrays must be matrices")
    if action.shape != outcome.shape or action.shape[1] != 2 or len(predictors) != len(action):
        raise Phase35ProtocolError("RM3 nuisance arrays must align as n×2 actions/outcomes")
    if not all(np.isfinite(value).all() for value in (predictors, action, outcome)):
        raise Phase35ProtocolError("RM3 nuisance arrays must be finite")
    validate_expanding_splits(splits, len(action))
    stacked = np.concatenate((action, outcome), axis=1)
    residual, fold_id = crossfit_residual_matrix(
        predictors, stacked, splits, alpha=ridge_alpha, epsilon=epsilon
    )
    evaluated = fold_id >= 0
    return RM3OOFResiduals(
        action=residual[:, :2],
        outcome=residual[:, 2:],
        fold_id=fold_id,
        evaluated=evaluated,
    )


def orthogonal_mimo_moment(
    action_residual: np.ndarray,
    outcome_residual: np.ndarray,
    *,
    ridge_alpha: float,
    epsilon: float,
    maximum_condition_number: float,
    minimum_differential_to_common_energy: float,
) -> RM3MomentAudit:
    action = np.asarray(action_residual, dtype=float)
    outcome = np.asarray(outcome_residual, dtype=float)
    finite = np.isfinite(action).all(axis=1) & np.isfinite(outcome).all(axis=1)
    action, outcome = action[finite], outcome[finite]
    if action.ndim != 2 or outcome.ndim != 2 or action.shape != outcome.shape or action.shape[1] != 2:
        raise Phase35ProtocolError("RM3 orthogonal moment requires aligned n×2 residuals")
    if len(action) < 8:
        raise Phase35ProtocolError("RM3 orthogonal moment has too few evaluated rows")
    gram = np.einsum("ni,nj->ij", action, action, optimize=False) / len(action)
    eigenvalues = np.linalg.eigvalsh(gram)
    condition = float(eigenvalues[-1] / max(eigenvalues[0], epsilon))
    common = 0.5 * (action[:, 0] + action[:, 1])
    differential = 0.5 * (action[:, 0] - action[:, 1])
    common_energy = float(np.mean(common**2))
    differential_energy = float(np.mean(differential**2))
    energy_ratio = differential_energy / max(common_energy, epsilon)
    regularized = gram + float(ridge_alpha) * np.eye(2)
    cross = np.einsum("ni,nj->ij", action, outcome, optimize=False) / len(action)
    matrix = np.linalg.solve(regularized, cross)
    supported = bool(
        condition <= maximum_condition_number
        and energy_ratio >= minimum_differential_to_common_energy
    )
    return RM3MomentAudit(
        matrix=matrix,
        action_gram=gram,
        condition_number=condition,
        common_energy=common_energy,
        differential_energy=differential_energy,
        differential_to_common_energy_ratio=energy_ratio,
        independent_channels_supported=supported,
    )


def orthogonal_r_loss(
    predicted_outcome_residual: torch.Tensor,
    outcome_residual: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if predicted_outcome_residual.shape != outcome_residual.shape:
        raise Phase35ProtocolError("RM3 R-loss prediction/target shapes differ")
    if delta <= 0 or not torch.isfinite(predicted_outcome_residual).all() or not torch.isfinite(outcome_residual).all():
        raise Phase35ProtocolError("RM3 R-loss inputs are invalid")
    return F.huber_loss(predicted_outcome_residual, outcome_residual, delta=float(delta))


def orthogonal_trajectory_moments(
    action_residual: np.ndarray,
    outcome_residual: np.ndarray,
    *,
    ridge_alpha: float,
    epsilon: float,
    maximum_condition_number: float,
    minimum_differential_to_common_energy: float,
    common_only: bool = False,
) -> tuple[RM3MomentAudit, ...]:
    """Estimate every prefix response point; never collapse H60/H180 to one endpoint."""
    action = np.asarray(action_residual, dtype=float)
    outcome = np.asarray(outcome_residual, dtype=float)
    if action.ndim != 3 or outcome.ndim != 3 or action.shape != outcome.shape:
        raise Phase35ProtocolError("RM3 trajectory moments require aligned n×h×2 arrays")
    audits = []
    for step in range(action.shape[1]):
        a = action[:, step]
        y = outcome[:, step]
        if common_only:
            common_action = a.mean(axis=1, keepdims=True)
            common_outcome = y.mean(axis=1, keepdims=True)
            gram = float(np.mean(common_action**2))
            gain = float(np.mean(common_action * common_outcome) / max(gram + ridge_alpha, epsilon))
            matrix = np.full((2, 2), gain / 2.0)
            audits.append(
                RM3MomentAudit(
                    matrix=matrix,
                    action_gram=np.full((2, 2), gram),
                    condition_number=float("inf"),
                    common_energy=gram,
                    differential_energy=0.0,
                    differential_to_common_energy_ratio=0.0,
                    independent_channels_supported=False,
                )
            )
        else:
            audits.append(
                orthogonal_mimo_moment(
                    a, y, ridge_alpha=ridge_alpha, epsilon=epsilon,
                    maximum_condition_number=maximum_condition_number,
                    minimum_differential_to_common_energy=minimum_differential_to_common_energy,
                )
            )
    return tuple(audits)


def generate_rm3_confounded_synthetic(
    *, seed: int, n_rows: int, collinear_actions: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_rows < 300:
        raise Phase35ProtocolError("RM3 synthetic sequence is too short")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 6))
    base_action = np.stack(
        (0.8 * x[:, 0] - 0.4 * x[:, 1], -0.5 * x[:, 0] + 0.7 * x[:, 2]), axis=1
    )
    innovation = rng.normal(scale=0.7, size=(n_rows, 2))
    if collinear_actions:
        innovation[:, 1] = innovation[:, 0]
    action = base_action + innovation
    true_matrix = np.array([[0.62, 0.08], [0.05, 0.48]], dtype=float)
    nuisance = np.stack((1.2 * x[:, 0] + 0.6 * x[:, 3], -0.9 * x[:, 1] + 0.5 * x[:, 4]), axis=1)
    outcome = nuisance + action @ true_matrix + rng.normal(scale=0.08, size=(n_rows, 2))
    return x, action, outcome, true_matrix


def synthetic_expanding_splits(n_rows: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    edges = (n_rows // 2, 2 * n_rows // 3, 5 * n_rows // 6, n_rows)
    return tuple(
        (np.arange(0, edges[index], dtype=np.int64), np.arange(edges[index], edges[index + 1], dtype=np.int64))
        for index in range(3)
    )
