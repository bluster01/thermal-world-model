"""Training-only orthogonalization and loss contracts for RM3-AV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ..schema import Phase35ProtocolError
from .rm3av_contracts import RM3AV_CANDIDATE_IDS


@dataclass(frozen=True)
class OOFRLossResiduals:
    action_residual: np.ndarray
    outcome_residual: np.ndarray
    fold_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OOFLinearFold:
    held_buckets: tuple[int, ...]
    x_center: np.ndarray
    action_center: np.ndarray
    outcome_center: np.ndarray
    action_coefficients: np.ndarray
    outcome_coefficients: np.ndarray


@dataclass(frozen=True)
class OOFRModelSet:
    folds: tuple[OOFLinearFold, ...]
    bucket_count: int
    uses_future_sp: bool = False

    def residualize(
        self,
        history: np.ndarray,
        action: np.ndarray,
        outcome: np.ndarray,
        groups: np.ndarray,
        future_sp: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action.shape != outcome.shape or len(history) != len(action) or len(groups) != len(action):
            raise Phase35ProtocolError("OOF residual application shape mismatch")
        if self.uses_future_sp and future_sp is None:
            raise Phase35ProtocolError("OOF residual application requires future SP")
        x = _design(history, future_sp if self.uses_future_sp else None)
        action_flat = action.reshape(len(action), -1).astype(np.float64)
        outcome_flat = outcome.reshape(len(outcome), -1).astype(np.float64)
        action_prediction = np.empty_like(action_flat)
        outcome_prediction = np.empty_like(outcome_flat)
        assigned = np.zeros(len(action), dtype=bool)
        buckets = np.mod(np.abs(np.asarray(groups, dtype=np.int64)), self.bucket_count)
        for fold in self.folds:
            held = np.isin(buckets, fold.held_buckets)
            if not np.any(held):
                continue
            action_prediction[held] = (
                (x[held] - fold.x_center) @ fold.action_coefficients + fold.action_center
            )
            outcome_prediction[held] = (
                (x[held] - fold.x_center) @ fold.outcome_coefficients + fold.outcome_center
            )
            assigned |= held
        if not assigned.all():
            raise Phase35ProtocolError("OOF residual model left samples unassigned")
        return (
            (action_flat - action_prediction).reshape(action.shape).astype(np.float32),
            (outcome_flat - outcome_prediction).reshape(outcome.shape).astype(np.float32),
        )


@dataclass(frozen=True)
class OOFActionProjection:
    projector: np.ndarray
    fold_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OOFActionOutcomeAudit:
    action_innovation: np.ndarray
    outcome_innovation: np.ndarray
    action_prediction: np.ndarray
    outcome_prediction: np.ndarray
    fold_records: tuple[dict[str, Any], ...]
    action_r2_by_side: tuple[float | None, float | None]
    outcome_r2_by_side: tuple[float | None, float | None]


def _design(history: np.ndarray, future_sp: np.ndarray | None = None) -> np.ndarray:
    if history.ndim != 3 or not np.isfinite(history).all():
        raise Phase35ProtocolError("OOF history must be finite [episode,window,feature]")
    blocks = [history[:, -1], history.mean(axis=1)]
    if future_sp is not None:
        if (
            future_sp.ndim != 3
            or future_sp.shape[0] != history.shape[0]
            or future_sp.shape[2] != 2
            or not np.isfinite(future_sp).all()
        ):
            raise Phase35ProtocolError("OOF future SP must be finite [episode,horizon,2]")
        blocks.append(future_sp.reshape(len(future_sp), -1))
    return np.concatenate(blocks, axis=1).astype(np.float64)


def _action_audit_design(history: np.ndarray, future_sp: np.ndarray) -> np.ndarray:
    if (
        history.ndim != 3
        or future_sp.ndim != 3
        or future_sp.shape[0] != history.shape[0]
        or future_sp.shape[2] != 2
        or not np.isfinite(future_sp).all()
    ):
        raise Phase35ProtocolError("OOF action-audit history/SP contract changed")
    horizon = future_sp.shape[1]
    history_state = np.broadcast_to(
        history[:, -1, None], (len(history), horizon, history.shape[-1])
    )
    history_mean = np.broadcast_to(
        history.mean(axis=1)[:, None], (len(history), horizon, history.shape[-1])
    )
    previous_sp = np.concatenate((future_sp[:, :1], future_sp[:, :-1]), axis=1)
    sp_delta = future_sp - previous_sp
    sp_integral_change = np.cumsum(
        future_sp - future_sp[:, :1], axis=1
    ) / np.arange(1, horizon + 1, dtype=np.float64)[None, :, None]
    return np.concatenate(
        (history_state, history_mean, future_sp, sp_delta, sp_integral_change), axis=2
    ).astype(np.float64)


def _side_r2(target: np.ndarray, prediction: np.ndarray) -> tuple[float | None, float | None]:
    result: list[float | None] = []
    for side in range(2):
        true = target[..., side]
        pred = prediction[..., side]
        denominator = float(np.sum((true - true.mean()) ** 2))
        result.append(
            float(1.0 - np.sum((true - pred) ** 2) / denominator)
            if denominator > 0.0 else None
        )
    return result[0], result[1]


def fit_oof_action_outcome_audit(
    history: np.ndarray,
    future_sp: np.ndarray,
    action: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    *,
    ridge: float,
) -> OOFActionOutcomeAudit:
    """Blocked OOF nuisance residuals for diagnostics, never a causal estimator."""

    if (
        action.shape != outcome.shape
        or action.ndim != 3
        or action.shape[-1] != 2
        or len(action) != len(history)
        or ridge <= 0.0
        or not np.isfinite(action).all()
        or not np.isfinite(outcome).all()
    ):
        raise Phase35ProtocolError("OOF action/outcome audit contract changed")
    design = _action_audit_design(history, future_sp)
    action_prediction = np.empty_like(action, dtype=np.float64)
    outcome_prediction = np.empty_like(outcome, dtype=np.float64)
    records = []
    for fold_index, (train, held, train_groups, held_groups) in enumerate(_group_folds(groups)):
        train_x = design[train].reshape(-1, design.shape[-1])
        held_x = design[held].reshape(-1, design.shape[-1])
        action_prediction[held] = _ridge_predict(
            train_x, action[train].reshape(-1, 2), held_x, ridge
        ).reshape(len(held), action.shape[1], 2)
        outcome_prediction[held] = _ridge_predict(
            train_x, outcome[train].reshape(-1, 2), held_x, ridge
        ).reshape(len(held), outcome.shape[1], 2)
        records.append({
            "fold_index": fold_index,
            "train_groups": train_groups,
            "held_out_groups": held_groups,
            "group_overlap_count": len(set(train_groups) & set(held_groups)),
            "held_episode_count": len(held),
        })
    return OOFActionOutcomeAudit(
        action_innovation=(action - action_prediction).astype(np.float32),
        outcome_innovation=(outcome - outcome_prediction).astype(np.float32),
        action_prediction=action_prediction.astype(np.float32),
        outcome_prediction=outcome_prediction.astype(np.float32),
        fold_records=tuple(records),
        action_r2_by_side=_side_r2(action, action_prediction),
        outcome_r2_by_side=_side_r2(outcome, outcome_prediction),
    )


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, held_x: np.ndarray, ridge: float) -> np.ndarray:
    center_x = train_x.mean(axis=0, keepdims=True)
    center_y = train_y.mean(axis=0, keepdims=True)
    x = train_x - center_x
    coefficients = np.linalg.solve(
        x.T @ x + ridge * np.eye(x.shape[1]), x.T @ (train_y - center_y)
    )
    return (held_x - center_x) @ coefficients + center_y


def _group_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, list[int], list[int]]]:
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(np.unique(groups)) < 2:
        raise Phase35ProtocolError("OOF residualization requires at least two independent groups")
    unique = np.unique(groups)
    bucket_count = min(5, len(unique))
    buckets = np.mod(np.abs(groups.astype(np.int64)), bucket_count)
    unique_buckets = np.unique(buckets)
    if len(unique_buckets) < 2:
        # Fallback for synthetic/sequential labels whose modulo collapses.
        rank = {value: index % bucket_count for index, value in enumerate(unique.tolist())}
        buckets = np.asarray([rank[value] for value in groups], dtype=np.int64)
        unique_buckets = np.unique(buckets)
    folds = []
    for held_buckets in np.array_split(unique_buckets, len(unique_buckets)):
        held_mask = np.isin(buckets, held_buckets)
        held = np.flatnonzero(held_mask)
        train = np.flatnonzero(~held_mask)
        held_groups = np.unique(groups[held]).tolist()
        folds.append((train, held, np.unique(groups[train]).tolist(), held_groups))
    return folds


def fit_oof_r_model(
    history: np.ndarray,
    action: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    *,
    ridge: float,
    future_sp: np.ndarray | None = None,
) -> tuple[OOFRModelSet, OOFRLossResiduals]:
    if action.shape != outcome.shape or action.ndim != 3 or len(history) != len(action):
        raise Phase35ProtocolError("OOF R-loss action/outcome contract changed")
    if ridge <= 0 or not np.isfinite(action).all() or not np.isfinite(outcome).all():
        raise Phase35ProtocolError("OOF R-loss inputs are invalid")
    x = _design(history, future_sp)
    action_flat = action.reshape(len(action), -1).astype(np.float64)
    outcome_flat = outcome.reshape(len(outcome), -1).astype(np.float64)
    action_prediction = np.empty_like(action_flat)
    outcome_prediction = np.empty_like(outcome_flat)
    records: list[dict[str, Any]] = []
    models: list[OOFLinearFold] = []
    group_folds = _group_folds(np.asarray(groups))
    bucket_count = len(group_folds)
    buckets = np.mod(np.abs(np.asarray(groups, dtype=np.int64)), bucket_count)
    if len(np.unique(buckets)) < 2:
        unique = np.unique(groups)
        mapping = {value: index % bucket_count for index, value in enumerate(unique.tolist())}
        buckets = np.asarray([mapping[value] for value in groups], dtype=np.int64)
    for fold_index, (train, held, train_groups, held_groups) in enumerate(group_folds):
        x_center = x[train].mean(axis=0, keepdims=True)
        action_center = action_flat[train].mean(axis=0, keepdims=True)
        outcome_center = outcome_flat[train].mean(axis=0, keepdims=True)
        centered = x[train] - x_center
        inverse = np.linalg.solve(
            centered.T @ centered + ridge * np.eye(centered.shape[1]), centered.T
        )
        action_coefficients = inverse @ (action_flat[train] - action_center)
        outcome_coefficients = inverse @ (outcome_flat[train] - outcome_center)
        action_prediction[held] = (x[held] - x_center) @ action_coefficients + action_center
        outcome_prediction[held] = (x[held] - x_center) @ outcome_coefficients + outcome_center
        held_buckets = tuple(sorted(np.unique(buckets[held]).astype(int).tolist()))
        models.append(
            OOFLinearFold(
                held_buckets=held_buckets,
                x_center=x_center,
                action_center=action_center,
                outcome_center=outcome_center,
                action_coefficients=action_coefficients,
                outcome_coefficients=outcome_coefficients,
            )
        )
        records.append(
            {
                "fold_index": fold_index,
                "indices": held.copy(),
                "train_groups": train_groups,
                "held_out_groups": held_groups,
                "held_buckets": list(held_buckets),
            }
        )
    residuals = OOFRLossResiduals(
        action_residual=(action_flat - action_prediction).reshape(action.shape).astype(np.float32),
        outcome_residual=(outcome_flat - outcome_prediction).reshape(outcome.shape).astype(np.float32),
        fold_records=tuple(records),
    )
    return OOFRModelSet(
        tuple(models), bucket_count, uses_future_sp=future_sp is not None
    ), residuals


def fit_oof_r_residuals(
    history: np.ndarray,
    action: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    *,
    ridge: float,
    future_sp: np.ndarray | None = None,
) -> OOFRLossResiduals:
    return fit_oof_r_model(
        history, action, outcome, groups, ridge=ridge, future_sp=future_sp
    )[1]


def fit_oof_action_projection(
    history: np.ndarray,
    action: np.ndarray,
    groups: np.ndarray,
    *,
    ridge: float,
) -> OOFActionProjection:
    if action.ndim != 3 or len(history) != len(action) or ridge <= 0:
        raise Phase35ProtocolError("OOF action projection inputs are invalid")
    x_full = _design(history)
    action_flat = action.reshape(len(action), -1).astype(np.float64)
    n_features = history.shape[-1]
    projectors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for train, held, train_groups, held_groups in _group_folds(groups):
        x = x_full[train] - x_full[train].mean(axis=0, keepdims=True)
        y = action_flat[train] - action_flat[train].mean(axis=0, keepdims=True)
        coefficients = np.linalg.solve(
            x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y
        )[:n_features]
        left, singular, _ = np.linalg.svd(coefficients, full_matrices=False)
        if singular.size and singular[0] > 0:
            energy = np.cumsum(singular**2) / np.sum(singular**2)
            rank = min(max(int(np.searchsorted(energy, 0.9)) + 1, 1), 4, n_features - 1)
            projector = np.eye(n_features) - left[:, :rank] @ left[:, :rank].T
        else:
            rank = 0
            projector = np.eye(n_features)
        projectors.append(projector)
        records.append(
            {
                "indices": held.copy(),
                "rank_removed": rank,
                "train_groups": train_groups,
                "held_out_groups": held_groups,
            }
        )
    averaged = np.mean(projectors, axis=0)
    return OOFActionProjection(
        projector=(0.5 * (averaged + averaged.T)).astype(np.float32),
        fold_records=tuple(records),
    )


def valve_dynamics_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise Phase35ProtocolError("RM3-AV valve dynamics shape mismatch")
    delta = F.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    roughness_parts = []
    for stride in (1, 3, 6):
        if prediction.shape[1] <= 2 * stride:
            continue
        pred_second = prediction[:, 2 * stride :] - 2 * prediction[:, stride:-stride] + prediction[:, : -2 * stride]
        true_second = target[:, 2 * stride :] - 2 * target[:, stride:-stride] + target[:, : -2 * stride]
        roughness_parts.append(F.smooth_l1_loss(pred_second, true_second))
    roughness = torch.stack(roughness_parts).mean()
    return {"delta": delta, "roughness": roughness}


def rm3av_multitask_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, torch.Tensor],
    *,
    candidate_id: str,
    target_scales: Mapping[str, float],
    action_residual: torch.Tensor | None = None,
    outcome_residual: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if candidate_id not in RM3AV_CANDIDATE_IDS:
        raise Phase35ProtocolError("RM3-AV loss candidate is invalid")
    prediction_keys = {
        "valve": "valve_prediction",
        "tin": "tin_prediction",
        "local": "local_drop_prediction",
        "terminal": "terminal_prediction",
    }
    if set(targets) != set(prediction_keys) or set(target_scales) != set(prediction_keys):
        raise Phase35ProtocolError("RM3-AV multitask target contract changed")
    components: dict[str, torch.Tensor] = {}
    for key, prediction_key in prediction_keys.items():
        if prediction_key not in output or target_scales[key] <= 0:
            raise Phase35ProtocolError("RM3-AV multitask output/scale is invalid")
        components[key] = F.smooth_l1_loss(
            output[prediction_key] / float(target_scales[key]),
            targets[key] / float(target_scales[key]),
        )
    total = 0.25 * sum(components.values())
    if candidate_id in {"C10", "C13"}:
        if "logged_local_drop_prediction" not in output:
            raise Phase35ProtocolError("logged-action auxiliary output is missing")
        components["logged_action_auxiliary"] = F.smooth_l1_loss(
            output["logged_local_drop_prediction"] / float(target_scales["local"]),
            targets["local"] / float(target_scales["local"]),
        )
        total = total + 0.10 * components["logged_action_auxiliary"]
    if candidate_id in {"C11", "C12"}:
        if action_residual is None or outcome_residual is None:
            raise Phase35ProtocolError("integrated R-loss requires OOF residuals")
        if "oof_response_prediction" not in output:
            raise Phase35ProtocolError("integrated R-loss response path is missing")
        if action_residual.shape != outcome_residual.shape or outcome_residual.shape != output["oof_response_prediction"].shape:
            raise Phase35ProtocolError("OOF residuals do not match response shape")
        components["oof_r_loss"] = F.smooth_l1_loss(
            output["oof_response_prediction"] / float(target_scales["local"]),
            outcome_residual / float(target_scales["local"]),
        )
        total = total + 0.10 * components["oof_r_loss"]
    if candidate_id == "C14":
        valve = valve_dynamics_loss(output["valve_prediction"], targets["valve"])
        components["delta_valve"] = valve["delta"]
        components["multiscale_roughness"] = valve["roughness"]
        total = total + 0.05 * valve["delta"] + 0.05 * valve["roughness"]
    components["total"] = total
    return components
