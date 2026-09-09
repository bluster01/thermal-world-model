"""Differentiable prediction objectives for the deterministic point-belief model.

This is a learning interface, not a training result or an RSSM objective. Batches
contain measured observations and a caller-supplied past-only physical anchor;
they have no hidden truth-state target. Declared scenario provenance still needs
to be checked by the data pipeline: a string cannot establish past-only origin.
"""

from dataclasses import dataclass
import math
from numbers import Real

import torch

from .contracts import BoundaryScenario
from .model import PersistentPhysicsWorldModel


@dataclass(frozen=True)
class LearningBatch:
    """One observed episode segment, with labels in engineering units.

    History feature order is observations, actions, boundaries, as required by
    ``model.initialize``. ``anchor`` must be estimated using information available
    at the history endpoint. Future actions and boundary scenarios describe the
    H transitions after that endpoint; label k measures the state after action k.
    Timestamps describe history only and must be finite and strictly increasing.
    Missing history/label entries may have arbitrary fills, including NaN/Inf.
    """

    anchor: torch.Tensor                  # B,S; no simulator hidden-state target
    history_values: torch.Tensor          # B,T,F; engineering units
    history_mask: torch.Tensor            # B,T,F; bool
    elapsed_seconds: torch.Tensor         # B,T
    future_actions: torch.Tensor          # B,H,A
    scenario: BoundaryScenario            # B,H,D; no oracle training scenarios
    future_observations: torch.Tensor     # B,H,O; engineering units
    future_observed_mask: torch.Tensor    # B,H,O; bool


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    rollout_loss: torch.Tensor
    prior_loss: torch.Tensor
    observed_count: int
    rollout_predictions: torch.Tensor     # B,H,O; engineering units
    prior_predictions: torch.Tensor       # B,H,O; before current-label assimilation


def _weight(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return float(value)


def _numeric_tensor(value, name, reference):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} requires a floating-point tensor')
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f'{name} must match the model device and dtype')


def _validate_future(model, batch):
    if not isinstance(batch, LearningBatch):
        raise ValueError('batch must be a LearningBatch')
    backend = model.backend
    actions, labels, mask = batch.future_actions, batch.future_observations, batch.future_observed_mask
    _numeric_tensor(actions, 'future_actions', backend.state_loc)
    if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[1] < 1 or actions.shape[2] != backend.action_dim:
        raise ValueError('future_actions must have nonempty matching (B,H,A) shape')
    if not bool(torch.isfinite(actions).all()) or bool(((actions < 0) | (actions > 1)).any()):
        raise ValueError('future_actions must be finite fractions in [0,1]')
    _numeric_tensor(labels, 'future_observations', backend.state_loc)
    if labels.shape != (*actions.shape[:2], backend.observation_dim):
        raise ValueError('future_observations must have matching (B,H,O) shape')
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != labels.shape:
        raise ValueError('future_observed_mask must be boolean with matching (B,H,O) shape')
    if mask.device != labels.device:
        raise ValueError('future_observed_mask must match the observation device')
    if not bool(torch.isfinite(labels[mask]).all()):
        raise ValueError('Observed future measurements must be finite')
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError('At least one observed future measurement is required')
    if not isinstance(batch.scenario, BoundaryScenario):
        raise ValueError('scenario must be a BoundaryScenario')
    if batch.scenario.origin == 'oracle':
        raise ValueError('Oracle boundary scenarios are not permitted for this learning objective')
    _numeric_tensor(batch.scenario.values, 'scenario values', backend.state_loc)
    if batch.scenario.values.shape != (*actions.shape[:2], backend.boundary_dim):
        raise ValueError('Scenario must match the action batch, horizon and boundary dimensions')
    # Recheck values because a frozen dataclass does not freeze tensor contents.
    if not bool(torch.isfinite(batch.scenario.values).all()):
        raise ValueError('Scenario values must be finite')
    return count


def _masked_mse(predictions, labels, mask, scale, count):
    # Remove missing NaN/Inf before any subtraction; a mask multiplied by a NaN
    # error afterwards would still corrupt both the objective and its gradients.
    clean_labels = torch.where(mask, labels, predictions)
    error = (predictions - clean_labels) / scale
    loss = error.square().sum() / count
    if not bool(torch.isfinite(loss)):
        raise ValueError('Normalized prediction loss is not finite')
    return loss


def prediction_objective(
    model: PersistentPhysicsWorldModel,
    batch: LearningBatch,
    *,
    rollout_weight: float = 1.,
    prior_weight: float = 1.,
) -> LossTerms:
    """Score free rollouts and online one-step priors, with full unroll gradients.

    Initialization occurs exactly once. Both branches start from that same
    initial belief; the model's functional transitions do not mutate it. The
    free branch never receives future labels. At online step k, decoding occurs
    after advance and BEFORE assimilating label k. That label can influence
    predictions only at later steps. Neither branch is detached or reset.

    Each branch uses mean squared observation error divided by the backend's
    observation scale, with one common denominator: the number of observed
    scalar labels across B,H,O. A wholly missing step is allowed; a wholly
    missing batch is rejected. Both branch diagnostics are computed even when
    one weight is zero. This function does not change the model's train/eval mode.
    """
    rollout_weight = _weight(rollout_weight, 'rollout_weight')
    prior_weight = _weight(prior_weight, 'prior_weight')
    if rollout_weight == 0 and prior_weight == 0:
        raise ValueError('At least one objective weight must be positive')
    count = _validate_future(model, batch)
    initial = model.initialize(batch.anchor, batch.history_values,
                               batch.history_mask, batch.elapsed_seconds)
    rollout_predictions = model.imagine(initial, batch.future_actions, batch.scenario).observations

    current = initial
    priors = []
    for step in range(batch.future_actions.shape[1]):
        boundary = batch.scenario.values[:, step]
        current = model.advance(current, batch.future_actions[:, step], boundary)
        predicted = model.backend.decode(current.physical, boundary)
        if predicted.shape != batch.future_observations[:, step].shape or not bool(torch.isfinite(predicted).all()):
            raise ValueError('Decoded prior observations must have finite matching (B,O) shape')
        priors.append(predicted)
        current = model.observe(current, batch.future_observations[:, step],
                                batch.future_observed_mask[:, step], boundary,
                                step_index=current.steps)
    prior_predictions = torch.stack(priors, 1)

    args = (batch.future_observations, batch.future_observed_mask,
            model.backend.observation_scale, count)
    rollout_loss = _masked_mse(rollout_predictions, *args)
    prior_loss = _masked_mse(prior_predictions, *args)
    total = rollout_weight * rollout_loss + prior_weight * prior_loss
    if not bool(torch.isfinite(total)):
        raise ValueError('Weighted prediction loss is not finite')
    return LossTerms(total, rollout_loss, prior_loss, count, rollout_predictions, prior_predictions)
