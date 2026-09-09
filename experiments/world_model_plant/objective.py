"""Retrospective position/boundary-conditioned learning, explicitly oracle."""
from dataclasses import dataclass, replace
import math

import torch

from .data import CONDITIONING, CONTEXT, PlantWindowBatch


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'Nonfinite {name}')


def validate_batch(model, batch, *, for_training):
    if not isinstance(batch, PlantWindowBatch) or batch.conditioning != CONDITIONING:
        raise ValueError('Explicit recorded oracle conditioning is required')
    if for_training and any(key.split != 'train' for key in batch.keys):
        raise ValueError('Neural fitting may only use prototype train windows')
    if any(key.split not in ('train', 'prototype_validation') for key in batch.keys):
        raise ValueError('No test split is permitted')
    b, h, a = batch.positions_left.shape
    if len(batch.keys) != b or b < 1 or not 1 <= h <= 128 or a != 2:
        raise ValueError('Invalid observational window dimensions')
    shapes = dict(history_values=(b, CONTEXT, 14), history_available_mask=(b, CONTEXT, 14),
                  history_present_mask=(b, CONTEXT, 14), history_age_s=(b, CONTEXT, 14),
                  elapsed_seconds=(b, CONTEXT), boundaries_left=(b, h, 7), boundaries_right=(b, h, 7),
                  observations_right=(b, h, 5), observed_right_mask=(b, h, 5), observations_right_age_s=(b, h, 5))
    for name, shape in shapes.items():
        value = getattr(batch, name)
        dtype = torch.bool if name.endswith('mask') else torch.float64 if name == 'elapsed_seconds' else model.backend.state_loc.dtype
        if value.shape != shape or value.dtype != dtype or value.device != model.backend.state_loc.device:
            raise ValueError(f'Invalid batch field: {name}')
    if batch.positions_left.dtype != model.backend.state_loc.dtype or batch.positions_left.device != model.backend.state_loc.device:
        raise ValueError('Position dtype/device mismatch')
    for name in ('positions_left', 'boundaries_left', 'boundaries_right', 'elapsed_seconds'):
        finite(getattr(batch, name), name)
    finite(batch.history_values[batch.history_available_mask], 'available history')
    finite(batch.observations_right[batch.observed_right_mask], 'present labels')
    if (bool(((batch.positions_left < 0) | (batch.positions_left > 1)).any()) or
            not bool(batch.history_available_mask[:, -1, :13].all()) or
            bool((batch.history_present_mask & ~batch.history_available_mask).any())):
        raise ValueError('Invalid positions or anchor availability')
    if (bool(batch.history_available_mask[:, :, -1].any()) or bool(batch.history_present_mask[:, :, -1].any()) or
            bool((batch.history_values[:, :, -1] != 0).any()) or
            bool((batch.boundaries_left[:, :, -1] != 0).any()) or bool((batch.boundaries_right[:, :, -1] != 0).any())):
        raise ValueError('Archived W cannot enter model inputs')
    if (not torch.equal(batch.positions_left[:, 0], batch.history_values[:, -1, 5:7]) or
            not torch.equal(batch.boundaries_left[:, 0], batch.history_values[:, -1, 7:])):
        raise ValueError('Left cut snapshot does not match final history row')
    count = int(batch.observed_right_mask.sum())
    if not count:
        raise ValueError('No fresh exported observation labels in requested batch')
    return count


def anchor_from_observations(model, batch):
    """Five-temperature inversion at the cut, not a position-equilibrium claim."""
    transition, provider = model.backend.transition, model.backend.transition.properties
    boundary, positions = batch.boundaries_left[:, 0], batch.positions_left[:, 0]
    observed = batch.history_values[:, -1, :5]
    provider.phase = 'five_temperature_anchor_inversion'
    anchor = transition.initial_steady_state(boundary, positions, observed)
    finite(anchor, 'five-temperature physical anchor')
    provider.phase = 'anchor_reconstruction'
    with torch.no_grad():
        reconstructed = transition.output_temperatures(anchor, boundary)
        finite(reconstructed, 'anchor reconstruction')
        rates = torch.stack(transition._spray_rates(boundary[:, 2], positions[:, 0], positions[:, 1], boundary[:, 6]), -1)
        lag = anchor[:, 9:11]
        provider.phase = 'diagnostic_endpoints'
        endpoints = []
        for level in (0., 100.):
            candidate = anchor.detach().clone()
            candidate[:, 7:9] = level * transition.val('tau_evap')
            candidate[:, 9:11] = level
            out = transition.output_temperatures(candidate, boundary)[:, (1, 3)]
            finite(out, 'anchor endpoint diagnostic')
            endpoints.append(out)
        lower, upper = torch.minimum(*endpoints), torch.maximum(*endpoints)
        tolerance = 2 * 100. / (2 ** 24)
        diagnostics = dict(
            initialization='five_observation_and_boundary_inversion_position_argument_is_not_used_by_initializer',
            signed_reconstruction_error_degC=(reconstructed - observed).cpu().tolist(),
            lag_kg_s=lag.cpu().tolist(), position_implied_rate_kg_s=rates.cpu().tolist(),
            lag_minus_position_rate_kg_s=(lag - rates).cpu().tolist(),
            near_zero_endpoint=(lag.abs() <= tolerance).cpu().tolist(),
            near_upper_endpoint=((lag - 100.).abs() <= tolerance).cpu().tolist(),
            outlet_targets_outside_endpoint_range=((observed[:, (1, 3)] < lower) | (observed[:, (1, 3)] > upper)).cpu().tolist(),
            anchor_observation_age_s=batch.history_age_s[:, -1, :5].cpu().tolist(),
            anchor_observation_present_mask=batch.history_present_mask[:, -1, :5].cpu().tolist(),
            endpoint_lag_range_kg_s=[0., 100.], iterations=24,
            anchor_error_based_selection=False, state_identifiability_established=False)
    return anchor, diagnostics


@dataclass(frozen=True)
class PlantLoss:
    total: torch.Tensor
    free_loss: torch.Tensor
    prior_loss: torch.Tensor
    observed_count: int
    free_predictions: torch.Tensor
    prior_predictions: torch.Tensor
    anchor_diagnostics: dict
    domain_diagnostics: dict


def oracle_conditioned_objective(model, batch, *, for_training=True, free_weight=1., prior_weight=1.):
    """Score both branches before fresh-label assimilation; retain full graphs.

    This deliberately does not invoke the generic objective that forbids oracle
    scenarios. Future recorded positions and boundaries condition every step.
    Finite unsupported windows retain their predictions and diagnostic flags;
    callers must record the requested denominator and never replace such draws.
    """
    for weight in (free_weight, prior_weight):
        if isinstance(weight, bool) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Loss weights must be finite nonnegative scalars')
    if free_weight + prior_weight == 0:
        raise ValueError('At least one loss weight must be positive')
    count = validate_batch(model, batch, for_training=for_training)
    provider = model.backend.transition.properties
    provider.reset(len(batch.keys))
    anchor, diagnostics = anchor_from_observations(model, batch)
    initial = model.initialize(anchor, batch.history_values, batch.history_available_mask, batch.elapsed_seconds)
    # History mask means available value, while this flag means actual fresh
    # export cells at the cut. Retained/held values are never new OBS events.
    initial = replace(initial, observed_at_step=batch.history_present_mask[:, -1, :5].any(-1))
    free, online = initial, initial
    free_predictions, priors = [], []
    for step in range(batch.positions_left.shape[1]):
        left, right, positions = batch.boundaries_left[:, step], batch.boundaries_right[:, step], batch.positions_left[:, step]
        provider.phase = 'free_advance_left'
        free = model.advance(free, positions, left)
        provider.phase = 'free_decode_right'
        predicted = model.backend.decode(free.physical, right)
        finite(predicted, 'free prediction')
        free_predictions.append(predicted)
        provider.phase = 'online_advance_left'
        online = model.advance(online, positions, left)
        provider.phase = 'online_decode_right_before_assimilation'
        prior = model.backend.decode(online.physical, right)
        finite(prior, 'online prior prediction')
        priors.append(prior)
        provider.phase = 'fresh_observe_right'
        online = model.observe(online, batch.observations_right[:, step], batch.observed_right_mask[:, step],
                                right, step_index=online.steps)
    free_predictions, priors = torch.stack(free_predictions, 1), torch.stack(priors, 1)

    def mse(prediction):
        labels = torch.where(batch.observed_right_mask, batch.observations_right, prediction)
        loss = ((prediction - labels) / model.backend.observation_scale).square().sum() / count
        finite(loss, 'normalized loss')
        return loss

    free_loss, prior_loss = mse(free_predictions), mse(priors)
    total = free_weight * free_loss + prior_weight * prior_loss
    finite(total, 'total loss')
    return PlantLoss(total, free_loss, prior_loss, count, free_predictions, priors, diagnostics, provider.summary())
