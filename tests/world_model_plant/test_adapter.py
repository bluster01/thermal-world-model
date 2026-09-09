"""Hand-authored oracle windows and property-asset software checks; no fitting."""
from dataclasses import replace
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.world_model_plant.data import PlantDataset, WindowKey, keys_from_records
from experiments.world_model_plant.model import (MonitoredGrid, PlantModelConfig, build_model, checkpoint_payload,
                                                load_checkpoint, model_identity, state_tensor_sha256)
from experiments.world_model_plant.objective import oracle_conditioned_objective
from src.final_wm.properties import GridThermoProperties

ASSET = Path(__file__).resolve().parents[2] / 'artifacts/final_wm/iapws_surrogate.npz'


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_data():
    n = 512
    ts = 10000 + np.arange(n, dtype=np.int64) * 10
    arrays = {'timestamps_epoch_s': ts, 'prototype_split': (np.arange(n) >= 256).astype(np.int8)}
    values = dict(observations=[500, 490, 530, 520, 565], positions=[.3, .4],
                  boundary=[350, 250, 17, 420, 320, 16, 99999])
    for group, base in values.items():
        value = np.tile(np.asarray(base, dtype=np.float32), (n, 2, 1))
        value += np.arange(n, dtype=np.float32)[:, None, None] * (1e-5 if group == 'positions' else .001)
        arrays[group] = value
        arrays[group + '_available_mask'] = np.ones(value.shape, bool)
        arrays[group + '_present_mask'] = np.ones(value.shape, bool)
        arrays[group + '_age_s'] = np.zeros(value.shape, np.float32)
        arrays[group + '_source_timestamp_epoch_s'] = np.broadcast_to(ts[:, None, None], value.shape).copy()
    # A stale-but-available cut observation is not a fresh exported update.
    for row in (63, 66):
        arrays['observations'][row, 0] = arrays['observations'][row - 1, 0]
        arrays['observations_present_mask'][row, 0] = False
        arrays['observations_age_s'][row, 0] = 10
        arrays['observations_source_timestamp_epoch_s'][row, 0] = ts[row - 1]
    for side in ('A', 'B'):
        arrays[f'{side}_train_property_input_window_starts'] = np.arange(65, dtype=np.int64)
        arrays[f'{side}_prototype_validation_property_input_window_starts'] = np.arange(256, 321, dtype=np.int64)
    manifest = dict(dt_seconds=10, window_context=64, window_future=128,
                    split_ranges={'train': [0, 256], 'prototype_validation': [256, 512]})
    return PlantDataset(arrays, manifest)


def small_model(variant='quota_attention', dtype='float64'):
    return build_model(PlantModelConfig(variant, 17, hidden_dim=8, heads=2, layers=1,
                                        cache_capacity=8, evidence_capacity=4, prediction_capacity=4, dtype=dtype), ASSET)


def test_left_interval_right_target_alignment_masks_ages_and_W_exclusion():
    dataset = fixture_data()
    batch = dataset.batch([WindowKey('A', 0, 'train'), WindowKey('B', 1, 'train')], horizon=4)
    assert batch.history_values.shape == (2, 64, 14)
    assert batch.elapsed_seconds[0].tolist() == list(range(-630, 1, 10))
    for index, (side, start) in enumerate([(0, 0), (1, 1)]):
        cut = start + 63
        torch.testing.assert_close(batch.positions_left[index], torch.tensor(dataset.arrays['positions'][cut:cut + 4, side]))
        torch.testing.assert_close(batch.boundaries_left[index, :, :6], torch.tensor(dataset.arrays['boundary'][cut:cut + 4, side, :6]))
        torch.testing.assert_close(batch.boundaries_right[index, :, :6], torch.tensor(dataset.arrays['boundary'][cut + 1:cut + 5, side, :6]))
        torch.testing.assert_close(batch.observations_right[index], torch.tensor(dataset.arrays['observations'][cut + 1:cut + 5, side]))
    assert bool(batch.history_available_mask[0, -1, :5].all())
    assert not bool(batch.history_present_mask[0, -1, :5].any())
    assert batch.history_age_s[0, -1, :5].tolist() == [10.] * 5
    assert not bool(batch.observed_right_mask[0, 2].any())
    assert bool((batch.history_values[:, :, -1] == 0).all())
    assert not bool(batch.history_available_mask[:, :, -1].any())
    assert bool((batch.boundaries_left[:, :, -1] == 0).all())
    assert bool((batch.boundaries_right[:, :, -1] == 0).all())


@pytest.mark.parametrize('fault', ['undeclared', 'cross_split', 'stale', 'cross_split_hold', 'bad_position', 'property_domain', 'bad_present_age', 'bad_time'])
def test_invalid_windows_fail_without_clipping_or_replacing(fault):
    data = fixture_data()
    key = WindowKey('A', 0, 'train')
    if fault == 'undeclared':
        key = WindowKey('A', 70, 'train')
    elif fault == 'cross_split':
        key = WindowKey('A', 256, 'train')
    elif fault == 'stale':
        data.arrays['positions_age_s'][10, 0, 0] = 40
    elif fault == 'cross_split_hold':
        key = WindowKey('A', 256, 'prototype_validation')
        data.arrays['positions_source_timestamp_epoch_s'][256, 0, 0] = data.arrays['timestamps_epoch_s'][255]
    elif fault == 'bad_position':
        data.arrays['positions'][100, 0, 0] = 1.2
    elif fault == 'property_domain':
        data.arrays['boundary'][150, 0, 4] = 280
    elif fault == 'bad_present_age':
        data.arrays['observations_present_mask'][63, 0, 0] = True
    elif fault == 'bad_time':
        data.arrays['timestamps_epoch_s'][4] += 1
    with pytest.raises(ValueError):
        if fault in ('bad_present_age', 'bad_time'):
            PlantDataset(data.arrays, data.manifest)
        else:
            data.batch([key], horizon=2)


def test_sampler_is_reproducible_balanced_train_only_and_anchors_spaced_per_side():
    data = fixture_data()
    plan = data.plan(updates=3, batch_size=4)
    assert plan == data.plan(updates=3, batch_size=4)
    assert plan != data.plan(updates=3, batch_size=4, sample_seed=8)
    for batch in plan['training']:
        keys = keys_from_records(batch)
        assert [key.side for key in keys] == ['A', 'A', 'B', 'B']
        assert all(key.split == 'train' for key in keys)
        data.batch(keys, horizon=1)
    for split, anchors in plan['diagnostic_anchors'].items():
        assert len(anchors) == 32 and all(r['split'] == split for r in anchors)
        assert len({(r['side'], r['start']) for r in anchors}) == 32
    with pytest.raises(ValueError):
        data.starts('A', 'test')


def test_four_variants_have_same_initial_tensors_fixed_priors_and_identity_scope():
    variants = ['shared_age', 'shared_attention', 'quota_age', 'quota_attention']
    models = [small_model(name) for name in variants]
    assert len({state_tensor_sha256(net.state_dict()) for net in models}) == 1
    for net in models:
        assert not any(p.requires_grad for p in net.backend.transition.parameters())
        assert net.backend.state_dim == 11
        identity = model_identity(net)
        assert identity['conditioning'] == 'recorded_positions_and_boundaries_oracle'
        assert 'historical' in identity['prior_knowledge_scope']
        assert identity['normalization']['observation_loc'] == [500., 520., 530., 535., 565.]
        assert net.initial_correction.weight.count_nonzero() == 0


def test_builder_rejects_unbound_property_asset(tmp_path):
    fake = tmp_path / 'bad.npz'
    fake.write_bytes(b'not the pinned grid')
    with pytest.raises(ValueError, match='identity'):
        build_model(PlantModelConfig('shared_age', 1), fake)


@pytest.mark.parametrize('mutation', ['policy', 'asset', 'prior', 'dataset', 'plan', 'physical', 'nan', 'shape'])
def test_checkpoint_checks_full_identity_and_frozen_physical_tensors(mutation):
    net = small_model()
    payload = checkpoint_payload(net, dataset_identity={'fixture': 1}, plan_sha256='fixture_plan', step=0, selection_value=1.)
    if mutation == 'policy':
        payload['identity']['retention_policy']['evidence_capacity'] = 3
    elif mutation == 'asset':
        payload['identity']['property_asset_sha256'] = 'other'
    elif mutation == 'prior':
        payload['identity']['priors']['tau_mix1'] = 10
    elif mutation == 'dataset':
        payload['dataset_identity'] = {'fixture': 2}
    elif mutation == 'plan':
        payload['plan_sha256'] = 'other'
    elif mutation == 'physical':
        payload['state_dict']['backend.transition.raw.tauB'].add_(.1)
        payload['state_tensor_sha256'] = state_tensor_sha256(payload['state_dict'])
    elif mutation == 'nan':
        payload['state_dict']['initial_correction.bias'][0] = float('nan')
        payload['state_tensor_sha256'] = state_tensor_sha256(payload['state_dict'])
    elif mutation == 'shape':
        payload['state_dict']['initial_correction.bias'] = torch.zeros(2, dtype=torch.float64)
        payload['state_tensor_sha256'] = state_tensor_sha256(payload['state_dict'])
    with pytest.raises(ValueError):
        load_checkpoint(net, payload, dataset_identity={'fixture': 1}, plan_sha256='fixture_plan')


def test_checkpoint_roundtrip_preserves_neural_values_without_mutating_physics():
    source, destination = small_model(), small_model()
    with torch.no_grad():
        source.initial_correction.bias.fill_(.001)  # fixed software value, not a fit
    payload = checkpoint_payload(source, dataset_identity={'fixture': 1}, plan_sha256='fixture_plan', step=0, selection_value=1.)
    load_checkpoint(destination, payload, dataset_identity={'fixture': 1}, plan_sha256='fixture_plan')
    assert state_tensor_sha256(source.state_dict()) == state_tensor_sha256(destination.state_dict())


def test_monitor_preserves_forward_and_gradients_and_separates_extensions():
    with np.load(ASSET, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    plain, monitor = GridThermoProperties(arrays).to(dtype=torch.float64), MonitoredGrid(arrays).to(dtype=torch.float64)
    monitor.reset(2)
    monitor.phase = 'actual_window'
    p = torch.tensor([17., 25.], dtype=torch.float64)
    h = torch.tensor([3000., 4000.], dtype=torch.float64, requires_grad=True)
    a, b = monitor.temperature_of_ph(p, h), plain.temperature_of_ph(p, h)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga, = torch.autograd.grad(a.sum(), h, retain_graph=True)
    gb, = torch.autograd.grad(b.sum(), h)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    assert monitor.summary()['unsupported_rows'] == [False, True]
    monitor.reset(2)
    values = torch.tensor([300., 450.], dtype=torch.float64)
    a, b = monitor.separator_enthalpy(p, values), plain.separator_enthalpy(p, values)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert monitor.summary()['unsupported_rows'] == [False, False]
    extension = monitor.summary()['separator_model_extensions'][0]
    assert extension['subcritical_temperature_lift_count'] == 1
    assert extension['supercritical_saturation_auxiliary_cap_count'] == 1


@pytest.mark.parametrize('variant', ['shared_age', 'shared_attention', 'quota_age', 'quota_attention'])
def test_oracle_loss_uses_left_advance_right_decode_and_fresh_only_updates_with_backward(variant):
    net = small_model(variant)
    data = fixture_data()
    batch = data.batch([WindowKey('A', 0, 'train'), WindowKey('B', 1, 'train')], horizon=3, dtype=torch.float64)
    before = {n: p.detach().clone() for n, p in net.backend.transition.named_parameters()}
    with patch.object(net, 'advance', wraps=net.advance) as advance, patch.object(net, 'observe', wraps=net.observe) as observe:
        result = oracle_conditioned_objective(net, batch)
    for step in range(3):
        for branch in (0, 1):
            args = advance.call_args_list[2 * step + branch].args
            torch.testing.assert_close(args[1], batch.positions_left[:, step], rtol=0, atol=0)
            torch.testing.assert_close(args[2], batch.boundaries_left[:, step], rtol=0, atol=0)
        args = observe.call_args_list[step].args
        torch.testing.assert_close(args[2], batch.observed_right_mask[:, step], rtol=0, atol=0)
        torch.testing.assert_close(args[3], batch.boundaries_right[:, step], rtol=0, atol=0)
    assert not bool(advance.call_args_list[0].args[0].observed_at_step[0])
    assert result.observed_count == int(batch.observed_right_mask.sum())
    clean = torch.where(batch.observed_right_mask, batch.observations_right, result.free_predictions)
    expected = ((result.free_predictions - clean) / net.backend.observation_scale).square().sum() / result.observed_count
    torch.testing.assert_close(result.free_loss, expected, rtol=0, atol=0)
    result.total.backward()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    for name, parameter in net.backend.transition.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert 'position_argument_is_not_used' in result.anchor_diagnostics['initialization']
    assert len(result.anchor_diagnostics['lag_minus_position_rate_kg_s']) == 2


def test_future_labels_never_affect_free_branch_and_prior_scores_before_assimilation():
    net = small_model()
    with torch.no_grad():
        net.observation_correction.weight.fill_(.001)
        net.power_head[-1].weight.fill_(.001)
    data = fixture_data()
    batch = data.batch([WindowKey('B', 0, 'train')], horizon=3, dtype=torch.float64)
    original = oracle_conditioned_objective(net, batch)
    labels = batch.observations_right.clone()
    labels[:, 0] += 5.
    changed = oracle_conditioned_objective(net, replace(batch, observations_right=labels))
    torch.testing.assert_close(original.free_predictions, changed.free_predictions, rtol=0, atol=0)
    torch.testing.assert_close(original.prior_predictions[:, 0], changed.prior_predictions[:, 0], rtol=0, atol=0)
    assert not torch.equal(original.prior_predictions[:, 1:], changed.prior_predictions[:, 1:])


def test_missing_labels_can_be_nan_without_loss_change_and_validation_cannot_train():
    net = small_model()
    data = fixture_data()
    batch = data.batch([WindowKey('A', 0, 'train')], horizon=3, dtype=torch.float64)
    base = oracle_conditioned_objective(net, batch)
    labels = torch.where(batch.observed_right_mask, batch.observations_right, torch.full_like(batch.observations_right, float('nan')))
    actual = oracle_conditioned_objective(net, replace(batch, observations_right=labels))
    torch.testing.assert_close(base.total, actual.total, rtol=0, atol=0)
    validation = data.batch([WindowKey('A', 256, 'prototype_validation')], horizon=1, dtype=torch.float64)
    with pytest.raises(ValueError, match='only'):
        oracle_conditioned_objective(net, validation)
    with torch.no_grad():
        result = oracle_conditioned_objective(net, validation, for_training=False)
    assert torch.isfinite(result.total)
