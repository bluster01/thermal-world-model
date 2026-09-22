#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bilateral pack + builder tests (spec 2026-09-22 §7 step 1 verification list).

Covers the interface checks named in the spec: A side bit-comparable to the old
pack, epoch alignment, valve cross-side mapping, no future true temperature in
the future inputs, normalization fitted on train rows only, common legal windows.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from .bilateral_data import (ACT_IDX, AUX_IDX, BND_IDX, CHANNELS, CONTEXT, NAMES, OUT_DEFAULT,
                             REF_MAP, TEMP_IDX, window_epochs)

HERE = Path(__file__).resolve().parent
PACK = HERE / 'data' / 'bilateral_AB_33pct.npz'
REF = HERE / 'data' / 'screen_A_33pct_h128.npz'
BYPASS = HERE / 'data' / 'hist_bypass_A_33pct_v1.npz'

pytestmark = pytest.mark.skipif(not PACK.exists(), reason='bilateral pack not built yet')


@pytest.fixture(scope='module')
def pack():
    return {k: np.load(PACK)[k] for k in np.load(PACK).files}


@pytest.fixture(scope='module')
def meta():
    return json.loads(PACK.with_suffix('.json').read_text(encoding='utf-8'))


def test_channel_layout_is_fixed_and_unique():
    assert len(CHANNELS) == 30 and len(set(NAMES)) == 30
    assert NAMES[:10] == ['A_T1', 'A_T2', 'A_T3', 'A_T4', 'A_T5',
                          'B_T1', 'B_T2', 'B_T3', 'B_T4', 'B_T5']
    assert [NAMES[i] for i in ACT_IDX] == ['u1A', 'u1B', 'u2A', 'u2B']
    assert [NAMES[i] for i in BND_IDX] == ['steam_flow', 'coal', 'sep_p', 'sep_T', 'fw_T',
                                           'out_p_left', 'out_p_right']
    assert len(AUX_IDX) == 9


def test_reference_channels_reproduce_old_pack(pack, meta):
    """Require exact identity, not merely high correlation or a loose tolerance."""
    ref = {k: np.load(REF)[k] for k in np.load(REF).files}
    for split in ('train','selector','evaluation'):
        for rc, oc in sorted(REF_MAP.items()):
            np.testing.assert_array_equal(pack[f'hist30_{split}'][:,:,oc],ref[split][:,:CONTEXT,rc])
        np.testing.assert_array_equal(pack[f'future_temp_{split}'][:,:,:5],ref[split][:,CONTEXT:,:5])
        np.testing.assert_array_equal(pack[f'future_act_{split}'][:,:,[0,3]],ref[split][:,63:,5:7])
        np.testing.assert_array_equal(pack[f'future_bnd_{split}'][:,:,:6],ref[split][:,63:,7:13])


def test_bypass_channels_bit_identical(pack):
    byp = {k: np.load(BYPASS)[k] for k in np.load(BYPASS).files}
    names = [c[0] for c in CHANNELS]
    for nm in ('corr_fuel_total', 'sa_flow', 'o2_a_3sel', 'load', 'sp1_a', 'sp2_b_set',
               'agc', 'load_rate', 'sh_spray_total'):
        j = names.index(nm)
        assert np.array_equal(pack['hist30_train'][:, :, j], byp['hist12_train'][:, :, byp['channel_names'].tolist().index(nm)])


def test_valve_cross_side_mapping_and_units(pack):
    """u2A is the B-chain stage-2 valve; u2B the A-chain one (crossed wiring)."""
    names = [c[0] for c in CHANNELS]
    u1a, u1b, u2a, u2b = (pack['hist30_train'][:, :, names.index(k)] for k in ('u1A', 'u1B', 'u2A', 'u2B'))
    # levels inside the physical band (fractions, small negative sensor readings allowed)
    assert u1a.min() >= -1e-6 and u1b.min() >= -1e-6
    assert -0.02 < u2a.min() < 0 and u2a.min() > -0.02
    assert max(u1a.max(), u1b.max(), u2a.max(), u2b.max()) <= 1.0 + 1e-6
    # cross-side mapping: stage-1 A/B are the strongly correlated pair (data-notes §4)
    v = pack['hist30_train'][:, :, ACT_IDX].reshape(-1, 4).astype(np.float64)
    corr = np.corrcoef(v.T)
    assert corr[0, 1] > 0.7 and corr[2, 3] > 0.2
    assert abs(corr[0, 2]) < 0.4 and abs(corr[1, 3]) < 0.4


def test_alignment_and_no_future_temperature_in_inputs(pack, meta):
    dt = meta['dt_seconds']
    for k, H in (('train', 128), ('selector', 128), ('evaluation', 512)):
        t_end = pack[f'{k}_time']
        ep = window_epochs(t_end, H)
        assert ep.shape[1] == CONTEXT + H
        assert np.all(np.diff(ep, axis=1) == dt)
        # future control/boundary rows are exactly the last H+1 rows of the epoch grid
        assert pack[f'future_act_{k}'].shape == (len(t_end), H + 1, 4)
        assert pack[f'future_bnd_{k}'].shape == (len(t_end), H + 1, 7)
        assert pack[f'future_temp_{k}'].shape == (len(t_end), H, 10)
        # history ends at t_end exactly
        assert np.array_equal(ep[:, CONTEXT - 1], t_end)
    # future temperature arrays are targets only: they are not part of the 30 history channels
    assert pack['hist30_train'].shape[2] == 30
    assert not any('future' in n for n in meta['channel_names'])


def test_normalization_uses_train_rows_only_and_old_channels_reuse_reference(pack):
    ref = {k: np.load(REF)[k] for k in np.load(REF).files}
    for rc, oc in REF_MAP.items():
        assert np.isclose(pack['mean'][oc], ref['mean'][rc], rtol=1e-6, atol=1e-4)
        assert np.isclose(pack['scale'][oc], ref['scale'][rc], rtol=1e-6, atol=1e-4)
    # new channels fitted on train history rows: mean must equal direct computation
    he = window_epochs(pack['train_time'], 128)[:, :CONTEXT].ravel()
    hv = pack['hist30_train'].reshape(-1, 30)
    _, first = np.unique(he, return_index=True)
    direct = np.nanmean(hv[first], axis=0)
    for j in (5, 6, 7, 8, 9, 11, 12, 20):
        assert np.isclose(pack['mean'][j], direct[j], rtol=1e-4)


def test_common_legal_windows_and_negative_feedback_flags(pack, meta):
    """Raw-feedback protocol: keep all old windows; flag the strict/negative subgroups."""
    for k in ('train', 'selector', 'evaluation'):
        assert pack[f'valid_{k}'].all(), 'B side must be finite on every old window'
    names = NAMES
    u2a_hist = pack['hist30_train'][:, :, names.index('u2A')]                 # rows 0..63
    u2a_future = pack['future_act_train'][:, 1:, names.index('u2A') - 10]     # rows 64..63+H
    real_neg = (np.concatenate((u2a_hist, u2a_future), axis=1) < 0).any(axis=1)
    assert np.array_equal(real_neg, pack['u2a_negative_train'])
    # whole-window subgroup counts reproduce the documented mmap-derived numbers exactly
    assert (int(pack['strict01_train'].sum()), int(pack['strict01_selector'].sum()),
            int(pack['strict01_evaluation'].sum())) == (13473, 70, 174)
    assert (int(pack['u2a_negative_train'].sum()), int(pack['u2a_negative_selector'].sum()),
            int(pack['u2a_negative_evaluation'].sum())) == (6898, 58, 82)
    assert meta['feedback_protocol']['tolerance'].startswith('raw feedback retained')
    # finite and inside a physically plausible band
    for j, nm in zip(TEMP_IDX, names[:10]):
        v = pack['hist30_train'][:, :, j]
        assert np.isfinite(v).all() and 300 <= v.min() and v.max() <= 700, nm
