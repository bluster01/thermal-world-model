"""Tests for the CF credential probes (checklist 2026-08-21).

CF-1 counterfactual_fidelity_synthetic, CF-3 position_binned_gain,
CF-4 constraint_checks, D1 calibration_coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.final_wm.contracts import FinalWMProtocolError, TransitionConfig
from src.final_wm.data import SPLIT_VAL, CanonicalRecord
from src.final_wm.evaluation import (
    calibration_coverage,
    constraint_checks,
    counterfactual_fidelity_synthetic,
)
from src.final_wm.properties import AnalyticThermoProperties
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.training import build_world_model
from src.final_wm.transition import Fan2020UDETransition
from experiments.final_wm import matrix_spec as ms


@pytest.fixture(scope="module")
def syn_record(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cf")
    arrays = synthetic_canonical_arrays(1_500, seed=7)
    path = tmp / "syn.npz"
    np.savez_compressed(path, **arrays)
    return CanonicalRecord(path)


@pytest.fixture(scope="module")
def student_model():
    spec = ms._base("dsyn", "student", 0, boundary_mode="oracle",
                    initial_state_mode="learned", closure_mode="conservative")
    torch.manual_seed(0)
    return build_world_model(spec, AnalyticThermoProperties())


def test_cf1_replay_identity_is_exact(syn_record, student_model):
    """Student transition == teacher transition + replay abduction on both
    sides => identical delta trajectories (zero counterfactual error)."""
    teacher = Fan2020UDETransition(TransitionConfig(), AnalyticThermoProperties())
    student_model.transition.load_state_dict(teacher.state_dict())
    out = counterfactual_fidelity_synthetic(
        student_model, teacher, syn_record, SPLIT_VAL,
        n_windows=8, history_steps=96, horizon=18, seed=0,
    )
    assert out["abduction"] == "replay"
    assert out["delta_mae"] < 1e-6
    assert out["terminal_sign_agreement"] >= 0.0  # defined even when teacher delta ~ 0


def test_cf1_schema_and_determinism(syn_record, student_model):
    teacher = Fan2020UDETransition(TransitionConfig(), AnalyticThermoProperties())
    a = counterfactual_fidelity_synthetic(
        student_model, teacher, syn_record, SPLIT_VAL,
        n_windows=8, history_steps=96, horizon=18, seed=3, abduction="observer",
    )
    b = counterfactual_fidelity_synthetic(
        student_model, teacher, syn_record, SPLIT_VAL,
        n_windows=8, history_steps=96, horizon=18, seed=3, abduction="observer",
    )
    assert a["delta_mae"] == b["delta_mae"]
    assert len(a["delta_mae_per_channel"]) == 5
    assert len(a["delta_magnitude_ratio_per_channel"]) == 5
    assert 0.0 <= a["terminal_sign_agreement"] <= 1.0
    assert np.isfinite(a["delta_mae"])
    with pytest.raises(FinalWMProtocolError):
        counterfactual_fidelity_synthetic(
            student_model, teacher, syn_record, SPLIT_VAL,
            n_windows=4, history_steps=96, horizon=18, abduction="bogus",
        )


def test_cf4_constraints_on_prior_model(syn_record, student_model):
    """An untrained model at priors must satisfy both hard constraints:
    more spray -> more cooling (monotone), and zero-spray drifts warm."""
    out = constraint_checks(
        student_model, syn_record, SPLIT_VAL,
        n_windows=8, history_steps=96, rollout_steps=30, seed=0,
    )
    mono = out["monotonicity"]
    assert len(mono["mean_terminal_delta_c"]) == 4
    assert mono["monotone_cooling"] is True
    drift = out["zero_spray_drift"]
    assert drift["frac_positive_sh1_out"] == 1.0
    assert drift["frac_positive_sh2_out"] == 1.0


def test_d1_calibration_schema_and_sanity(syn_record, student_model):
    out = calibration_coverage(
        student_model, syn_record, SPLIT_VAL,
        n_windows=16, batch_size=8, history_steps=96, horizon=18, seed=0,
    )
    assert out["levels"] == [0.5, 0.8, 0.95]
    assert out["horizons"] == [1, 6, 18]
    for key, cov in out["overall"].items():
        assert 0.0 <= cov <= 1.0, key
    assert len(out["per_channel"]["H18_L0.95"]) == 5
    assert 0.0 <= out["mean_abs_coverage_gap"] <= 1.0


def test_d1_perfect_sigma_recovers_nominal():
    """If mu == y and sigma is exact for known Gaussian noise, empirical
    coverage must track the nominal level (pure math sanity of the probe)."""
    from statistics import NormalDist

    torch.manual_seed(0)
    n = 20_000
    sigma_true = 2.0
    y = sigma_true * torch.randn(n)
    mu = torch.zeros(n)
    for level in (0.5, 0.8, 0.95):
        z = NormalDist().inv_cdf((1.0 + level) / 2.0)
        cov = float(((y - mu).abs() <= z * sigma_true).float().mean())
        assert abs(cov - level) < 0.02


def test_cf3_position_binned_gain_handbuilt(tmp_path):
    """Hand-built record with one isolated valve-up event lands in the bin
    matching its pre-event opening, with gain = dT/dv."""
    from src.final_wm.analysis import position_binned_gain

    n = 400
    boundary = np.zeros((n, 7), dtype=np.float32) + 500.0
    actions = np.full((n, 2), 0.20, dtype=np.float32)
    actions[:, 1] = 0.30  # v2 baseline opening (v1 stays 0.20, constant)
    obs = np.zeros((n, 5), dtype=np.float32) + 520.0
    # Piecewise-constant v2 with two isolated +0.05 up-steps (>=130 apart so
    # the 60-step exclusion windows never overlap): 0.30 -> 0.35 at t=130
    # (pre-opening 0.30) and 0.35 -> 0.40 at t=270 (pre-opening 0.35).
    # Each step drops the final outlet (index 4) by 0.5 degC over 60 steps,
    # i.e. gain = -0.5/0.05 = -10 degC per full opening.
    actions[130:, 1] = 0.35
    actions[270:, 1] = 0.40
    obs[189:, 4] -= 0.5   # 130 - 1 + 60
    obs[329:, 4] -= 0.5   # 270 - 1 + 60 (cumulative; per-event delta is -0.5)
    arrays = dict(
        boundary=boundary, actions=actions, obs=obs,
        timestamps=np.arange(n) * 10,
        split=np.ones(n, dtype=np.int64),  # all validation: the probe reads SPLIT_VAL
    )
    path = tmp_path / "rec.npz"
    np.savez_compressed(path, **arrays)
    record = CanonicalRecord(path)
    out = position_binned_gain(record, SPLIT_VAL, 1, n_bins=2, horizon=60)
    assert out["n_events"] == 2
    low, high = out["bins"]
    assert low["data"]["n"] == 1 and high["data"]["n"] == 1
    assert abs(low["data"]["mean_gain"] - (-10.0)) < 1e-4
    assert abs(high["data"]["mean_gain"] - (-10.0)) < 1e-4


def test_cf3_too_few_events(tmp_path):
    from src.final_wm.analysis import position_binned_gain

    n = 200
    arrays = dict(
        boundary=np.zeros((n, 7), dtype=np.float32) + 500.0,
        actions=np.full((n, 2), 0.3, dtype=np.float32),
        obs=np.zeros((n, 5), dtype=np.float32) + 520.0,
        timestamps=np.arange(n) * 10,
        split=np.zeros(n, dtype=np.int64),
    )
    path = tmp_path / "rec.npz"
    np.savez_compressed(path, **arrays)
    out = position_binned_gain(CanonicalRecord(path), SPLIT_VAL, 1, n_bins=4)
    assert out["bins"] == []
    assert "too few" in out["note"]


# ---------------------------------------------------------------------------
# Multi-shuffle leakage null (seed1 marginal-case audit instrument)
# ---------------------------------------------------------------------------

def test_leakage_multishuffle_schema_and_compat(syn_record, student_model):
    from src.final_wm.diagnostics import leakage_probe

    args = dict(n_windows=32, history_steps=96, epochs=2, seed=0)
    single = leakage_probe(student_model, syn_record, **args)
    assert "shuffle_null" not in single
    assert single["n_shuffles"] == 1
    multi = leakage_probe(student_model, syn_record, n_shuffles=4, **args)
    null = multi["shuffle_null"]
    assert len(null["improvements"]) == 4
    # Backward compat: shuffle 0 of the multi run IS the frozen single shuffle
    # (same permutation seed), and blind/aware are identical across runs.
    assert multi["shuffled_relative_improvement"] == single["shuffled_relative_improvement"]
    assert multi["aware_relative_improvement"] == single["aware_relative_improvement"]
    assert multi["blind"]["val_mse_norm"] == single["blind"]["val_mse_norm"]
    assert multi["leakage_delta"] == single["leakage_delta"]
    assert null["improvements"][0] == single["shuffled_relative_improvement"]
    assert null["std"] >= 0.0
    assert 0.0 <= null["aware_percentile"] <= 1.0
    again = leakage_probe(student_model, syn_record, n_shuffles=4, **args)
    assert again["shuffle_null"]["improvements"] == null["improvements"]
