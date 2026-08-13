from __future__ import annotations

import numpy as np
import pytest

from src.phase35.multistep.rm3av_diagnostics import (
    ASSUMPTION_LEDGER_METHODS,
    REQUIRED_DIAGNOSTIC_MODES,
    build_assumption_ledger,
    build_manual_verdict_template,
    build_prediction_diagnostics,
    build_state_closure_audit,
    convergence_diagnostics,
    daily_gain_context_diagnostics,
    dependence_diagnostics,
    mechanism_residual_dependence,
    response_trajectory_diagnostics,
    stratified_error_diagnostics,
    valve_policy_probe_diagnostics,
    valve_trajectory_diagnostics,
    valve_innovation_rank,
)
from src.phase35.schema import Phase35ProtocolError


def test_prediction_diagnostics_report_persistence_skill_per_side_and_task() -> None:
    target = np.ones((3, 60, 2), dtype=np.float32)
    predictions = {key: target.copy() for key in ("valve", "tin", "local", "terminal")}
    baselines = {key: np.zeros((3, 2), dtype=np.float32) for key in predictions}
    payload = build_prediction_diagnostics(predictions, {key: target for key in predictions}, baselines)
    for key in predictions:
        assert payload[key]["mae_pooled"] == 0.0
        assert payload[key]["persistence_mae_pooled"] == 1.0
        assert payload[key]["skill_vs_persistence_pooled"] == 1.0
        assert set(payload[key]["mae_by_side"]) == {"A", "B"}


def test_valve_rank_reports_common_and_differential_energy_without_overclaim() -> None:
    rng = np.random.default_rng(22)
    common = rng.normal(size=(200, 1))
    innovations = np.concatenate((common, common + 1e-5 * rng.normal(size=(200, 1))), axis=1)
    payload = valve_innovation_rank(innovations)
    assert payload["singular_values"][0] > payload["singular_values"][1]
    assert payload["differential_energy_fraction"] < 1e-6
    assert payload["independent_channels_supported"] is False


def test_mode_contract_requires_every_mode_or_explicit_not_applicable_reason() -> None:
    predictions = {key: np.zeros((2, 60, 2), dtype=np.float32) for key in ("valve", "tin", "local", "terminal")}
    targets = {key: value.copy() for key, value in predictions.items()}
    baselines = {key: np.zeros((2, 2), dtype=np.float32) for key in predictions}
    modes = {mode: {"status": "computed"} for mode in REQUIRED_DIAGNOSTIC_MODES}
    payload = build_prediction_diagnostics(predictions, targets, baselines, mode_records=modes)
    assert set(payload["mode_records"]) == set(REQUIRED_DIAGNOSTIC_MODES)
    modes["lead"] = {"status": "not_applicable"}
    with pytest.raises(Phase35ProtocolError, match="reason"):
        build_prediction_diagnostics(predictions, targets, baselines, mode_records=modes)


def test_state_closure_never_calls_true_future_context_a_simulator() -> None:
    blocked = build_state_closure_audit(
        generated={"valve", "tin", "tout", "terminal", "sp"},
        declared_external=set(),
        required={"valve", "tin", "tout", "terminal", "sp", "load", "pressure"},
    )
    assert blocked["status"] == "STATE_CLOSURE_BLOCKED"
    assert blocked["missing"] == ["load", "pressure"]
    declared = build_state_closure_audit(
        generated={"valve", "tin", "tout", "terminal", "sp"},
        declared_external={"load", "pressure"},
        required={"valve", "tin", "tout", "terminal", "sp", "load", "pressure"},
    )
    assert declared["status"] == "DECLARED_CONTEXT_ROLLOUT_ONLY"
    assert declared["state_closed_simulator"] is False


def test_assumption_ledger_and_33_verdicts_cannot_auto_upgrade_identification() -> None:
    ledger = build_assumption_ledger()
    assert set(ledger) == set(ASSUMPTION_LEDGER_METHODS)
    assert all(row["identification_claim_allowed"] is False for row in ledger.values())
    assert all(row["status"] in {"unmet", "not_testable"} for row in ledger.values())
    verdicts = build_manual_verdict_template()
    assert list(verdicts) == [f"Q{index:02d}" for index in range(1, 34)]
    assert all(value is None for value in verdicts.values())


def test_valve_trajectory_metrics_expose_persistence_like_smoothing() -> None:
    target = np.zeros((4, 60, 2), dtype=np.float32)
    target[:, 1::2] = 2.0
    prediction = np.ones_like(target)
    baseline = np.ones((4, 2), dtype=np.float32)
    payload = valve_trajectory_diagnostics(prediction, target, baseline)
    assert payload["prediction_mean_abs_delta"] == 0.0
    assert payload["target_mean_abs_delta"] > 0.0
    assert payload["prediction_mean_range"] == 0.0
    assert payload["target_multiscale_roughness"] > payload["prediction_multiscale_roughness"]
    assert payload["persistence_skill"] <= 0.0
    assert payload["prediction_small_step_fraction_lt_0_05_point"] == 1.0
    assert payload["deadband_is_descriptive_not_controller_identification"] is True


def test_response_diagnostics_report_horizons_modes_and_identity() -> None:
    effect = np.zeros((3, 60, 2), dtype=np.float32)
    effect[:, :, 0] = np.arange(60, dtype=np.float32)
    payload = response_trajectory_diagnostics(
        effect,
        constant_action_effect=np.zeros_like(effect),
        stable_poles=np.array([[0.8, 0.9, 0.99], [0.7, 0.85, 0.95]]),
    )
    assert set(payload["mean_absolute_effect_by_horizon_steps"]) == {"6", "18", "60"}
    assert payload["constant_action_identity_max_abs"] == 0.0
    assert payload["common_mode_energy"] > 0.0
    assert payload["differential_mode_energy"] > 0.0
    assert payload["all_poles_stable"] is True
    assert payload["time_constant_boundary_diagnostic"]["architecture_order_claim"] is False
    assert set(payload["signed_absolute_timing_grid"]) == {
        "0", "1", "3", "6", "12", "18", "30", "60"
    }
    assert payload["signed_absolute_timing_grid"]["0"]["signed_effect_by_side"] == [0.0, 0.0]


def test_mechanism_residual_dependence_never_claims_structural_noise_recovery() -> None:
    rng = np.random.default_rng(221)
    residuals = {
        key: rng.normal(size=(20, 60, 2)).astype(np.float32)
        for key in ("valve", "tin", "local", "terminal")
    }
    payload = mechanism_residual_dependence(residuals)
    assert len(payload["feature_names"]) == 8
    assert payload["independent_mechanism_noise_claim"] is False


def test_stratification_and_convergence_are_descriptive_not_automatic_passes() -> None:
    target = np.zeros((8, 60, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[4:] += 2.0
    strata = {
        "load": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "date": np.arange(8),
    }
    payload = stratified_error_diagnostics(prediction, target, strata)
    assert payload["load"]["0"]["mae_pooled"] == 0.0
    assert payload["load"]["1"]["mae_pooled"] == 2.0
    assert payload["automatic_invariance_pass"] is None
    convergence = convergence_diagnostics(
        list(np.linspace(2.0, 1.0, 1000)), best_update=900, update_cap=1000
    )
    assert convergence["last_500_update_slope"] < 0.0
    assert convergence["best_update_distance_to_cap"] == 100
    assert convergence["converged"] is None


def test_blocked_valve_probes_compare_sp_pi_and_full_history_without_causal_claim() -> None:
    rng = np.random.default_rng(28)
    n = 80
    history = rng.normal(size=(n, 6, 15)).astype(np.float32)
    future_sp = rng.normal(size=(n, 60, 2)).astype(np.float32)
    # Make future valve learnable from future SP plus current terminal proxy.
    target = 0.7 * future_sp + 0.2 * history[:, -1, 7:9, None].transpose(0, 2, 1)
    groups = np.repeat(np.arange(10), 8)
    payload = valve_policy_probe_diagnostics(
        history,
        future_sp,
        target.astype(np.float32),
        groups,
        sp_indices=(8, 13),
        temperature_indices=(7, 12),
        ridge=1e-3,
    )
    assert set(payload["probes"]) == {
        "sp_only", "sp_plus_current_temperature", "pi_features", "full_history"
    }
    assert payload["probes"]["sp_plus_current_temperature"]["mae"] < payload["probes"]["sp_only"]["mae"]
    assert payload["group_overlap_count"] == 0
    assert payload["causal_direction_claim"] is False


def test_horizon_skill_gain_context_and_cross_side_dependence_are_reported() -> None:
    target = np.ones((10, 60, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 18:] += 1.0
    baselines = np.zeros((10, 2), dtype=np.float32)
    tasks = {key: prediction for key in ("valve", "tin", "local", "terminal")}
    targets = {key: target for key in tasks}
    baseline_map = {key: baselines for key in tasks}
    horizons = build_prediction_diagnostics(tasks, targets, baseline_map)
    assert set(horizons["terminal"]["horizon_curve_steps"]) == {"6", "18", "60"}
    assert horizons["terminal"]["horizon_curve_steps"]["6"]["mae_pooled"] == 0.0

    rng = np.random.default_rng(31)
    innovations = rng.normal(size=(50, 2))
    dependence = dependence_diagnostics(innovations)
    assert set(dependence) >= {"pearson_correlation", "cross_covariance", "independence_claim"}
    assert dependence["independence_claim"] is False

    n = 30
    payload = daily_gain_context_diagnostics(
        gain=rng.normal(size=(n, 2)),
        context=rng.normal(size=(n, 3)),
        activity=rng.normal(size=(n, 2)),
        groups=np.repeat(np.arange(10), 3),
        context_names=("load", "pressure", "coal"),
        ridge=1e-3,
    )
    assert payload["group_overlap_count"] == 0
    assert set(payload["coefficients_by_side"]) == {"A", "B"}
    assert payload["causal_gain_explanation"] is False
