from __future__ import annotations

import numpy as np
import pytest

from src.phase35.multistep.gatec_synthetic import (
    assert_independent_channel_support,
    evaluate_synthetic_controls,
    generate_gatec_known_truth,
    recover_local_gain,
    run_attribution_competition,
    train_synthetic_response_operator,
)
from src.phase35.schema import Phase35ProtocolError


def test_supported_excitation_recovers_local_direction_and_amplitude() -> None:
    batch = generate_gatec_known_truth(seed=17, n_episodes=48, horizon=60)
    audit = assert_independent_channel_support(batch.opening_dose)
    assert audit.differential_energy_ratio > 0.05
    estimate = recover_local_gain(batch.opening_dose, batch.local_effect)
    assert np.all(estimate.gain > 0)
    relative = np.sqrt(np.sum((estimate.gain - batch.true_gain) ** 2)) / np.sqrt(
        np.sum(batch.true_gain**2)
    )
    assert relative < 0.08
    assert abs(estimate.decay - batch.true_decay) < 0.03


def test_collinear_inputs_refuse_independent_channel_claims() -> None:
    batch = generate_gatec_known_truth(
        seed=19, n_episodes=24, horizon=40, collinear_inputs=True
    )
    with pytest.raises(Phase35ProtocolError, match="common spray mode"):
        assert_independent_channel_support(batch.opening_dose)


def test_generator_preserves_boundary_and_cross_side_terminal_truth() -> None:
    batch = generate_gatec_known_truth(seed=23, n_episodes=8, horizon=30)
    assert batch.tin.shape == batch.tout.shape == batch.terminal.shape == (8, 30, 2)
    assert np.allclose(batch.tout, batch.tin - batch.base_local_drop - batch.local_effect)
    assert abs(batch.terminal_mixing[0, 1]) > 0
    assert abs(batch.terminal_mixing[1, 0]) > 0


def test_leakage_and_response_collapse_mutants_fail_closed() -> None:
    clean = evaluate_synthetic_controls()
    assert clean.eligible is True
    assert evaluate_synthetic_controls(leakage_mutant=True).eligible is False
    assert evaluate_synthetic_controls(collapse_mutant=True).eligible is False


@pytest.mark.parametrize(
    "route",
    [
        "a1phys_three_pole",
        "stable_koopman_lpv",
        "pi_neural_ode",
        "deeponet_response",
    ],
)
def test_route_specific_training_recovers_heldout_known_truth(route: str) -> None:
    batch = generate_gatec_known_truth(seed=31, n_episodes=40, horizon=36)
    result = train_synthetic_response_operator(
        route=route,
        batch=batch,
        seed=7,
        steps=140,
        learning_rate=0.03,
    )
    assert result.final_train_loss < 0.4 * result.initial_train_loss
    assert result.heldout_relative_rollout_error < 0.65
    assert result.heldout_direction_accuracy > 0.85
    assert 0.45 < result.heldout_amplitude_ratio < 1.55
    assert result.stable_pole_max < 1.0
    assert result.finite is True


def test_route_training_refuses_collinear_independent_channels() -> None:
    batch = generate_gatec_known_truth(
        seed=37, n_episodes=20, horizon=24, collinear_inputs=True
    )
    with pytest.raises(Phase35ProtocolError, match="common spray mode"):
        train_synthetic_response_operator(
            route="a1phys_three_pole", batch=batch, seed=3, steps=2
        )


def test_free_capacity_by_residual_excitation_scan_is_diagnostic_only() -> None:
    results = [
        run_attribution_competition(
            residual_capacity=capacity,
            excitation=excitation,
            seed=41,
            steps=120,
        )
        for excitation in ("low", "high")
        for capacity in ("small", "large")
    ]
    low = [result for result in results if result.excitation == "low"]
    high = [result for result in results if result.excitation == "high"]
    assert max(result.residual_excitation_fraction for result in low) < 0.05
    assert min(result.residual_excitation_fraction for result in high) > 0.25
    assert all(result.finite for result in results)
    assert all(result.local_supervision is False for result in results)
    assert all(result.free_reads_future_action is False for result in results)
    assert all(result.automatic_scientific_pass is None for result in results)
    assert max(result.heldout_response_amplitude_ratio for result in low) < min(
        result.heldout_response_amplitude_ratio for result in high
    )
    assert all(result.heldout_total_relative_error >= 0 for result in results)
