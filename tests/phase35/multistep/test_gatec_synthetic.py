from __future__ import annotations

import numpy as np
import pytest

from src.phase35.multistep.gatec_synthetic import (
    assert_independent_channel_support,
    evaluate_synthetic_controls,
    generate_gatec_known_truth,
    recover_local_gain,
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
