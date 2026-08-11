from __future__ import annotations

import numpy as np


def test_circular_day_block_bootstrap_is_deterministic_and_preserves_sign():
    from experiments.phase3_5.audit_ms3_real_adaptation import (
        circular_day_block_bootstrap,
    )

    values = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    first = circular_day_block_bootstrap(
        values, block_length=2, samples=2000, seed=17
    )
    second = circular_day_block_bootstrap(
        values, block_length=2, samples=2000, seed=17
    )
    assert first == second
    assert first[0] > 0
    assert first[1] > first[0]


def test_day_diagnostics_use_utc_day_as_the_independent_unit():
    from experiments.phase3_5.audit_ms3_real_adaptation import day_diagnostics

    episodes = {
        "utc_days": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
        "dynamic_mask": [True, True, True, False],
        "logged_mae_c": [1.0, 1.0, 1.0, 1.0],
        "baseline_action_mae_c": [1.2, 1.4, 0.9, 5.0],
        "shuffled_action_mae_c": [1.3, 1.5, 1.1, 5.0],
    }
    result = day_diagnostics(episodes, samples=1000, seed=23)
    baseline = result["logged_vs_baseline"]
    shuffled = result["logged_vs_shuffled"]
    assert baseline["day_count"] == 2
    assert baseline["window_count"] == 3
    assert baseline["positive_day_count"] == 1
    assert np.isclose(baseline["mean_improvement_c"], 0.1)
    assert shuffled["positive_day_count"] == 2
    assert np.allclose(
        shuffled["leave_one_day_out_mean_range_c"], [0.1, 0.4]
    )
