import numpy as np

from src.phase35.evaluation import (
    bootstrap_mean_curve,
    event_response_metrics,
    forecast_metrics,
    onset_lag_seconds,
)


def test_forecast_metrics_use_declared_horizons():
    target = np.zeros((4, 6))
    pred = np.ones((4, 6))
    out = forecast_metrics(target, pred, horizons=(1, 6, 10))
    assert out["integrated_mae"] == 1.0
    assert out["mae_h1"] == 1.0 and out["mae_h6"] == 1.0
    assert "mae_h10" not in out


def test_onset_lag_and_event_direction_are_physical():
    empirical = np.tile([0.0, -0.05, -0.2, -0.4], (8, 1))
    model = np.tile([0.0, -0.04, -0.18, -0.35], (8, 1))
    dose = np.arange(1, 9)
    out = event_response_metrics(empirical, model, dose, step_seconds=10, bootstrap_replicates=100)
    assert onset_lag_seconds(empirical.mean(0), 10) == 30.0
    assert out["empirical_direction_rate"] == 1.0
    assert out["model_direction_rate"] == 1.0
    assert out["irf_wmae"] < 0.1
    assert out["empirical_effect_h1"] == 0.0


def test_cluster_bootstrap_is_deterministic_and_counts_blocks():
    curves = np.arange(30, dtype=float).reshape(10, 3)
    clusters = np.repeat([0, 1], 5)
    a = bootstrap_mean_curve(curves, n_boot=100, seed=7, cluster_ids=clusters)
    b = bootstrap_mean_curve(curves, n_boot=100, seed=7, cluster_ids=clusters)
    np.testing.assert_allclose(a["low"], b["low"])
    assert a["n_clusters"] == 2
