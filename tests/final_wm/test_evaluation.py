"""Evaluation contracts: CRPS closed form, day-paired bootstrap, audits."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.final_wm.contracts import ClosureConfig, ObserverConfig, WorldModelConfig
from src.final_wm.data import CanonicalRecord
from src.final_wm.evaluation import (
    WindowMetrics,
    constant_condition_stability,
    day_block_mean_ci,
    gaussian_crps,
    relative_improvement_ci,
    residual_quantiles,
    rollout_stability,
    state_continuity_metrics,
    step_response_direction,
)
from src.final_wm.model import FinalWorldModel
from src.final_wm.properties import AnalyticThermoProperties
from src.final_wm.synthetic import synthetic_canonical_arrays


def _record(tmp_path, n: int = 1500) -> str:
    arrays = synthetic_canonical_arrays(total_steps=n, seed=1)
    path = tmp_path / "record.npz"
    np.savez_compressed(path, **arrays)
    return str(path)


def test_gaussian_crps_closed_form() -> None:
    mu = torch.zeros(3)
    sigma = torch.ones(3)
    y = torch.zeros(3)
    expected = 2.0 * (1.0 / math.sqrt(2.0 * math.pi)) - 1.0 / math.sqrt(math.pi)
    crps = gaussian_crps(mu, sigma, y)
    assert torch.allclose(crps, torch.full((3,), expected), atol=1e-6)
    # worse predictions must have larger CRPS
    assert bool((gaussian_crps(mu, sigma, torch.full((3,), 2.0)) > crps).all())


def test_relative_improvement_ci_day_paired() -> None:
    days = torch.arange(10).repeat_interleave(4)
    gen = torch.Generator().manual_seed(0)
    base_vals = 2.0 + 0.05 * torch.randn(40, 18, generator=gen)
    arm_vals = 1.6 + 0.05 * torch.randn(40, 18, generator=gen)
    base = WindowMetrics(nll=base_vals, mae=base_vals, crps=base_vals, day_ids=days)
    arm = WindowMetrics(nll=arm_vals, mae=arm_vals, crps=arm_vals, day_ids=days)
    ci = relative_improvement_ci(base, arm, horizon=18, n_boot=200)
    assert ci.point == pytest.approx(0.2, abs=0.03)
    assert ci.ci_lo > 0.0
    assert ci.n_days == 10


def test_day_block_mean_ci_reports_identifiability() -> None:
    values = torch.tensor([-2.0, -1.0, -4.0, -3.0])
    days = torch.tensor([10, 10, 11, 11])
    ci = day_block_mean_ci(values, days, n_boot=200, seed=4)
    assert ci["point"] == pytest.approx(-2.5)
    assert ci["ci_hi"] < 0.0
    assert ci["n_days"] == 2
    assert ci["identifiable"] is True

    one_day = day_block_mean_ci(values[:2], days[:2], n_boot=20, seed=4)
    assert one_day["identifiable"] is False
    assert one_day["ci_lo"] is None and one_day["ci_hi"] is None


def test_protocol_continuity_and_stability_metrics_run(tmp_path) -> None:
    record = CanonicalRecord(_record(tmp_path))
    model = FinalWorldModel(
        WorldModelConfig(
            observer=ObserverConfig(history_steps=16),
            closure=ClosureConfig(injection_mode="conservative"),
        ),
        AnalyticThermoProperties(),
    )
    continuity = state_continuity_metrics(
        model, record, 1, n_windows=4, history_steps=16, gap_steps=6, seed=2,
    )
    assert continuity.values.shape == (4,)
    assert continuity.day_ids.shape == (4,)
    assert bool(torch.isfinite(continuity.values).all())

    constant = constant_condition_stability(
        model, record, 1, n_windows=4, history_steps=16, rollout_steps=12, seed=3,
    )
    assert constant["rollout_steps"] == 12
    assert constant["all_finite"] is True
    assert constant["max_abs_drift_c"] >= 0.0

    rollout = rollout_stability(
        model, record, 1, n_windows=4, history_steps=16, horizon=12,
        boundary_mode="oracle", seed=4,
    )
    assert rollout["horizon"] == 12
    assert rollout["all_finite"] is True


def test_step_response_direction_and_residual_quantiles(tmp_path) -> None:
    from src.final_wm.data import CanonicalRecord

    record = CanonicalRecord(_record(tmp_path))
    model = FinalWorldModel(
        WorldModelConfig(closure=ClosureConfig(injection_mode="conservative")),
        AnalyticThermoProperties(),
    )
    direction = step_response_direction(
        model, record, 1, n_windows=8, history_steps=16, rollout_steps=30, seed=0
    )
    assert direction["frac_negative"] == 1.0
    quant = residual_quantiles(model, record, 1, n_windows=8, history_steps=16)
    assert quant["max_kw"] == pytest.approx(0.0, abs=1e-6)  # zero-init closure
