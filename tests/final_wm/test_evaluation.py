"""Evaluation contracts: CRPS closed form, day-paired bootstrap, audits."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.final_wm.contracts import ClosureConfig, WorldModelConfig
from src.final_wm.evaluation import (
    WindowMetrics,
    gaussian_crps,
    relative_improvement_ci,
    residual_quantiles,
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
