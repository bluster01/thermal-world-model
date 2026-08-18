from __future__ import annotations

import pytest
import torch

from src.final_wm.contracts import ControllerConfig, FinalWMProtocolError
from src.final_wm.controller import CascadePIController


def _controller(**kwargs) -> CascadePIController:
    return CascadePIController(ControllerConfig(**kwargs))


def test_positive_error_opens_valve() -> None:
    ctrl = _controller(rate_limit_per_step=1.0, actuator_tau_seconds=1e-6)
    ctrl.reset(torch.tensor([0.4]))
    for _ in range(5):
        valve = ctrl.step(torch.tensor([565.0]), torch.tensor([570.0]))
    assert valve.item() > 0.4


def test_negative_error_closes_valve() -> None:
    ctrl = _controller(rate_limit_per_step=1.0, actuator_tau_seconds=1e-6)
    ctrl.reset(torch.tensor([0.4]))
    for _ in range(5):
        valve = ctrl.step(torch.tensor([565.0]), torch.tensor([560.0]))
    assert valve.item() < 0.4


def test_saturation_bounds() -> None:
    ctrl = _controller(rate_limit_per_step=1.0, actuator_tau_seconds=1e-6)
    ctrl.reset(torch.tensor([0.9]))
    for _ in range(50):
        valve = ctrl.step(torch.tensor([565.0]), torch.tensor([700.0]))
    assert valve.item() <= 1.0
    ctrl2 = _controller(rate_limit_per_step=1.0, actuator_tau_seconds=1e-6)
    ctrl2.reset(torch.tensor([0.1]))
    for _ in range(50):
        valve2 = ctrl2.step(torch.tensor([565.0]), torch.tensor([400.0]))
    assert valve2.item() >= 0.0


def test_rate_limit() -> None:
    ctrl = _controller(kp=1.0, ki=0.0, rate_limit_per_step=0.02, actuator_tau_seconds=1e-6)
    ctrl.reset(torch.tensor([0.5]))
    valve = ctrl.step(torch.tensor([565.0]), torch.tensor([600.0]))
    assert abs(valve.item() - 0.5) <= 0.02 + 1e-6


def test_deadband_freezes_small_errors() -> None:
    ctrl = _controller(deadband_c=1.0, rate_limit_per_step=1.0, actuator_tau_seconds=1e-6)
    ctrl.reset(torch.tensor([0.5]))
    valve = ctrl.step(torch.tensor([565.0]), torch.tensor([565.5]))
    assert valve.item() == pytest.approx(0.5)


def test_actuator_lag_smooths_step() -> None:
    ctrl = _controller(kp=1.0, ki=0.0, rate_limit_per_step=1.0, actuator_tau_seconds=100.0, dt_seconds=10.0)
    ctrl.reset(torch.tensor([0.5]))
    valve = ctrl.step(torch.tensor([565.0]), torch.tensor([600.0]))
    # command saturates at 1.0; exact first-order lag moves 1-exp(-dt/tau) of the gap
    import math
    alpha = 1.0 - math.exp(-0.1)
    assert valve.item() == pytest.approx(0.5 + alpha * 0.5, abs=1e-6)


def test_anti_windup_freezes_integral_at_saturation() -> None:
    ctrl = _controller(kp=0.0, ki=0.1, rate_limit_per_step=1.0, actuator_tau_seconds=1e-6)
    ctrl.reset(torch.tensor([0.99]))
    for _ in range(20):
        ctrl.step(torch.tensor([565.0]), torch.tensor([700.0]))
    # Integral must not have accumulated past the saturation point.
    assert ctrl._integral.abs().max().item() < 5.0


def test_reset_and_shape_contracts() -> None:
    ctrl = _controller()
    with pytest.raises(FinalWMProtocolError):
        ctrl.step(torch.tensor([565.0]), torch.tensor([566.0]))
    with pytest.raises(FinalWMProtocolError):
        ctrl.reset(torch.tensor([[0.5]]))
    ctrl.reset(torch.tensor([0.4, 0.5]))
    with pytest.raises(FinalWMProtocolError):
        ctrl.step(torch.tensor([565.0]), torch.tensor([566.0]))
