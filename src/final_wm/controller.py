"""Layered control chain: SP -> PI controller -> actuator -> valve position.

The three layers are kept semantically separate, matching the project's
action hierarchy: the setpoint is a supervisory signal, the controller
output is a command, and the valve position is the plant-level action that
enters the transition through varphi(u).

Plant sign: opening a spray valve cools the steam (negative plant gain), so
the PI drives the valve *up* when the measured temperature is above SP.

The chain includes output saturation, an error deadband, a rate limit, and
a first-order actuator lag.  This is an interface model for closed-loop
rollout, not an identified controller; tag mapping and per-regime parameter
fits remain open evidence gaps.
"""

from __future__ import annotations

import torch

from src.final_wm.contracts import ControllerConfig, FinalWMProtocolError, validate_controller_config


class CascadePIController:
    """Batched PI controller with actuator dynamics (stateful)."""

    def __init__(self, config: ControllerConfig) -> None:
        validate_controller_config(config)
        self.config = config
        self._integral: torch.Tensor | None = None
        self._valve: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None

    def reset(self, valve0: torch.Tensor) -> None:
        valve0 = valve0.detach().clone().float()
        if valve0.ndim != 1:
            raise FinalWMProtocolError("controller reset expects a (B,) valve tensor")
        self._valve = valve0.clamp(self.config.valve_min, self.config.valve_max)
        self._integral = torch.zeros_like(self._valve)
        # Position-form PI bias: at zero error the controller holds the
        # valve at its reset position (bumpless takeover).
        self._bias = self._valve.clone()

    def step(self, sp_c: torch.Tensor, measured_c: torch.Tensor) -> torch.Tensor:
        """Advance one dt; returns the new physical valve position (B,)."""
        if self._valve is None or self._integral is None:
            raise FinalWMProtocolError("controller must be reset before stepping")
        sp_c = sp_c.float()
        measured_c = measured_c.float()
        if sp_c.shape != self._valve.shape or measured_c.shape != self._valve.shape:
            raise FinalWMProtocolError("controller inputs must be (B,) matching reset")

        error = measured_c - sp_c  # positive error -> open valve (cooling)
        error = torch.where(error.abs() <= self.config.deadband_c, torch.zeros_like(error), error)

        cfg = self.config
        trial_integral = self._integral + error * cfg.dt_seconds
        raw_command = self._bias + cfg.kp * error + cfg.ki * trial_integral
        command = raw_command.clamp(cfg.valve_min, cfg.valve_max)
        # Clamping anti-windup: freeze the integral when the command is
        # saturated and the error keeps driving further into the limit.
        stuck_high = (raw_command >= cfg.valve_max) & (error > 0)
        stuck_low = (raw_command <= cfg.valve_min) & (error < 0)
        self._integral = torch.where(stuck_high | stuck_low, self._integral, trial_integral)
        target = self._valve + (command - self._valve).clamp(
            -cfg.rate_limit_per_step, cfg.rate_limit_per_step
        )
        # Exact first-order actuator update; alpha in (0, 1) for any dt/tau.
        alpha = 1.0 - float(torch.exp(torch.tensor(-cfg.dt_seconds / cfg.actuator_tau_seconds)))
        valve = self._valve + alpha * (target - self._valve)
        self._valve = valve.clamp(cfg.valve_min, cfg.valve_max)
        return self._valve.clone()

    @property
    def valve(self) -> torch.Tensor:
        if self._valve is None:
            raise FinalWMProtocolError("controller must be reset before reading the valve")
        return self._valve.clone()
