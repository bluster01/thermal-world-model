"""Shared contracts for Phase 3.5 multi-step action-response operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn as nn


ROUTES = {"graybox", "koopman", "pi_ode", "deeponet"}
OPENING_MAPS = {"identity", "equal_percentage_r50", "monotone"}
DELAY_MODES = {"none", "fixed", "learned"}


class Phase35MultiStepError(ValueError):
    """Raised when a multi-step experiment violates its frozen contract."""


@dataclass(frozen=True)
class OperatorCapabilities:
    stateful_rollout: bool
    fixed_horizon: bool
    direction_constrained: bool
    continuous_time: bool


@dataclass(frozen=True)
class OperatorConfig:
    """One response representation under the common Phase 3.5-MS protocol."""

    route: str
    horizon: int
    context_dim: int
    dt_seconds: float = 10.0
    opening_map: str = "identity"
    poles: int = 2
    latent_dim: int = 4
    hidden_dim: int = 32
    tau_min_seconds: float = 20.0
    tau_max_seconds: float = 900.0
    ode_substeps: int = 2
    closure_scale: float = 0.02
    context_scheduled: bool = False
    schedule_log_scale: float = 0.5
    delay_mode: str = "none"
    fixed_delay_steps: int = 0
    max_delay_steps: int = 0

    def validate(self) -> None:
        if self.route not in ROUTES:
            raise Phase35MultiStepError(f"unknown response route={self.route!r}")
        if self.horizon < 2 or self.context_dim < 1:
            raise Phase35MultiStepError("horizon must be >=2 and context_dim must be positive")
        if self.dt_seconds <= 0:
            raise Phase35MultiStepError("dt_seconds must be positive")
        if self.opening_map not in OPENING_MAPS:
            raise Phase35MultiStepError(f"unknown opening_map={self.opening_map!r}")
        if self.poles not in {1, 2, 3}:
            raise Phase35MultiStepError("graybox poles must be 1, 2, or 3")
        if self.latent_dim < 1 or self.hidden_dim < 2:
            raise Phase35MultiStepError("latent_dim and hidden_dim are outside supported ranges")
        if not 0 < self.tau_min_seconds < self.tau_max_seconds:
            raise Phase35MultiStepError("time-constant bounds must satisfy 0 < min < max")
        if self.tau_min_seconds < self.dt_seconds:
            raise Phase35MultiStepError("tau_min_seconds must be at least one sampling interval")
        if self.ode_substeps < 1 or self.closure_scale < 0:
            raise Phase35MultiStepError("ODE substeps must be positive and closure_scale non-negative")
        if self.schedule_log_scale <= 0:
            raise Phase35MultiStepError("schedule_log_scale must be positive")
        if self.context_scheduled and self.route != "graybox":
            raise Phase35MultiStepError("context_scheduled is currently supported only by graybox")
        if self.delay_mode not in DELAY_MODES:
            raise Phase35MultiStepError(f"unknown delay_mode={self.delay_mode!r}")
        if not isinstance(self.fixed_delay_steps, int) or not isinstance(
            self.max_delay_steps, int
        ):
            raise Phase35MultiStepError("delay steps must be integers")
        if self.fixed_delay_steps < 0 or self.max_delay_steps < 0:
            raise Phase35MultiStepError("delay steps must be non-negative")
        if self.delay_mode != "none" and self.route != "graybox":
            raise Phase35MultiStepError("delay is currently only supported by graybox")
        if self.delay_mode == "none" and (
            self.fixed_delay_steps != 0 or self.max_delay_steps != 0
        ):
            raise Phase35MultiStepError("delay_mode=none requires zero delay steps")
        if self.delay_mode in {"fixed", "learned"} and self.max_delay_steps < 1:
            raise Phase35MultiStepError("active delay requires max_delay_steps >= 1")
        if self.delay_mode == "fixed" and not (
            0 <= self.fixed_delay_steps <= self.max_delay_steps
        ):
            raise Phase35MultiStepError(
                "fixed_delay_steps must be within max_delay_steps"
            )
        if self.delay_mode == "learned" and self.fixed_delay_steps != 0:
            raise Phase35MultiStepError(
                "learned delay does not accept fixed_delay_steps"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "OperatorConfig":
        obj = cls(**raw)
        obj.validate()
        return obj

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ResponseOutput:
    effect: torch.Tensor
    state_trajectory: torch.Tensor
    diagnostics: dict[str, torch.Tensor]

    @property
    def final_state(self) -> torch.Tensor:
        return self.state_trajectory[:, -1]


class ActionResponseOperator(nn.Module, ABC):
    """Common, reference-subtracted action-response interface."""

    capabilities: OperatorCapabilities

    def __init__(self, config: OperatorConfig):
        super().__init__()
        config.validate()
        self.config = config

    def _validate_inputs(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if context.ndim != 2 or context.shape[1] != self.config.context_dim:
            raise Phase35MultiStepError(
                f"context must have shape [B,{self.config.context_dim}], got {tuple(context.shape)}"
            )
        if action.ndim != 2 or reference.ndim != 2 or action.shape != reference.shape:
            raise Phase35MultiStepError("action and reference must have the same [B,H] shape")
        if action.shape[0] != context.shape[0]:
            raise Phase35MultiStepError("context and action batch sizes differ")
        if action.shape[1] < 1:
            raise Phase35MultiStepError("action horizon must contain at least one step")
        if self.capabilities.fixed_horizon and action.shape[1] != self.config.horizon:
            raise Phase35MultiStepError(
                f"action horizon must be {self.config.horizon}, got {action.shape[1]}"
            )
        if not (torch.isfinite(context).all() and torch.isfinite(action).all() and torch.isfinite(reference).all()):
            raise Phase35MultiStepError("operator inputs must be finite")

    @staticmethod
    def _identity_error(effect: torch.Tensor, action: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if torch.equal(action, reference):
            return effect.detach().abs().max()
        return torch.zeros((), dtype=effect.dtype, device=effect.device)

    @abstractmethod
    def forward(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> ResponseOutput:
        raise NotImplementedError
