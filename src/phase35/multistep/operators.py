"""Causal multi-step response representations under one auditable contract."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import MonotoneValveMap
from .contracts import (
    ActionResponseOperator,
    OperatorCapabilities,
    OperatorConfig,
    Phase35MultiStepError,
    ResponseOutput,
)


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse softplus requires a positive value")
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class _ValveResponseOperator(ActionResponseOperator):
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.opening_map = MonotoneValveMap(config.opening_map)

    def effective_dose(self, action: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.opening_map(action) - self.opening_map(reference)


class StableGrayboxOperator(_ValveResponseOperator):
    """Stable cascaded valve-to-temperature model with optional context scheduling."""

    capabilities = OperatorCapabilities(True, False, True, True)

    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.raw_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.04), dtype=torch.float32))
        initial_taus = torch.linspace(
            max(config.tau_min_seconds * 1.5, 60.0),
            min(config.tau_max_seconds * 0.75, 240.0),
            config.poles,
        )
        fractions = (initial_taus - config.tau_min_seconds) / (
            config.tau_max_seconds - config.tau_min_seconds
        )
        self.raw_tau = nn.Parameter(torch.tensor([_logit(float(x)) for x in fractions]))
        if config.context_scheduled:
            self.gain_schedule = nn.Linear(config.context_dim, 1, bias=False)
            self.tau_schedule = nn.Linear(config.context_dim, config.poles, bias=False)
            nn.init.zeros_(self.gain_schedule.weight)
            nn.init.zeros_(self.tau_schedule.weight)
        else:
            self.gain_schedule = None
            self.tau_schedule = None
        if config.delay_mode == "learned":
            initial_delay_logits = torch.zeros(
                config.max_delay_steps + 1, dtype=torch.float32
            )
            initial_delay_logits[0] = 2.0
            self.delay_logits = nn.Parameter(initial_delay_logits)
        else:
            self.register_parameter("delay_logits", None)
        if config.delay_mode == "fixed":
            fixed_weights = torch.zeros(
                config.max_delay_steps + 1, dtype=torch.float32
            )
            fixed_weights[config.fixed_delay_steps] = 1.0
            self.register_buffer("fixed_delay_weights", fixed_weights)
        else:
            self.register_buffer("fixed_delay_weights", None)

    @property
    def delay_state_width(self) -> int:
        if self.config.delay_mode == "none":
            return 0
        return self.config.max_delay_steps + 1

    def delay_weights(self) -> torch.Tensor | None:
        if self.config.delay_mode == "none":
            return None
        if self.config.delay_mode == "fixed":
            return self.fixed_delay_weights
        return torch.softmax(self.delay_logits, dim=0)

    def physical_parameters(self, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        base_gain = -F.softplus(self.raw_gain)
        base_tau = self.config.tau_min_seconds + (
            self.config.tau_max_seconds - self.config.tau_min_seconds
        ) * torch.sigmoid(self.raw_tau)
        if self.gain_schedule is None:
            return base_gain, base_tau
        if context is None:
            raise Phase35MultiStepError("context is required for scheduled graybox parameters")
        gain_shift = self.config.schedule_log_scale * torch.tanh(self.gain_schedule(context)).squeeze(1)
        tau_shift = self.config.schedule_log_scale * torch.tanh(self.tau_schedule(context))
        gain = base_gain * torch.exp(gain_shift)
        tau = base_tau[None, :] * torch.exp(tau_shift)
        tau = tau.clamp(self.config.tau_min_seconds, self.config.tau_max_seconds)
        return gain, tau

    def forward(self, context, action, reference, initial_state=None) -> ResponseOutput:
        self._validate_inputs(context, action, reference)
        dose = self.effective_dose(action, reference)
        gain, tau = self.physical_parameters(context)
        decay = torch.exp(-self.config.dt_seconds / tau)
        batch = context.shape[0]
        delay_width = self.delay_state_width
        state_width = delay_width + self.config.poles
        if initial_state is None:
            state = torch.zeros(batch, state_width, dtype=action.dtype, device=action.device)
        else:
            expected = (batch, state_width)
            if tuple(initial_state.shape) != expected:
                raise Phase35MultiStepError(f"graybox initial_state must have shape {expected}")
            state = initial_state
        delay_buffer = state[:, :delay_width] if delay_width else None
        pole_state = state[:, delay_width:]
        delay_weights = self.delay_weights()
        trajectory = []
        for step in range(action.shape[1]):
            if delay_buffer is None:
                stage_input = dose[:, step]
            else:
                delay_buffer = torch.cat(
                    (dose[:, step, None], delay_buffer[:, :-1]), dim=1
                )
                stage_input = (delay_buffer * delay_weights[None, :]).sum(dim=1)
            updated = []
            for pole in range(self.config.poles):
                pole_decay = decay[pole] if decay.ndim == 1 else decay[:, pole]
                stage = (
                    pole_decay * pole_state[:, pole]
                    + (1.0 - pole_decay) * stage_input
                )
                updated.append(stage)
                stage_input = stage
            pole_state = torch.stack(updated, dim=1)
            state = (
                pole_state
                if delay_buffer is None
                else torch.cat((delay_buffer, pole_state), dim=1)
            )
            trajectory.append(state)
        states = torch.stack(trajectory, dim=1)
        effect = (
            gain * states[..., -1]
            if gain.ndim == 0
            else gain[:, None] * states[..., -1]
        )
        diagnostics = {
            "gain_c_per_effective_pct": gain.mean(),
            "gain_range_c_per_effective_pct": torch.stack((gain.min(), gain.max())),
            "tau_seconds": tau.mean(dim=0) if tau.ndim == 2 else tau,
            "tau_range_seconds": torch.stack(
                (tau.amin(dim=0), tau.amax(dim=0)), dim=1
            ) if tau.ndim == 2 else torch.stack((tau, tau), dim=1),
            "spectral_radius": decay.max(),
            "context_scheduled": self.config.context_scheduled,
            "delay_mode": self.config.delay_mode,
            "reference_identity_max_error": self._identity_error(effect, action, reference),
        }
        if delay_weights is not None:
            delay_steps = torch.arange(
                delay_weights.numel(),
                dtype=delay_weights.dtype,
                device=delay_weights.device,
            )
            diagnostics["delay_weights"] = delay_weights
            diagnostics["expected_delay_seconds"] = (
                delay_weights * delay_steps
            ).sum() * self.config.dt_seconds
        return ResponseOutput(effect, states, diagnostics)


class ControlledKoopmanOperator(_ValveResponseOperator):
    """Stable diagonal controlled Koopman response, distinct from a free forecast head."""

    capabilities = OperatorCapabilities(True, False, True, False)

    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        taus = torch.linspace(60.0, min(config.tau_max_seconds * 0.8, 480.0), config.latent_dim)
        decay = torch.exp(-config.dt_seconds / taus.clamp_min(config.tau_min_seconds))
        self.raw_decay = nn.Parameter(torch.tensor([_logit(float(x)) for x in decay]))
        self.raw_control = nn.Parameter(torch.full((config.latent_dim,), _inverse_softplus(1.0)))
        self.raw_decoder = nn.Parameter(torch.full((config.latent_dim,), _inverse_softplus(0.04)))

    def stable_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.raw_decay).clamp(1e-5, 1.0 - 1e-5)
        control = (1.0 - decay) * F.softplus(self.raw_control) / self.config.latent_dim
        decoder = -F.softplus(self.raw_decoder)
        return decay, control, decoder

    def forward(self, context, action, reference, initial_state=None) -> ResponseOutput:
        self._validate_inputs(context, action, reference)
        dose = self.effective_dose(action, reference)
        decay, control, decoder = self.stable_parameters()
        batch = context.shape[0]
        if initial_state is None:
            state = torch.zeros(batch, self.config.latent_dim, dtype=action.dtype, device=action.device)
        else:
            expected = (batch, self.config.latent_dim)
            if tuple(initial_state.shape) != expected:
                raise Phase35MultiStepError(f"Koopman initial_state must have shape {expected}")
            state = initial_state
        states, effects = [], []
        for step in range(action.shape[1]):
            state = decay * state + control * dose[:, step, None]
            states.append(state)
            effects.append((state * decoder).sum(dim=1))
        trajectory = torch.stack(states, dim=1)
        effect = torch.stack(effects, dim=1)
        diagnostics = {
            "koopman_decay": decay,
            "control_weight": control,
            "decoder_weight": decoder,
            "spectral_radius": decay.max(),
            "reference_identity_max_error": self._identity_error(effect, action, reference),
        }
        return ResponseOutput(effect, trajectory, diagnostics)


class PhysicsInformedODEOperator(_ValveResponseOperator):
    """Nominal two-pole ODE with a small, auditable neural closure."""

    # The nominal block has the physical direction, but the learned closure can
    # violate it.  Report that honestly and gate the realised response at eval.
    capabilities = OperatorCapabilities(True, False, False, True)

    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.raw_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.04)))
        initial_taus = (max(60.0, config.tau_min_seconds * 1.5), min(240.0, config.tau_max_seconds * 0.75))
        fractions = [
            (tau - config.tau_min_seconds) / (config.tau_max_seconds - config.tau_min_seconds)
            for tau in initial_taus
        ]
        self.raw_tau = nn.Parameter(torch.tensor([_logit(x) for x in fractions]))
        self.closure = nn.Sequential(
            nn.Linear(config.context_dim + 3, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 2),
        )
        nn.init.zeros_(self.closure[-1].weight)
        nn.init.zeros_(self.closure[-1].bias)

    def physical_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        gain = -F.softplus(self.raw_gain)
        tau = self.config.tau_min_seconds + (
            self.config.tau_max_seconds - self.config.tau_min_seconds
        ) * torch.sigmoid(self.raw_tau)
        return gain, tau

    def _derivative(
        self, context: torch.Tensor, state: torch.Tensor, dose: torch.Tensor, tau: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nominal = torch.stack(((dose - state[:, 0]) / tau[0], (state[:, 0] - state[:, 1]) / tau[1]), dim=1)
        closure_raw = torch.tanh(self.closure(torch.cat((context, state, dose[:, None]), dim=1)))
        magnitude = dose.abs() + state.abs().sum(dim=1)
        gate = (magnitude / (1.0 + magnitude))[:, None]
        closure = self.config.closure_scale * gate * closure_raw
        return nominal + closure, closure

    def forward(self, context, action, reference, initial_state=None) -> ResponseOutput:
        self._validate_inputs(context, action, reference)
        dose = self.effective_dose(action, reference)
        gain, tau = self.physical_parameters()
        batch = context.shape[0]
        if initial_state is None:
            state = torch.zeros(batch, 2, dtype=action.dtype, device=action.device)
        else:
            if tuple(initial_state.shape) != (batch, 2):
                raise Phase35MultiStepError(f"PI-ODE initial_state must have shape {(batch, 2)}")
            state = initial_state
        step_dt = self.config.dt_seconds / self.config.ode_substeps
        states, closures = [], []
        for step in range(action.shape[1]):
            u = dose[:, step]
            last_closure = torch.zeros_like(state)
            for _ in range(self.config.ode_substeps):
                k1, c1 = self._derivative(context, state, u, tau)
                k2, c2 = self._derivative(context, state + 0.5 * step_dt * k1, u, tau)
                k3, c3 = self._derivative(context, state + 0.5 * step_dt * k2, u, tau)
                k4, c4 = self._derivative(context, state + step_dt * k3, u, tau)
                state = state + step_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                last_closure = (c1 + 2 * c2 + 2 * c3 + c4) / 6.0
            states.append(state)
            closures.append(last_closure)
        trajectory = torch.stack(states, dim=1)
        closure_trajectory = torch.stack(closures, dim=1)
        effect = gain * trajectory[..., 1]
        nominal_decay = torch.exp(-self.config.dt_seconds / tau)
        diagnostics = {
            "gain_c_per_effective_pct": gain,
            "tau_seconds": tau,
            "spectral_radius": nominal_decay.max(),
            "physics_residual_mse": closure_trajectory.square().mean(),
            "reference_identity_max_error": self._identity_error(effect, action, reference),
        }
        return ResponseOutput(effect, trajectory, diagnostics)


class CausalDeepONetOperator(_ValveResponseOperator):
    """Prefix-causal branch/trunk operator with exact reference subtraction."""

    capabilities = OperatorCapabilities(False, True, False, False)

    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.branch = nn.GRU(config.context_dim + 1, config.latent_dim, batch_first=True)
        self.trunk = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        time = torch.arange(1, config.horizon + 1, dtype=torch.float32)[:, None] / config.horizon
        self.register_buffer("normalized_time", time)

    def _raw_operator(self, context: torch.Tensor, opening: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        repeated_context = context[:, None, :].expand(-1, self.config.horizon, -1)
        branch_input = torch.cat((opening[..., None] / 100.0, repeated_context), dim=2)
        coefficients, _ = self.branch(branch_input)
        basis = self.trunk(self.normalized_time).to(dtype=coefficients.dtype)
        value = self.output_scale * (coefficients * basis[None, :, :]).sum(dim=2) / math.sqrt(self.config.latent_dim)
        return value, coefficients

    def forward(self, context, action, reference, initial_state=None) -> ResponseOutput:
        self._validate_inputs(context, action, reference)
        if initial_state is not None:
            raise Phase35MultiStepError("causal DeepONet is a fixed-horizon operator and does not accept initial_state")
        action_opening = self.opening_map(action)
        reference_opening = self.opening_map(reference)
        action_value, action_state = self._raw_operator(context, action_opening)
        reference_value, reference_state = self._raw_operator(context, reference_opening)
        effect = action_value - reference_value
        state = action_state - reference_state
        diagnostics = {
            "spectral_radius": torch.full((), float("nan"), dtype=effect.dtype, device=effect.device),
            "reference_identity_max_error": self._identity_error(effect, action, reference),
        }
        return ResponseOutput(effect, state, diagnostics)


def build_response_operator(config: OperatorConfig) -> ActionResponseOperator:
    config.validate()
    if config.route == "graybox":
        return StableGrayboxOperator(config)
    if config.route == "koopman":
        return ControlledKoopmanOperator(config)
    if config.route == "pi_ode":
        return PhysicsInformedODEOperator(config)
    if config.route == "deeponet":
        return CausalDeepONetOperator(config)
    raise Phase35MultiStepError(f"unknown response route={config.route!r}")
