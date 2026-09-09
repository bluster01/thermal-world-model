"""Explicit command-to-position lag around an instantaneous-input physics backend.

Time constants are caller-supplied assumptions, not identified equipment values.
The inner backend receives held mean opening. For nonlinear opening-to-flow
maps this is not the exact average flow, nor an exact coupled thermal solution.
"""

from enum import Enum
import math
from numbers import Real

import torch

from src.world_model_vnext.backends import PhysicsBackend
from src.world_model_vnext.thermal_backend import LumpedThermalBackend


class BackendInputSemantics(str, Enum):
    INSTANTANEOUS_ACTUAL_FRACTION = 'instantaneous_actual_fraction'
    COMMAND_WITH_ACTUATOR_DYNAMICS = 'command_with_actuator_dynamics'


class ActuatorLagBackend(PhysicsBackend):
    """Append actual actuator fractions to physical state; accept command fractions.

    The declaration is mandatory because arbitrary custom backends cannot be
    inspected reliably for hidden actuator dynamics. Known incompatible types
    and explicit dynamic-backend markers are additionally rejected.
    """

    input_semantics = BackendInputSemantics.COMMAND_WITH_ACTUATOR_DYNAMICS

    def __init__(self, backend: PhysicsBackend, *, time_constants_seconds,
                 input_semantics: BackendInputSemantics | str):
        if not isinstance(backend, PhysicsBackend):
            raise ValueError('An explicit PhysicsBackend is required')
        if input_semantics != BackendInputSemantics.INSTANTANEOUS_ACTUAL_FRACTION:
            raise ValueError('Inner input_semantics must declare instantaneous_actual_fraction')
        if (isinstance(backend, (ActuatorLagBackend, LumpedThermalBackend)) or
                getattr(backend, 'input_semantics', None) == BackendInputSemantics.COMMAND_WITH_ACTUATOR_DYNAMICS):
            raise ValueError('Inner backend already contains actuator dynamics; repeated lag is forbidden')
        dt = getattr(backend, 'dt_seconds', None)
        if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Inner backend must declare positive finite dt_seconds')
        if isinstance(time_constants_seconds, torch.Tensor):
            if not time_constants_seconds.is_floating_point() or time_constants_seconds.requires_grad:
                raise ValueError('Time constants require fixed floating-point values, not trainable tensors')
            source_tau = time_constants_seconds.detach()
        else:
            if (not isinstance(time_constants_seconds, (list, tuple)) or
                    any(isinstance(value, bool) or not isinstance(value, Real) for value in time_constants_seconds)):
                raise ValueError('Time constants require an explicit numeric vector')
            source_tau = time_constants_seconds
        tau = torch.as_tensor(source_tau, dtype=backend.state_loc.dtype,
                              device=backend.state_loc.device).detach().clone()
        if tau.shape != (backend.action_dim,) or not bool(torch.isfinite(tau).all()) or bool((tau <= 0).any()):
            raise ValueError('Time constants must be positive finite and match action_dim in backend dtype')

        def norms(prefix):
            return torch.stack([getattr(backend, prefix + '_loc'), getattr(backend, prefix + '_scale')], -1).detach().cpu().tolist()

        super().__init__(state_norm=norms('state') + norms('action'),
                         observation_norm=norms('observation'), action_norm=norms('action'),
                         boundary_norm=norms('boundary'),
                         allowed_boundary_indices=backend.allowed_boundary_indices.detach().cpu().tolist(),
                         correction_mask=backend.correction_mask.detach().cpu().tolist() + [0] * backend.action_dim,
                         residual_scale=backend.residual_scale.detach().cpu().tolist())
        self.inner = backend
        self.dt_seconds = float(dt)
        self.inner_state_dim = backend.state_dim
        # Copy normalization tensors exactly, including an already-double inner
        # backend; avoid losing precision through the base constructor's float32.
        for prefix in ('observation', 'action', 'boundary', 'feature'):
            for suffix in ('loc', 'scale'):
                setattr(self, prefix + '_' + suffix, getattr(backend, prefix + '_' + suffix).detach().clone())
        self.state_loc = torch.cat([backend.state_loc, backend.action_loc]).detach().clone()
        self.state_scale = torch.cat([backend.state_scale, backend.action_scale]).detach().clone()
        self.correction_mask = torch.cat([backend.correction_mask, backend.action_loc.new_zeros(backend.action_dim)]).detach().clone()
        self.residual_scale = backend.residual_scale.detach().clone()
        self.allowed_boundary_indices = backend.allowed_boundary_indices.detach().clone()
        self.register_buffer('time_constants_seconds', tau)

    def _matrix(self, value, width, label, batch=None):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or
                value.dtype != self.state_loc.dtype or value.device != self.state_loc.device or
                value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != width or
                (batch is not None and value.shape[0] != batch) or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{label} must be finite (B,{width}) with matching backend dtype/device/batch')

    def _timebase(self):
        inner_dt = getattr(self.inner, 'dt_seconds', None)
        if (isinstance(inner_dt, bool) or not isinstance(inner_dt, Real) or
                not math.isfinite(inner_dt) or inner_dt <= 0 or inner_dt != self.dt_seconds):
            raise ValueError('Inner dt_seconds must remain positive finite and match the actuator timebase')

    def _physical(self, physical):
        self._timebase()
        self._matrix(physical, self.state_dim, 'physical state')
        actual = physical[:, self.inner_state_dim:]
        if bool(((actual < 0) | (actual > 1)).any()):
            raise ValueError('Actual actuator fractions must be in [0,1]')
        if self.inner.state_loc.dtype != self.state_loc.dtype or self.inner.state_loc.device != self.state_loc.device:
            raise ValueError('Inner and wrapper backend dtype/device must match')
        return actual

    def actuator_step(self, actual, command):
        """Return exact endpoint and step-average fractions under held commands."""
        self._timebase()
        self._matrix(actual, self.action_dim, 'actual actuator fractions')
        self._matrix(command, self.action_dim, 'command', actual.shape[0])
        if bool(((actual < 0) | (actual > 1) | (command < 0) | (command > 1)).any()):
            raise ValueError('Command and actual actuator fractions must be in [0,1]')
        tau = self.time_constants_seconds
        if not bool(torch.isfinite(tau).all()) or bool((tau <= 0).any()):
            raise ValueError('Fixed time constants must remain positive finite')
        ratio = self.dt_seconds / tau
        response = -torch.expm1(-ratio)
        # Mean response = 1 - (1-exp(-ratio))/ratio. Direct subtraction loses
        # the small response; the series keeps its leading ratio/2 derivative.
        small_ratio = ratio.clamp(max=1e-3)
        series = small_ratio * (.5 + small_ratio * (-1 / 6 + small_ratio * (1 / 24 - small_ratio / 120)))
        denominator = torch.where(ratio > 0, ratio, torch.ones_like(ratio))
        mean_response = torch.where(ratio < 1e-3, series, 1 - response / denominator)
        next_actual = (1 - response) * actual + response * command
        mean_actual = (1 - mean_response) * actual + mean_response * command
        return next_actual, mean_actual

    def advance(self, physical, action, boundary, residual):
        actual = self._physical(physical)
        batch = physical.shape[0]
        self._matrix(action, self.action_dim, 'command', batch)
        self._matrix(boundary, self.boundary_dim, 'boundary', batch)
        self._matrix(residual, self.residual_dim, 'residual', batch)
        next_actual, mean_actual = self.actuator_step(actual, action)
        # Keep this call and its autograd graph intact: the Legacy backend's
        # singular zero-opening guard must still apply to the effective input.
        next_physical = self.inner.advance(physical[:, :self.inner_state_dim], mean_actual, boundary, residual)
        self._matrix(next_physical, self.inner_state_dim, 'inner next state', batch)
        return torch.cat([next_physical, next_actual], -1)

    def decode(self, physical, boundary):
        self._physical(physical)
        self._matrix(boundary, self.boundary_dim, 'boundary', physical.shape[0])
        observations = self.inner.decode(physical[:, :self.inner_state_dim], boundary)
        self._matrix(observations, self.observation_dim, 'decoded observation', physical.shape[0])
        return observations
