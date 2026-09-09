"""Physical transitions and their units, separate from neural state inference.

ToyThermalBackend is a synthetic software fixture, never industrial evidence.
LegacyThermalBackend reuses an explicitly supplied v0.7 transition/provider. It
does not load checkpoints, change old experiments, or silently use a fake grid.
"""
from abc import ABC, abstractmethod
from numbers import Integral

import torch
from torch import nn


class PhysicsBackend(nn.Module, ABC):
    def __init__(self, *, state_norm, observation_norm, action_norm, boundary_norm,
                 allowed_boundary_indices, correction_mask, residual_scale):
        super().__init__()
        self.state_dim = len(state_norm)
        self.observation_dim = len(observation_norm)
        self.action_dim = len(action_norm)
        self.boundary_dim = len(boundary_norm)
        self.residual_dim = len(residual_scale)
        for name, pairs in [('state', state_norm), ('observation', observation_norm),
                            ('action', action_norm), ('boundary', boundary_norm),
                            ('feature', (*observation_norm, *action_norm, *boundary_norm))]:
            norms = torch.tensor(pairs, dtype=torch.float32)
            if norms.shape != (len(pairs), 2) or not bool(torch.isfinite(norms).all()) or bool((norms[:, 1] <= 0).any()):
                raise ValueError(f'Invalid {name} normalization')
            self.register_buffer(name + '_loc', norms[:, 0])
            self.register_buffer(name + '_scale', norms[:, 1])
        indices = tuple(allowed_boundary_indices)
        if (any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices) or
                len(set(indices)) != len(indices) or any(i < 0 or i >= self.boundary_dim for i in indices)):
            raise ValueError('Invalid boundary whitelist')
        if len(correction_mask) != self.state_dim or not all(v in (0, 1) for v in correction_mask):
            raise ValueError('State correction mask must be binary and match states')
        scales = torch.tensor(residual_scale, dtype=torch.float32)
        if scales.ndim != 1 or scales.numel() == 0 or not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError('Residual scales must be positive')
        self.register_buffer('allowed_boundary_indices', torch.tensor(indices, dtype=torch.long))
        self.register_buffer('correction_mask', torch.tensor(correction_mask, dtype=torch.float32))
        self.register_buffer('residual_scale', scales)

    def allowed_boundary(self, boundary):
        return ((boundary - self.boundary_loc) / self.boundary_scale).index_select(-1, self.allowed_boundary_indices)

    @abstractmethod
    def advance(self, physical, action, boundary, residual):
        """Advance one declared physical sampling interval; return B,S."""

    @abstractmethod
    def decode(self, physical, boundary):
        """Return B,O observations in engineering units."""


class ToyThermalBackend(PhysicsBackend):
    """Two thermal inventories, one heater, ambient temperature; one-second step.

    Internal exchange residuals cancel in the continuous energy balance. The
    backward-Euler solve advances both reservoirs simultaneously. This fixture
    intentionally omits the real boiler's flow, steam properties and wet states.
    """

    def __init__(self):
        super().__init__(state_norm=((20., 10.), (20., 10.)),
                         observation_norm=((20., 10.),), action_norm=((.5, .5),),
                         boundary_norm=((20., 10.),), allowed_boundary_indices=(0,),
                         correction_mask=(1, 1), residual_scale=(1.,))
        self.dt_seconds = 1.0
        # Energy / temperature capacities and conductances in synthetic units.
        self.register_buffer('capacity', torch.tensor([10., 20.]))
        self.register_buffer('conductance', torch.tensor(1.))
        self.register_buffer('loss', torch.tensor(.2))

    def advance(self, physical, action, boundary, residual):
        c1, c2 = self.capacity.unbind()
        k, loss, dt = self.conductance, self.loss, self.dt_seconds
        matrix = torch.stack([torch.stack([c1 / dt + k, -k]),
                              torch.stack([-k, c2 / dt + k + loss])])
        r = residual[:, 0]
        rhs = torch.stack([c1 / dt * physical[:, 0] + 10. * action[:, 0] + r,
                           c2 / dt * physical[:, 1] + loss * boundary[:, 0] - r], -1)
        return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)

    def decode(self, physical, boundary):
        return physical[:, 1:2]


class LegacyThermalBackend(PhysicsBackend):
    """The existing eleven-state, action-driven thermal transition as a backend.

    New neural heads/weights are distinct from old observers and closures.
    Passing a trained transition here does not make this a v0.7 replay.
    """

    def __init__(self, transition):
        from src.final_wm.contracts import (PHYSICAL_STATE_NORM, OBSERVATION_NORM,
                                            ACTION_NORM, BOUNDARY_NORM)
        if transition.layout.latent_dim != 0 or transition.config.spray_total_mode != 'action':
            raise ValueError('Adapter requires an 11-state action-driven transition')
        super().__init__(state_norm=PHYSICAL_STATE_NORM, observation_norm=OBSERVATION_NORM,
                         action_norm=ACTION_NORM, boundary_norm=BOUNDARY_NORM,
                         allowed_boundary_indices=tuple(range(6)),
                         correction_mask=(0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0),
                         residual_scale=(30000.,) * 3)
        self.transition = transition
        self.dt_seconds = transition.config.dt_seconds

    def advance(self, physical, action, boundary, residual):
        from src.final_wm.transition import ResidualInjection
        # The inherited v**gamma mapping has an infinite action derivative at
        # v=0 if 0<gamma<1. Preserve historical forward behavior, but reject this
        # known singular endpoint when a new differentiable planner asks for it.
        if torch.is_grad_enabled() and action.requires_grad:
            gamma = torch.stack([self.transition.val('gamma1'), self.transition.val('gamma2')])
            if bool(((action == 0) & (gamma < 1)).any()):
                raise ValueError('Action gradient is singular at zero opening with gamma < 1')
        injection = ResidualInjection(residual, -residual, None)
        return self.transition.step(physical, boundary, action, injection).state

    def decode(self, physical, boundary):
        return self.transition.output_temperatures(physical, boundary)
