"""Approximate lumped thermal model for the independent synthetic benchmark.

Temperatures: degC; heat capacities: kJ/K; conductances: kW/K; time: seconds.
These are illustrative thermal-process parameters, not a calibrated boiler.
The independent truth plant includes an extra heated wall inventory absent here.
"""
import math

import torch

from .backends import PhysicsBackend


class LumpedThermalBackend(PhysicsBackend):
    """Two heat inventories and a bounded first-order heater actuator.

    Residual port 0 is EXTERNAL hot-side power, port 1 is an equal-and-opposite
    internal exchange. External residuals are a declared missing-power term,
    not energy conservation. No residual or observation update alters actuator.
    """

    def __init__(self, *, dt_seconds=1., actuator_tau=6.):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (dt_seconds, actuator_tau)):
            raise ValueError('dt_seconds and actuator_tau must be positive finite')
        super().__init__(state_norm=((40., 20.), (40., 20.), (.5, .5)),
                         observation_norm=((40., 20.),), action_norm=((.5, .5),),
                         boundary_norm=((20., 10.),), allowed_boundary_indices=(0,),
                         correction_mask=(1, 1, 0), residual_scale=(5., 3.))
        self.dt_seconds = float(dt_seconds)
        self.actuator_tau = float(actuator_tau)
        self.register_buffer('capacity', torch.tensor([10., 20.]))
        self.register_buffer('conductance', torch.tensor(1.))
        self.register_buffer('loss', torch.tensor(.2))
        self.register_buffer('heater_power', torch.tensor(10.))

    def advance(self, physical, action, boundary, residual):
        batch = physical.shape[0] if isinstance(physical, torch.Tensor) and physical.ndim == 2 else None
        for name, value, width in (('physical', physical, 3), ('action', action, 1),
                                   ('boundary', boundary, 1), ('residual', residual, 2)):
            if (not isinstance(value, torch.Tensor) or value.ndim != 2 or
                    value.shape != (batch, width) or batch is None or batch < 1 or
                    not value.is_floating_point() or value.dtype != self.state_loc.dtype or
                    value.device != self.state_loc.device or not bool(torch.isfinite(value).all())):
                raise ValueError(f'{name} must be a finite (B,{width}) tensor matching the backend dtype/device')
        if bool(((physical[:, 2] < 0) | (physical[:, 2] > 1)).any()):
            raise ValueError('Actuator state must be in [0,1]')
        if bool(((action < 0) | (action > 1)).any()):
            raise ValueError('Command must be in [0,1]')
        dt = self.dt_seconds
        ratio = dt / self.actuator_tau
        response = -math.expm1(-ratio)
        mean_factor = response / ratio if ratio else 1.
        actuator = physical[:, 2] + response * (action[:, 0] - physical[:, 2])
        # Exact mean actuator over a step with held command. Its integral sets
        # heater energy, while the temperatures use a simultaneous implicit step.
        mean_actuator = action[:, 0] + (physical[:, 2] - action[:, 0]) * mean_factor
        c1, c2 = self.capacity.unbind()
        k, loss = self.conductance, self.loss
        matrix = torch.stack([torch.stack([c1 / dt + k, -k]),
                              torch.stack([-k, c2 / dt + k + loss])])
        external, internal = residual.unbind(-1)
        rhs = torch.stack([c1 / dt * physical[:, 0] + self.heater_power * mean_actuator + external + internal,
                           c2 / dt * physical[:, 1] + loss * boundary[:, 0] - internal], -1)
        if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(rhs).all()):
            raise ValueError('Thermal solve exceeds the backend dtype finite range')
        temperatures = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
        result = torch.cat([temperatures, actuator[:, None]], -1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Thermal solve produced a nonfinite state')
        return result

    def decode(self, physical, boundary):
        return physical[:, 1:2]
