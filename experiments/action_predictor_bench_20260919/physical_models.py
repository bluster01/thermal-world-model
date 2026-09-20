"""Constant-effective-cp thermal network with optional predictive-state training.

Energy units are kg-equivalent * degC, not calibrated steam enthalpy. Five
fluid cells and three metal cells exchange heat conservatively. Unobserved
metal states are estimates, never privileged physical ground truth.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import Base

ARMS = ('P0', 'P1', 'P2')
HEATED = (0, 2, 4)
PHYSICAL_FEATURES = (0, 1, 2, 3, 4, 7, 9, 10, 11, 12)


def head(inputs, outputs):
    net = nn.Sequential(nn.Linear(inputs*8, 64), nn.SiLU(), nn.Linear(64, outputs))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Physical(Base):
    def __init__(self, mean, scale, aux_mean, aux_scale, arm='P0'):
        super().__init__(mean, scale)
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.register_buffer('aux_mean', torch.as_tensor(aux_mean).float().clone())
        self.register_buffer('aux_scale', torch.as_tensor(aux_scale).float().clone())
        self.register_buffer('state_mean', self.mean[list(range(5))+list(HEATED)].clone())
        self.register_buffer('state_scale', self.scale[list(range(5))+list(HEATED)].clone())
        self.encoder = head(13+len(aux_mean), 6)
        self.physical_encoder = head(len(PHYSICAL_FEATURES), 3)
        if arm != 'P2':
            self.physical_encoder.requires_grad_(False)
        self.register_buffer('nominal_flow', self.mean[7].clamp_min(1).clone())
        # Initialize only from TRAIN means: steady mixing and heat balance.
        # This is a surrogate prior, not an identified per-stage water flow.
        water1 = ((self.mean[0]-self.mean[1])/(self.mean[1]-self.mean[11]).clamp_min(20)).clamp(.001, .3)
        water2 = ((1+water1)*(self.mean[2]-self.mean[3])/(self.mean[3]-self.mean[11]).clamp_min(20)).clamp(.001, .3)
        rises = torch.stack((self.mean[0]-self.mean[10], (1+water1)*(self.mean[2]-self.mean[1]),
                             (1+water1+water2)*(self.mean[4]-self.mean[3]))).clamp(2, 150)
        self.register_buffer('initial_wall_logit', torch.logit((rises/2/200).clamp(.005, .95)))
        self.raw_heat = nn.Parameter(torch.log(torch.expm1(rises/50)))
        self.raw_exchange = nn.Parameter(torch.full((3,), math.log(math.expm1(2.))))
        tau = torch.tensor([60., 20., 120., 20., 180., 180., 360., 540.])
        self.raw_capacity = nn.Parameter(torch.logit((tau-5)/1795))
        gain = torch.stack((water1,water2))/self.mean[5:7].clamp_min(.05)
        self.raw_water_gain = nn.Parameter(torch.logit(((gain-.0005)/.4995).clamp(.0001,.9999)))
        self.raw_valve_power = nn.Parameter(torch.full((2,), math.log(.2/.8)))
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False) if arm == 'P1' else None

    def features(self, history, physical=False):
        normalized = self.norm(history[:, :, :13])
        if physical:
            normalized = normalized[:, :, PHYSICAL_FEATURES]
        else:
            auxiliary = (history[:, :, 13:]-self.aux_mean)/self.aux_scale
            normalized = torch.cat((normalized, auxiliary), -1)
        return F.adaptive_avg_pool1d(normalized.transpose(1, 2), 8).flatten(1)

    def observe(self, history, *, physical=False, teacher=False):
        if teacher:
            if self.teacher is None:
                raise ValueError('No EMA teacher')
            encoded = self.teacher(self.features(history))
        else:
            encoded = (self.physical_encoder if physical else self.encoder)(self.features(history, physical))
        fluid = history[:, -1, :5]
        metal = fluid[:, HEATED] + 200*torch.sigmoid(self.initial_wall_logit+encoded[:, :3])
        state = torch.cat((fluid, metal), -1)
        return state, encoded

    def coefficients(self, history, actions, boundaries):
        """Build implicit-Euler matrices; coefficients never read candidate T."""
        _, encoded = self.observe(history)
        heat_multiplier = 1+.5*torch.tanh(encoded[:, 3:6])
        capacity = self.nominal_flow*(5+1795*torch.sigmoid(self.raw_capacity))
        exchange = self.nominal_flow*F.softplus(self.raw_exchange)
        max_water = self.nominal_flow*(.0005+.4995*torch.sigmoid(self.raw_water_gain))
        power = .5+2.5*torch.sigmoid(self.raw_valve_power)
        water = max_water*actions.clamp(0, 1).pow(power)
        # All flows are kg/s proxies; constant effective heat capacity cancels.
        flow = boundaries[:, :, 0].clamp_min(1)
        zero = torch.zeros_like(flow)
        injection = torch.stack((zero, water[:, :, 0], zero, water[:, :, 1], zero), -1)
        outflow = flow[:, :, None]+injection.cumsum(-1)
        inflow = outflow-injection
        batch, horizon = actions.shape[:2]
        loss = history.new_zeros(batch, horizon, 8, 8)
        for i in range(5):
            loss[:, :, i, i] = outflow[:, :, i]
            if i:
                loss[:, :, i, i-1] = -inflow[:, :, i]
        for j, i in enumerate(HEATED):
            loss[:, :, i, i] = loss[:, :, i, i]+exchange[j]
            loss[:, :, i, 5+j] = -exchange[j]
            loss[:, :, 5+j, i] = -exchange[j]
            loss[:, :, 5+j, 5+j] = exchange[j]
        # Feedwater T is an explicit coolant-temperature proxy, not water enthalpy.
        source_fluid = injection*boundaries[:, :, 4:5]
        source_fluid = source_fluid + F.one_hot(torch.tensor(0, device=history.device), 5)*flow[:, :, None]*boundaries[:, :, 3:4]
        fuel_ratio = (boundaries[:, :, 1]/self.mean[8].clamp_min(1)).clamp(.2, 2.)
        heat = self.nominal_flow*50*F.softplus(self.raw_heat)
        source_metal = fuel_ratio[:, :, None]*heat_multiplier[:, None]*heat
        source = torch.cat((source_fluid, source_metal), -1)
        diagonal = torch.diag(capacity/10).expand(batch, horizon, -1, -1)
        matrix = diagonal+loss
        solved = torch.linalg.solve(matrix, torch.cat((diagonal, source[:, :, :, None]), -1))
        return solved[:, :, :, :8], solved[:, :, :, 8], capacity, loss, source, water

    def rollout(self, history, actions, boundaries, *, physical=False, coefficients=None):
        coefficients = self.coefficients(history, actions, boundaries) if coefficients is None else coefficients
        transition, forcing = coefficients[:2]
        state, _ = self.observe(history, physical=physical)
        states = []
        for t in range(actions.shape[1]):
            state = torch.bmm(transition[:, t], state[:, :, None])[:, :, 0]+forcing[:, t]
            states.append(state)
        return torch.stack(states, 1)

    def forward(self, history, actions, boundaries):
        return self.rollout(history, actions, boundaries)[:, :, :5]

    def forecast_plan(self, history, actions, boundaries, *, mode):
        if mode not in ('block', 'native'):
            raise ValueError(mode)
        # Both APIs preserve the same physical state; no artificial energy reset.
        return self(history, actions[:, :-1], boundaries[:, :-1])

    def normalized_state(self, states):
        return (states-self.state_mean)/self.state_scale

    def future_states(self, bank, horizons, *, physical=False, teacher=False):
        history = bank[:, :64]
        states = []
        for h in horizons:
            # Future measured base history is a TARGET only. Auxiliary context
            # remains the starting history; no future auxiliary data is loaded.
            window = torch.cat((bank[:, h:h+64, :13], history[:, :, 13:]), -1)
            states.append(self.observe(window, physical=physical, teacher=teacher)[0])
        return self.normalized_state(torch.stack(states, 1))

    def training_objective(self, bank, weight=.05):
        h, u, d = bank[:, :64], bank[:, 63:-1, 5:7], bank[:, 63:-1, 7:13]
        states = self.rollout(h, u, d)
        target = bank[:, 64:, :5]
        weights = bank.new_tensor([.125, .125, .125, .125, .5])
        error = (((states[:, :, :5]-target)/self.scale[:5]).square()*weights).sum(-1)
        temperature = .5*error[:, :32].mean()+.5*error[:, 32:].mean()
        if self.arm == 'P0':
            return temperature
        horizons = (32, 64, 128)
        predicted = self.normalized_state(states[:, [k-1 for k in horizons]])
        if self.arm == 'P1':
            with torch.no_grad():
                future = self.future_states(bank, horizons, teacher=True)
            auxiliary = F.mse_loss(predicted, future)
        else:
            # Two initial-state views, one shared physics transition and same
            # exogenous heat scenario. Both target observers stay online.
            predicted_physical = self.normalized_state(self.rollout(h, u, d, physical=True)[:, [k-1 for k in horizons]])
            future = self.future_states(bank, horizons)
            future_physical = self.future_states(bank, horizons, physical=True)
            auxiliary = sum(F.mse_loss(p, t) for p in (predicted, predicted_physical)
                            for t in (future, future_physical))/4
        return temperature+weight*auxiliary

    @torch.no_grad()
    def after_optimizer_step(self):
        if self.teacher is not None:
            for target, source in zip(self.teacher.parameters(), self.encoder.parameters()):
                target.lerp_(source, .01)

    @torch.no_grad()
    def parameter_report(self):
        return dict(capacity_kg_equivalent=(self.nominal_flow*(5+1795*torch.sigmoid(self.raw_capacity))).cpu().tolist(),
            exchange_kg_per_second_equivalent=(self.nominal_flow*F.softplus(self.raw_exchange)).cpu().tolist(),
            heat_kg_degC_per_second_equivalent=(self.nominal_flow*50*F.softplus(self.raw_heat)).cpu().tolist(),
            maximum_water_kg_per_second_proxy=(self.nominal_flow*(.0005+.4995*torch.sigmoid(self.raw_water_gain))).cpu().tolist(),
            valve_power=(.5+2.5*torch.sigmoid(self.raw_valve_power)).cpu().tolist(),
            note='Effective parameters of constant-cp surrogate; not identified plant quantities')
