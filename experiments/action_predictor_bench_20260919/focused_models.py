"""Four-arm reference/auxiliary-history comparison; original history stays fixed."""
import torch
from torch import nn
from torch.nn import functional as F

from .models import Base, build

ARMS = ('A', 'B', 'C', 'D')


class Focused(Base):
    def __init__(self, mean, scale, aux_mean, aux_scale, arm):
        super().__init__(mean, scale)
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.protected = arm in ('C', 'D')
        self.background = build('ssm', mean, scale)
        self.register_buffer('aux_mean', torch.as_tensor(aux_mean).float().clone())
        self.register_buffer('aux_scale', torch.as_tensor(aux_scale).float().clone())
        if arm in ('B', 'D'):
            self.adapter = nn.Sequential(nn.Linear(len(aux_mean)*8, 48), nn.SiLU(), nn.Linear(48, 48))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        else:
            self.adapter = None
        self.response = build('r4', mean, scale) if self.protected else None
        if self.response is not None:
            self.response.requires_grad_(False)

    def context(self, history):
        if self.adapter is None:
            return None
        z = (history[:, :, 13:] - self.aux_mean) / self.aux_scale
        z = F.adaptive_avg_pool1d(z.transpose(1, 2), 8).flatten(1)
        return .1 * torch.tanh(self.adapter(z))

    def forward(self, history, actions, boundaries):
        raw, context = history[:, :, :13], self.context(history)
        nominal = raw[:, -1:, 5:7].expand_as(actions) if self.protected else actions
        value = self.background(raw, nominal, boundaries, initial_context=context)
        if self.protected:
            value = value + self.response.core(raw, actions, boundaries)['action_response_C']
        return value

    def forecast_plan(self, history, actions, boundaries, *, mode):
        if mode not in ('block', 'native'):
            raise ValueError(mode)
        raw, context = history[:, :, :13], self.context(history)
        nominal = raw[:, -1:, 5:7].expand_as(actions) if self.protected else actions
        horizon = actions.shape[1] - 1
        if mode == 'native':
            value = self.background(raw, nominal[:, :-1], boundaries[:, :-1], initial_context=context)
        else:
            current, parts = raw, []
            for start in range(0, horizon, 32):
                stop = min(start+32, horizon)
                p = self.background(current, nominal[:, start:stop], boundaries[:, start:stop], initial_context=context)
                parts.append(p)
                rows = torch.cat((p, nominal[:, start+1:stop+1], boundaries[:, start+1:stop+1]), -1)
                current = torch.cat((current, rows), 1)[:, -64:]
            value = torch.cat(parts, 1)
        if self.protected:
            # Original history and continuous carrier: no candidate temperature
            # feeds back into the nominal background, including across blocks.
            value = value + self.response.core(raw, actions[:, :-1], boundaries[:, :-1])['action_response_C']
        return value
