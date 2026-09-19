"""Small baseline families. AIT is a compact inspired baseline, not CEDAR reproduction."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import CONTEXT, TRAIN_H
from .r4_transport import TransportWorldModel

TRAINED = ['direct_no_action', 'direct', 'gru', 'ssm', 'attention_concat', 'ait', 'r4', 'r4_mlp', 'r4_directref']
R4_PLAN = dict(width=32, backbone_width=32, global_hidden=27, response_hidden=3,
               response_context_width=6, response_network_width=16,
               initial_global_tau_seconds=60, initial_stage_tau_seconds=[60, 30, 120, 30, 180],
               initial_hidden_tau_seconds=180, initial_cooling_gain_raw=-1.5,
               response_rebound_scale=0., valve_modes_seconds=[20, 60, 180, 540], dt_seconds=10)


class Base(nn.Module):
    native_long = True

    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).float().clone())
        self.register_buffer('scale', torch.as_tensor(scale).float().clone())

    def norm(self, x):
        return (x - self.mean) / self.scale


class Persistence(Base):
    def forward(self, history, actions, boundaries):
        return history[:, -1:, :5].expand(-1, actions.shape[1], -1)


class Direct(Base):
    """RevIN + per-variable patch encoding + future-input fusion; fixed H32."""
    native_long = False

    def __init__(self, mean, scale, use_action=True):
        super().__init__(mean, scale)
        self.use_action = use_action
        self.patch = nn.Conv1d(1, 16, 8, stride=8)
        self.temporal = nn.Sequential(nn.Flatten(), nn.Linear(16 * (CONTEXT // 8), 32), nn.GELU())
        self.future = nn.Sequential(nn.Linear(TRAIN_H * (8 if use_action else 6), 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(13 * 32 + 64, 128), nn.GELU(), nn.Linear(128, TRAIN_H * 5))
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, actions, boundaries):
        horizon = actions.shape[1]
        if not 0 < horizon <= TRAIN_H:
            raise ValueError('Direct head requires <=32 steps; use block rollout for longer horizons')
        mu = history.mean(1, keepdim=True)
        sd = history.std(1, keepdim=True, unbiased=False).clamp_min(self.scale[None, None] * .05)
        x = ((history - mu) / sd).transpose(1, 2).reshape(-1, 1, CONTEXT)
        z = self.temporal(F.gelu(self.patch(x))).reshape(len(history), -1)
        inputs = torch.cat((actions, boundaries), -1) if self.use_action else boundaries
        start = 5 if self.use_action else 7
        inputs = (inputs - mu[:, :, start:]) / sd[:, :, start:]
        if horizon < TRAIN_H:
            inputs = torch.cat((inputs, inputs[:, -1:].expand(-1, TRAIN_H-horizon, -1)), 1)
        delta = self.head(torch.cat((z, self.future(inputs.flatten(1))), -1)).reshape(-1, TRAIN_H, 5)
        return history[:, -1:, :5] + delta[:, :horizon] * sd[:, :, :5]


class Recurrent(Base):
    def __init__(self, mean, scale, kind):
        super().__init__(mean, scale)
        self.kind = kind
        self.encoder = nn.GRU(13, 48, batch_first=True)
        if kind == 'gru':
            self.cell = nn.GRUCell(13, 48)
        elif kind == 'ssm':
            self.drive = nn.Sequential(nn.Linear(48 + 13, 64), nn.SiLU(), nn.Linear(64, 48), nn.Tanh())
            self.log_tau = nn.Parameter(torch.linspace(math.log(30), math.log(900), 48))
        else:
            raise ValueError(kind)
        self.readout = nn.Linear(48, 5)
        nn.init.normal_(self.readout.weight, std=.01)
        nn.init.zeros_(self.readout.bias)

    def forward(self, history, actions, boundaries):
        _, hidden = self.encoder(self.norm(history))
        h = hidden[0]
        h0 = h
        anchor = self.norm(history)[:, -1, :5]
        state = anchor
        future = (torch.cat((actions, boundaries), -1) - self.mean[5:]) / self.scale[5:]
        output = []
        for t in range(actions.shape[1]):
            x = torch.cat((state, future[:, t]), -1)
            if self.kind == 'gru':
                h = self.cell(x, h)
                state = state + .1 * self.readout(h)
            else:
                alpha = -torch.expm1(-10 / self.log_tau.exp())
                h = h + alpha * (self.drive(torch.cat((h, x), -1)) - h)
                state = anchor + self.readout(h - h0)
            output.append(state * self.scale[:5] + self.mean[:5])
        return torch.stack(output, 1)


class ActionInterleaved(Base):
    """Bounded-context causal state/action tokens, trained with H32 free rollout.

    History GRU initializes context. Last 16 tokens attend causally; one block,
    width32, no text module, no teacher forcing, no future state inputs.
    """
    def __init__(self, mean, scale, interleaved=True):
        super().__init__(mean, scale)
        self.interleaved = interleaved
        self.encoder = nn.GRU(13, 32, batch_first=True)
        self.state_enc = nn.Linear(5, 32)
        self.action_enc = nn.Linear(8, 32)
        self.role = nn.Parameter(torch.randn(2, 32) * .02)
        self.attn = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(32), nn.LayerNorm(32)
        self.ff = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.readout = nn.Linear(32, 5)
        nn.init.normal_(self.readout.weight, std=.001)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer('frequency', torch.exp(-math.log(10000) * torch.arange(16) / 16))

    def forward(self, history, actions, boundaries):
        normalized = self.norm(history)
        _, encoded = self.encoder(normalized)
        context = encoded[0]
        state = normalized[:, -1, :5]
        future = (torch.cat((actions, boundaries), -1) - self.mean[5:]) / self.scale[5:]
        tokens, output = [], []
        for t in range(actions.shape[1]):
            if self.interleaved:
                for role, token in enumerate((self.state_enc(state), self.action_enc(future[:, t]))):
                    angle = (2 * t + role) * self.frequency
                    pos = torch.cat((angle.sin(), angle.cos()))
                    tokens.append(token + context + self.role[role] + pos)
            else:
                # Feature-fused control: same parameters and 8-transition context.
                angle = (2*t+1) * self.frequency
                pos = torch.cat((angle.sin(), angle.cos()))
                tokens.append(self.state_enc(state) + self.action_enc(future[:, t]) + context + self.role.sum(0) + pos)
            tokens = tokens[-(16 if self.interleaved else 8):]
            x = torch.stack(tokens, 1)
            # Only the last (action) token is queried, so all keys precede it.
            z = self.norm1(x)
            attended, _ = self.attn(z[:, -1:], z, z, need_weights=False)
            q = x[:, -1:] + attended
            q = q + self.ff(self.norm2(q))
            state = state + .1 * self.readout(q[:, 0])
            output.append(state * self.scale[:5] + self.mean[:5])
        return torch.stack(output, 1)


class HistoryAdapter(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.adapter = nn.Sequential(nn.Linear(13 * 8, 64), nn.GELU(), nn.Linear(64, 27))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, normalized):
        initial, modes = self.original(normalized)
        pooled = F.adaptive_avg_pool1d(normalized.transpose(1, 2), 8).flatten(1)
        return torch.cat((initial[:, :5], initial[:, 5:] + .1 * torch.tanh(self.adapter(pooled))), -1), modes


class R4(Base):
    def __init__(self, mean, scale, kind='r4'):
        super().__init__(mean, scale)
        self.kind = kind
        self.core = TransportWorldModel(mean, scale, R4_PLAN)
        if kind == 'r4_mlp':
            self.core.encoder = HistoryAdapter(self.core.encoder)
        if kind == 'r4_directref':
            self.reference = Direct(mean, scale, use_action=False)
            self.native_long = False

    def forward(self, history, actions, boundaries):
        if self.kind != 'r4_directref':
            return self.core(history, actions, boundaries)['prediction']
        c = self.core
        initial, _ = c.encoder(c.normalize(history))
        hidden = initial[:, 5:]
        reference = self.reference(history, actions, boundaries)
        reference_action = (history[:, -1, 5:7] - c.mean[5:7]) / c.scale[5:7]
        carrier = history.new_zeros((len(history), 2, 5))
        memory = history.new_zeros((len(history), 5, c.local_size))
        modes = history.new_zeros((len(history), 2, 4))
        output = []
        for t in range(actions.shape[1]):
            boundary = (boundaries[:, t] - c.mean[7:]) / c.scale[7:]
            hidden = c.reference_transition(hidden, reference_action, boundary)
            action = (actions[:, t] - c.mean[5:7]) / c.scale[5:7] - reference_action
            ref = (reference[:, t] - c.mean[:5]) / c.scale[:5]
            carrier, memory, modes, response, _ = c.transport_transition(carrier, memory, modes, action, boundary, ref, hidden)
            output.append(reference[:, t] + response)
        return torch.stack(output, 1)


class Anchored(Base):
    """Fixed hold-last nominal plan. No separate fitting; sum component costs."""
    native_long = True  # hybrid: direct nominal blocks + uninterrupted R4 difference

    def __init__(self, predictor, response):
        super().__init__(predictor.mean, predictor.scale)
        self.predictor, self.response = predictor, response

    def forward(self, history, actions, boundaries):
        nominal = history[:, -1:, 5:7].expand_as(actions)
        return (self.predictor(history, nominal, boundaries)
                + (self.response(history, actions, boundaries)
                   - self.response(history, nominal, boundaries)))


def build(name, mean, scale):
    if name == 'persistence':
        return Persistence(mean, scale)
    if name in ('direct', 'direct_no_action'):
        return Direct(mean, scale, name == 'direct')
    if name in ('gru', 'ssm'):
        return Recurrent(mean, scale, name)
    if name in ('ait', 'attention_concat'):
        return ActionInterleaved(mean, scale, name == 'ait')
    if name.startswith('r4'):
        return R4(mean, scale, name)
    raise ValueError(name)
