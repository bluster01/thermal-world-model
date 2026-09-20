"""Predict a candidate-independent nominal plan, then replace its R4 response."""
import torch
from torch import nn

from .models import Base
from .round2_models import ProtectedReference


class NominalPolicy(Base):
    """History + causal boundary sequence -> nominal valves, no future true T/u.

    First control is observed u[t]. Each later control uses earlier boundaries.
    Zero output initialization is exactly the hold policy, even at valve bounds.
    """
    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        self.encoder = nn.GRU(13, 24, batch_first=True)
        self.cell = nn.GRUCell(8, 24)
        self.readout = nn.Linear(24, 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, history, boundaries):
        _, encoded = self.encoder(self.norm(history))
        hidden = encoded[0]
        initial = history[:, -1, 5:7]
        logits = torch.logit(initial.clamp(1e-5, 1-1e-5))
        offset = logits.sigmoid()
        current, output = initial, [initial]
        for t in range(boundaries.shape[1]-1):
            d = (boundaries[:, t]-self.mean[7:])/self.scale[7:]
            u = (current-self.mean[5:7])/self.scale[5:7]
            hidden = self.cell(torch.cat((u, d), -1), hidden)
            current = (initial + ((logits+self.readout(hidden)).sigmoid()-offset)).clamp(0, 1)
            output.append(current)
        return torch.stack(output, 1)


class PlannedReference(ProtectedReference):
    """F(nominal) + R(candidate) - R(nominal), with one nominal for whole rollout.

    The policy is pretrained on training valves, then frozen. Candidate actions
    never enter policy or reference feedback, preserving paired R4 responses.
    """
    def __init__(self, predictor, response, policy=None):
        super().__init__(predictor, response)
        self.policy = policy
        if self.policy is not None: self.policy.requires_grad_(False)

    def nominal(self, history, boundaries):
        if self.policy is None:
            return history[:, -1:, 5:7].expand(-1, boundaries.shape[1], -1)
        return self.policy(history, boundaries)

    def action_adjustment(self, history, actions, nominal, boundaries):
        candidate = self.response.core(history, actions, boundaries)['action_response_C']
        if self.policy is None: return candidate  # hold response is identically zero
        reference = self.response.core(history, nominal, boundaries)['action_response_C']
        return candidate-reference

    def forward(self, history, actions, boundaries):
        nominal = self.nominal(history, boundaries)
        return self.predictor(history, nominal, boundaries) + self.action_adjustment(history, actions, nominal, boundaries)

    def forecast_plan(self, history, actions, boundaries, *, mode):
        from .evaluation import forecast
        nominal = self.nominal(history, boundaries)
        reference = forecast(self.predictor, history, nominal, boundaries, mode=mode)
        adjustment = self.action_adjustment(history, actions[:, :-1], nominal[:, :-1], boundaries[:, :-1])
        return reference + adjustment
