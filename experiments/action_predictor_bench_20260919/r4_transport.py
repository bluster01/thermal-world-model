"""R0 reference backbone with unit-DC latent transport and independent readout."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


from .r4_base import DualWorldModel


class TransportWorldModel(DualWorldModel):
    def __init__(self, mean, scale, plan):
        # Preserve initialization/RNG ordering of every surviving R0 parameter.
        super().__init__(mean, scale, plan)
        assert self.rebound_scale == 0.
        del self.link_gain, self.rebound, self.cooling_gain_raw
        self.path_gain_raw = nn.Parameter(torch.full((6,), plan["initial_cooling_gain_raw"]))
        self.register_buffer("path_valve", torch.tensor([0, 0, 0, 0, 1, 1]))
        self.register_buffer("path_stage", torch.tensor([1, 2, 3, 4, 3, 4]))
        self.register_buffer("reachable", torch.tensor([[0., 1., 1., 1., 1.], [0., 0., 0., 1., 1.]]))
        self.register_buffer("entry", self.spray_map.T.clone())
        self.register_buffer("path_output", F.one_hot(self.path_stage, 5).float())

    def transport_transition(self, carrier, memory, valve_modes, action_delta,
                             boundary, reference_temperature, global_hidden):
        shifted = torch.cat((torch.zeros_like(carrier[:, :, :1]), carrier[:, :, :-1]), -1)
        own, upstream = carrier.sum(1), shifted.sum(1)
        context = torch.tanh(self.response_context(global_hidden))
        features = torch.cat((memory, reference_temperature[..., None], own[..., None],
            upstream[..., None], boundary[:, None].expand(-1, 5, -1),
            context[:, None].expand(-1, 5, -1),
            self.stage_embedding[None].expand(len(carrier), -1, -1)), -1)
        local = self.conditioner(features)
        tau_v = self.valve_tau * torch.exp(.5 * torch.tanh(self.valve_rate(local[:, [1, 3]])))
        modes_next = valve_modes - torch.expm1(-self.dt / tau_v) * (action_delta[..., None] - valve_modes)
        effect = (modes_next * self.valve_mix.softmax(-1)).sum(-1)
        goal = shifted * (self.reachable - self.entry) + effect[..., None] * self.entry
        tau = self.log_stage_tau.exp() * torch.exp(.5 * torch.tanh(self.stage_rate(local).squeeze(-1)))
        alpha = -torch.expm1(-self.dt / tau)
        carrier_next = (carrier + alpha[:, None] * (goal - carrier)) * self.reachable

        memory_input = torch.cat((memory, own[..., None], upstream[..., None]), -1)
        memory_goal = torch.tanh(self.hidden_target(memory_input)) * self.hidden_gate(local).sigmoid()
        alpha_h = -torch.expm1(-self.dt / self.log_hidden_tau.exp())
        memory_next = memory + alpha_h[None, :, None] * (memory_goal - memory)

        modulation = .25 * torch.tanh(self.cooling_modulation(local).squeeze(-1))
        gain_C = F.softplus(self.path_gain_raw + modulation[:, self.path_stage]) * self.scale[self.path_stage]
        response_C = -(gain_C * carrier_next[:, self.path_valve, self.path_stage]) @ self.path_output
        return carrier_next, memory_next, modes_next, response_C, gain_C

    def forward(self, history, actions, boundaries):
        initial, _ = self.encoder(self.normalize(history))
        hidden_initial = initial[:, 5:]
        hidden = hidden_initial
        reference_action = (history[:, -1, 5:7] - self.mean[5:7]) / self.scale[5:7]
        normalized_actions = (actions - self.mean[5:7]) / self.scale[5:7]
        normalized_boundaries = (boundaries - self.mean[7:]) / self.scale[7:]
        carrier = history.new_zeros((len(history), 2, 5))
        memory = history.new_zeros((len(history), 5, self.local_size))
        modes = history.new_zeros((len(history), 2, 4))
        predictions, representations, references, responses, carriers, gains = [], [], [], [], [], []
        for step in range(actions.shape[1]):
            boundary = normalized_boundaries[:, step]
            hidden = self.reference_transition(hidden, reference_action, boundary)
            reference_C = history[:, -1, :5] + self.global_temperature(hidden - hidden_initial) * self.scale[:5]
            reference_normalized = (reference_C - self.mean[:5]) / self.scale[:5]
            carrier, memory, modes, response_C, gain_C = self.transport_transition(
                carrier, memory, modes, normalized_actions[:, step] - reference_action,
                boundary, reference_normalized, hidden)
            prediction = reference_C + response_C
            latent = torch.cat(((prediction - self.mean[:5]) / self.scale[:5], hidden), -1)
            predictions.append(prediction)
            representations.append(latent)
            references.append(reference_C)
            responses.append(response_C)
            carriers.append(carrier)
            gains.append(gain_C)
        return {"prediction": torch.stack(predictions, 1), "latent": torch.stack(representations, 1),
            "initial_latent": initial, "reference_prediction": torch.stack(references, 1),
            "action_response_C": torch.stack(responses, 1), "transport_state": torch.stack(carriers, 1),
            "path_gain_C_per_normalized_action": torch.stack(gains, 1)}
