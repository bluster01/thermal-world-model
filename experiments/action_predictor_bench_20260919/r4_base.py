"""Rich held-valve reference dynamics plus a directed action-increment model."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SharedEncoder(nn.Module):
    def __init__(self, width, memory):
        super().__init__()
        self.backbone = nn.GRU(13, width, batch_first=True)
        self.memory = nn.Linear(width, memory)

    def forward(self, normalized_history):
        _, shared = self.backbone(normalized_history)
        memory = torch.tanh(self.memory(shared[-1]))
        latent = torch.cat((normalized_history[:, -1, :5], memory), -1)
        return latent, normalized_history[:, -1, 5:7].repeat_interleave(4, -1)


class DualWorldModel(nn.Module):
    def __init__(self, mean, scale, plan):
        super().__init__()
        hidden = plan["global_hidden"]
        local = plan["response_hidden"]
        context = plan["response_context_width"]
        width = plan["response_network_width"]
        assert plan["width"] == hidden + 5
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("valve_tau", torch.tensor(plan["valve_modes_seconds"], dtype=torch.float32))
        self.register_buffer("spray_map", torch.tensor([[0., 0.], [1., 0.], [0., 0.], [0., 1.], [0., 0.]]))
        self.dt = plan["dt_seconds"]
        self.local_size = local
        self.rebound_scale = plan["response_rebound_scale"]
        self.encoder = SharedEncoder(plan["backbone_width"], hidden)
        self.global_conditioner = nn.Sequential(nn.Linear(hidden + 2 + 6, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.global_target = nn.Linear(32, hidden)
        self.global_temperature = nn.Linear(hidden, 5, bias=False)
        self.log_global_tau = nn.Parameter(torch.full((hidden,), math.log(plan["initial_global_tau_seconds"])))

        self.response_context = nn.Linear(hidden, context)
        self.stage_embedding = nn.Parameter(torch.randn(5, 2) * .1)
        # Each node sees shared action-independent context, its own memory and
        # response, and only its immediate upstream response.
        self.conditioner = nn.Sequential(nn.Linear(local + 1 + 1 + 1 + 6 + context + 2, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.stage_rate = nn.Linear(width, 1)
        self.link_gain = nn.Linear(width, 1)
        self.cooling_modulation = nn.Linear(width, 1)
        self.valve_rate = nn.Linear(width, 4)
        self.hidden_gate = nn.Linear(width, local)
        self.hidden_target = nn.Linear(local + 2, local, bias=False)
        self.rebound = nn.Linear(local, 1, bias=False)
        self.log_stage_tau = nn.Parameter(torch.tensor(plan["initial_stage_tau_seconds"], dtype=torch.float32).log())
        self.log_hidden_tau = nn.Parameter(torch.full((5,), math.log(plan["initial_hidden_tau_seconds"])))
        self.valve_mix = nn.Parameter(torch.zeros(2, 4))
        self.cooling_gain_raw = nn.Parameter(torch.full((2,), plan["initial_cooling_gain_raw"]))

    def normalize(self, values):
        return (values - self.mean) / self.scale

    def reference_transition(self, hidden, reference_action, boundary):
        shared = self.global_conditioner(torch.cat((hidden, reference_action, boundary), -1))
        alpha = -torch.expm1(-self.dt / self.log_global_tau.exp())
        return hidden + alpha * (torch.tanh(self.global_target(shared)) - hidden)

    def response_transition(self, response_C, memory, valve_modes, action_delta, boundary, reference_temperature, global_hidden):
        upstream_C = torch.cat((torch.zeros_like(response_C[:, :1]), response_C[:, :-1]), -1)
        own = response_C / self.scale[:5]
        upstream = upstream_C / self.scale[:5]
        global_context = torch.tanh(self.response_context(global_hidden))
        features = torch.cat((memory, reference_temperature[..., None], own[..., None], upstream[..., None], boundary[:, None].expand(-1, 5, -1), global_context[:, None].expand(-1, 5, -1), self.stage_embedding[None].expand(response_C.shape[0], -1, -1)), -1)
        local = self.conditioner(features)
        spray_local = local[:, [1, 3]]
        tau_v = self.valve_tau * torch.exp(.5 * torch.tanh(self.valve_rate(spray_local)))
        alpha_v = -torch.expm1(-self.dt / tau_v)
        modes_next = valve_modes + alpha_v * (action_delta[..., None] - valve_modes)
        effect = (modes_next * self.valve_mix.softmax(-1)).sum(-1)
        gain = F.softplus(self.cooling_gain_raw + .25 * torch.tanh(self.cooling_modulation(spray_local).squeeze(-1)))
        cooling_C = (effect * gain) @ self.spray_map.T * self.scale[:5]

        tau = self.log_stage_tau.exp() * torch.exp(.5 * torch.tanh(self.stage_rate(local).squeeze(-1)))
        alpha = -torch.expm1(-self.dt / tau)
        transported_C = self.link_gain(local).squeeze(-1).sigmoid() * upstream_C
        rebound_C = self.rebound_scale * torch.tanh(self.rebound(memory).squeeze(-1)) * self.scale[:5]
        response_next = response_C + alpha * (transported_C - cooling_C + rebound_C - response_C)
        memory_input = torch.cat((memory, own[..., None], upstream[..., None]), -1)
        goal = torch.tanh(self.hidden_target(memory_input)) * self.hidden_gate(local).sigmoid()
        alpha_h = -torch.expm1(-self.dt / self.log_hidden_tau.exp())
        memory_next = memory + alpha_h[None, :, None] * (goal - memory)
        return response_next, memory_next, modes_next

    def forward(self, history, actions, boundaries):
        initial, _ = self.encoder(self.normalize(history))
        hidden_initial = initial[:, 5:]
        hidden = hidden_initial
        reference_action = (history[:, -1, 5:7] - self.mean[5:7]) / self.scale[5:7]
        normalized_actions = (actions - self.mean[5:7]) / self.scale[5:7]
        normalized_boundaries = (boundaries - self.mean[7:]) / self.scale[7:]
        response_C = history.new_zeros((len(history), 5))
        response_memory = history.new_zeros((len(history), 5, self.local_size))
        modes = history.new_zeros((len(history), 2, 4))
        predictions, representations, references, responses = [], [], [], []
        for step in range(actions.shape[1]):
            boundary = normalized_boundaries[:, step]
            hidden = self.reference_transition(hidden, reference_action, boundary)
            reference_C = history[:, -1, :5] + self.global_temperature(hidden - hidden_initial) * self.scale[:5]
            reference_normalized = (reference_C - self.mean[:5]) / self.scale[:5]
            response_C, response_memory, modes = self.response_transition(response_C, response_memory, modes, normalized_actions[:, step] - reference_action, boundary, reference_normalized, hidden)
            prediction = reference_C + response_C
            # JEPA observes the actual combined rollout (not a separate future
            # predictor or the action-independent reference temperature alone).
            latent = torch.cat(((prediction - self.mean[:5]) / self.scale[:5], hidden), -1)
            predictions.append(prediction)
            representations.append(latent)
            references.append(reference_C)
            responses.append(response_C)
        return {"prediction": torch.stack(predictions, 1), "latent": torch.stack(representations, 1), "initial_latent": initial, "reference_prediction": torch.stack(references, 1), "action_response_C": torch.stack(responses, 1)}
