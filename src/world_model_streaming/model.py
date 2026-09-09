"""Physical dynamics with persistent, typed event-memory retrieval.

Attention reads the old information state before a new action transition. The
new predicted event is written afterwards; observations contribute only after
arrival. Stored predictions remain explicitly different from measured events.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from src.world_model_vnext.contracts import PointBelief
from src.world_model_vnext.model import PersistentPhysicsWorldModel
from .memory import EventMemory, PhysicalMemoryReader, append_event, empty_memory, select_memory


HISTORY, PREDICTION, OBSERVATION = 0, 1, 2


@dataclass(frozen=True)
class StreamingBelief(PointBelief):
    events: EventMemory | None = None
    event_kinds: torch.Tensor | None = None  # B,K int64; -1 for invalid slots


class StreamingPhysicsWorldModel(PersistentPhysicsWorldModel):
    """Same physical ports and point belief, with online cache reads and writes.

    Cache capacity counts events, not seconds. One transition and its subsequent
    measurement can occupy two slots at the same physical time. Float64 event
    times are relative to the CURRENT belief time; every transition ages old
    entries by dt. This avoids an ever-growing float32 absolute clock.

    Content attention, uniform mean, and zero-context modes use the same cache
    content. Cache entries are embeddings, not preprojected K/V: current reader
    projections remain trainable. Bounded storage does NOT bound the autograd
    graph; detach_belief is an explicit whole-state TBPTT boundary.
    """

    def __init__(self, backend, *, hidden_dim=64, heads=4, layers=2, correction_fraction=.1,
                 encoder_kind='gru', use_neural_residual=True, cache_capacity=128,
                 read_mode='attention', store_predictions=True):
        super().__init__(backend, hidden_dim=hidden_dim, heads=heads, layers=layers,
                         correction_fraction=correction_fraction, encoder_kind=encoder_kind,
                         use_neural_residual=use_neural_residual)
        if isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int) or cache_capacity < 1:
            raise ValueError('cache_capacity must be a positive integer')
        if not isinstance(store_predictions, bool):
            raise ValueError('store_predictions must be boolean')
        dt = getattr(backend, 'dt_seconds', None)
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('The physical backend must declare a positive finite dt_seconds')
        self.cache_capacity = cache_capacity
        self.store_predictions = store_predictions
        self.dt_seconds = float(dt)
        s, a, o, c = backend.state_dim, backend.action_dim, backend.observation_dim, backend.allowed_boundary_indices.numel()
        self.reader = PhysicalMemoryReader(s + hidden_dim + c, hidden_dim, heads, read_mode)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_gate = nn.Parameter(torch.tensor(-2.))
        # State, actually applied/planned action, allowed boundary, innovation,
        # measurement mask, and an explicit action-present bit.
        self.event_projection = nn.Linear(s + a + c + 2 * o + 1, hidden_dim)
        self.event_type_embedding = nn.Embedding(3, hidden_dim)
        if read_mode == 'none':
            self.context_projection.requires_grad_(False)
            self.context_gate.requires_grad_(False)
            self.event_projection.requires_grad_(False)
            self.event_type_embedding.requires_grad_(False)

    def _belief(self, belief):
        if not isinstance(belief, StreamingBelief):
            raise ValueError('Streaming model requires a complete StreamingBelief including its event cache')
        if getattr(self.backend, 'dt_seconds', None) != self.dt_seconds:
            raise ValueError('Backend sampling interval changed relative to the event-memory clock')
        super()._belief(belief)
        if not isinstance(belief.events, EventMemory):
            raise ValueError('Streaming belief is missing event memory')
        belief.events.__post_init__()
        b = belief.physical.shape[0]
        if belief.events.tokens.shape != (b, self.cache_capacity, self.hidden_dim):
            raise ValueError('Event cache dimensions do not match this model')
        self._numeric_contract(belief.events.tokens, 'event tokens')
        if belief.events.times.dtype != torch.float64:
            raise ValueError('Streaming event times must use float64 seconds')
        if bool((belief.events.times[belief.events.valid] > 0).any()):
            raise ValueError('Streaming belief contains future event times')
        kinds = belief.event_kinds
        if (not isinstance(kinds, torch.Tensor) or kinds.dtype != torch.long or
                kinds.shape != (b, self.cache_capacity) or kinds.device != belief.physical.device):
            raise ValueError('Event kinds require matching int64 (B,K) metadata')
        if (bool((kinds[~belief.events.valid] != -1).any()) or
                bool(((kinds[belief.events.valid] < HISTORY) | (kinds[belief.events.valid] > OBSERVATION)).any())):
            raise ValueError('Invalid event-kind metadata')

    def initialize(self, physical_anchor, history_values, history_mask, elapsed_seconds):
        self._finite_matrix(physical_anchor, self.backend.state_dim, 'physical anchor')
        self._numeric_contract(history_values, 'history values')
        if (not isinstance(history_mask, torch.Tensor) or history_mask.shape != history_values.shape or
                history_values.ndim != 3 or history_mask.dtype != torch.bool or
                history_mask.device != history_values.device):
            raise ValueError('History and mask require matching (B,T,F) shapes/device')
        b, length, features = history_values.shape
        if b != physical_anchor.shape[0] or features != self.backend.feature_loc.numel():
            raise ValueError('History dimensions do not match the physical backend')
        clean = torch.where(history_mask, history_values, self.backend.feature_loc)
        tokens = self.encoder((clean - self.backend.feature_loc) / self.backend.feature_scale,
                              history_mask, elapsed_seconds)
        memory = tokens[:, -1]
        state = physical_anchor + self._correction(self.initial_correction(memory))
        seen = history_mask[:, -1, :self.backend.observation_dim].any(-1)
        keep = min(length, self.cache_capacity)
        reference = physical_anchor
        cache = empty_memory(b, self.cache_capacity, self.hidden_dim, reference)
        # HISTORY tokens represent available historical features, including masks
        # and past actions; they are NOT invented measurement innovations.
        values = tokens[:, -keep:] + self.event_type_embedding.weight[HISTORY]
        time64 = elapsed_seconds.to(torch.float64)
        relative_time = time64[:, -keep:] - time64[:, -1:]
        pad = self.cache_capacity - keep
        cache = EventMemory(torch.cat([cache.tokens[:, :pad], values], 1),
                            torch.cat([cache.times[:, :pad], relative_time], 1),
                            torch.cat([cache.valid[:, :pad], torch.ones(b, keep, device=reference.device, dtype=torch.bool)], 1))
        kinds = torch.cat([torch.full((b, pad), -1, dtype=torch.long, device=reference.device),
                           torch.full((b, keep), HISTORY, dtype=torch.long, device=reference.device)], 1)
        result = StreamingBelief(state, memory, 0, seen, cache, kinds)
        self._belief(result)
        return result

    def inspect_memory(self, belief, boundary):
        """Return readout and weights; weights are retrieval diagnostics, not causes."""
        self._belief(belief)
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', belief.physical.shape[0])
        query = torch.cat([self._state_features(belief.physical), belief.memory,
                           self.backend.allowed_boundary(boundary)], -1)
        return self.reader(query, belief.events, belief.events.times.new_zeros(belief.physical.shape[0]))

    def _retrieved_memory(self, belief, boundary):
        context = self.inspect_memory(belief, boundary).value
        # Bias-free transform makes zero-context mode exactly the original GRU
        # path. The bounded additive update is distinct from a Bayesian gain.
        return belief.memory + torch.sigmoid(self.context_gate) * torch.tanh(self.context_projection(context))

    def _power_with_memory(self, physical, memory, boundary):
        if not self.use_neural_residual:
            return physical.new_zeros((physical.shape[0], self.backend.residual_dim))
        value = self.power_head(torch.cat([self._state_features(physical), memory,
                                          self.backend.allowed_boundary(boundary)], -1))
        return self.backend.residual_scale * torch.tanh(value)

    def residual_power(self, belief, boundary):
        return self._power_with_memory(belief.physical, self._retrieved_memory(belief, boundary), boundary)

    def _event_token(self, state, action, boundary, innovation, mask, kind):
        b = state.shape[0]
        if action is None:
            actions = state.new_zeros((b, self.backend.action_dim))
            present = state.new_zeros((b, 1))
        else:
            actions = (action - self.backend.action_loc) / self.backend.action_scale
            present = state.new_ones((b, 1))
        features = torch.cat([self._state_features(state), actions, self.backend.allowed_boundary(boundary),
                              innovation, mask.to(state.dtype), present], -1)
        return self.event_projection(features) + self.event_type_embedding.weight[kind]

    def _append(self, cache, kinds, token, active, kind):
        cache = append_event(cache, token, cache.times.new_zeros(token.shape[0]), active)
        shifted = torch.cat([kinds[:, 1:], torch.full_like(kinds[:, :1], kind)], 1)
        return cache, torch.where(active[:, None], shifted, kinds)

    def advance(self, belief, action, boundary):
        self._belief(belief)
        b = belief.physical.shape[0]
        self._finite_matrix(action, self.backend.action_dim, 'action', b)
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', b)
        if bool(((action < 0) | (action > 1)).any()):
            raise ValueError('Action fractions must be in [0,1]')
        memory = self._retrieved_memory(belief, boundary)  # old cache, before u_t is appended
        residual = self._power_with_memory(belief.physical, memory, boundary)
        state = self.backend.advance(belief.physical, action, boundary, residual)
        action_features = (action - self.backend.action_loc) / self.backend.action_scale
        memory = self.transition_memory(torch.cat([self._state_features(state), action_features,
                                                   self.backend.allowed_boundary(boundary)], -1), memory)
        cache = EventMemory(belief.events.tokens,
                            torch.where(belief.events.valid, belief.events.times - self.dt_seconds, belief.events.times),
                            belief.events.valid)
        kinds = belief.event_kinds
        if self.store_predictions:
            zero = state.new_zeros((b, self.backend.observation_dim))
            token = self._event_token(state, action, boundary, zero, zero.bool(), PREDICTION)
            cache, kinds = self._append(cache, kinds, token, torch.ones(b, dtype=torch.bool, device=state.device), PREDICTION)
        result = StreamingBelief(state, memory, belief.steps + 1,
                                 torch.zeros(b, dtype=torch.bool, device=state.device), cache, kinds)
        self._belief(result)
        return result

    def observe(self, prior, observation, observed_mask, boundary, *, step_index):
        self._belief(prior)
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index != prior.steps:
            raise ValueError('Observation step must match the prior step index')
        b, o = prior.physical.shape[0], self.backend.observation_dim
        self._numeric_contract(observation, 'observation')
        if (observation.shape != (b, o) or not isinstance(observed_mask, torch.Tensor) or
                observed_mask.shape != observation.shape or observed_mask.dtype != torch.bool or
                observed_mask.device != observation.device):
            raise ValueError('Observation and mask require matching (B,O) shapes/device')
        if not bool(torch.isfinite(observation[observed_mask]).all()):
            raise ValueError('Observed measurements must be finite')
        self._finite_matrix(boundary, self.backend.boundary_dim, 'boundary', b)
        active = observed_mask.any(-1)
        seen = torch.zeros_like(active) if prior.observed_at_step is None else prior.observed_at_step
        if bool((active & seen).any()):
            raise ValueError('A measurement update was already applied at this step')
        if not bool(active.any()):
            return prior
        prediction = self.backend.decode(prior.physical, boundary)
        self._finite_matrix(prediction, o, 'decoded prior observation', b)
        innovation = (torch.where(observed_mask, observation, prediction) - prediction) / self.backend.observation_scale
        features = self._state_features(prior.physical)
        memory = self.observation_memory(torch.cat([features, innovation, observed_mask.to(features.dtype),
                                                    self.backend.allowed_boundary(boundary)], -1),
                                          self._retrieved_memory(prior, boundary))
        correction = self._correction(self.observation_correction(
            torch.cat([memory, features, innovation, observed_mask.to(features.dtype)], -1)))
        state = torch.where(active[:, None], prior.physical + correction, prior.physical)
        memory = torch.where(active[:, None], memory, prior.memory)
        token = self._event_token(state, None, boundary, innovation, observed_mask, OBSERVATION)
        cache, kinds = self._append(prior.events, prior.event_kinds, token, active, OBSERVATION)
        result = StreamingBelief(state, memory, prior.steps, seen | active, cache, kinds)
        self._belief(result)
        return result

    def select_belief(self, belief, indices):
        """Copy complete candidate rows, including cache, provenance and flags."""
        self._belief(belief)
        cache = select_memory(belief.events, indices)
        result = StreamingBelief(belief.physical.index_select(0, indices), belief.memory.index_select(0, indices),
                                 belief.steps, None if belief.observed_at_step is None else
                                 belief.observed_at_step.index_select(0, indices), cache,
                                 belief.event_kinds.index_select(0, indices))
        self._belief(result)
        return result

    def repeat_belief(self, belief, repeats):
        self._belief(belief)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError('repeats must be a positive integer')
        indices = torch.arange(belief.physical.shape[0], device=belief.physical.device).repeat_interleave(repeats)
        return self.select_belief(belief, indices)

    def clone_belief(self, belief):
        """Copy the complete information state without truncating its gradients."""
        self._belief(belief)
        return self.select_belief(belief, torch.arange(belief.physical.shape[0], device=belief.physical.device))

    def detach_belief(self, belief):
        """Detach AND copy all fields at an explicit global training boundary.

        This does not make a stale belief valid after changing model parameters.
        Reinitialize/replay observations when the model weights are updated.
        """
        self._belief(belief)
        clone = lambda value: value.detach().clone()
        return StreamingBelief(clone(belief.physical), clone(belief.memory), belief.steps,
                               None if belief.observed_at_step is None else clone(belief.observed_at_step),
                               EventMemory(clone(belief.events.tokens), clone(belief.events.times), clone(belief.events.valid)),
                               clone(belief.event_kinds))
