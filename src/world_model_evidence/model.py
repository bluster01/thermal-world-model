"""Evidence retention changes cache composition, leaving the physical reader intact."""

import torch

from src.world_model_actuated import ActuatedStreamingBelief, ActuatedStreamingPhysicsWorldModel
from src.world_model_streaming.memory import EventMemory, empty_memory
from src.world_model_streaming.model import HISTORY, StreamingBelief, StreamingPhysicsWorldModel
from .memory import EventRetentionPolicy, append_quota_event, validate_quota_memory


class _EvidenceQuotaMixin:
    """Shared write/validation/history policy; no parameters or physics formulas."""

    def __init__(self, backend, *, cache_capacity=128, evidence_capacity=64,
                 prediction_capacity=64, **model_kwargs):
        policy = EventRetentionPolicy(evidence_capacity, prediction_capacity)
        if isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int) or cache_capacity != policy.capacity:
            raise ValueError('cache_capacity must equal evidence_capacity plus prediction_capacity')
        super().__init__(backend, cache_capacity=cache_capacity, **model_kwargs)
        self.retention_policy = policy

    def _belief(self, belief):
        super()._belief(belief)
        validate_quota_memory(belief.events, belief.event_kinds, self.retention_policy)

    def _append(self, cache, kinds, token, active, kind):
        return append_quota_event(cache, kinds, token, cache.times.new_zeros(token.shape[0]),
                                  active, kind, self.retention_policy)

    def _initialize_history(self, reference, history_values, history_mask, elapsed_seconds):
        self._numeric_contract(history_values, 'history values')
        if (not isinstance(history_mask, torch.Tensor) or history_mask.shape != history_values.shape or
                history_values.ndim != 3 or history_mask.dtype != torch.bool or
                history_mask.device != history_values.device):
            raise ValueError('History and mask require matching (B,T,F) shapes/device')
        b, length, features = history_values.shape
        if b != reference.shape[0] or features != self.backend.feature_loc.numel():
            raise ValueError('History dimensions do not match the physical backend')
        clean = torch.where(history_mask, history_values, self.backend.feature_loc)
        # Encoder sees the full allowed history, irrespective of either quota.
        tokens = self.encoder((clean - self.backend.feature_loc) / self.backend.feature_scale,
                              history_mask, elapsed_seconds)
        memory = tokens[:, -1]
        correction = self._correction(self.initial_correction(memory))
        seen = history_mask[:, -1, :self.backend.observation_dim].any(-1)
        keep = min(length, self.retention_policy.evidence_capacity)
        cache = empty_memory(b, self.cache_capacity, self.hidden_dim, reference)
        kinds = torch.full((b, self.cache_capacity), -1, dtype=torch.long, device=reference.device)
        if keep:
            values = tokens[:, -keep:] + self.event_type_embedding.weight[HISTORY]
            time64 = elapsed_seconds.to(torch.float64)
            times = time64[:, -keep:] - time64[:, -1:]
            pad = self.cache_capacity - keep
            cache = EventMemory(torch.cat([cache.tokens[:, :pad], values], 1),
                                torch.cat([cache.times[:, :pad], times], 1),
                                torch.cat([cache.valid[:, :pad], torch.ones(b, keep, device=reference.device, dtype=torch.bool)], 1))
            kinds = torch.cat([kinds[:, :pad], torch.full((b, keep), HISTORY, dtype=torch.long, device=reference.device)], 1)
        return memory, correction, seen, cache, kinds


class EvidenceStreamingPhysicsWorldModel(_EvidenceQuotaMixin, StreamingPhysicsWorldModel):
    """Streaming world model with one reader and separate E/P FIFO quotas."""

    def initialize(self, physical_anchor, history_values, history_mask, elapsed_seconds):
        self._finite_matrix(physical_anchor, self.backend.state_dim, 'physical anchor')
        memory, correction, seen, cache, kinds = self._initialize_history(
            physical_anchor, history_values, history_mask, elapsed_seconds)
        result = StreamingBelief(physical_anchor + correction, memory, 0, seen, cache, kinds)
        self._belief(result)
        return result


class ActuatedEvidenceStreamingPhysicsWorldModel(_EvidenceQuotaMixin, ActuatedStreamingPhysicsWorldModel):
    """The same quota policy, preserving the independent actuator graph."""

    def initialize_from_parts(self, thermal_anchor, actual_positions, history_values,
                              history_mask, elapsed_seconds):
        self._actuator_contract()
        self._finite_matrix(thermal_anchor, self.inner_state_dim, 'thermal anchor')
        self._finite_matrix(actual_positions, self.backend.action_dim, 'initial actual positions',
                            thermal_anchor.shape[0])
        self.backend._physical(torch.cat([thermal_anchor, actual_positions], -1))
        memory, correction, seen, cache, kinds = self._initialize_history(
            thermal_anchor, history_values, history_mask, elapsed_seconds)
        actual = actual_positions.clone()
        thermal = thermal_anchor + correction[:, :self.inner_state_dim]
        result = ActuatedStreamingBelief(torch.cat([thermal, actual], -1), memory, 0, seen, cache, kinds, actual)
        self._belief(result)
        return result
