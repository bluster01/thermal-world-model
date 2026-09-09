"""One event cache with disjoint evidence/prediction retention budgets."""

from dataclasses import dataclass

import torch

from src.world_model_streaming.memory import EventMemory, _floating
from src.world_model_streaming.model import HISTORY, OBSERVATION, PREDICTION


@dataclass(frozen=True)
class EventRetentionPolicy:
    """HISTORY and OBS share E; every PRED uses P. No unused quota is borrowed."""

    evidence_capacity: int = 64
    prediction_capacity: int = 64

    def __post_init__(self):
        for name in ('evidence_capacity', 'prediction_capacity'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.capacity < 1:
            raise ValueError('At least one event slot is required')

    @property
    def capacity(self):
        return self.evidence_capacity + self.prediction_capacity

    def as_dict(self):
        """Serialize alongside weights: a state_dict alone does not bind policy."""
        return dict(kind='same_class_fifo_v1', evidence_capacity=self.evidence_capacity,
                    prediction_capacity=self.prediction_capacity, cache_capacity=self.capacity,
                    borrow_unused_quota=False, evidence_kinds=['HISTORY', 'OBSERVATION'])


def validate_quota_memory(cache, kinds, policy):
    if not isinstance(policy, EventRetentionPolicy):
        raise ValueError('An EventRetentionPolicy is required')
    policy.__post_init__()
    if not isinstance(cache, EventMemory):
        raise ValueError('An EventMemory is required')
    cache.__post_init__()
    if cache.tokens.shape[1] != policy.capacity:
        raise ValueError('Cache capacity must equal E plus P quotas')
    if (not isinstance(kinds, torch.Tensor) or kinds.dtype != torch.long or
            kinds.shape != cache.valid.shape or kinds.device != cache.tokens.device):
        raise ValueError('Kinds require matching int64 (B,K) metadata')
    if (bool((kinds[~cache.valid] != -1).any()) or
            bool(((kinds[cache.valid] < HISTORY) | (kinds[cache.valid] > OBSERVATION)).any())):
        raise ValueError('Invalid event-kind metadata')
    evidence = (kinds == HISTORY) | (kinds == OBSERVATION)
    prediction = kinds == PREDICTION
    if (bool((evidence.sum(1) > policy.evidence_capacity).any()) or
            bool((prediction.sum(1) > policy.prediction_capacity).any())):
        raise ValueError('A retained event class exceeds its quota')


def append_quota_event(cache, kinds, token, time, active, kind, policy):
    """Append causally; evict the oldest same-class event and compact stably.

    Equal times preserve append order, not a sort on floating timestamps.
    Gather/where copy storage without detaching retained tokens. A zero quota
    suppresses that class's write, while still validating every input.
    """
    validate_quota_memory(cache, kinds, policy)
    b, capacity, dim = cache.tokens.shape
    if isinstance(kind, bool) or not isinstance(kind, int) or kind not in (HISTORY, PREDICTION, OBSERVATION):
        raise ValueError('Unknown event kind')
    _floating(token, 'event token', cache.tokens)
    _floating(time, 'event time', cache.times)
    if token.shape != (b, dim) or time.shape != (b,):
        raise ValueError('Event token/time require matching (B,D)/(B) shapes')
    if (not isinstance(active, torch.Tensor) or active.dtype != torch.bool or
            active.shape != (b,) or active.device != cache.tokens.device):
        raise ValueError('Active requires boolean (B) on the memory device')
    if bool((active & cache.valid[:, -1] & (time < cache.times[:, -1])).any()):
        raise ValueError('An appended event cannot move time backwards')
    prediction = kind == PREDICTION
    quota = policy.prediction_capacity if prediction else policy.evidence_capacity
    same_class = kinds == PREDICTION if prediction else (kinds == HISTORY) | (kinds == OBSERVATION)
    writing = active & (quota > 0)
    oldest = same_class.to(torch.long).argmax(1)
    index = torch.arange(capacity, device=kinds.device)[None]
    evict = writing & (same_class.sum(1) == quota)
    keep = cache.valid & ~(evict[:, None] & (index == oldest[:, None]))
    keep = torch.cat([keep, writing[:, None]], 1)
    # Invalid candidates sort left. Remaining integer positions retain the
    # original append order, including PRED then OBS at an identical timestamp.
    positions = torch.arange(capacity + 1, device=kinds.device)[None].expand(b, -1)
    order = torch.where(keep, positions, -1).argsort(dim=1, stable=True)[:, -capacity:]
    valid = keep.gather(1, order)
    tokens = torch.cat([cache.tokens, token[:, None]], 1).gather(1, order[:, :, None].expand(-1, -1, dim))
    times = torch.cat([cache.times, time[:, None]], 1).gather(1, order)
    types = torch.cat([kinds, torch.full_like(kinds[:, :1], kind)], 1).gather(1, order)
    result = EventMemory(torch.where(valid[:, :, None], tokens, torch.zeros_like(tokens)),
                         torch.where(valid, times, torch.zeros_like(times)), valid)
    types = torch.where(valid, types, -1)
    validate_quota_memory(result, types, policy)
    return result, types
