"""Bounded event memory and physical-query reads for a streaming prototype.

Memory is explicit, copyable, and differentiable across appends. The reader is
ordinary attention with a learned nonnegative age decay, not a calibrated
filter or an established improvement on industrial control tasks.
Fixed cache capacity bounds stored event slots, not the size of the retained
autograd graph across a long unroll; this module intentionally never detaches.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _floating(value, name, reference=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} requires a floating-point tensor')
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f'{name} must match the reference dtype and device')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


@dataclass(frozen=True)
class EventMemory:
    """B,K,D tokens with left-padded, contiguous valid suffixes in each row.

    K is fixed capacity. Token and time floating dtypes may differ, but their
    devices must agree. All tensors must be finite; invalid token/time slots
    must be exactly zero. Valid times may repeat but may never decrease. Frozen
    fields do not freeze tensor storage, so operations revalidate their input.
    """

    tokens: torch.Tensor
    times: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self):
        _validate_memory(self)


def _validate_memory(cache):
    if not isinstance(cache, EventMemory):
        raise ValueError('cache must be an EventMemory')
    _floating(cache.tokens, 'memory tokens')
    if cache.tokens.ndim != 3 or any(size < 1 for size in cache.tokens.shape):
        raise ValueError('memory tokens require nonempty (B,K,D) shape')
    _floating(cache.times, 'memory times')
    if cache.times.device != cache.tokens.device:
        raise ValueError('memory times must match the token device')
    if cache.times.shape != cache.tokens.shape[:2]:
        raise ValueError('memory times require matching (B,K) shape')
    if (not isinstance(cache.valid, torch.Tensor) or cache.valid.dtype != torch.bool or
            cache.valid.shape != cache.times.shape or cache.valid.device != cache.tokens.device):
        raise ValueError('memory valid requires boolean (B,K) on the memory device')
    if bool((cache.tokens[~cache.valid] != 0).any()) or bool((cache.times[~cache.valid] != 0).any()):
        raise ValueError('Invalid memory slots must have zero tokens and times')
    if bool((cache.valid[:, :-1] & ~cache.valid[:, 1:]).any()):
        raise ValueError('Valid memory entries must form a contiguous suffix')
    pairs = cache.valid[:, :-1] & cache.valid[:, 1:]
    if bool((pairs & (cache.times[:, 1:] < cache.times[:, :-1])).any()):
        raise ValueError('Valid memory times must be nondecreasing')


def empty_memory(batch: int, capacity: int, dim: int, reference: torch.Tensor) -> EventMemory:
    """Allocate invalid tokens matching reference, with float64 event times."""
    for name, value in (('batch', batch), ('capacity', capacity), ('dim', dim)):
        _positive_integer(value, name)
    _floating(reference, 'reference')
    return EventMemory(reference.new_zeros((batch, capacity, dim)),
                       torch.zeros(batch, capacity, dtype=torch.float64, device=reference.device),
                       torch.zeros(batch, capacity, dtype=torch.bool, device=reference.device))


def append_event(cache: EventMemory, token: torch.Tensor, time: torch.Tensor,
                 active: torch.Tensor) -> EventMemory:
    """Append on active rows, shifting their fixed-capacity buffers left once.

    Inactive rows keep every tensor value unchanged. Equal event timestamps are
    permitted, for example advance and observe at the same physical time. No
    argument is modified or detached. Even ignored inputs must be finite.
    """
    _validate_memory(cache)
    b, _, d = cache.tokens.shape
    _floating(token, 'event token', cache.tokens)
    _floating(time, 'event time', cache.times)
    if token.shape != (b, d) or time.shape != (b,):
        raise ValueError('event token/time require matching (B,D)/(B) shapes')
    if (not isinstance(active, torch.Tensor) or active.shape != (b,) or
            active.dtype != torch.bool or active.device != cache.tokens.device):
        raise ValueError('active requires boolean (B) on the memory device')
    if bool((active & cache.valid[:, -1] & (time < cache.times[:, -1])).any()):
        raise ValueError('An appended event cannot move time backwards')
    tokens = torch.cat([cache.tokens[:, 1:], token[:, None]], dim=1)
    times = torch.cat([cache.times[:, 1:], time[:, None]], dim=1)
    valid = torch.cat([cache.valid[:, 1:], torch.ones_like(active[:, None])], dim=1)
    return EventMemory(torch.where(active[:, None, None], tokens, cache.tokens),
                       torch.where(active[:, None], times, cache.times),
                       torch.where(active[:, None], valid, cache.valid))


def clone_memory(cache: EventMemory) -> EventMemory:
    """Clone every tensor while preserving autograd links to the source cache."""
    _validate_memory(cache)
    return EventMemory(cache.tokens.clone(), cache.times.clone(), cache.valid.clone())


def select_memory(cache: EventMemory, indices: torch.Tensor) -> EventMemory:
    """Copy selected batch rows, including duplicates, without sharing storage."""
    _validate_memory(cache)
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.ndim != 1 or
            indices.numel() < 1 or indices.device != cache.tokens.device):
        raise ValueError('indices require a nonempty int64 vector on the memory device')
    if bool(((indices < 0) | (indices >= cache.tokens.shape[0])).any()):
        raise ValueError('Memory selection index is out of bounds')
    return EventMemory(cache.tokens.index_select(0, indices), cache.times.index_select(0, indices),
                       cache.valid.index_select(0, indices))


def repeat_memory(cache: EventMemory, repeats: int) -> EventMemory:
    """Repeat each row consecutively (repeat_interleave order), copying storage."""
    _validate_memory(cache)
    _positive_integer(repeats, 'repeats')
    indices = torch.arange(cache.tokens.shape[0], device=cache.tokens.device).repeat_interleave(repeats)
    return select_memory(cache, indices)


@dataclass(frozen=True)
class AttentionReadout:
    value: torch.Tensor        # B,D
    weights: torch.Tensor      # B,heads,K+1; last position is a fixed zero null
    valid_count: torch.Tensor  # B; int64 count of actual cached events


class PhysicalMemoryReader(nn.Module):
    """Read events by content plus age, age only, a mean, or exact zero.

    Attention logits are dot(query,key)/sqrt(head_dim) minus
    softplus(raw_decay)*log1p(age_seconds). Decay is learned separately per head;
    its initial value is softplus(0). This fixes the time reference scale to one
    second. Keys and values of an always-valid final null slot are fixed zeros.

    Age mode omits the query-key content term but keeps the same trainable decay,
    value/output projections and null softmax slot. Query/key projections remain
    stored for state-dict compatibility but are frozen and never evaluated.
    Mean mode uniformly averages valid events PLUS the null through the SAME
    value/output projection structure, so each gets 1/(event_count+1). None
    mode always puts weight one on null and returns exact zeros. Unused query,
    key, and decay parameters are frozen in mean; all projections are frozen in
    none. Their stored tensor shapes stay compatible across read-mode ablations.
    Times and `now` share a floating dtype independent of token/model dtype;
    empty_memory defaults to float64 times. Age/log1p is computed in that time
    dtype, then explicitly cast to the logits dtype. No token/query casts or
    dropout are used. Relative negative times and repeated zero times are valid.
    """

    def __init__(self, query_dim: int, hidden_dim: int, heads: int = 4, mode: str = 'attention'):
        super().__init__()
        for name, value in (('query_dim', query_dim), ('hidden_dim', hidden_dim), ('heads', heads)):
            _positive_integer(value, name)
        if hidden_dim % heads:
            raise ValueError('hidden_dim must be divisible by heads')
        if mode not in ('attention', 'age', 'mean', 'none'):
            raise ValueError('mode must be attention, age, mean or none')
        self.query_dim, self.hidden_dim, self.heads, self.mode = query_dim, hidden_dim, heads, mode
        self.query_projection = nn.Linear(query_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.raw_decay = nn.Parameter(torch.zeros(heads))
        if mode != 'attention':
            self.query_projection.requires_grad_(False)
            self.key_projection.requires_grad_(False)
        if mode in ('mean', 'none'):
            self.raw_decay.requires_grad_(False)
        if mode == 'none':
            self.value_projection.requires_grad_(False)
            self.output_projection.requires_grad_(False)

    def forward(self, query: torch.Tensor, cache: EventMemory, now: torch.Tensor) -> AttentionReadout:
        _validate_memory(cache)
        b, k, d = cache.tokens.shape
        reference = self.query_projection.weight
        _floating(cache.tokens, 'memory tokens', reference)
        _floating(query, 'physical query', reference)
        _floating(now, 'now', cache.times)
        if d != self.hidden_dim or query.shape != (b, self.query_dim) or now.shape != (b,):
            raise ValueError('Query, cache and now dimensions do not match the reader')
        if bool((cache.valid & (cache.times > now[:, None])).any()):
            raise ValueError('Memory contains a future event relative to now')
        count = cache.valid.sum(dim=1)
        if self.mode == 'none':
            weights = torch.cat([cache.tokens.new_zeros((b, self.heads, k)),
                                 cache.tokens.new_ones((b, self.heads, 1))], dim=-1)
            return AttentionReadout(cache.tokens.new_zeros((b, d)), weights, count)

        head_dim = d // self.heads
        values = self.value_projection(cache.tokens).reshape(b, k, self.heads, head_dim).transpose(1, 2)
        values = torch.cat([values, values.new_zeros((b, self.heads, 1, head_dim))], dim=2)
        if self.mode == 'mean':
            denominator = (count + 1).to(query.dtype)
            valid_weights = cache.valid.to(query.dtype) / denominator[:, None]
            weights = torch.cat([valid_weights, denominator.reciprocal()[:, None]], dim=-1)
            weights = weights[:, None].expand(b, self.heads, k + 1)
        else:
            if self.mode == 'age':
                logits = cache.tokens.new_zeros((b, self.heads, k))
            else:
                projected_query = self.query_projection(query).reshape(b, self.heads, head_dim)
                keys = self.key_projection(cache.tokens).reshape(b, k, self.heads, head_dim).transpose(1, 2)
                logits = (projected_query[:, :, None] * keys).sum(dim=-1) / math.sqrt(head_dim)
            # A negative `now` is allowed. Zero invalid ages BEFORE log1p;
            # zero-padding timestamps must not create a negative-age NaN.
            age = torch.where(cache.valid, now[:, None] - cache.times, torch.zeros_like(cache.times))
            if not bool(torch.isfinite(age).all()):
                raise ValueError('Event age exceeds the finite range of the memory dtype')
            log_age = torch.log1p(age).to(dtype=logits.dtype)
            logits = logits - F.softplus(self.raw_decay)[None, :, None] * log_age[:, None]
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Attention logits are not finite')
            logits = logits.masked_fill(~cache.valid[:, None], -torch.inf)
            logits = torch.cat([logits, logits.new_zeros((b, self.heads, 1))], dim=-1)
            weights = torch.softmax(logits, dim=-1)
        joined = (weights[:, :, :, None] * values).sum(dim=2).reshape(b, d)
        value = self.output_projection(joined)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Memory readout is not finite')
        return AttentionReadout(value, weights, count)
