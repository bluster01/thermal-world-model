"""History-only comparison models for unified industrial screening, v1.

All forward methods read exactly four history fields. No measured future action,
boundary, label, mask, or timestamp is consumed. Lead queries use t / 60, so H180
means extrapolation of an H60-trained map, never interpolation/rescaling to H180.

The mechanism is an equivalent-state hypothesis. Thermal temperatures are put
on ONE common affine scale before paired exchange; capacities are dimensionless
equivalent capacities, not identified equipment heat capacities. Only each
measured node and its own latent stores exchange heat: no process chain is
invented from temperature tag names, including the unconfirmed reheat topology.
External relaxation ports and sensor lag are explicit. SCR uses nonnegative NO
concentration proxies, declared source/removal ports, and same-side inlet-to-
outlet transport. NO destruction is an allowed sink: the sum of the four NO
readings is not conserved. Reagent histories condition coefficients jointly;
there is no assumed assignment of the six reagent groups to A/B.

Neither thermal nor SCR forecasts establish intervention monotonicity: their
coefficients depend on the observation history. Slow-state ablation removes
latent stores while retaining the observer, external ports, and sensor filter.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARMS = ("gru_direct", "ssm_rollout", "itransformer", "mechanism", "mechanism_no_slow")
HISTORY_KEYS = ("history_normalized", "history_valid", "history_observed", "history_age_s")


def _lead_features(horizon: int, reference: Tensor) -> Tensor:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    t = torch.arange(1, horizon + 1, device=reference.device, dtype=reference.dtype) / 60.0
    return torch.stack((t, t.square(), torch.sin(t), torch.cos(t), 1 - torch.exp(-t)), -1)


def _vector(value, size, default):
    result = torch.full((size,), default, dtype=torch.float32) if value is None else torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (size,) or not torch.isfinite(result).all():
        raise ValueError(f"Expected a finite vector of length {size}")
    return result.clone()


class HistoryModel(nn.Module):
    def __init__(self, input_names: Sequence[str], target_names: Sequence[str], task_name: str,
                 hidden=64, *, target_input_indices=None, target_center=None,
                 target_scale=None, input_center=None, input_scale=None,
                 history_length=360, history_stride=6, task_spec=None, dt_seconds=10.0,
                 **unused_metadata):
        super().__init__()
        self.input_names, self.target_names = list(input_names), list(target_names)
        self.task_name, self.hidden = task_name, int(hidden)
        self.channels, self.outputs = len(input_names), len(target_names)
        self.history_length, self.history_stride = int(history_length), int(history_stride)
        self.task_spec = task_spec or {}
        self.dt_seconds = float(dt_seconds)
        if min(self.channels, self.outputs, self.hidden, self.history_length, self.history_stride) < 1:
            raise ValueError("Positive channel, hidden, and history sizes are required")
        if not math.isfinite(self.dt_seconds) or self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be finite and positive")
        indices = ([self.input_names.index(name) for name in target_names]
                   if target_input_indices is None else list(target_input_indices))
        if len(indices) != self.outputs or any(i < 0 or i >= self.channels for i in indices):
            raise ValueError("target_input_indices must locate every target in input_names")
        self.register_buffer("target_indices", torch.tensor(indices, dtype=torch.long))
        if target_center is None and input_center is not None:
            target_center = torch.as_tensor(input_center)[indices]
        if target_scale is None and input_scale is not None:
            target_scale = torch.as_tensor(input_scale)[indices]
        self.register_buffer("target_center", _vector(target_center, self.outputs, 0.0))
        self.register_buffer("target_scale", _vector(target_scale, self.outputs, 1.0))
        if (self.target_scale <= 0).any():
            raise ValueError("Target normalization scales must be positive")

    def history(self, batch: Mapping[str, Tensor]):
        x = batch["history_normalized"]
        if x.ndim != 3 or x.shape[1:] != (self.history_length, self.channels):
            raise ValueError("history_normalized must have shape [B, history_length, input_channels]")
        valid = batch["history_valid"].bool() & torch.isfinite(x)
        observed = batch["history_observed"].bool() & valid
        age = torch.nan_to_num(batch["history_age_s"], nan=3600.0, posinf=3600.0, neginf=0.0)
        age = torch.log1p(age.clamp(0, 3600) / 60.0) / math.log(61.0)
        clean = torch.where(valid, x, torch.zeros_like(x))
        feature = torch.cat((clean, valid.to(x.dtype), observed.to(x.dtype), age.to(x.dtype)), -1)
        target_valid, target_values = valid.index_select(-1, self.target_indices), clean.index_select(-1, self.target_indices)
        position = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)
        last_position = torch.where(target_valid, position, -1).amax(1)
        anchor = target_values.gather(1, last_position.clamp_min(0).unsqueeze(1)).squeeze(1)
        anchor = torch.where(last_position >= 0, anchor, torch.zeros_like(anchor))
        return feature, anchor

    def description(self):
        return {"task_name": self.task_name, "input_names": self.input_names,
                "target_names": self.target_names, "history_length": self.history_length,
                "history_stride": self.history_stride, "future_inputs": "none",
                "native_training_horizon_steps": 60, "dt_seconds": self.dt_seconds,
                "trainable_parameters": sum(p.numel() for p in self.parameters())}


class GRUObserver(nn.Module):
    """All history values enter causal fixed-block mean pooling before the GRU.

    Pools values AND validity/arrival/age features, retaining missingness evidence;
    the latest valid target anchor is extracted from the original unpooled input.
    """
    def __init__(self, channels, hidden, stride):
        super().__init__()
        self.stride = stride
        self.gru = nn.GRU(channels * 4, hidden, batch_first=True)

    def forward(self, feature):
        pooled = F.avg_pool1d(feature.transpose(1, 2), self.stride, self.stride,
                              ceil_mode=True).transpose(1, 2)
        return self.gru(pooled)[1][-1]


class GRUDirect(HistoryModel):
    """History GRU with a continuous lead-query output, trained at H60."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer = GRUObserver(self.channels, self.hidden, self.history_stride)
        self.head = nn.Sequential(nn.Linear(self.hidden + 5, self.hidden), nn.SiLU(),
                                  nn.Linear(self.hidden, self.outputs))

    def forward(self, batch, horizon=60):
        feature, anchor = self.history(batch)
        z = self.observer(feature)
        lead = _lead_features(horizon, z)
        query = torch.cat((z[:, None].expand(-1, horizon, -1), lead[None].expand(z.shape[0], -1, -1)), -1)
        # Residual approaches zero continuously at lead zero.
        return anchor[:, None] + self.head(query) * lead[:, :1][None]


class StableSSMRollout(HistoryModel):
    """Diagonal positive-time-constant recurrence with history-predicted drive.

    h+ = alpha*h + (1-alpha)*(b + amplitude*exp(-t/tau_drive)).
    Its time-varying exogenous drive is predicted solely at the history origin.
    Outputs are anchored changes of the hidden state, a black-box SSM baseline.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer = GRUObserver(self.channels, self.hidden, self.history_stride)
        self.drive = nn.Linear(self.hidden, self.hidden * 2)
        self.readout = nn.Linear(self.hidden, self.outputs, bias=False)
        self.log_tau = nn.Parameter(torch.linspace(math.log(30), math.log(1800), self.hidden))
        self.log_drive_tau = nn.Parameter(torch.tensor(math.log(600.0)))

    def forward(self, batch, horizon=60):
        _lead_features(horizon, batch["history_normalized"])
        feature, anchor = self.history(batch)
        initial = self.observer(feature)
        equilibrium, amplitude = self.drive(initial).chunk(2, -1)
        alpha = torch.exp(-self.dt_seconds / (10 + self.log_tau.exp()))
        tau_drive = 10 + self.log_drive_tau.exp()
        state, predictions = initial, []
        for step in range(1, horizon + 1):
            forcing = equilibrium + amplitude * torch.exp(-step * self.dt_seconds / tau_drive)
            state = alpha * state + (1 - alpha) * forcing
            predictions.append(anchor + self.readout(state - initial))
        return torch.stack(predictions, 1)


class InvertedTransformer(HistoryModel):
    """Local inverted-variable attention baseline; NOT an official reproduction.

    Each variable's complete value/mask/arrival/age history becomes one token.
    Two pre-norm self-attention layers mix variables. Per-target tokens receive
    continuous lead queries. No future labels, RevIN, or official checkpoint.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Linear(self.history_length * 4, self.hidden)
        heads = next(n for n in (4, 2, 1) if self.hidden % n == 0)
        layer = nn.TransformerEncoderLayer(self.hidden, heads, self.hidden * 2,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.channel_embedding = nn.Parameter(torch.zeros(1, self.channels, self.hidden))
        self.head = nn.Sequential(nn.Linear(self.hidden + 5, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1))

    def forward(self, batch, horizon=60):
        feature, anchor = self.history(batch)
        b, length, _ = feature.shape
        tokens = feature.reshape(b, length, 4, self.channels).permute(0, 3, 2, 1).reshape(b, self.channels, -1)
        z = self.encoder(self.embedding(tokens) + self.channel_embedding).index_select(1, self.target_indices)
        lead = _lead_features(horizon, z)
        query = torch.cat((z[:, None].expand(-1, horizon, -1, -1),
                           lead[None, :, None].expand(b, -1, self.outputs, -1)), -1)
        return anchor[:, None] + self.head(query).squeeze(-1) * lead[:, :1][None]


def implicit_star_step(fast, slow, capacity, conductance, source, sink, dt):
    """Backward Euler solve of independent fast/slow exchange stars.

    Fast capacity = 1, slow capacity > 0, conductance >= 0. Equation is
    fast' - fast = dt*[sum k(slow'-fast') + source - sink*fast'].
    With zero source/sink, fast + sum(capacity*slow) is conserved. Nonnegative
    initial states/source and nonnegative sink imply nonnegative states.
    This applies to this declared surrogate and not unmeasured plant inventories.
    """
    if slow.shape[-1]:
        a = dt * conductance / capacity
        weight = dt * conductance / (1 + a)
        next_fast = (fast + dt * source + (weight * slow).sum(-1)) / (1 + dt * sink + weight.sum(-1))
        next_slow = (slow + a * next_fast[..., None]) / (1 + a)
    else:
        next_fast = (fast + dt * source) / (1 + dt * sink)
        next_slow = slow
    return next_fast, next_slow


class MechanismModel(HistoryModel):
    """History-observed positive exchange/source/removal/sensor state model.

    Thermal hard constraints: internal equivalent-storage balance, positive
    conductance/capacity and dissipative exchange, stable implicit source ports.
    SCR hard constraint: nonnegative state and sensor concentration when the
    initial observed concentration is projected onto the nonnegative domain.
    SCR outputs remain in the source data's raw concentration units: units are
    not silently interpreted as ppm or converted to total NOx.
    """
    def __init__(self, *args, use_slow=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scr = self.task_name.lower() == "scr"
        self.slow_count = 3 if use_slow else 0
        self.observer = GRUObserver(self.channels, self.hidden, self.history_stride)
        # Same observer/port parameterization in full and no-slow arms.
        self.ports = nn.Linear(self.hidden, self.outputs * 7)
        if self.slow_count:
            self.slow_init = nn.Linear(self.hidden, self.outputs * self.slow_count)
            self.slow_rates = nn.Linear(self.hidden, self.outputs * self.slow_count)
            self.raw_capacity = nn.Parameter(torch.zeros(self.outputs, self.slow_count))
        self.register_buffer("common_scale", self.target_scale.mean().clamp_min(1e-4))
        self.register_buffer("common_origin", torch.zeros(()) if self.is_scr else self.target_center.mean())
        self.register_buffer("base_slow_tau", torch.tensor([120.0, 600.0, 2400.0]))
        incoming = torch.zeros(self.outputs, self.outputs)
        if self.is_scr:
            for side in ("a", "b"):
                inlet, outlet = f"{side}_in", f"{side}_out"
                if inlet in self.target_names and outlet in self.target_names:
                    incoming[self.target_names.index(outlet), self.target_names.index(inlet)] = 1
        self.register_buffer("scr_incoming", incoming)

    def description(self):
        d = super().description()
        d.update(slow_stores_per_target=self.slow_count,
                 process_topology="same_side_inlet_outlet" if self.is_scr else "independent_target_local_storage_stars",
                 capacity_interpretation="dimensionless equivalent capacity",
                 observation_readout="sensor state only; no latent residual output head",
                 reagent_group_to_side_assignment="none", future_action_policy="history-only initialization")
        return d

    def forward(self, batch, horizon=60):
        _lead_features(horizon, batch["history_normalized"])
        feature, anchor = self.history(batch)
        z = self.observer(feature)
        coefficients = self.ports(z).reshape(-1, self.outputs, 7)
        raw = anchor * self.target_scale + self.target_center
        initial = (raw - self.common_origin) / self.common_scale
        if self.is_scr:
            initial = initial.clamp_min(0)
        fast, sensor = initial, initial
        port_tau = 30 + 300 * F.softplus(coefficients[..., 1])
        port_rate = 1 / port_tau
        sensor_alpha = torch.exp(-self.dt_seconds / (5 + 30 * F.softplus(coefficients[..., 2])))
        carrier_tau = 30 + 600 * F.softplus(coefficients[..., 4])
        carrier_amplitude = coefficients[..., 3]
        if self.slow_count:
            shift = self.slow_init(z).reshape(-1, self.outputs, self.slow_count)
            slow = initial[..., None] + torch.tanh(shift)
            if self.is_scr:
                slow = initial[..., None] * torch.exp(0.5 * torch.tanh(shift))
            capacity = 0.1 + F.softplus(self.raw_capacity)
            tau = self.base_slow_tau * (0.25 + F.softplus(self.slow_rates(z).reshape_as(slow)))
            conductance = capacity / tau
        else:
            slow = initial[..., None][..., :0]
            capacity, conductance = slow, slow
        if self.is_scr:
            reaction_rate = F.softplus(coefficients[..., 5] - 3) / 300.0
            transport_rate = F.softplus(coefficients[..., 6]) / 120.0
            has_upstream = self.scr_incoming.sum(-1)
            # Stable inverse softplus: zero learned shift approximately preserves anchor.
            positive_anchor = initial.clamp_min(1e-4)
            equilibrium_base = positive_anchor + torch.log(-torch.expm1(-positive_anchor))
        predictions = []
        for step in range(1, horizon + 1):
            carrier = carrier_amplitude * torch.exp(-step * self.dt_seconds / carrier_tau)
            shift = coefficients[..., 0] + carrier
            if self.is_scr:
                equilibrium = F.softplus(equilibrium_base + shift)
                upstream = fast @ self.scr_incoming.T
                source = port_rate * equilibrium + transport_rate * upstream
                sink = port_rate + reaction_rate + transport_rate * has_upstream
            else:
                equilibrium = initial + shift
                source, sink = port_rate * equilibrium, port_rate
            fast, slow = implicit_star_step(fast, slow, capacity, conductance,
                                            source, sink, self.dt_seconds)
            sensor = sensor_alpha * sensor + (1 - sensor_alpha) * fast
            predicted_raw = sensor * self.common_scale + self.common_origin
            predictions.append((predicted_raw - self.target_center) / self.target_scale)
        return torch.stack(predictions, 1)


def build_model(arm, input_names, target_names, task_name, hidden=64, **metadata):
    """Construct one of five arms. Pass O-vector target_center/target_scale.

    Alternatively pass C-vector input_center/input_scale in input_names order.
    Global 80-channel statistics must be sliced by the runner before this call.
    """
    common = dict(input_names=input_names, target_names=target_names, task_name=task_name,
                  hidden=hidden, **metadata)
    if arm == "gru_direct":
        return GRUDirect(**common)
    if arm == "ssm_rollout":
        return StableSSMRollout(**common)
    if arm == "itransformer":
        return InvertedTransformer(**common)
    if arm in ("mechanism", "mechanism_no_slow"):
        return MechanismModel(**common, use_slow=arm == "mechanism")
    raise ValueError(f"Unknown arm {arm!r}; choose from {ARMS}")
