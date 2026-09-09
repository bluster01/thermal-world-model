"""Fresh neural candidates over fixed historical physical priors and hashed grid."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math

import numpy as np
import torch

from src.final_wm.contracts import TransitionConfig
from src.final_wm.properties import GridThermoProperties
from src.final_wm.transition import Fan2020UDETransition, TRANSITION_PARAM_PRIORS
from src.world_model_evidence import EvidenceStreamingPhysicsWorldModel
from src.world_model_streaming.model import StreamingPhysicsWorldModel
from src.world_model_vnext.backends import LegacyThermalBackend
from .data import CONDITIONING, file_sha256

PROPERTY_SHA256 = '9fd7a1dba96a5b968661f644fa185ac755266c85853d689d065432a2d41f6e92'
VARIANTS = ('shared_age', 'shared_attention', 'quota_age', 'quota_attention')
PRIOR_SCOPE = ('Fixed inherited priors include historical side-A validation spray-gain calibration and '
               'historically learned mixing-lag knowledge. New neural fitting uses only the selected '
               'prototype training partition; this is not a claim that all prior knowledge came from that tenth.')


@dataclass(frozen=True)
class PlantModelConfig:
    variant: str
    seed: int
    hidden_dim: int = 32
    heads: int = 4
    layers: int = 2
    correction_fraction: float = .25
    cache_capacity: int = 128
    evidence_capacity: int = 64
    prediction_capacity: int = 64
    dtype: str = 'float32'

    def __post_init__(self):
        if self.variant not in VARIANTS or self.dtype not in ('float32', 'float64'):
            raise ValueError('Unknown candidate or dtype')
        for name in ('seed', 'hidden_dim', 'heads', 'layers', 'cache_capacity', 'evidence_capacity', 'prediction_capacity'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'Invalid {name}')
        if (self.hidden_dim % self.heads or self.cache_capacity != self.evidence_capacity + self.prediction_capacity or
                isinstance(self.correction_fraction, bool) or not math.isfinite(self.correction_fraction) or
                not 0 < self.correction_fraction <= 1):
            raise ValueError('Invalid attention/cache/correction dimensions')


class MonitoredGrid(GridThermoProperties):
    """Read primitive inputs before unchanged provider math, without graph edits.

    Counters are scalar visits, including repeated bisection queries. They do
    not see transition-internal clamps or prove phase correctness. Nonfinite
    arguments fail; finite domain excursions are retained and counted.
    """

    def __init__(self, arrays, *, device='cpu'):
        super().__init__(arrays, device=device)
        self.phase, self.records = 'construction', {}
        self.batch_size, self.unsupported_rows, self.separator_extensions = None, [], []

    def reset(self, batch_size=None):
        self.records = {}
        self.batch_size = batch_size
        self.unsupported_rows = [False] * batch_size if batch_size is not None else []
        self.separator_extensions = []

    def record(self, name, label, value, lo, hi, *, critical_cap=False):
        raw = value.detach()
        finite = torch.isfinite(raw)
        key = f'{self.phase}/{name}/{label}'
        record = self.records.setdefault(key, dict(phase=self.phase, primitive=name, axis=label,
                    lo=float(lo), hi=float(hi), calls=0, queries=0, nonfinite=0, below_domain=0,
                    above_domain=0, critical_policy_cap=0, raw_min=None, raw_max=None))
        record['calls'] += 1
        record['queries'] += raw.numel()
        record['nonfinite'] += int((~finite).sum())
        record['below_domain'] += int((finite & (raw < lo)).sum())
        record['critical_policy_cap' if critical_cap else 'above_domain'] += int((finite & (raw > hi)).sum())
        unsupported = (~finite) | (raw < lo) | ((raw > hi) & (not critical_cap))
        if self.batch_size is not None and self.phase not in ('diagnostic_endpoints', 'five_temperature_anchor_inversion'):
            if raw.ndim and raw.shape[0] == self.batch_size:
                flags = unsupported.reshape(self.batch_size, -1).any(1).cpu().tolist()
            else:
                flags = [bool(unsupported.any())] * self.batch_size
            self.unsupported_rows = [old or new for old, new in zip(self.unsupported_rows, flags)]
        if bool(finite.any()):
            low, high = float(raw[finite].min()), float(raw[finite].max())
            record['raw_min'] = low if record['raw_min'] is None else min(low, record['raw_min'])
            record['raw_max'] = high if record['raw_max'] is None else max(high, record['raw_max'])
        if not bool(finite.all()):
            raise ValueError(f'Nonfinite primitive input: {key}')

    def temperature_of_ph(self, p, h):
        bounds = self.bounds
        self.record('temperature_of_ph', 'p', p, bounds.p_lo, bounds.p_hi)
        self.record('temperature_of_ph', 'h', h, bounds.h_lo, bounds.h_hi)
        return super().temperature_of_ph(p, h)

    def enthalpy_of_pt(self, p, temperature):
        bounds = self.bounds
        self.record('enthalpy_of_pt', 'p', p, bounds.p_lo, bounds.p_hi)
        self.record('enthalpy_of_pt', 'temperature', temperature, bounds.t_lo, bounds.t_hi)
        return super().enthalpy_of_pt(p, temperature)

    def saturation_temperature(self, p):
        self.record('saturation_temperature', 'p', p, self._psub_lo, self._p_crit, critical_cap=True)
        return super().saturation_temperature(p)

    def saturated_vapor_enthalpy(self, p):
        self.record('saturated_vapor_enthalpy', 'p', p, float(self._psub[0]), float(self._psub[-1]))
        return super().saturated_vapor_enthalpy(p)

    def liquid_enthalpy(self, temperature):
        self.record('liquid_enthalpy', 'temperature', temperature, float(self._tliq[0]), float(self._tliq[-1]))
        return super().liquid_enthalpy(temperature)

    def separator_enthalpy(self, pm, tm_sep):
        # Read-only diagnostic calculation bypasses the recording override;
        # inherited separator execution below still records its actual calls.
        with torch.no_grad():
            tsat = GridThermoProperties.saturation_temperature(self, pm.detach())
            lifted = (pm <= self.critical_pressure) & (tm_sep < tsat + .5)
            self.separator_extensions.append(dict(phase=self.phase,
                subcritical_temperature_lift_count=int(lifted.sum()),
                supercritical_saturation_auxiliary_cap_count=int((pm > self.critical_pressure).sum()),
                policy='unchanged_separator_max_Tsat_plus_0.5_and_supercritical_auxiliary_cap'))
        return super().separator_enthalpy(pm, tm_sep)

    def summary(self):
        return dict(primitive_axis_records=copy.deepcopy(list(self.records.values())),
                    unsupported_rows=list(self.unsupported_rows),
                    separator_model_extensions=copy.deepcopy(self.separator_extensions),
                    endpoint_probe_excursions_excluded_from_window_support=True,
                    inverse_search_probes_excluded_final_anchor_reconstruction_checked=True,
                    counting_unit='primitive_input_axis_scalar_visits_not_independent_states',
                    excludes=['transition_internal_clamps', 'phase_correctness', 'property_ground_truth_accuracy'])


def build_model(config, properties_path, *, expected_properties_sha256=PROPERTY_SHA256, device='cpu'):
    if not isinstance(config, PlantModelConfig):
        raise ValueError('Explicit PlantModelConfig required')
    config.__post_init__()
    if expected_properties_sha256 != PROPERTY_SHA256 or file_sha256(properties_path) != expected_properties_sha256:
        raise ValueError('Physical property asset identity mismatch')
    with np.load(properties_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if any(value.dtype.hasobject or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite or object property grid')
    # Parameters are constructed on CPU so CUDA generator state cannot alter
    # matching initial tensors. No old fitted checkpoint is loaded here.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        provider = MonitoredGrid(arrays)
        transition = Fan2020UDETransition(TransitionConfig(dt_seconds=10., substep_seconds=2., spray_total_mode='action'),
                                          provider, priors=dict(TRANSITION_PARAM_PRIORS))
        transition.requires_grad_(False)
        backend = LegacyThermalBackend(transition)
        kwargs = dict(hidden_dim=config.hidden_dim, heads=config.heads, layers=config.layers,
                      correction_fraction=config.correction_fraction, encoder_kind='gru',
                      use_neural_residual=True, cache_capacity=config.cache_capacity,
                      read_mode=config.variant.split('_', 1)[1], store_predictions=True)
        cls = EvidenceStreamingPhysicsWorldModel if config.variant.startswith('quota_') else StreamingPhysicsWorldModel
        if cls is EvidenceStreamingPhysicsWorldModel:
            kwargs.update(evidence_capacity=config.evidence_capacity, prediction_capacity=config.prediction_capacity)
        model = cls(backend, **kwargs)
    model = model.to(device=device, dtype=getattr(torch, config.dtype))
    model.plant_config = config
    model.property_asset_sha256 = expected_properties_sha256
    return model


def model_identity(model):
    config, transition = model.plant_config, model.backend.transition
    if any(p.requires_grad for p in transition.parameters()):
        raise ValueError('Prototype physical parameters must remain frozen')
    retention = (model.retention_policy.as_dict() if hasattr(model, 'retention_policy') else
                 dict(kind='shared_fifo_v1', cache_capacity=model.cache_capacity))
    return dict(schema_version=1, config=asdict(config), model_class=type(model).__name__,
                conditioning=CONDITIONING, transition_config=asdict(transition.config),
                priors=dict(transition.priors), prior_knowledge_scope=PRIOR_SCOPE,
                effective_physical_parameters={name: float(transition.val(name).detach()) for name in transition.priors},
                physical_parameter_policy='all_transition_parameters_frozen',
                property_asset_sha256=model.property_asset_sha256, retention_policy=retention,
                store_predictions=model.store_predictions, dt_seconds=model.dt_seconds,
                read_mode=model.reader.mode, encoder_kind=model.encoder_kind,
                trainable_parameter_names=[n for n, p in model.named_parameters() if p.requires_grad],
                normalization={n: b.detach().cpu().tolist() for n, b in model.backend.named_buffers() if n.endswith(('_loc', '_scale'))},
                interval_alignment='advance_left_recorded_snapshot_decode_observe_right_recorded_boundary')


def state_tensor_sha256(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        header = json.dumps([name, value.dtype.str, list(value.shape)], separators=(',', ':')).encode('ascii')
        h.update(header + b'\n' + value.tobytes())
    return h.hexdigest()


def checkpoint_payload(model, *, dataset_identity, plan_sha256, step, selection_value):
    state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    if any(not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError('Nonfinite checkpoint tensor')
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or not math.isfinite(selection_value):
        raise ValueError('Invalid checkpoint selection metadata')
    return dict(schema_version=1, identity=model_identity(model), dataset_identity=dataset_identity,
                plan_sha256=plan_sha256, step=step, selection_value=selection_value,
                state_dict=state, state_tensor_sha256=state_tensor_sha256(state))


def load_checkpoint(model, payload, *, dataset_identity, plan_sha256):
    if (payload['identity'] != model_identity(model) or payload['dataset_identity'] != dataset_identity or
            payload['plan_sha256'] != plan_sha256):
        raise ValueError('Full checkpoint identity mismatch')
    state = payload['state_dict']
    if payload['state_tensor_sha256'] != state_tensor_sha256(state):
        raise ValueError('Checkpoint tensor digest mismatch')
    if any(not bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError('Nonfinite checkpoint tensor')
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError('Checkpoint tensor inventory mismatch')
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError('Checkpoint tensor schema mismatch')
        if name.startswith('backend.') and not torch.equal(tensor.cpu(), expected[name].cpu()):
            raise ValueError('Frozen physical parameters or normalization were altered')
    model.load_state_dict(state, strict=True)
    return model
