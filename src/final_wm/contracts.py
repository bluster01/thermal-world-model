"""Fail-closed contracts for the final world-model pipeline.

This module freezes the shared vocabulary of the final pipeline: state,
boundary, action and observation registries, the packed-state layout, the
permission labels for future information, and the configuration dataclasses
that every module validates against before doing any work.

Contract rules enforced here (violations raise `FinalWMProtocolError`):

1. Forecast mode never reads measured future boundary or spray flow.  The
   measured total spray flow `W` is an oracle-only diagnostic channel.
2. Actions enter the transition only through the declared action channels
   and the monotone valve mapping; the closure never reads actions.
3. Residual injection positions are fixed to the declared set; the rejected
   "double injection" pattern (same residual into metal and steam energy)
   is not representable.
4. Stability-relevant parameters are bounded by construction (softplus /
   tanh parameterisations are defined in the modules; configs only carry
   admissible values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class FinalWMProtocolError(RuntimeError):
    """Raised when a final world-model contract is violated."""


# ---------------------------------------------------------------------------
# Registries (order is contractual; tensors use these orders everywhere)
# ---------------------------------------------------------------------------

PHYSICAL_STATE_ELEMENTS = (
    "h1", "h2", "h3",        # segment steam enthalpies, kJ/kg
    "tm1", "tm2", "tm3",     # metal heat-storage temperatures, degC
    "rb",                    # fuel/mills lag state, t/h
    "m_liq1", "m_liq2",      # attemperator liquid droplet masses, kg
)
BOUNDARY_ELEMENTS = (
    "steam_flow",            # D, kg/s
    "coal_command",          # uB, t/h
    "separator_pressure",    # pm, MPa
    "separator_temperature", # Tm_sep, degC
    "feedwater_temperature", # Tfw, degC
    "outlet_pressure",       # p_out, MPa
    "spray_flow_total",      # W, t/h -- unreliable; oracle diagnostics only
)
ACTION_ELEMENTS = (
    "valve1_position",       # v1, fraction 0..1
    "valve2_position",       # v2, fraction 0..1
)
OBSERVATION_ELEMENTS = (
    "sh1_inlet_temp",
    "sh1_outlet_temp",
    "sh2_inlet_temp",
    "sh2_outlet_temp",
    "final_outlet_temp",
)

BOUNDARY_MODES = ("forecast", "oracle")
SPRAY_TOTAL_MODES = ("action", "boundary")
INJECTION_MODES = ("none", "steam_only", "conservative")

# Physical constants shared by the modules (kept identical to the recovered
# Fan2020-UDE legacy snapshot so validation numbers stay comparable).
KAPPA_TPH_TO_KGS = 1.0 / 3.6
CRITICAL_PRESSURE_MPA = 22.064
BLEND_HALF_WIDTH_MPA = 0.3
PRESSURE_REF_MPA = 20.5

# Oracle-only boundary channels: indices into BOUNDARY_ELEMENTS whose *future*
# values must not be consumed in forecast mode.
ORACLE_ONLY_BOUNDARY_CHANNELS = ("spray_flow_total",)

# Boundary channels the action-blind closure may read (current/past only).
# `spray_flow_total` is excluded: it is action-correlated and oracle-only
# (legacy evidence E4: residuals reading W flipped the valve causal sign).
CLOSURE_BOUNDARY_CHANNELS = tuple(
    name for name in BOUNDARY_ELEMENTS if name not in ORACLE_ONLY_BOUNDARY_CHANNELS
)

# Default normalization constants (loc, scale) in registry order.  Used by
# observer, boundary model, closure and observation for stable feature
# scales; they are constants, not fitted statistics.
PHYSICAL_STATE_NORM = (
    (3000.0, 500.0), (3000.0, 500.0), (3000.0, 500.0),      # h1..h3 kJ/kg
    (550.0, 150.0), (550.0, 150.0), (550.0, 150.0),         # tm1..tm3 degC
    (250.0, 150.0),                                          # rb t/h
    (60.0, 60.0), (60.0, 60.0),                              # m_liq kg
)
BOUNDARY_NORM = (
    (350.0, 80.0),    # steam_flow kg/s
    (250.0, 60.0),    # coal_command t/h
    (17.0, 5.0),      # separator_pressure MPa
    (420.0, 80.0),    # separator_temperature degC
    (280.0, 30.0),    # feedwater_temperature degC
    (16.0, 5.0),      # outlet_pressure MPa
    (10.0, 8.0),      # spray_flow_total t/h
)
OBSERVATION_NORM = (
    (500.0, 60.0), (520.0, 60.0), (530.0, 60.0), (535.0, 60.0), (565.0, 40.0),
)
ACTION_NORM = ((0.5, 0.5), (0.5, 0.5))


def norm_constants(registry: Sequence[str], norms: Sequence[tuple[float, float]], label: str) -> None:
    if len(registry) != len(norms):
        raise FinalWMProtocolError(f"{label} normalization constants rank mismatch")
    if any(scale <= 0 for _loc, scale in norms):
        raise FinalWMProtocolError(f"{label} normalization scales must be positive")


norm_constants(PHYSICAL_STATE_ELEMENTS, PHYSICAL_STATE_NORM, "physical state")
norm_constants(BOUNDARY_ELEMENTS, BOUNDARY_NORM, "boundary")
norm_constants(OBSERVATION_ELEMENTS, OBSERVATION_NORM, "observation")
norm_constants(ACTION_ELEMENTS, ACTION_NORM, "action")


def _require_unique(registry: Sequence[str], label: str) -> None:
    if len(set(registry)) != len(registry) or not registry:
        raise FinalWMProtocolError(f"{label} registry must be non-empty and unique")


for _label, _registry in (
    ("physical state", PHYSICAL_STATE_ELEMENTS),
    ("boundary", BOUNDARY_ELEMENTS),
    ("action", ACTION_ELEMENTS),
    ("observation", OBSERVATION_ELEMENTS),
):
    _require_unique(_registry, _label)


# ---------------------------------------------------------------------------
# Packed state layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateLayout:
    """Packed state vector layout: physical block first, latent block last.

    The physical block always follows PHYSICAL_STATE_ELEMENTS order.  Latent
    dimensions are an optional stability-bounded block for unmeasured
    mixing/wall/disturbance effects; they never reinterpret the physical
    block.
    """

    latent_dim: int = 0

    def __post_init__(self) -> None:
        if int(self.latent_dim) < 0:
            raise FinalWMProtocolError("latent_dim must be >= 0")

    @property
    def physical_dim(self) -> int:
        return len(PHYSICAL_STATE_ELEMENTS)

    @property
    def dim(self) -> int:
        return self.physical_dim + int(self.latent_dim)

    @property
    def h_slice(self) -> slice:
        return slice(0, 3)

    @property
    def tm_slice(self) -> slice:
        return slice(3, 6)

    @property
    def rb_index(self) -> int:
        return 6

    @property
    def m_liq_slice(self) -> slice:
        return slice(7, 9)

    @property
    def latent_slice(self) -> slice:
        return slice(self.physical_dim, self.dim)

    def physical_index(self, name: str) -> int:
        if name not in PHYSICAL_STATE_ELEMENTS:
            raise FinalWMProtocolError(f"unknown physical state element: {name}")
        return PHYSICAL_STATE_ELEMENTS.index(name)


# ---------------------------------------------------------------------------
# Module configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransitionConfig:
    dt_seconds: float = 10.0
    substep_seconds: float = 2.0
    spray_total_mode: str = "action"        # canonical: spray via varphi(u)
    init_metal_offset: bool = True          # steady-state dTm correction
    latent_dim: int = 0

    @property
    def n_substeps(self) -> int:
        ratio = float(self.dt_seconds) / float(self.substep_seconds)
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise FinalWMProtocolError("dt_seconds must be a positive integer multiple of substep_seconds")
        return int(round(ratio))


@dataclass(frozen=True)
class ClosureConfig:
    hidden_dim: int = 64
    injection_mode: str = "conservative"
    residual_scale_kw: float = 3.0e4
    latent_scale: float = 1.0
    stochastic: bool = False
    reads_actions: bool = False             # contractual: must stay False


@dataclass(frozen=True)
class ObserverConfig:
    history_steps: int = 96
    d_hidden: int = 128
    latent_dim: int = 0


@dataclass(frozen=True)
class BoundaryModelConfig:
    history_steps: int = 96
    horizon: int = 60
    d_hidden: int = 128
    scenario_dim: int = 0


@dataclass(frozen=True)
class ObservationConfig:
    heteroscedastic: bool = True
    min_sigma_c: float = 0.05
    max_sigma_c: float = 10.0
    init_sigma_c: float = 0.5


@dataclass(frozen=True)
class ControllerConfig:
    kp: float = 0.02                        # valve fraction per degC
    ki: float = 0.002                       # valve fraction per degC per step
    valve_min: float = 0.0
    valve_max: float = 1.0
    rate_limit_per_step: float = 0.05       # |dv| per dt
    deadband_c: float = 0.2
    actuator_tau_seconds: float = 30.0
    dt_seconds: float = 10.0


@dataclass(frozen=True)
class WorldModelConfig:
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    closure: ClosureConfig = field(default_factory=ClosureConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    boundary: BoundaryModelConfig = field(default_factory=BoundaryModelConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    boundary_mode: str = "forecast"          # "forecast" | "oracle"
    initial_state_mode: str = "steady"       # "steady" | "learned" | "hybrid"
    support_margin: float = 0.05            # action support slack (fraction)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_transition_config(config: TransitionConfig) -> None:
    _ = config.n_substeps
    if config.dt_seconds <= 0 or config.substep_seconds <= 0:
        raise FinalWMProtocolError("transition time steps must be positive")
    if config.spray_total_mode not in SPRAY_TOTAL_MODES:
        raise FinalWMProtocolError(f"spray_total_mode must be one of {SPRAY_TOTAL_MODES}")
    if config.latent_dim < 0:
        raise FinalWMProtocolError("transition latent_dim must be >= 0")


def validate_closure_config(config: ClosureConfig) -> None:
    if config.injection_mode not in INJECTION_MODES:
        raise FinalWMProtocolError(
            f"injection_mode must be one of {INJECTION_MODES}; "
            "the rejected double-injection pattern is not representable"
        )
    if config.reads_actions is not False:
        raise FinalWMProtocolError("closure must remain action-blind (reads_actions=False)")
    if config.hidden_dim < 1 or config.residual_scale_kw <= 0 or config.latent_scale <= 0:
        raise FinalWMProtocolError("closure capacity/scale config is invalid")


def validate_observer_config(config: ObserverConfig) -> None:
    if config.history_steps < 1 or config.d_hidden < 1 or config.latent_dim < 0:
        raise FinalWMProtocolError("observer config is invalid")


def validate_boundary_config(config: BoundaryModelConfig) -> None:
    if config.history_steps < 1 or config.horizon < 1 or config.d_hidden < 1 or config.scenario_dim < 0:
        raise FinalWMProtocolError("boundary model config is invalid")


def validate_observation_config(config: ObservationConfig) -> None:
    if not (0.0 < config.min_sigma_c <= config.init_sigma_c <= config.max_sigma_c):
        raise FinalWMProtocolError("observation sigma bounds are invalid")


def validate_controller_config(config: ControllerConfig) -> None:
    if config.valve_min >= config.valve_max:
        raise FinalWMProtocolError("controller valve bounds are invalid")
    if config.rate_limit_per_step <= 0 or config.actuator_tau_seconds <= 0 or config.dt_seconds <= 0:
        raise FinalWMProtocolError("controller rate/actuator config is invalid")
    if config.kp < 0 or config.ki < 0 or config.deadband_c < 0:
        raise FinalWMProtocolError("controller gains must be non-negative")


def validate_world_model_config(config: WorldModelConfig) -> StateLayout:
    validate_transition_config(config.transition)
    validate_closure_config(config.closure)
    validate_observer_config(config.observer)
    validate_boundary_config(config.boundary)
    validate_observation_config(config.observation)
    validate_controller_config(config.controller)
    if config.boundary_mode not in BOUNDARY_MODES:
        raise FinalWMProtocolError(f"boundary_mode must be one of {BOUNDARY_MODES}")
    if config.initial_state_mode not in ("steady", "learned", "hybrid"):
        raise FinalWMProtocolError("initial_state_mode must be steady | learned | hybrid")
    if config.support_margin < 0:
        raise FinalWMProtocolError("support_margin must be >= 0")
    latent_dims = {
        config.transition.latent_dim,
        config.observer.latent_dim,
    }
    if len(latent_dims) != 1:
        raise FinalWMProtocolError("transition/observer latent_dim mismatch")
    if config.boundary_mode == "forecast" and config.transition.spray_total_mode == "boundary":
        raise FinalWMProtocolError(
            "forecast mode must not consume measured spray flow; "
            "spray_total_mode='boundary' is restricted to oracle diagnostics"
        )
    return StateLayout(latent_dim=config.transition.latent_dim)


# ---------------------------------------------------------------------------
# Action support (counterfactual support-domain gate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSupport:
    """Axis-aligned action support box estimated from history."""

    lo: tuple[float, ...]
    hi: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lo) != len(ACTION_ELEMENTS) or len(self.hi) != len(ACTION_ELEMENTS):
            raise FinalWMProtocolError("action support rank mismatch")
        if any(a > b for a, b in zip(self.lo, self.hi)):
            raise FinalWMProtocolError("action support box is empty")

    def contains(self, actions: Any) -> Any:
        """Return a boolean mask (...,) marking in-support action rows."""
        import torch

        tensor = torch.as_tensor(actions, dtype=torch.float32)
        if tensor.shape[-1] != len(ACTION_ELEMENTS):
            raise FinalWMProtocolError("action tensor last dim must match ACTION_ELEMENTS")
        lo = torch.tensor(self.lo, dtype=tensor.dtype)
        hi = torch.tensor(self.hi, dtype=tensor.dtype)
        return ((tensor >= lo) & (tensor <= hi)).all(dim=-1)


def action_support_from_history(history_actions: Any, margin: float) -> ActionSupport:
    import torch

    tensor = torch.as_tensor(history_actions, dtype=torch.float32)
    if tensor.ndim < 2 or tensor.shape[-1] != len(ACTION_ELEMENTS):
        raise FinalWMProtocolError("history actions must have shape (..., steps, 2)")
    flat = tensor.reshape(-1, tensor.shape[-1])
    lo = (flat.min(dim=0).values - margin).clamp(0.0, 1.0)
    hi = (flat.max(dim=0).values + margin).clamp(0.0, 1.0)
    return ActionSupport(lo=tuple(float(v) for v in lo), hi=tuple(float(v) for v in hi))
