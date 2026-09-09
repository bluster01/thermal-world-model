"""Independent known-truth thermal process; NumPy only, no learned-model imports.

This is a controlled synthetic benchmark, not a thermal-power-plant simulator.
Truth has a heated wall, hot and outlet inventories, and actuator lag. The
student omits the wall. Source parameters/hidden states are evaluation-only.
"""
from dataclasses import dataclass, fields
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class ThermalPlantSpec:
    wall_capacity: float = 6.
    hot_capacity: float = 10.
    outlet_capacity: float = 20.
    wall_conductance: float = 1.7
    exchange_conductance: float = 1.
    ambient_loss: float = .2
    heater_power: float = 10.
    actuator_tau: float = 6.
    command_exponent: float = 1.
    dt_seconds: float = 1.
    substeps: int = 4

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field.name} must be positive finite')
        if not isinstance(self.substeps, int):
            raise ValueError('substeps must be an integer')


def equilibrium(spec, command, ambient, hidden_load_kw=0.):
    """Analytic steady state for testing/generation; never an inference anchor."""
    _validate_inputs(np.zeros(4), command, ambient, hidden_load_kw)
    actuator = float(command) ** spec.command_exponent
    heater = spec.heater_power * actuator
    outlet = ambient + (heater + hidden_load_kw) / spec.ambient_loss
    hot = outlet + (heater + hidden_load_kw) / spec.exchange_conductance
    wall = hot + heater / spec.wall_conductance
    state = np.array([wall, hot, outlet, actuator], dtype=np.float64)
    _validate_inputs(state, command, ambient, hidden_load_kw)
    return state


def _validate_inputs(state, command, ambient, load):
    if (not isinstance(state, np.ndarray) or state.shape != (4,) or
            not np.issubdtype(state.dtype, np.number) or not np.isrealobj(state) or
            not np.isfinite(state).all()):
        raise ValueError('Truth state must be a finite NumPy vector of width 4')
    if not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)
               for v in (command, ambient, load)):
        raise ValueError('Truth inputs must be finite')
    if not 0 <= command <= 1 or not 0 <= state[3] <= 1:
        raise ValueError('Command and actuator must be in [0,1]')


def truth_step(spec, state, command, ambient, hidden_load_kw=0.):
    """Held-input RK4, with no student physics, neural residual, or observation."""
    _validate_inputs(state, command, ambient, hidden_load_kw)
    demanded = float(command) ** spec.command_exponent

    def derivative(x):
        wall, hot, outlet, actuator = x
        q_wall = spec.wall_conductance * (wall - hot)
        q_out = spec.exchange_conductance * (hot - outlet)
        return np.array([(spec.heater_power * actuator - q_wall) / spec.wall_capacity,
                         (q_wall - q_out + hidden_load_kw) / spec.hot_capacity,
                         (q_out - spec.ambient_loss * (outlet - ambient)) / spec.outlet_capacity,
                         (demanded - actuator) / spec.actuator_tau])

    result = state.astype(np.float64, copy=True)
    dt = spec.dt_seconds / spec.substeps
    for _ in range(spec.substeps):
        k1 = derivative(result)
        k2 = derivative(result + dt * k1 / 2)
        k3 = derivative(result + dt * k2 / 2)
        k4 = derivative(result + dt * k3)
        result = result + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    _validate_inputs(result, command, ambient, hidden_load_kw)
    return result


@dataclass(frozen=True)
class ObservedEpisode:
    """Only data permitted to the learner; y_t and previous command u_(t-1)."""
    observations: np.ndarray  # N+1,1; missing values NaN
    observed_mask: np.ndarray  # N+1,1 boolean
    commands: np.ndarray  # N,1; command applied from t to t+1
    ambient: float  # constant, declared known context, not a future measurement
    initial_previous_command: float
    dt_seconds: float
    episode_id: str


@dataclass(frozen=True)
class TruthEpisode:
    observed: ObservedEpisode
    states: np.ndarray  # N+1,4 evaluation only
    hidden_load_kw: np.ndarray  # N evaluation only
    spec: ThermalPlantSpec
    regime: str


def generate_episode(seed, *, steps=512, regime='nominal', missing_fraction=.1,
                     noise_std=.05, warmup=256):
    """Generate one independently seeded trajectory with bounded command ramps.

    Parameter shifts are factorized: slow_actuator, extra_storage, net_load.
    Nominal and other shift conditions have zero external disturbance. Hidden
    loads, observation noise and full initial state can be reused across branches.
    """
    if regime not in ('nominal', 'slow_actuator', 'extra_storage', 'net_load'):
        raise ValueError('Unknown benchmark regime')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (steps, warmup)):
        raise ValueError('steps and warmup must be positive integers')
    if not 0 <= missing_fraction < 1 or not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError('Invalid missing fraction or noise')
    # Separate streams preserve all nuisance draws when only a regime changes.
    # The same seed across regimes forms a paired group, not independent splits.
    streams = np.random.SeedSequence(seed).spawn(7)
    parameter_rng, shift_rng, context_rng, command_rng, load_rng, noise_rng, mask_rng = (
        np.random.default_rng(stream) for stream in streams)
    parameters = dict(wall_capacity=parameter_rng.uniform(4., 8.), hot_capacity=parameter_rng.uniform(9., 11.),
                      outlet_capacity=parameter_rng.uniform(18., 22.), wall_conductance=parameter_rng.uniform(1.4, 2.),
                      exchange_conductance=parameter_rng.uniform(.9, 1.1), ambient_loss=parameter_rng.uniform(.18, .22),
                      actuator_tau=parameter_rng.uniform(4., 8.))
    if regime == 'slow_actuator':
        parameters['actuator_tau'] = shift_rng.uniform(12., 18.)
    elif regime == 'extra_storage':
        parameters['wall_capacity'] = shift_rng.uniform(12., 18.)
    spec = ThermalPlantSpec(**parameters)
    ambient = float(context_rng.uniform(18., 25.))
    previous = float(context_rng.uniform(.25, .75))
    state = equilibrium(spec, previous, ambient)
    commands, loads, states = [], [], [state]
    command, target, hold, load = previous, previous, 0, 0.
    for index in range(warmup + steps):
        if hold == 0:
            target = float(command_rng.uniform(.15, .85))
            hold = int(command_rng.integers(20, 61))
        command += float(np.clip(target - command, -.04, .04))
        hold -= 1
        if regime == 'net_load':
            load = .995 * load + float(load_rng.normal(0., .05))
        state = truth_step(spec, state, command, ambient, load)
        commands.append([command])
        loads.append(load)
        states.append(state)
    states = np.stack(states[warmup:])
    initial_previous_command = float(commands[warmup - 1][0])
    commands = np.asarray(commands[warmup:], dtype=np.float64)
    loads = np.asarray(loads[warmup:], dtype=np.float64)
    observations = states[:, 2:3] + noise_rng.normal(0., noise_std, size=(steps + 1, 1))
    mask = mask_rng.uniform(size=observations.shape) >= missing_fraction
    mask[0] = True  # a past measurement for every new episode's anchor
    observations = np.where(mask, observations, np.nan)
    observed = ObservedEpisode(observations, mask, commands, ambient,
                               initial_previous_command, spec.dt_seconds, f'{regime}:{seed}')
    return TruthEpisode(observed, states, loads, spec, regime)


def history_window(episode: ObservedEpisode, *, cut, context):
    """Return a past-only anchor/history ending at t=cut, before u_cut is applied.

    Anchor assumes a nominal steady hot/outlet difference, using the last valid
    measured outlet and previous command. It never reads a truth state/spec.
    """
    if not isinstance(episode, ObservedEpisode):
        raise TypeError('Learning history requires ObservedEpisode, without truth fields')
    if (episode.observations.ndim != 2 or episode.observations.shape[1] != 1 or
            episode.observed_mask.shape != episode.observations.shape or
            episode.observed_mask.dtype != np.bool_ or
            episode.commands.shape != (len(episode.observations) - 1, 1)):
        raise ValueError('Episode observations/masks/commands have incompatible shapes or mask dtype')
    if (isinstance(context, bool) or not isinstance(context, int) or context < 1 or
            isinstance(cut, bool) or not isinstance(cut, int) or
            cut < context - 1 or cut >= len(episode.observations)):
        raise ValueError('Invalid history window')
    start = cut - context + 1
    if (not math.isfinite(episode.dt_seconds) or episode.dt_seconds <= 0 or
            not math.isfinite(episode.ambient) or
            not math.isfinite(episode.initial_previous_command) or
            not 0 <= episode.initial_previous_command <= 1):
        raise ValueError('Episode time, ambient, and previous command must be valid')
    # Check only the observable past, never future values, including u_cut.
    past_y = episode.observations[:cut + 1]
    past_mask = episode.observed_mask[:cut + 1]
    executed = episode.commands[:cut]
    if (not np.isfinite(past_y[past_mask]).all() or not np.isfinite(executed).all() or
            ((executed < 0) | (executed > 1)).any()):
        raise ValueError('Past observed values and executed commands must be valid')
    y = episode.observations[start:cut + 1]
    mask = episode.observed_mask[start:cut + 1]
    past_actions = np.concatenate([[[episode.initial_previous_command]], executed], axis=0)
    actions = past_actions[start:cut + 1]
    boundary = np.full((context, 1), episode.ambient)
    history = np.concatenate([y, actions, boundary], -1)
    history_mask = np.concatenate([mask, np.ones((context, 2), dtype=bool)], -1)
    # The anchor may use a measurement older than the encoder context. This is
    # explicitly past-only, shared by all arms, and no arbitrary future fill.
    valid = np.flatnonzero(episode.observed_mask[:cut + 1, 0])
    if not valid.size:
        raise ValueError('No past observation is available for the physical anchor')
    outlet = episode.observations[valid[-1], 0]
    last_command = actions[-1, 0]
    anchor = np.array([outlet + 10. * last_command, outlet, last_command])
    time = np.arange(-context + 1, 1, dtype=np.float64) * episode.dt_seconds
    return anchor, history, history_mask, time
