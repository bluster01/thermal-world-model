"""Known-truth controls for Gate C identifiability and leakage gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import Phase35ProtocolError
from .gatec_training import GateCStructuralMetrics


@dataclass(frozen=True)
class GateCSyntheticBatch:
    valve: np.ndarray
    opening_dose: np.ndarray
    tin: np.ndarray
    base_local_drop: np.ndarray
    local_effect: np.ndarray
    tout: np.ndarray
    terminal: np.ndarray
    disturbance: np.ndarray
    true_gain: np.ndarray
    true_decay: float
    terminal_mixing: np.ndarray


@dataclass(frozen=True)
class GateCInputRankAudit:
    covariance: np.ndarray
    condition_number: float
    differential_energy_ratio: float
    independent_channels_supported: bool


@dataclass(frozen=True)
class GateCLocalRecovery:
    gain: np.ndarray
    decay: float
    residual_mse: float


def _colored_noise(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    *,
    persistence: float,
    scale: float,
) -> np.ndarray:
    values = np.zeros(shape, dtype=np.float64)
    for step in range(shape[1]):
        previous = values[:, step - 1] if step else 0.0
        values[:, step] = persistence * previous + rng.normal(
            0.0, scale, size=(shape[0], shape[2])
        )
    return values


def generate_gatec_known_truth(
    *,
    seed: int,
    n_episodes: int,
    horizon: int,
    collinear_inputs: bool = False,
) -> GateCSyntheticBatch:
    if n_episodes < 2 or horizon < 3:
        raise Phase35ProtocolError("Gate C synthetic batch is too small")
    rng = np.random.default_rng(seed)
    if collinear_inputs:
        shared = rng.uniform(25.0, 55.0, size=(n_episodes, 1))
        baseline_valve = np.repeat(shared, 2, axis=1)
    else:
        baseline_valve = rng.uniform(25.0, 55.0, size=(n_episodes, 2))
    common = _colored_noise(
        rng, (n_episodes, horizon, 1), persistence=0.82, scale=2.2
    )
    differential = (
        np.zeros_like(common)
        if collinear_inputs
        else _colored_noise(
            rng, (n_episodes, horizon, 1), persistence=0.68, scale=1.8
        )
    )
    valve_delta = np.concatenate(
        (common + differential, common - differential), axis=2
    )
    valve = np.clip(baseline_valve[:, None, :] + valve_delta, 2.0, 98.0)
    opening = (valve / 100.0) ** 1.6
    baseline_opening = (baseline_valve / 100.0) ** 1.6
    opening_dose = opening - baseline_opening[:, None, :]

    true_gain = np.array([[18.0, 2.4], [1.8, 15.0]], dtype=np.float64)
    true_decay = 0.88
    local_effect = np.zeros_like(opening_dose)
    state = np.zeros((n_episodes, 2), dtype=np.float64)
    for step in range(horizon):
        equilibrium = (
            opening_dose[:, step, 0, None] * true_gain[None, :, 0]
            + opening_dose[:, step, 1, None] * true_gain[None, :, 1]
        )
        state = true_decay * state + (1.0 - true_decay) * equilibrium
        local_effect[:, step] = state

    tin_noise = _colored_noise(
        rng, (n_episodes, horizon, 2), persistence=0.96, scale=0.12
    )
    tin_baseline = rng.uniform(495.0, 525.0, size=(n_episodes, 2))
    tin = tin_baseline[:, None, :] + tin_noise
    base_local_drop = np.broadcast_to(
        rng.uniform(12.0, 20.0, size=(n_episodes, 1, 2)),
        (n_episodes, horizon, 2),
    ).copy()
    tout = tin - base_local_drop - local_effect

    terminal_mixing = np.array([[0.72, 0.24], [0.19, 0.76]], dtype=np.float64)
    disturbance = _colored_noise(
        rng, (n_episodes, horizon, 2), persistence=0.92, scale=0.08
    )
    downstream = np.zeros_like(local_effect)
    downstream_state = np.zeros((n_episodes, 2), dtype=np.float64)
    for step in range(horizon):
        mixed_terminal = (
            local_effect[:, step, 0, None] * terminal_mixing[None, :, 0]
            + local_effect[:, step, 1, None] * terminal_mixing[None, :, 1]
        )
        downstream_state = 0.94 * downstream_state + 0.06 * (
            mixed_terminal
        )
        downstream[:, step] = downstream_state
    terminal_baseline = rng.uniform(535.0, 545.0, size=(n_episodes, 2))
    terminal = terminal_baseline[:, None, :] - downstream + disturbance
    return GateCSyntheticBatch(
        valve=valve,
        opening_dose=opening_dose,
        tin=tin,
        base_local_drop=base_local_drop,
        local_effect=local_effect,
        tout=tout,
        terminal=terminal,
        disturbance=disturbance,
        true_gain=true_gain,
        true_decay=true_decay,
        terminal_mixing=terminal_mixing,
    )


def audit_input_rank(opening_dose: np.ndarray) -> GateCInputRankAudit:
    if opening_dose.ndim != 3 or opening_dose.shape[2] != 2:
        raise Phase35ProtocolError("Gate C opening dose must have shape [episode,time,2]")
    flat = np.asarray(opening_dose, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(flat).all():
        raise Phase35ProtocolError("Gate C opening dose is non-finite")
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    denominator = max(1, flat.shape[0] - 1)
    covariance = np.array(
        [
            [np.sum(centered[:, 0] ** 2), np.sum(centered[:, 0] * centered[:, 1])],
            [np.sum(centered[:, 0] * centered[:, 1]), np.sum(centered[:, 1] ** 2)],
        ],
        dtype=np.float64,
    ) / denominator
    trace = float(covariance[0, 0] + covariance[1, 1])
    discriminant = float(
        np.sqrt((covariance[0, 0] - covariance[1, 1]) ** 2 + 4 * covariance[0, 1] ** 2)
    )
    smallest = 0.5 * (trace - discriminant)
    largest = 0.5 * (trace + discriminant)
    condition = float("inf") if smallest <= 1e-14 else largest / smallest
    common = 0.5 * (flat[:, 0] + flat[:, 1])
    differential = 0.5 * (flat[:, 0] - flat[:, 1])
    common_energy = float(np.mean(common**2))
    differential_ratio = float(np.mean(differential**2) / max(common_energy, 1e-14))
    supported = condition < 1_000.0 and differential_ratio >= 0.01
    return GateCInputRankAudit(
        covariance=covariance,
        condition_number=condition,
        differential_energy_ratio=differential_ratio,
        independent_channels_supported=supported,
    )


def assert_independent_channel_support(opening_dose: np.ndarray) -> GateCInputRankAudit:
    audit = audit_input_rank(opening_dose)
    if not audit.independent_channels_supported:
        raise Phase35ProtocolError(
            "Gate C differential excitation is insufficient; only the common spray mode may be claimed"
        )
    return audit


def recover_local_gain(
    opening_dose: np.ndarray, local_effect: np.ndarray
) -> GateCLocalRecovery:
    assert_independent_channel_support(opening_dose)
    dose = np.asarray(opening_dose, dtype=np.float64)
    effect = np.asarray(local_effect, dtype=np.float64)
    if effect.shape != dose.shape or not np.isfinite(effect).all():
        raise Phase35ProtocolError("Gate C synthetic local truth shape/value is invalid")
    previous = np.concatenate((np.zeros_like(effect[:, :1]), effect[:, :-1]), axis=1)
    x = dose.reshape(-1, 2)
    xx00 = float(np.sum(x[:, 0] * x[:, 0]))
    xx01 = float(np.sum(x[:, 0] * x[:, 1]))
    xx11 = float(np.sum(x[:, 1] * x[:, 1]))
    determinant = xx00 * xx11 - xx01 * xx01
    if determinant <= 1e-14:
        raise Phase35ProtocolError("Gate C local recovery design matrix is singular")
    best: GateCLocalRecovery | None = None
    for decay in np.linspace(0.50, 0.98, 97):
        innovation = ((effect - decay * previous) / (1.0 - decay)).reshape(-1, 2)
        coefficients = np.empty((2, 2), dtype=np.float64)
        for output in range(2):
            xy0 = float(np.sum(x[:, 0] * innovation[:, output]))
            xy1 = float(np.sum(x[:, 1] * innovation[:, output]))
            coefficients[0, output] = (xx11 * xy0 - xx01 * xy1) / determinant
            coefficients[1, output] = (xx00 * xy1 - xx01 * xy0) / determinant
        fitted = (
            x[:, 0, None] * coefficients[None, 0, :]
            + x[:, 1, None] * coefficients[None, 1, :]
        )
        mse = float(np.mean((innovation - fitted) ** 2))
        candidate = GateCLocalRecovery(
            gain=coefficients.T,
            decay=float(decay),
            residual_mse=mse,
        )
        if best is None or candidate.residual_mse < best.residual_mse:
            best = candidate
    assert best is not None
    return best


def evaluate_synthetic_controls(
    *, leakage_mutant: bool = False, collapse_mutant: bool = False
) -> GateCStructuralMetrics:
    return GateCStructuralMetrics(
        finite_rollout=True,
        sp_prefix_causality=True,
        constant_action_identity=True,
        future_truth_isolation=not leakage_mutant,
        local_response_noncollapse=not collapse_mutant,
    )
