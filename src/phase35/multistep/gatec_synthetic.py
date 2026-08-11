"""Known-truth controls for Gate C identifiability and leakage gates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn

from ..schema import Phase35ProtocolError
from .gatec_training import GateCStructuralMetrics


@dataclass(frozen=True)
class GateCSyntheticBatch:
    baseline_valve: np.ndarray
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


@dataclass(frozen=True)
class GateCSyntheticTrainingResult:
    route: str
    seed: int
    steps: int
    initial_train_loss: float
    final_train_loss: float
    heldout_relative_rollout_error: float
    heldout_direction_accuracy: float
    heldout_amplitude_ratio: float
    stable_pole_max: float
    finite: bool


@dataclass(frozen=True)
class GateCAttributionCompetitionResult:
    residual_capacity: str
    excitation: str
    seed: int
    residual_excitation_fraction: float
    heldout_total_relative_error: float
    heldout_response_amplitude_ratio: float
    heldout_response_relative_error: float
    heldout_free_relative_error: float
    local_supervision: bool
    free_reads_future_action: bool
    finite: bool
    automatic_scientific_pass: None = None


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
        baseline_valve=baseline_valve,
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


def train_synthetic_response_operator(
    *,
    route: str,
    batch: GateCSyntheticBatch,
    seed: int,
    steps: int,
    learning_rate: float = 0.02,
) -> GateCSyntheticTrainingResult:
    """Fit one route on train episodes and score held-out known-truth rollouts."""

    from .gatec_model import build_local_response_operator

    if steps < 1 or learning_rate <= 0:
        raise Phase35ProtocolError("Gate C synthetic training budget is invalid")
    assert_independent_channel_support(batch.opening_dose)
    n_episodes, horizon, sides = batch.valve.shape
    if sides != 2 or n_episodes < 8:
        raise Phase35ProtocolError("Gate C synthetic training batch is too small")
    split = max(4, int(n_episodes * 0.75))
    if split >= n_episodes:
        raise Phase35ProtocolError("Gate C synthetic training requires held-out episodes")
    torch.manual_seed(seed)
    dtype = torch.float32
    valve = torch.as_tensor(batch.valve, dtype=dtype)
    baseline = torch.as_tensor(batch.baseline_valve, dtype=dtype)
    target = torch.as_tensor(batch.local_effect, dtype=dtype)
    context = torch.zeros((n_episodes, 8), dtype=dtype)
    operator = build_local_response_operator(
        route=route,
        context_dim=context.shape[1],
        state_dim=6,
        horizon=horizon,
        dt_seconds=10.0,
        scheduled=True,
    )
    optimizer = torch.optim.Adam(operator.parameters(), lr=learning_rate)
    scale = target[:split].square().mean().sqrt().clamp_min(1e-3)

    def train_loss() -> torch.Tensor:
        output = operator(context[:split], valve[:split], baseline[:split])
        return ((output["effect"] - target[:split]) / scale).square().mean()

    operator.train()
    initial_loss = float(train_loss().detach().item())
    final_loss = initial_loss
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(operator.parameters(), 2.0)
        optimizer.step()
        final_loss = float(loss.detach().item())

    operator.eval()
    with torch.no_grad():
        heldout = operator(context[split:], valve[split:], baseline[split:])
        prediction = heldout["effect"]
        truth = target[split:]
        relative = (
            (prediction - truth).square().mean().sqrt()
            / truth.square().mean().sqrt().clamp_min(1e-6)
        )
        mask = truth.abs() > 1e-3
        direction = (
            (torch.sign(prediction[mask]) == torch.sign(truth[mask])).float().mean()
            if bool(mask.any())
            else prediction.new_tensor(1.0)
        )
        amplitude = prediction.abs().sum() / truth.abs().sum().clamp_min(1e-6)
        poles = heldout["stable_poles"]
        pole_max = float(poles.max().item()) if poles.numel() else 0.0
        finite = bool(
            torch.isfinite(prediction).all()
            and torch.isfinite(poles).all()
            and pole_max < 1.0
        )
    return GateCSyntheticTrainingResult(
        route=route,
        seed=int(seed),
        steps=int(steps),
        initial_train_loss=initial_loss,
        final_train_loss=final_loss,
        heldout_relative_rollout_error=float(relative.item()),
        heldout_direction_accuracy=float(direction.item()),
        heldout_amplitude_ratio=float(amplitude.item()),
        stable_pole_max=pole_max,
        finite=finite,
    )


def run_attribution_competition(
    *,
    residual_capacity: str,
    excitation: str,
    seed: int,
    steps: int,
    learning_rate: float = 0.02,
) -> GateCAttributionCompetitionResult:
    """Train free+response branches under known conditional action innovation.

    This is an attribution sensitivity diagnostic. It deliberately returns no
    automatic scientific decision.
    """

    from .gatec_model import build_local_response_operator

    if residual_capacity not in {"small", "base", "large"}:
        raise Phase35ProtocolError("Gate C attribution residual capacity is invalid")
    if excitation not in {"low", "high"}:
        raise Phase35ProtocolError("Gate C attribution excitation label is invalid")
    if steps < 1:
        raise Phase35ProtocolError("Gate C attribution training budget is invalid")
    torch.manual_seed(seed)
    n_episodes, horizon, context_dim = 48, 30, 4
    context = torch.randn(n_episodes, context_dim)
    normalized_time = torch.linspace(0.0, 1.0, horizon)[None, :, None]
    sinusoid = torch.sin(torch.linspace(0.0, math.pi, horizon))[None, :, None]
    predictable_a = 3.0 * (
        context[:, None, 0:1] * normalized_time
        + context[:, None, 1:2] * sinusoid
    )
    predictable_b = 3.0 * (
        context[:, None, 0:1] * normalized_time
        - context[:, None, 1:2] * sinusoid
    )
    predictable = torch.cat((predictable_a, predictable_b), dim=2)
    innovation_scale = 0.12 if excitation == "low" else 3.0
    innovation = torch.zeros_like(predictable)
    for step in range(horizon):
        previous = innovation[:, step - 1] if step else 0.0
        innovation[:, step] = 0.55 * previous + innovation_scale * torch.randn(
            n_episodes, 2
        )
    baseline = 35.0 + 4.0 * torch.tanh(context[:, :2])
    valve = torch.clamp(baseline[:, None, :] + predictable + innovation, 2.0, 98.0)
    opening = (valve / 100.0).pow(1.6)
    opening_baseline = (baseline / 100.0).pow(1.6)
    dose = opening - opening_baseline[:, None, :]
    true_gain = torch.tensor(((18.0, 2.4), (1.8, 15.0)))
    local_truth = torch.zeros_like(dose)
    state = torch.zeros(n_episodes, 2)
    for step in range(horizon):
        equilibrium = torch.einsum("bi,oi->bo", dose[:, step], true_gain)
        state = 0.88 * state + 0.12 * equilibrium
        local_truth[:, step] = state
    nuisance = torch.cat(
        (
            0.12 * context[:, None, 2:3] * sinusoid
            + 0.08 * context[:, None, 3:4] * normalized_time,
            -0.10 * context[:, None, 2:3] * sinusoid
            + 0.07 * context[:, None, 3:4] * normalized_time,
        ),
        dim=2,
    )
    target = local_truth + nuisance
    residual_fraction = float(
        innovation.square().mean()
        / (predictable + innovation).square().mean().clamp_min(1e-8)
    )

    hidden = {"small": 8, "base": 24, "large": 64}[residual_capacity]
    if residual_capacity == "small":
        free_head: nn.Module = nn.Linear(context_dim, horizon * 2)
    elif residual_capacity == "base":
        free_head = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.GELU(), nn.Linear(hidden, horizon * 2)
        )
    else:
        free_head = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * 2),
        )
    response = build_local_response_operator(
        route="a1phys_three_pole",
        context_dim=context_dim,
        state_dim=6,
        horizon=horizon,
        dt_seconds=10.0,
        scheduled=False,
    )
    optimizer = torch.optim.Adam(
        [*free_head.parameters(), *response.parameters()],
        lr=learning_rate,
        weight_decay=1e-5,
    )
    split = 36
    scale = target[:split].square().mean().sqrt().clamp_min(1e-3)
    free_head.train()
    response.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        free_prediction = free_head(context[:split]).reshape(split, horizon, 2)
        response_prediction = response(
            context[:split], valve[:split], baseline[:split]
        )["effect"]
        loss = ((free_prediction + response_prediction - target[:split]) / scale).square().mean()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*free_head.parameters(), *response.parameters()], 2.0
        )
        optimizer.step()

    free_head.eval()
    response.eval()
    with torch.no_grad():
        free_prediction = free_head(context[split:]).reshape(-1, horizon, 2)
        response_output = response(context[split:], valve[split:], baseline[split:])
        response_prediction = response_output["effect"]
        total_prediction = free_prediction + response_prediction
        heldout_target = target[split:]
        heldout_local = local_truth[split:]
        heldout_nuisance = nuisance[split:]
        total_relative = (
            (total_prediction - heldout_target).square().mean().sqrt()
            / heldout_target.square().mean().sqrt().clamp_min(1e-6)
        )
        response_relative = (
            (response_prediction - heldout_local).square().mean().sqrt()
            / heldout_local.square().mean().sqrt().clamp_min(1e-6)
        )
        free_relative = (
            (free_prediction - heldout_nuisance).square().mean().sqrt()
            / heldout_nuisance.square().mean().sqrt().clamp_min(1e-6)
        )
        amplitude = response_prediction.abs().sum() / heldout_local.abs().sum().clamp_min(1e-6)
        poles = response_output["stable_poles"]
        finite = bool(
            torch.isfinite(total_prediction).all()
            and torch.isfinite(poles).all()
            and float(poles.max()) < 1.0
        )
    return GateCAttributionCompetitionResult(
        residual_capacity=residual_capacity,
        excitation=excitation,
        seed=int(seed),
        residual_excitation_fraction=residual_fraction,
        heldout_total_relative_error=float(total_relative),
        heldout_response_amplitude_ratio=float(amplitude),
        heldout_response_relative_error=float(response_relative),
        heldout_free_relative_error=float(free_relative),
        local_supervision=False,
        free_reads_future_action=False,
        finite=finite,
        automatic_scientific_pass=None,
    )


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
