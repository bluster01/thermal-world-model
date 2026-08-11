"""Known-truth multi-step systems for representation and optimization checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


PROFILE_NAMES = ("hold", "step", "pulse", "ramp", "multi_step")
_SPLIT_OFFSETS = {"train": 0, "validation": 100_003, "test": 200_003}
TRUTH_REGIMES = {
    "two_pole_linear",
    "nonlinear_valve",
    "context_scheduled",
    "delayed_context_scheduled",
    "disturbed_context_scheduled",
    "full_coupled_context_scheduled",
}
TRUTH_OPENING_MAPS = {"identity", "equal_percentage_r50"}


@dataclass(frozen=True)
class SyntheticSpec:
    samples: int = 1024
    horizon: int = 60
    context_dim: int = 4
    dt_seconds: float = 10.0
    seed: int = 20260809
    noise_std: float = 0.02
    gain_c_per_pct: float = -0.04
    tau_seconds: tuple[float, ...] = (70.0, 210.0)
    truth_regime: str = "two_pole_linear"
    truth_opening_map: str = "identity"
    context_gain_log_scale: float = 0.0
    context_tau_log_scale: float = 0.0
    input_delay_steps: int = 0
    disturbance_std: float = 0.0
    disturbance_tau_seconds: float = 0.0
    free_trajectory_scale: float = 0.0
    action_context_coupling_pct: float = 0.0

    def validate(self) -> None:
        if self.samples < len(PROFILE_NAMES) or self.horizon < 8 or self.context_dim < 1:
            raise ValueError("synthetic samples, horizon, and context_dim are too small")
        if self.dt_seconds <= 0 or self.noise_std < 0:
            raise ValueError("synthetic dt must be positive and noise non-negative")
        if self.gain_c_per_pct >= 0 or min(self.tau_seconds) < self.dt_seconds:
            raise ValueError("synthetic truth requires negative gain and resolvable positive time constants")
        if len(self.tau_seconds) not in {1, 2, 3}:
            raise ValueError("synthetic truth supports one to three cascaded poles")
        if self.truth_regime not in TRUTH_REGIMES:
            raise ValueError(f"unknown truth_regime={self.truth_regime!r}")
        if self.truth_opening_map not in TRUTH_OPENING_MAPS:
            raise ValueError(f"unknown truth_opening_map={self.truth_opening_map!r}")
        if min(self.context_gain_log_scale, self.context_tau_log_scale) < 0:
            raise ValueError("context scheduling scales must be non-negative")
        if not isinstance(self.input_delay_steps, int) or self.input_delay_steps < 0:
            raise ValueError("input_delay_steps must be a non-negative integer")
        if self.input_delay_steps >= self.horizon:
            raise ValueError("input delay must be smaller than horizon")
        if self.truth_regime == "nonlinear_valve" and self.truth_opening_map == "identity":
            raise ValueError("nonlinear_valve truth requires a nonlinear opening map")
        if self.truth_regime in {
            "context_scheduled",
            "delayed_context_scheduled",
            "disturbed_context_scheduled",
            "full_coupled_context_scheduled",
        } and max(self.context_gain_log_scale, self.context_tau_log_scale) <= 0:
            raise ValueError("scheduled truth requires a non-zero scheduling scale")
        if (
            self.truth_regime == "delayed_context_scheduled"
            and self.input_delay_steps <= 0
        ):
            raise ValueError("delayed_context_scheduled truth requires a positive input delay")
        if self.disturbance_std < 0 or self.disturbance_tau_seconds < 0:
            raise ValueError("disturbance scale and time constant must be non-negative")
        if self.truth_regime == "disturbed_context_scheduled":
            if self.disturbance_std <= 0:
                raise ValueError("disturbed truth requires a positive disturbance scale")
            if self.disturbance_tau_seconds < self.dt_seconds:
                raise ValueError(
                    "disturbance time constant must be at least one sampling interval"
                )
        elif self.disturbance_std != 0 or self.disturbance_tau_seconds != 0:
            raise ValueError(
                "disturbance parameters require disturbed_context_scheduled truth"
            )
        if self.truth_regime == "full_coupled_context_scheduled":
            if (
                self.context_dim < 4
                or self.free_trajectory_scale <= 0
                or self.action_context_coupling_pct <= 0
            ):
                raise ValueError(
                    "full coupling truth requires context_dim>=4 and positive free/policy scales"
                )
        elif self.free_trajectory_scale != 0 or self.action_context_coupling_pct != 0:
            raise ValueError(
                "full coupling parameters require full_coupled_context_scheduled truth"
            )


@dataclass
class SyntheticBatch:
    context: torch.Tensor
    action: torch.Tensor
    reference: torch.Tensor
    clean_effect: torch.Tensor
    target_effect: torch.Tensor
    target_temperature: torch.Tensor
    clean_free: torch.Tensor
    clean_total: torch.Tensor
    colored_disturbance: torch.Tensor
    profile_ids: torch.Tensor
    profile_names: tuple[str, ...]
    truth: dict[str, object]


def _uniform(generator: torch.Generator, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator)


def _generate_actions(spec: SyntheticSpec, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, horizon = spec.samples, spec.horizon
    reference_level = _uniform(generator, (n,), 12.0, 48.0)
    reference = reference_level[:, None].expand(-1, horizon).clone()
    action = reference.clone()
    profile_ids = torch.arange(n, dtype=torch.long) % len(PROFILE_NAMES)
    permutation = torch.randperm(n, generator=generator)
    profile_ids = profile_ids[permutation]
    for row in range(n):
        profile = int(profile_ids[row])
        if profile == 0:
            continue
        onset = int(torch.randint(1, max(2, horizon // 3), (1,), generator=generator))
        magnitude = float(_uniform(generator, (1,), 2.0, 8.0))
        sign = -1.0 if float(torch.rand((), generator=generator)) < 0.5 else 1.0
        amplitude = sign * magnitude
        if profile == 1:
            action[row, onset:] += amplitude
        elif profile == 2:
            duration = int(torch.randint(max(2, horizon // 6), max(3, horizon // 2), (1,), generator=generator))
            action[row, onset:min(horizon, onset + duration)] += amplitude
        elif profile == 3:
            ramp = torch.linspace(0.0, amplitude, horizon - onset)
            action[row, onset:] += ramp
        else:
            second = int(torch.randint(max(onset + 2, horizon // 2), horizon, (1,), generator=generator))
            action[row, onset:] += amplitude
            action[row, second:] -= 0.6 * amplitude
    return action.clamp(0.0, 100.0), reference, profile_ids


def _effective_opening(opening: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "identity":
        return opening
    if mode == "equal_percentage_r50":
        ratio = torch.as_tensor(50.0, dtype=opening.dtype, device=opening.device)
        raw = torch.pow(ratio, opening / 100.0 - 1.0)
        return (raw - 1.0 / ratio) / (1.0 - 1.0 / ratio) * 100.0
    raise ValueError(f"unknown truth opening map={mode!r}")


def _scheduled_parameters(spec: SyntheticSpec, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch = context.shape[0]
    gain = torch.full(
        (batch,), spec.gain_c_per_pct, dtype=context.dtype, device=context.device
    )
    tau = torch.as_tensor(
        spec.tau_seconds, dtype=context.dtype, device=context.device
    )[None, :].expand(batch, -1).clone()
    if spec.context_gain_log_scale > 0:
        gain = gain * torch.exp(spec.context_gain_log_scale * torch.tanh(context[:, 0]))
    if spec.context_tau_log_scale > 0:
        shifts = []
        for pole in range(len(spec.tau_seconds)):
            shifts.append(torch.tanh(context[:, (pole + 1) % context.shape[1]]))
        tau = tau * torch.exp(spec.context_tau_log_scale * torch.stack(shifts, dim=1))
    return gain, tau


def _cascade_response(
    dose: torch.Tensor, dt_seconds: float, tau_seconds: torch.Tensor, gain: torch.Tensor
) -> torch.Tensor:
    decay = torch.exp(-dt_seconds / tau_seconds)
    state = torch.zeros(
        dose.shape[0], tau_seconds.shape[1], dtype=dose.dtype, device=dose.device
    )
    outputs = []
    for step in range(dose.shape[1]):
        stage_input = dose[:, step]
        updated = []
        for pole in range(tau_seconds.shape[1]):
            stage = decay[:, pole] * state[:, pole] + (1.0 - decay[:, pole]) * stage_input
            updated.append(stage)
            stage_input = stage
        state = torch.stack(updated, dim=1)
        outputs.append(gain * state[:, -1])
    return torch.stack(outputs, dim=1)


def _pure_delay(dose: torch.Tensor, steps: int) -> torch.Tensor:
    if steps == 0:
        return dose
    delayed = torch.zeros_like(dose)
    delayed[:, steps:] = dose[:, :-steps]
    return delayed


def _colored_disturbance(
    reference: torch.Tensor,
    *,
    dt_seconds: float,
    stationary_std: float,
    tau_seconds: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    if stationary_std == 0:
        return torch.zeros_like(reference), 0.0
    rho = float(torch.exp(torch.tensor(-dt_seconds / tau_seconds)))
    innovation_std = stationary_std * (1.0 - rho**2) ** 0.5
    state = stationary_std * torch.randn(
        reference.shape[0], generator=generator, dtype=reference.dtype
    )
    values = []
    for _ in range(reference.shape[1]):
        state = rho * state + innovation_std * torch.randn(
            reference.shape[0], generator=generator, dtype=reference.dtype
        )
        values.append(state)
    return torch.stack(values, dim=1), rho


def _lag_one_correlation(value: torch.Tensor) -> float:
    if value.shape[1] < 2 or torch.count_nonzero(value).item() == 0:
        return 0.0
    left = value[:, :-1].reshape(-1)
    right = value[:, 1:].reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 1e-12:
        return 0.0
    return float((left * right).sum() / denominator)


def _apply_context_policy(
    action: torch.Tensor,
    reference: torch.Tensor,
    context: torch.Tensor,
    coupling_pct: float,
) -> torch.Tensor:
    if coupling_pct == 0:
        return action
    active = (action - reference).abs() > 1e-8
    offset = coupling_pct * torch.tanh(context[:, :1])
    return (action + active.to(action.dtype) * offset).clamp(0.0, 100.0)


def _free_trajectory(spec: SyntheticSpec, context: torch.Tensor) -> torch.Tensor:
    if spec.truth_regime != "full_coupled_context_scheduled":
        level = 565.0 + 0.7 * context[:, :1]
        return level.expand(-1, spec.horizon).clone()
    step = torch.arange(
        1, spec.horizon + 1, dtype=context.dtype, device=context.device
    )[None, :]
    unit_time = step / float(spec.horizon)
    seconds = step * spec.dt_seconds
    free = (
        0.8 * torch.tanh(context[:, 0:1])
        + 0.5 * torch.tanh(context[:, 1:2]) * unit_time
        + 0.4
        * torch.tanh(context[:, 2:3])
        * (1.0 - torch.exp(-seconds / 180.0))
        + 0.2 * torch.tanh(context[:, 3:4]) * torch.sin(torch.pi * unit_time)
    )
    return spec.free_trajectory_scale * free


def generate_synthetic_split(spec: SyntheticSpec, split: str) -> SyntheticBatch:
    spec.validate()
    if split not in _SPLIT_OFFSETS:
        raise ValueError(f"unknown synthetic split={split!r}")
    generator = torch.Generator().manual_seed(spec.seed + _SPLIT_OFFSETS[split])
    context = torch.randn((spec.samples, spec.context_dim), generator=generator)
    action, reference, profile_ids = _generate_actions(spec, generator)
    action = _apply_context_policy(
        action,
        reference,
        context,
        spec.action_context_coupling_pct,
    )
    dose = _effective_opening(action, spec.truth_opening_map) - _effective_opening(
        reference, spec.truth_opening_map
    )
    dose = _pure_delay(dose, spec.input_delay_steps)
    gain, tau = _scheduled_parameters(spec, context)
    clean_effect = _cascade_response(dose, spec.dt_seconds, tau, gain)
    noise = spec.noise_std * torch.randn(clean_effect.shape, generator=generator)
    colored_disturbance, disturbance_rho = _colored_disturbance(
        clean_effect,
        dt_seconds=spec.dt_seconds,
        stationary_std=spec.disturbance_std,
        tau_seconds=spec.disturbance_tau_seconds,
        generator=generator,
    )
    target_effect = clean_effect + noise + colored_disturbance
    clean_free = _free_trajectory(spec, context)
    clean_total = clean_free + clean_effect
    target_temperature = clean_free + target_effect
    truth = {
        "generator": "stable_cascade_v2" if spec.truth_regime != "two_pole_linear" else "stable_two_pole_v1",
        "truth_regime": spec.truth_regime,
        "truth_opening_map": spec.truth_opening_map,
        "gain_c_per_pct": spec.gain_c_per_pct,
        "tau_seconds": list(spec.tau_seconds),
        "context_gain_log_scale": spec.context_gain_log_scale,
        "context_tau_log_scale": spec.context_tau_log_scale,
        "input_delay_steps": spec.input_delay_steps,
        "input_delay_seconds": spec.input_delay_steps * spec.dt_seconds,
        "disturbance_std": spec.disturbance_std,
        "disturbance_tau_seconds": spec.disturbance_tau_seconds,
        "disturbance_rho": disturbance_rho,
        "disturbance_realized_mean": float(colored_disturbance.mean()),
        "disturbance_realized_std": float(colored_disturbance.std()),
        "disturbance_realized_lag1_correlation": _lag_one_correlation(
            colored_disturbance
        ),
        "free_trajectory_scale": spec.free_trajectory_scale,
        "action_context_coupling_pct": spec.action_context_coupling_pct,
        "realized_gain_range": [float(gain.min()), float(gain.max())],
        "realized_tau_range": [
            [float(tau[:, pole].min()), float(tau[:, pole].max())]
            for pole in range(tau.shape[1])
        ],
        "dt_seconds": spec.dt_seconds,
        "seed": spec.seed + _SPLIT_OFFSETS[split],
        "split": split,
    }
    return SyntheticBatch(
        context=context,
        action=action,
        reference=reference,
        clean_effect=clean_effect,
        target_effect=target_effect,
        target_temperature=target_temperature,
        clean_free=clean_free,
        clean_total=clean_total,
        colored_disturbance=colored_disturbance,
        profile_ids=profile_ids,
        profile_names=PROFILE_NAMES,
        truth=truth,
    )
