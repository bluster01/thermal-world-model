"""Known-truth multi-step systems for representation and optimization checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


PROFILE_NAMES = ("hold", "step", "pulse", "ramp", "multi_step")
_SPLIT_OFFSETS = {"train": 0, "validation": 100_003, "test": 200_003}
TRUTH_REGIMES = {"two_pole_linear", "nonlinear_valve", "context_scheduled"}
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
        if self.truth_regime == "nonlinear_valve" and self.truth_opening_map == "identity":
            raise ValueError("nonlinear_valve truth requires a nonlinear opening map")
        if self.truth_regime == "context_scheduled" and max(
            self.context_gain_log_scale, self.context_tau_log_scale
        ) <= 0:
            raise ValueError("context_scheduled truth requires a non-zero scheduling scale")


@dataclass
class SyntheticBatch:
    context: torch.Tensor
    action: torch.Tensor
    reference: torch.Tensor
    clean_effect: torch.Tensor
    target_effect: torch.Tensor
    target_temperature: torch.Tensor
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


def generate_synthetic_split(spec: SyntheticSpec, split: str) -> SyntheticBatch:
    spec.validate()
    if split not in _SPLIT_OFFSETS:
        raise ValueError(f"unknown synthetic split={split!r}")
    generator = torch.Generator().manual_seed(spec.seed + _SPLIT_OFFSETS[split])
    context = torch.randn((spec.samples, spec.context_dim), generator=generator)
    action, reference, profile_ids = _generate_actions(spec, generator)
    dose = _effective_opening(action, spec.truth_opening_map) - _effective_opening(
        reference, spec.truth_opening_map
    )
    gain, tau = _scheduled_parameters(spec, context)
    clean_effect = _cascade_response(dose, spec.dt_seconds, tau, gain)
    noise = spec.noise_std * torch.randn(clean_effect.shape, generator=generator)
    target_effect = clean_effect + noise
    free_level = 565.0 + 0.7 * context[:, :1]
    target_temperature = free_level + target_effect
    truth = {
        "generator": "stable_cascade_v2" if spec.truth_regime != "two_pole_linear" else "stable_two_pole_v1",
        "truth_regime": spec.truth_regime,
        "truth_opening_map": spec.truth_opening_map,
        "gain_c_per_pct": spec.gain_c_per_pct,
        "tau_seconds": list(spec.tau_seconds),
        "context_gain_log_scale": spec.context_gain_log_scale,
        "context_tau_log_scale": spec.context_tau_log_scale,
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
        profile_ids=profile_ids,
        profile_names=PROFILE_NAMES,
        truth=truth,
    )
