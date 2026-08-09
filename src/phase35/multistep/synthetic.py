"""Known-truth multi-step systems for representation and optimization checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


PROFILE_NAMES = ("hold", "step", "pulse", "ramp", "multi_step")
_SPLIT_OFFSETS = {"train": 0, "validation": 100_003, "test": 200_003}


@dataclass(frozen=True)
class SyntheticSpec:
    samples: int = 1024
    horizon: int = 60
    context_dim: int = 4
    dt_seconds: float = 10.0
    seed: int = 20260809
    noise_std: float = 0.02
    gain_c_per_pct: float = -0.04
    tau_seconds: tuple[float, float] = (70.0, 210.0)

    def validate(self) -> None:
        if self.samples < len(PROFILE_NAMES) or self.horizon < 8 or self.context_dim < 1:
            raise ValueError("synthetic samples, horizon, and context_dim are too small")
        if self.dt_seconds <= 0 or self.noise_std < 0:
            raise ValueError("synthetic dt must be positive and noise non-negative")
        if self.gain_c_per_pct >= 0 or min(self.tau_seconds) < self.dt_seconds:
            raise ValueError("synthetic truth requires negative gain and resolvable positive time constants")


@dataclass
class SyntheticBatch:
    context: torch.Tensor
    action: torch.Tensor
    reference: torch.Tensor
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


def _two_pole_response(
    dose: torch.Tensor, dt_seconds: float, tau_seconds: tuple[float, float], gain: float
) -> torch.Tensor:
    decay = [torch.exp(torch.tensor(-dt_seconds / tau)) for tau in tau_seconds]
    first = torch.zeros(dose.shape[0], dtype=dose.dtype)
    second = torch.zeros_like(first)
    outputs = []
    for step in range(dose.shape[1]):
        first = decay[0] * first + (1.0 - decay[0]) * dose[:, step]
        second = decay[1] * second + (1.0 - decay[1]) * first
        outputs.append(gain * second)
    return torch.stack(outputs, dim=1)


def generate_synthetic_split(spec: SyntheticSpec, split: str) -> SyntheticBatch:
    spec.validate()
    if split not in _SPLIT_OFFSETS:
        raise ValueError(f"unknown synthetic split={split!r}")
    generator = torch.Generator().manual_seed(spec.seed + _SPLIT_OFFSETS[split])
    context = torch.randn((spec.samples, spec.context_dim), generator=generator)
    action, reference, profile_ids = _generate_actions(spec, generator)
    clean_effect = _two_pole_response(
        action - reference, spec.dt_seconds, spec.tau_seconds, spec.gain_c_per_pct
    )
    noise = spec.noise_std * torch.randn(clean_effect.shape, generator=generator)
    target_effect = clean_effect + noise
    free_level = 565.0 + 0.7 * context[:, :1]
    target_temperature = free_level + target_effect
    truth = {
        "generator": "stable_two_pole_v1",
        "gain_c_per_pct": spec.gain_c_per_pct,
        "tau_seconds": list(spec.tau_seconds),
        "dt_seconds": spec.dt_seconds,
        "seed": spec.seed + _SPLIT_OFFSETS[split],
        "split": split,
    }
    return SyntheticBatch(
        context=context,
        action=action,
        reference=reference,
        target_effect=target_effect,
        target_temperature=target_temperature,
        profile_ids=profile_ids,
        profile_names=PROFILE_NAMES,
        truth=truth,
    )
