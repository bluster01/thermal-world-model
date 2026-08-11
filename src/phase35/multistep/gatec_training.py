"""Training losses and checkpoint eligibility for MS3-R Gate C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import Phase35ProtocolError


LOSS_KEYS = {"valve", "tin", "local", "terminal", "rollout", "structure"}
TARGET_KEYS = {"valve", "tin", "local", "terminal"}


@dataclass(frozen=True)
class GateCRobustScales:
    values: dict[str, float]
    scale_floor: float
    source_split: str = "train"

    @classmethod
    def fit(
        cls,
        targets: Mapping[str, torch.Tensor],
        *,
        split: str,
        scale_floor: float,
    ) -> "GateCRobustScales":
        if split != "train":
            raise Phase35ProtocolError("Gate C robust scales must be fit on the train split")
        if not torch.isfinite(torch.tensor(scale_floor)) or scale_floor <= 0:
            raise Phase35ProtocolError("Gate C scale floor must be finite and positive")
        if set(targets) != TARGET_KEYS:
            raise Phase35ProtocolError("Gate C scale targets are not closed")
        values: dict[str, float] = {}
        for key, raw in targets.items():
            tensor = raw.detach().to(dtype=torch.float64).reshape(-1)
            if tensor.numel() == 0 or not torch.isfinite(tensor).all():
                raise Phase35ProtocolError(f"Gate C scale target {key!r} is empty/non-finite")
            median = torch.median(tensor)
            mad = float(torch.median(torch.abs(tensor - median)).item())
            values[key] = max(float(scale_floor), mad)
        return cls(values=values, scale_floor=float(scale_floor))


@dataclass(frozen=True)
class GateCLossBreakdown:
    total: torch.Tensor
    valve: torch.Tensor
    tin: torch.Tensor
    local: torch.Tensor
    terminal: torch.Tensor
    rollout: torch.Tensor
    structure: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            key: float(getattr(self, key).detach().cpu().item())
            for key in ("total", "valve", "tin", "local", "terminal", "rollout", "structure")
        }


@dataclass(frozen=True)
class GateCStructuralMetrics:
    finite_rollout: bool
    sp_prefix_causality: bool
    constant_action_identity: bool
    future_truth_isolation: bool
    local_response_noncollapse: bool

    @property
    def eligible(self) -> bool:
        return all(
            (
                self.finite_rollout,
                self.sp_prefix_causality,
                self.constant_action_identity,
                self.future_truth_isolation,
                self.local_response_noncollapse,
            )
        )


@dataclass(frozen=True)
class GateCSelectorRecord:
    checkpoint_id: str
    boundary_mode: str
    composite_score: float
    structural: GateCStructuralMetrics

    @property
    def eligible(self) -> bool:
        return self.boundary_mode == "forecast_boundary" and self.structural.eligible


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if set(weights) != LOSS_KEYS:
        raise Phase35ProtocolError("Gate C loss weight fields are not closed")
    numeric = {key: float(value) for key, value in weights.items()}
    if any(value < 0 or not torch.isfinite(torch.tensor(value)) for value in numeric.values()):
        raise Phase35ProtocolError("Gate C loss weights must be finite and non-negative")
    if abs(sum(numeric.values()) - 1.0) > 1e-12:
        raise Phase35ProtocolError("Gate C loss weights must sum to one")
    return numeric


def _normalized_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, scale: float
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise Phase35ProtocolError("Gate C prediction/target shapes do not match")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise Phase35ProtocolError("Gate C prediction/target contains non-finite values")
    return F.smooth_l1_loss(prediction / scale, target / scale)


def compute_gatec_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    scales: GateCRobustScales,
    weights: Mapping[str, float],
    *,
    local_supervision: bool = True,
) -> GateCLossBreakdown:
    numeric_weights = _validate_weights(weights)
    required_predictions = {
        "valve_prediction",
        "tin_prediction",
        "local_drop_prediction",
        "terminal_prediction",
        "local_effect",
    }
    if not required_predictions.issubset(predictions) or set(targets) != TARGET_KEYS:
        raise Phase35ProtocolError("Gate C loss inputs are incomplete")
    if set(scales.values) != TARGET_KEYS or scales.source_split != "train":
        raise Phase35ProtocolError("Gate C loss requires frozen train-only scales")
    valve = _normalized_smooth_l1(
        predictions["valve_prediction"], targets["valve"], scales.values["valve"]
    )
    tin = _normalized_smooth_l1(
        predictions["tin_prediction"], targets["tin"], scales.values["tin"]
    )
    local = (
        _normalized_smooth_l1(
            predictions["local_drop_prediction"], targets["local"], scales.values["local"]
        )
        if local_supervision
        else valve.new_zeros(())
    )
    terminal = _normalized_smooth_l1(
        predictions["terminal_prediction"], targets["terminal"], scales.values["terminal"]
    )
    horizon = predictions["terminal_prediction"].shape[1]
    rollout_start = max(0, horizon * 2 // 3)
    rollout = _normalized_smooth_l1(
        predictions["terminal_prediction"][:, rollout_start:],
        targets["terminal"][:, rollout_start:],
        scales.values["terminal"],
    )
    raw_structure = predictions.get("structure_penalty")
    structure = valve.new_zeros(()) if raw_structure is None else raw_structure.mean()
    components = {
        "valve": valve,
        "tin": tin,
        "local": local,
        "terminal": terminal,
        "rollout": rollout,
        "structure": structure,
    }
    total = sum(numeric_weights[key] * value for key, value in components.items())
    return GateCLossBreakdown(total=total, **components)


def select_gatec_checkpoint(
    records: Sequence[GateCSelectorRecord],
) -> GateCSelectorRecord:
    eligible = [
        record
        for record in records
        if record.eligible and torch.isfinite(torch.tensor(record.composite_score))
    ]
    if not eligible:
        raise Phase35ProtocolError("Gate C has no eligible forecast-boundary checkpoint")
    return min(eligible, key=lambda record: (record.composite_score, record.checkpoint_id))


def validate_warmup_updates(warmup_updates: int, total_updates: int) -> int:
    if total_updates <= 0 or warmup_updates < 0:
        raise Phase35ProtocolError("Gate C update counts must be non-negative and non-empty")
    if warmup_updates > int(total_updates * 0.10):
        raise Phase35ProtocolError("Gate C warm-up cannot exceed 10% of optimizer updates")
    return int(warmup_updates)


def freeze_for_warmup(
    model: nn.Module, *, trainable_name_fragments: Sequence[str]
) -> None:
    if not trainable_name_fragments:
        raise Phase35ProtocolError("Gate C warm-up must name at least one trainable block")
    matched = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(fragment in name for fragment in trainable_name_fragments))
        matched = matched or parameter.requires_grad
    if not matched:
        raise Phase35ProtocolError("Gate C warm-up trainable block did not match the model")


def unfreeze_for_joint_training(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
