from __future__ import annotations

import pytest
import torch

from src.phase35.multistep.gatec_training import (
    GateCRobustScales,
    GateCSelectorRecord,
    GateCStructuralMetrics,
    compute_gatec_loss,
    freeze_for_warmup,
    select_gatec_checkpoint,
    unfreeze_for_joint_training,
    validate_warmup_updates,
)
from src.phase35.schema import Phase35ProtocolError


WEIGHTS = {
    "valve": 0.15,
    "tin": 0.15,
    "local": 0.25,
    "terminal": 0.25,
    "rollout": 0.10,
    "structure": 0.10,
}


def _targets() -> dict[str, torch.Tensor]:
    base = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
    return {
        "valve": base / 10,
        "tin": 500 + base / 20,
        "local": 10 + base / 30,
        "terminal": 540 + base / 40,
    }


def _metrics(**changes: bool) -> GateCStructuralMetrics:
    values = {
        "finite_rollout": True,
        "sp_prefix_causality": True,
        "constant_action_identity": True,
        "future_truth_isolation": True,
        "local_response_noncollapse": True,
    }
    values.update(changes)
    return GateCStructuralMetrics(**values)


def test_train_only_robust_scales_have_a_frozen_floor() -> None:
    scales = GateCRobustScales.fit(_targets(), split="train", scale_floor=0.1)
    assert set(scales.values) == {"valve", "tin", "local", "terminal"}
    assert all(value >= 0.1 for value in scales.values.values())
    with pytest.raises(Phase35ProtocolError, match="train split"):
        GateCRobustScales.fit(_targets(), split="validation", scale_floor=0.1)
    constant = {key: torch.ones_like(value) for key, value in _targets().items()}
    floored = GateCRobustScales.fit(constant, split="train", scale_floor=0.1)
    assert set(floored.values.values()) == {0.1}


def test_multitask_loss_is_dimensionless_and_weights_are_closed() -> None:
    targets = _targets()
    scales = GateCRobustScales.fit(targets, split="train", scale_floor=0.1)
    predictions = {
        "valve_prediction": targets["valve"].clone(),
        "tin_prediction": targets["tin"].clone(),
        "local_drop_prediction": targets["local"].clone(),
        "terminal_prediction": targets["terminal"].clone(),
        "local_effect": torch.ones_like(targets["local"]),
    }
    breakdown = compute_gatec_loss(predictions, targets, scales, WEIGHTS)
    assert breakdown.total.item() == pytest.approx(0.0)
    bad = dict(WEIGHTS)
    bad["terminal"] += 0.01
    with pytest.raises(Phase35ProtocolError, match="sum to one"):
        compute_gatec_loss(predictions, targets, scales, bad)


def test_selector_applies_structural_gates_before_terminal_score() -> None:
    records = [
        GateCSelectorRecord("good", "forecast_boundary", 0.30, _metrics()),
        GateCSelectorRecord(
            "low_mae_but_collapsed",
            "forecast_boundary",
            0.05,
            _metrics(local_response_noncollapse=False),
        ),
        GateCSelectorRecord("oracle_best", "oracle_boundary", 0.01, _metrics()),
    ]
    selected = select_gatec_checkpoint(records)
    assert selected.checkpoint_id == "good"
    leakage = GateCSelectorRecord(
        "leaky", "forecast_boundary", 0.0, _metrics(future_truth_isolation=False)
    )
    with pytest.raises(Phase35ProtocolError, match="eligible forecast"):
        select_gatec_checkpoint([leakage])


def test_warmup_is_bounded_and_joint_training_unfreezes_every_module() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    assert validate_warmup_updates(10, 100) == 10
    with pytest.raises(Phase35ProtocolError, match="10%"):
        validate_warmup_updates(11, 100)
    freeze_for_warmup(model, trainable_name_fragments=("1",))
    assert any(parameter.requires_grad for parameter in model[1].parameters())
    assert not any(parameter.requires_grad for parameter in model[0].parameters())
    unfreeze_for_joint_training(model)
    assert all(parameter.requires_grad for parameter in model.parameters())
