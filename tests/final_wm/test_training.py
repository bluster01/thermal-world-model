"""Training-loop contracts: spec validation, ledger, checkpointing, learning."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import CanonicalRecord
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.training import TrainSpec, build_world_model, train_arm


def _record(tmp_path, n: int = 1500) -> CanonicalRecord:
    arrays = synthetic_canonical_arrays(total_steps=n, seed=2)
    path = tmp_path / "record.npz"
    np.savez_compressed(path, **arrays)
    return CanonicalRecord(path)


def _quick_spec(**kw) -> TrainSpec:
    defaults = dict(
        unit="smoke", arm="arm", seed=0, epochs=2, batches_per_epoch=3, batch_size=8,
        history_steps=16, horizon=12, patience=2, eval_windows=16, eval_batch=8,
    )
    defaults.update(kw)
    return TrainSpec(**defaults)


def test_spec_validation_fail_closed() -> None:
    with pytest.raises(FinalWMProtocolError):
        TrainSpec(unit="x", arm="a", seed=0, boundary_mode="forecast", train_boundary=False).validate()
    with pytest.raises(FinalWMProtocolError):
        TrainSpec(unit="x", arm="a", seed=0, boundary_loss_only=True, boundary_mode="oracle").validate()
    # bogus initial_state_mode is caught at model build, not spec validate
    with pytest.raises(FinalWMProtocolError):
        build_world_model(TrainSpec(unit="x", arm="a", seed=0, initial_state_mode="bogus"))


def test_train_arm_writes_ledger_and_checkpoint(tmp_path) -> None:
    record = _record(tmp_path)
    spec = _quick_spec(initial_state_mode="learned", closure_mode="conservative")
    final = train_arm(spec, record, tmp_path / "out")
    assert np.isfinite(final["best_val_nll"])
    ledger = (tmp_path / "out" / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger) >= 2  # at least one epoch + final
    last = json.loads(ledger[-1])
    assert last["final"] is True
    # convergence diagnostics (matrix v0.2)
    assert last["stop_reason"] in ("patience", "cap")
    assert last["converged"] == (last["stop_reason"] == "patience")
    assert 1 <= len(last["val_tail"]) <= 5
    ckpt = tmp_path / "out" / "checkpoints" / "smoke_arm_seed0.pt"
    assert ckpt.exists()


def test_properties_move_with_model_apply() -> None:
    # Executor-fix regression (cc81cb3): properties tensors must follow the
    # module's device/dtype moves; nn.Module._apply skips plain attributes.
    from src.final_wm.contracts import TransitionConfig
    from src.final_wm.properties import AnalyticThermoProperties
    from src.final_wm.transition import Fan2020UDETransition

    transition = Fan2020UDETransition(TransitionConfig(), AnalyticThermoProperties())
    transition.to(dtype=torch.float64)
    assert transition.properties._tsat_p.dtype == torch.float64
    assert transition.properties._hsatv.dtype == torch.float64


def test_train_arm_learns_on_teacher_record(tmp_path) -> None:
    import torch

    from src.final_wm.contracts import WorldModelConfig
    from src.final_wm.model import FinalWorldModel
    from src.final_wm.properties import AnalyticThermoProperties

    torch.manual_seed(0)
    teacher = FinalWorldModel(WorldModelConfig(), AnalyticThermoProperties())
    with torch.no_grad():
        for name, p in teacher.transition.named_parameters():
            if name.startswith("raw_"):
                p.add_(0.15 * torch.randn_like(p))
    arrays = synthetic_canonical_arrays(total_steps=1500, seed=3, teacher=teacher.transition)
    path = tmp_path / "teacher_record.npz"
    np.savez_compressed(path, **arrays)
    record = CanonicalRecord(path)
    spec = _quick_spec(initial_state_mode="learned", epochs=3)
    final = train_arm(spec, record, tmp_path / "out")
    ledger = [json.loads(l) for l in (tmp_path / "out" / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    epoch_vals = [e["val_nll"] for e in ledger if "epoch" in e]
    assert epoch_vals[-1] <= epoch_vals[0] + 1e-6  # no degradation on same-type data


def test_boundary_only_training(tmp_path) -> None:
    record = _record(tmp_path)
    spec = _quick_spec(boundary_mode="forecast", boundary_loss_only=True)
    final = train_arm(spec, record, tmp_path / "out")
    assert np.isfinite(final["best_val_nll"])
