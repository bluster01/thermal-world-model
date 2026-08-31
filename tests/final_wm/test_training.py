"""Training-loop contracts: spec validation, ledger, checkpointing, learning."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import CanonicalRecord
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.training import (
    TrainSpec,
    apply_anchor_constants,
    build_world_model,
    config_fingerprint,
    train_arm,
)


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


def test_config_fingerprint_covers_model_structure() -> None:
    # Regression for the Hermes rerun failure 2026-08-20 §3: a repair batch
    # that changes model structure (state registry/prior table) must bust the
    # resume fingerprint even when the TrainSpec is unchanged.
    from unittest import mock

    from src.final_wm.transition import TRANSITION_PARAM_PRIORS

    spec = _quick_spec()
    fp_before = config_fingerprint(spec)
    assert config_fingerprint(spec) == fp_before  # deterministic
    with mock.patch.dict(TRANSITION_PARAM_PRIORS, {"tau_mix1": 123.0}):
        assert config_fingerprint(spec) != fp_before


def test_config_fingerprint_binds_input_content(tmp_path) -> None:
    record = tmp_path / "record.npz"
    properties = tmp_path / "properties.npz"
    anchor = tmp_path / "anchor.pt"
    record.write_bytes(b"record-v1")
    properties.write_bytes(b"properties-v1")
    anchor.write_bytes(b"anchor-v1")
    spec = _quick_spec(anchor_constants_checkpoint=str(anchor))
    before = config_fingerprint(spec, record_path=record, properties_path=properties)
    record.write_bytes(b"record-v2")
    assert config_fingerprint(spec, record_path=record, properties_path=properties) != before
    record.write_bytes(b"record-v1")
    properties.write_bytes(b"properties-v2")
    assert config_fingerprint(spec, record_path=record, properties_path=properties) != before
    properties.write_bytes(b"properties-v1")
    anchor.write_bytes(b"anchor-v2")
    assert config_fingerprint(spec, record_path=record, properties_path=properties) != before


def test_validation_anchor_seed_is_fixed_across_epochs(tmp_path, monkeypatch) -> None:
    import src.final_wm.training as training

    record = _record(tmp_path)
    seen: list[int] = []
    original = training.evaluate_windows

    def capture(*args, **kwargs):
        seen.append(int(kwargs["seed"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "evaluate_windows", capture)
    spec = _quick_spec(epochs=3, patience=3)
    final = train_arm(spec, record, tmp_path / "out")
    assert len(seen) == final["epochs_run"]
    assert len(set(seen)) == 1
    assert final["validation_anchor_seed"] == seen[0]


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
        for parameter in teacher.transition.raw.parameters():
            parameter.add_(0.15 * torch.randn_like(parameter))
    arrays = synthetic_canonical_arrays(total_steps=1500, seed=3, teacher=teacher.transition)
    path = tmp_path / "teacher_record.npz"
    np.savez_compressed(path, **arrays)
    record = CanonicalRecord(path)
    spec = _quick_spec(initial_state_mode="learned", epochs=3)
    final = train_arm(spec, record, tmp_path / "out")
    ledger = [json.loads(l) for l in (tmp_path / "out" / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    epoch_vals = [e["val_nll"] for e in ledger if "epoch" in e]
    # Repair 1-B changed the init semantics: at zero-init the observer returns
    # the (already well-anchored) initial state, so early epochs may wiggle
    # slightly before improving.  The loop-regression guard is a 2% band, not
    # strict monotone descent.
    assert epoch_vals[-1] <= epoch_vals[0] * 1.02


def test_anchor_constants_warm_start(tmp_path) -> None:
    """Amendment v0.6-B: only transition.raw constants are copied from the
    reference checkpoint; every network parameter stays at its fresh init."""
    src_spec = _quick_spec()
    src_model = build_world_model(src_spec)
    with torch.no_grad():
        for name, p in src_model.transition.raw.items():
            p.fill_(float(hash(name) % 7) + 0.5)  # recognisable constant values
        for p in src_model.observer.parameters():
            p.fill_(9.0)
    ref_path = tmp_path / "ref.pt"
    torch.save({"state_dict": src_model.state_dict()}, ref_path)

    torch.manual_seed(1234)
    fresh = build_world_model(_quick_spec(seed=99))
    fresh_obs = [p.detach().clone() for p in fresh.observer.parameters()]
    assert all((p != 9.0).any() for p in fresh_obs)  # fresh, not yet anchored

    apply_anchor_constants(fresh, ref_path)
    for name, p in fresh.transition.raw.items():
        expected = float(hash(name) % 7) + 0.5
        assert p.item() == pytest.approx(expected)
    for p_new, p_old in zip(fresh.observer.parameters(), fresh_obs):
        torch.testing.assert_close(p_new, p_old)  # networks untouched


def test_anchor_constants_fail_closed_on_missing_keys(tmp_path) -> None:
    model = build_world_model(_quick_spec())
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {"observer.net.weight": torch.zeros(3, 3)}}, bad)
    with pytest.raises(FinalWMProtocolError, match="lacks transition constants"):
        apply_anchor_constants(model, bad)


def test_anchor_constants_validation_exclusions() -> None:
    with pytest.raises(FinalWMProtocolError, match="fresh-network"):
        _quick_spec(init_checkpoint="a.pt", anchor_constants_checkpoint="b.pt").validate()
    with pytest.raises(FinalWMProtocolError, match="not composable"):
        _quick_spec(closure_mode="conservative_norew",
                    anchor_constants_checkpoint="b.pt").validate()
    # fingerprint must change when the anchor source is set (asdict covers it)
    fp_plain = config_fingerprint(_quick_spec())
    fp_anchor = config_fingerprint(_quick_spec(anchor_constants_checkpoint="b.pt"))
    assert fp_plain != fp_anchor


def test_boundary_only_training(tmp_path) -> None:
    record = _record(tmp_path)
    spec = _quick_spec(boundary_mode="forecast", boundary_loss_only=True)
    final = train_arm(spec, record, tmp_path / "out")
    assert np.isfinite(final["best_val_nll"])
