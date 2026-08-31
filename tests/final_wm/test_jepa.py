from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.final_wm.contracts import BOUNDARY_ELEMENTS, FinalWMProtocolError, PHYSICAL_STATE_ELEMENTS
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL
from src.final_wm.data_v2 import AUX_ELEMENTS, BOUNDARY_EXT_ELEMENTS, N_MILLS
from src.final_wm.jepa import (
    B2SlowState,
    JepaBRecord,
    JepaWindowBatch,
    PrivilegedNormalizer,
    SlicedGaussianCFLoss,
    build_jepa_model,
    fit_privileged_normalizer,
    sample_jepa_windows,
)
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import AnalyticThermoProperties
from experiments.final_wm import jepa_b5_spec
from experiments.final_wm.jepa_b_spec import (
    FROZEN_MATRIX_SHA256,
    load_matrix,
    matrix_sha256,
)
from experiments.final_wm.run_jepa_b import (
    _spec_for,
    _verified_existing_train,
    parse_args,
    sanity_report,
    train_arm,
)


def _record(path: Path, n: int = 320) -> Path:
    t = np.arange(n, dtype=np.float32)
    boundary = np.zeros((n, len(BOUNDARY_ELEMENTS)), dtype=np.float32)
    boundary[:, 0] = 330.0 + 5.0 * np.sin(t / 20.0)
    boundary[:, 1] = 240.0 + 3.0 * np.cos(t / 23.0)
    boundary[:, 2] = 18.0
    boundary[:, 3] = 420.0
    boundary[:, 4] = 280.0
    boundary[:, 5] = 17.0
    boundary[:, 6] = 10.0
    actions = np.stack([0.4 + 0.05 * np.sin(t / 17.0), 0.5 + 0.05 * np.cos(t / 19.0)], axis=1)
    obs = np.stack([500 + i * 10 + np.sin(t / (9 + i)) for i in range(5)], axis=1).astype(np.float32)
    ext = np.zeros((n, len(BOUNDARY_EXT_ELEMENTS)), dtype=np.float32)
    ext[:, BOUNDARY_EXT_ELEMENTS.index("fuel_corrected")] = 240.0
    ext[:, BOUNDARY_EXT_ELEMENTS.index("water_coal_ratio")] = 4.0 + 0.1 * np.sin(t / 13.0)
    ext[:, BOUNDARY_EXT_ELEMENTS.index("unit_load")] = 500.0 + 20.0 * np.sin(t / 29.0)
    for j in range(ext.shape[1]):
        if not np.any(ext[:, j]):
            ext[:, j] = j + 0.01 * t
    aux = np.stack([j + 0.02 * t for j in range(len(AUX_ELEMENTS))], axis=1).astype(np.float32)
    mill = np.tile(np.arange(N_MILLS) % 2, (n, 1)).astype(np.uint8)
    split = np.full(n, SPLIT_TRAIN, dtype=np.int64)
    split[220:] = SPLIT_VAL
    np.savez_compressed(
        path,
        boundary=boundary,
        actions=actions,
        obs=obs,
        valid=np.ones(n, dtype=bool),
        timestamps=np.arange(n, dtype=np.int64) * 10,
        split=split,
        boundary_ext=ext,
        aux=aux,
        mill_on=mill,
    )
    return path


def test_jepa_record_applies_a5_quality_gate(tmp_path: Path):
    path = _record(tmp_path / "v2.npz")
    arrays = dict(np.load(path))
    arrays["boundary_ext"][10, BOUNDARY_EXT_ELEMENTS.index("unit_load")] = 100.0
    np.savez_compressed(path, **arrays)
    record = JepaBRecord(path)
    assert int((record.split == -1).sum()) == 1
    assert record.privileged.shape == (record.n, 32)


def test_privileged_normalizer_is_train_only(tmp_path: Path):
    record = JepaBRecord(_record(tmp_path / "v2.npz"))
    normalizer = fit_privileged_normalizer(record)
    train = record.privileged[record.split == SPLIT_TRAIN]
    assert torch.allclose(normalizer.mean, train.mean(0), atol=1e-6)
    changed = record.privileged.clone()
    changed[record.split == SPLIT_VAL] += 10000.0
    record.privileged = changed
    again = fit_privileged_normalizer(record)
    assert torch.equal(normalizer.mean, again.mean)


def test_fixed_derangement_has_no_fixed_points_and_preserves_windows(tmp_path: Path):
    record = JepaBRecord(_record(tmp_path / "v2.npz"))
    batch = sample_jepa_windows(
        record, SPLIT_TRAIN, 12, 24, 6, torch.Generator().manual_seed(7)
    )
    assert torch.all(batch.partner_future_indices != batch.future_indices)
    assert batch.partner_history_privileged.shape == batch.history_privileged.shape
    assert batch.partner_future_privileged.shape == batch.future_privileged.shape


def test_sliced_gaussian_cf_is_differentiable_and_rejects_bad_shape():
    loss_fn = SlicedGaussianCFLoss(dim=4, num_slices=8, num_knots=9, seed=3)
    x = torch.randn(16, 4, requires_grad=True)
    loss = loss_fn(x)
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(FinalWMProtocolError, match="embedding width"):
        loss_fn(torch.randn(8, 3))


def _batch(history_steps: int = 12, horizon: int = 6, batch_size: int = 4) -> JepaWindowBatch:
    history = HistoryWindow(
        obs=torch.full((batch_size, history_steps, 5), 520.0),
        actions=torch.full((batch_size, history_steps, 2), 0.5),
        boundary=torch.tensor([330.0, 240.0, 18.0, 420.0, 280.0, 17.0, 10.0])
        .view(1, 1, 7).repeat(batch_size, history_steps, 1),
    )
    future_boundary = history.boundary[:, :horizon].clone()
    future_actions = history.actions[:, :horizon].clone()
    future_obs = history.obs[:, :horizon].clone()
    ph = torch.randn(batch_size, history_steps, 32)
    pf = torch.randn(batch_size, horizon, 32)
    return JepaWindowBatch(
        history=history,
        future_boundary=future_boundary,
        future_actions=future_actions,
        future_obs=future_obs,
        history_privileged=ph,
        future_privileged=pf,
        partner_history_privileged=ph.roll(1, 0),
        partner_future_privileged=pf.roll(1, 0),
        future_indices=torch.arange(batch_size),
        partner_future_indices=torch.arange(batch_size).roll(1),
        day_ids=torch.arange(batch_size),
        unit_load=torch.full((batch_size,), 500.0),
    )


def _normalizer() -> PrivilegedNormalizer:
    return PrivilegedNormalizer(torch.zeros(32), torch.ones(32))


def test_b1_and_b3_do_not_change_inference_when_auxiliary_is_disabled():
    props = AnalyticThermoProperties()
    batch = _batch()
    torch.manual_seed(11)
    c0 = build_jepa_model("c0", history_steps=12, properties=props, normalizer=_normalizer())
    for arm in ("b1", "b3", "b3_shuffle"):
        torch.manual_seed(11)
        model = build_jepa_model(arm, history_steps=12, properties=props, normalizer=_normalizer())
        model.base.load_state_dict(c0.base.state_dict())
        with torch.no_grad():
            r0 = c0.forecast(batch.history, batch.future_actions, boundary_mode="oracle",
                             true_future_boundary=batch.future_boundary)
            r1 = model.forecast(batch.history, batch.future_actions, boundary_mode="oracle",
                                true_future_boundary=batch.future_boundary)
        assert torch.equal(r0.temps_mu, r1.temps_mu)


def test_b3_target_encoder_is_action_blind():
    model = build_jepa_model(
        "b3", history_steps=12, properties=AnalyticThermoProperties(), normalizer=_normalizer()
    )
    batch = _batch()
    target0 = model.auxiliary.target_embeddings(model.base, batch)[0]
    changed = batch._replace(future_actions=torch.rand_like(batch.future_actions))
    target1 = model.auxiliary.target_embeddings(model.base, changed)[0]
    assert torch.equal(target0, target1)


def test_b2_slow_state_holds_between_registered_updates():
    slow = B2SlowState(physical_dim=len(PHYSICAL_STATE_ELEMENTS), boundary_dim=7,
                       slow_dim=4, stride=3)
    z = torch.randn(2, 4)
    physical = torch.randn(2, len(PHYSICAL_STATE_ELEMENTS))
    boundary = torch.randn(2, 7)
    assert torch.equal(slow.update(z, physical, boundary, step=1), z)
    assert torch.equal(slow.update(z, physical, boundary, step=2), z)
    assert not torch.equal(slow.update(z, physical, boundary, step=3), z)


def test_b5_slow_state_is_action_blind():
    # B5 update must NOT read the physical state (a function of logged actions).
    slow = B2SlowState(physical_dim=len(PHYSICAL_STATE_ELEMENTS), boundary_dim=7,
                       slow_dim=4, stride=3, use_physical=False)
    z = torch.randn(2, 4)
    physical_a = torch.randn(2, len(PHYSICAL_STATE_ELEMENTS))
    physical_b = torch.randn(2, len(PHYSICAL_STATE_ELEMENTS))
    boundary = torch.randn(2, 7)
    a = slow.update(z, physical_a, boundary, step=3)
    b = slow.update(z, physical_b, boundary, step=3)
    assert torch.equal(a, b), "B5 slow update must be invariant to the physical state"
    assert not torch.equal(a, z), "B5 slow update must still move on registered steps"


def test_b5_identity_gate_off_is_exact():
    # With the slow mechanism scaled to 0, b5 rollout must be bitwise equal to c0.
    model = build_jepa_model(
        "b5", history_steps=12, properties=AnalyticThermoProperties(), normalizer=_normalizer()
    )
    control = build_jepa_model(
        "c0", history_steps=12, properties=AnalyticThermoProperties(), normalizer=_normalizer()
    )
    model.base.load_state_dict(control.base.state_dict())
    model.slow_mechanism_scale = 0.0
    batch = _batch()
    r0 = control.forecast(batch.history, batch.future_actions, boundary_mode="oracle",
                          true_future_boundary=batch.future_boundary)
    r1 = model.forecast(batch.history, batch.future_actions, boundary_mode="oracle",
                        true_future_boundary=batch.future_boundary)
    assert torch.equal(r0.temps_mu, r1.temps_mu)


def test_b4_auxiliary_splits_physical_and_residual_state():
    model = build_jepa_model(
        "b4", history_steps=12, properties=AnalyticThermoProperties(), normalizer=_normalizer()
    )
    batch = _batch()
    result = model.forecast(batch.history, batch.future_actions, boundary_mode="oracle",
                            true_future_boundary=batch.future_boundary)
    terms = model.auxiliary_terms(batch, result)
    assert {"prediction", "gaussian_cf", "static", "dynamic"} <= set(terms)
    assert all(torch.isfinite(value) for value in terms.values())
    assert result.states.shape[-1] == len(PHYSICAL_STATE_ELEMENTS) + 4


def test_frozen_matrix_has_all_and_only_registered_arms():
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/final_wm/jepa_b_series_v1.json"
    matrix = json.loads(path.read_text())
    ids = [arm["id"] for arm in matrix["arms"]]
    assert ids == ["c0", "b1", "b2", "b3", "b3_shuffle", "b4"]
    assert matrix_sha256(path) == FROZEN_MATRIX_SHA256
    assert matrix["execution_contract"]["paper_verdict_authorized"] is False


def test_registry_closes_jepa_batches_after_linux_audit():
    root = Path(__file__).resolve().parents[2]
    registry = json.loads(
        (root / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    assert registry["linux_authorized_gate"] is None
    for experiment_id in ("jepa_b_series", "jepa_b5"):
        experiment = registry["experiments"][experiment_id]
        assert experiment["status"] == "audited"
        assert experiment["protocol_state"]["test_locked"] is True
        assert experiment["protocol_state"]["audited"] is True


def test_b5_arm_is_validated_after_matrix_selects_b5_spec(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    matrix = root / "configs/final_wm/jepa_b5_series_v1.json"
    monkeypatch.setattr(sys, "argv", ["run_jepa_b.py", "--matrix", str(matrix), "--arm", "b5"])
    args = parse_args()
    assert args.arm == "b5"
    spec = _spec_for(args.matrix)
    assert spec is jepa_b5_spec
    assert "b5" in spec.ORDERED_ARMS


def test_b5_registry_authorization_is_fail_closed(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    registry = json.loads(
        (root / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    registry["active_gate"] = "jepa_b5"
    registry["linux_authorized_gate"] = "jepa_b5"
    experiment = registry["experiments"]["jepa_b5"]
    experiment["status"] = "ready_for_linux"
    state = experiment["protocol_state"]
    state.update({"active": True, "ready_for_linux": True, "linux_completed": False,
                  "results_returned": False, "audited": False})
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    assert jepa_b5_spec.require_linux_authorization(path)["status"] == "ready_for_linux"
    state["seed_scope"] = [0, 1]
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(FinalWMProtocolError, match="seed/retry"):
        jepa_b5_spec.require_linux_authorization(path)


def test_sanity_identity_gates_all_registered_mechanisms(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    matrix = load_matrix(root / "configs/final_wm/jepa_b_series_v1.json")
    record = JepaBRecord(_record(tmp_path / "v2.npz", n=420))
    report = sanity_report(
        matrix, record, fit_privileged_normalizer(record),
        AnalyticThermoProperties(), torch.device("cpu"),
    )
    assert all(item["exact"] for item in report["identities"].values())
    assert report["privileged_dim"] == 32


def test_quick_control_training_smoke_writes_bound_checkpoint(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    matrix_path = root / "configs/final_wm/jepa_b_series_v1.json"
    matrix = load_matrix(matrix_path)
    record = JepaBRecord(_record(tmp_path / "v2.npz", n=420))
    normalizer = fit_privileged_normalizer(record)
    final = train_arm(
        "c0", matrix, matrix_sha256(matrix_path), record, normalizer,
        AnalyticThermoProperties(), tmp_path / "out", torch.device("cpu"), quick=True,
    )
    checkpoint = tmp_path / "out/c0/checkpoints/jepa_b_c0_seed0.pt"
    assert final["final"] is True
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["matrix_sha256"] == matrix_sha256(matrix_path)


def test_resume_requires_report_final_ledger_and_matching_checkpoint(tmp_path: Path):
    report = tmp_path / "report.json"
    ledger = tmp_path / "ledger.jsonl"
    checkpoint = tmp_path / "checkpoint.pt"
    payload = {"arm": "c0", "commit": "abc", "matrix_sha256": "matrix"}
    report.write_text(json.dumps({**payload, "train": {"best_epoch": 2}}), encoding="utf-8")
    ledger.write_text(json.dumps({**payload, "final": True}) + "\n", encoding="utf-8")
    with pytest.raises(FinalWMProtocolError, match="complete ledger/checkpoint"):
        _verified_existing_train("c0", report, ledger, checkpoint, "abc", "matrix")
    torch.save({**payload, "matrix_sha256": "wrong"}, checkpoint)
    with pytest.raises(FinalWMProtocolError, match="matrix mismatch"):
        _verified_existing_train("c0", report, ledger, checkpoint, "abc", "matrix")
    torch.save(payload, checkpoint)
    assert _verified_existing_train(
        "c0", report, ledger, checkpoint, "abc", "matrix"
    ) == {"best_epoch": 2}
