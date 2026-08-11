from __future__ import annotations

import json

import pytest
import torch

from src.phase35.multistep.contracts import OperatorConfig
from src.phase35.multistep.full_training import (
    FullCouplingTrainingConfig,
    build_full_model,
    component_metrics,
    evaluate_full_model,
    train_full_synthetic_run,
)
from src.phase35.multistep.synthetic import SyntheticSpec, generate_synthetic_split


def _spec(samples: int = 40) -> SyntheticSpec:
    return SyntheticSpec(
        samples=samples,
        horizon=12,
        context_dim=4,
        seed=20260817,
        noise_std=0.02,
        gain_c_per_pct=-0.10,
        tau_seconds=(40.0, 70.0, 210.0),
        truth_regime="full_coupled_context_scheduled",
        truth_opening_map="equal_percentage_r50",
        context_gain_log_scale=0.35,
        context_tau_log_scale=0.30,
        free_trajectory_scale=1.0,
        action_context_coupling_pct=4.0,
    )


def _operator() -> OperatorConfig:
    return OperatorConfig(
        route="graybox",
        horizon=12,
        context_dim=4,
        opening_map="monotone",
        poles=3,
        context_scheduled=True,
    )


def test_component_metrics_separate_total_free_and_response():
    batch = generate_synthetic_split(_spec(), "validation")
    prediction = batch.clean_total.clone()
    free = batch.clean_free.clone()
    effect = batch.clean_effect.clone()
    metrics, episodes = component_metrics(batch, prediction, free, effect)
    assert metrics["total_clean_nmae"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["free_clean_nmae"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["response_clean_nmae"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["response_amplitude_ratio"] == pytest.approx(1.0)
    assert len(episodes["episode_ids"]) == 40
    assert episodes["profile_ids"] == batch.profile_ids.tolist()


def test_free_only_model_has_exact_zero_response_and_action_blind_free_head():
    batch = generate_synthetic_split(_spec(), "validation")
    model = build_full_model(_operator(), free_hidden_dim=16, mode="free_only")
    first = model(batch.context, batch.action, batch.reference)
    changed = batch.action.clone()
    changed[:, 6:] = (changed[:, 6:] + 5.0).clamp(0.0, 100.0)
    second = model(batch.context, changed, batch.reference)
    assert torch.count_nonzero(first["effect"]).item() == 0
    torch.testing.assert_close(
        first["free_prediction"], second["free_prediction"], atol=0, rtol=0
    )


def test_full_training_config_rejects_unclosed_stage_budget():
    with pytest.raises(ValueError, match="sum"):
        FullCouplingTrainingConfig(
            epochs=12,
            stage_a_epochs=3,
            stage_b_epochs=3,
            stage_c_epochs=3,
        ).validate()


@pytest.mark.parametrize(
    "mode", ["free_only", "joint_total", "staged_total", "component_oracle"]
)
def test_all_full_training_modes_complete_cpu_smoke(tmp_path, mode):
    output = tmp_path / mode
    result = train_full_synthetic_run(
        operator_config=_operator(),
        full_config=FullCouplingTrainingConfig(
            batch_size=16,
            epochs=6,
            patience=2,
            learning_rate=0.002,
            stage_a_epochs=2,
            stage_b_epochs=2,
            stage_c_epochs=2,
            stage_patience=2,
        ),
        synthetic_spec=_spec(samples=40),
        validation_samples=40,
        seed=0,
        mode=mode,
        route_id=f"smoke_{mode}",
        output_dir=output,
        device="cpu",
        overwrite=False,
        protocol_version="phase3.5-ms5-smoke",
    )
    assert result.checkpoint.is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (output / "metrics_validation.json").read_text(encoding="utf-8")
    )
    episodes = json.loads(
        (output / "episode_metrics_validation.json").read_text(encoding="utf-8")
    )
    assert manifest["training_mode"] == mode
    assert manifest["test_accessed"] is False
    assert metrics["sample_count"] == 40
    assert len(episodes["episode_ids"]) == 40
    assert metrics["structural_diagnostics"]["free_future_action_leakage_max_error"] == 0
    if mode == "free_only":
        assert metrics["response_amplitude_ratio"] == 0
    if mode == "staged_total":
        assert [item["stage"] for item in manifest["stage_summaries"]] == [
            "stage_a_free_hold",
            "stage_b_response_frozen_free",
            "stage_c_low_lr_joint",
        ]
        assert len(manifest["stage_checkpoints"]) == 3
        history = json.loads((output / "history.json").read_text(encoding="utf-8"))
        assert all(
            item["response_gradient_norm"] == 0
            for item in history
            if item["phase"] == "stage_a_free_hold"
        )
        assert all(
            item["free_gradient_norm"] == 0
            for item in history
            if item["phase"] == "stage_b_response_frozen_free"
        )
    else:
        assert manifest["stage_checkpoints"] == []


def test_evaluate_full_model_reports_finite_component_diagnostics():
    batch = generate_synthetic_split(_spec(), "validation")
    model = build_full_model(_operator(), free_hidden_dim=16, mode="joint_total")
    metrics, episodes = evaluate_full_model(model, batch, torch.device("cpu"))
    assert metrics["structural_diagnostics"]["finite_prediction"] is True
    assert metrics["structural_diagnostics"]["reference_identity_max_error"] == 0
    assert len(episodes["response_clean_mae"]) == len(batch.context)
