from __future__ import annotations

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.multistep.gatec_real_smoke import (
    GateCRealSmokeConfig,
    run_gatec_real_subset_smoke,
)
from src.phase35.schema import MS3_HISTORY_FEATURES


def _caches(rows: int = 1000) -> dict[str, Phase35Cache]:
    rng = np.random.default_rng(51)
    columns = tuple(MS3_HISTORY_FEATURES)
    timestamps = np.arange(rows, dtype=np.int64) * 10_000_000_000
    shared = {
        "机组负荷": 360 + rng.normal(0, 2, rows),
        "主蒸汽压力": 16 + rng.normal(0, 0.1, rows),
        "主给水流量": 1000 + rng.normal(0, 5, rows),
        "未校正总煤量": 160 + rng.normal(0, 2, rows),
        "主蒸汽流量": 1750 + rng.normal(0, 8, rows),
    }
    output = {}
    for side_index, side in enumerate(("A", "B")):
        values = np.zeros((rows, len(columns)), dtype=np.float32)
        for name, series in shared.items():
            values[:, columns.index(name)] = series
        sp = 540 + side_index + np.sin(np.arange(rows) / 35)
        valve = 32 + 2 * side_index + 0.7 * np.sin(np.arange(rows) / 15)
        tin = 552 + side_index + np.sin(np.arange(rows) / 50)
        local = 12 + 0.04 * valve
        values[:, columns.index("二级减温器入口温度")] = tin
        values[:, columns.index("二级减温器出口温度")] = tin - local
        values[:, columns.index("末级过热器出口汽温")] = 540 + side_index + np.cos(
            np.arange(rows) / 60
        )
        values[:, columns.index("二级减温调节阀设定")] = sp
        values[:, columns.index("二级减温调节门阀位")] = valve
        output[side] = Phase35Cache(
            timestamps_ns=timestamps.copy(),
            values=values,
            ages_s=np.zeros_like(values),
            columns=columns,
            metadata={
                "side": side,
                "step_seconds": 10,
                "source": {"sha256": "frozen-real-like"},
            },
        )
    return output


def test_real_subset_smoke_uses_train_validation_and_never_test() -> None:
    config = GateCRealSmokeConfig(
        route="a1phys_three_pole",
        fraction_denominator=10,
        seed=3,
        window=12,
        horizon=6,
        d_model=8,
        latent_dim=6,
        batch_size=16,
        optimizer_updates=3,
        validation_batch_size=32,
        max_validation_anchors=32,
    )
    result = run_gatec_real_subset_smoke(_caches(), config, device="cpu")
    assert result["scope"] == "local_real_train_validation_subset_not_causal"
    assert result["route"] == "a1phys_three_pole"
    assert result["fraction_denominator"] == 10
    assert result["train_anchor_count"] > 0
    assert result["validation_anchor_count"] > 0
    assert result["test_accessed"] is False
    assert result["boundary_mode"] == "forecast_boundary"
    assert result["oracle_boundary_role"] == "diagnostic_ceiling_only"
    assert result["metrics_validation"]["finite"] is True
    assert result["metrics_validation"]["persistence_local_drop_mae_c"] >= 0
    assert "local_not_worse_than_1p05_persistence" in result["baseline_diagnostics"]
    assert result["structural_validation"]["constant_action_identity"] is True
    assert result["automatic_scientific_pass"] is None


def test_real_subset_smoke_is_deterministic_for_frozen_seed() -> None:
    config = GateCRealSmokeConfig(
        route="pi_neural_ode",
        fraction_denominator=10,
        seed=5,
        window=12,
        horizon=6,
        d_model=8,
        latent_dim=6,
        batch_size=16,
        optimizer_updates=2,
        validation_batch_size=32,
        max_validation_anchors=24,
    )
    first = run_gatec_real_subset_smoke(_caches(), config, device="cpu")
    second = run_gatec_real_subset_smoke(_caches(), config, device="cpu")
    assert first["train_anchor_sha256"] == second["train_anchor_sha256"]
    assert first["validation_anchor_sha256"] == second["validation_anchor_sha256"]
    assert first["metrics_validation"] == second["metrics_validation"]


def test_paired_free_marks_response_noncollapse_not_applicable() -> None:
    result = run_gatec_real_subset_smoke(
        _caches(),
        GateCRealSmokeConfig(
            route="none",
            candidate_id="C0_paired_free",
            response_scheduling="none",
            fraction_denominator=10,
            window=12,
            horizon=6,
            d_model=8,
            latent_dim=6,
            batch_size=16,
            optimizer_updates=1,
            validation_batch_size=32,
            max_validation_anchors=24,
        ),
        device="cpu",
    )
    assert result["candidate_id"] == "C0_paired_free"
    assert result["metrics_validation"]["logged_action_effect_mean_abs_c"] == 0.0
    assert result["structural_validation"]["local_response_noncollapse"] is True
    assert (
        result["structural_applicability"]["local_response_noncollapse"]
        == "not_applicable_paired_free"
    )
    assert result["logged_action_auxiliary_used_for_training"] is False
    assert sum(result["training_weights"].values()) == pytest.approx(1.0)
    assert result["training_weights"]["structure"] == 0.0


def test_terminal_only_really_disables_local_and_logged_action_auxiliary() -> None:
    result = run_gatec_real_subset_smoke(
        _caches(),
        GateCRealSmokeConfig(
            route="a1phys_three_pole",
            candidate_id="C5_sched_base_terminal_only",
            local_supervision=False,
            fraction_denominator=10,
            window=12,
            horizon=6,
            d_model=8,
            latent_dim=6,
            batch_size=16,
            optimizer_updates=1,
            validation_batch_size=32,
            max_validation_anchors=24,
        ),
        device="cpu",
    )
    assert result["local_supervision"] is False
    assert result["logged_action_auxiliary_used_for_training"] is False
    assert result["training_weights"]["local"] == 0.0
    assert result["training_weights"]["structure"] == 0.0
    assert "logged_vs_shuffled_local_advantage_c" in result["metrics_validation"]


def test_capacity_candidates_share_anchors_and_order_parameter_counts() -> None:
    results = []
    for capacity in ("small", "base", "large"):
        results.append(
            run_gatec_real_subset_smoke(
                _caches(),
                GateCRealSmokeConfig(
                    route="a1phys_three_pole",
                    residual_capacity=capacity,
                    fraction_denominator=10,
                    seed=7,
                    window=12,
                    horizon=6,
                    d_model=8,
                    latent_dim=6,
                    batch_size=16,
                    optimizer_updates=1,
                    validation_batch_size=32,
                    max_validation_anchors=24,
                ),
                device="cpu",
            )
        )
    assert len({item["train_anchor_sha256"] for item in results}) == 1
    assert len({item["validation_anchor_sha256"] for item in results}) == 1
    counts = [item["trainable_parameter_count"] for item in results]
    assert counts[0] < counts[1] < counts[2]
