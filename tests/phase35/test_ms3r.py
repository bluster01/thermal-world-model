from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.phase35.data import Phase35Cache
from src.phase35.ms3r import (
    rank_diagnostics,
    run_gate1_analysis,
    validate_aligned_caches,
    validate_ms3r_gate1_config,
)
from src.phase35.schema import MS3_HISTORY_FEATURES, Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/phase3_5/ms3r_gate1_point_identifiability.json"


def _config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["data_contract"]["source_sha256"] = "synthetic"
    config["data_contract"]["step_seconds"] = 3600
    config["analysis"].update(
        {
            "history_lags_steps": [1, 2],
            "horizons_steps": [1, 2, 4],
            "rolling_folds": 2,
            "minimum_training_days": 10,
            "minimum_evaluation_days": 1,
            "minimum_rows_per_fold": 50,
            "day_shift_steps": 24,
        }
    )
    config["rank_analysis"].update(
        {
            "hankel_windows_steps": [2, 4],
            "hankel_stride_steps": 1,
            "maximum_hankel_rows": 1000,
        }
    )
    config["statistics"].update(
        {
            "bootstrap_samples": 200,
            "minimum_day_slope_rows": 20,
        }
    )
    return config


def _cache(side: str, seed: int) -> Phase35Cache:
    rng = np.random.default_rng(seed)
    n = 3000
    columns = tuple(MS3_HISTORY_FEATURES)
    values = np.zeros((n, len(columns)), dtype=np.float32)
    defaults = {
        "机组负荷": 350.0,
        "主蒸汽压力": 15.0,
        "主给水流量": 1000.0,
        "未校正总煤量": 150.0,
        "主蒸汽流量": 1800.0,
        "二级减温器入口温度": 550.0,
        "二级减温器出口温度": 545.0,
        "末级过热器出口汽温": 540.0,
        "二级减温调节阀设定": 540.0,
        "二级减温调节门阀位": 50.0,
    }
    for column, value in defaults.items():
        values[:, columns.index(column)] = value

    dv = rng.normal(0.0, 0.25, n)
    valve = 50.0 + np.cumsum(dv) * 0.05
    sp = 540.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    tin = 550.0 + rng.normal(0.0, 0.03, n)
    local_drop = 5.0 + rng.normal(0.0, 0.03, n)
    local_drop[1:] += 1.5 * dv[:-1]
    terminal = 540.0 + rng.normal(0.0, 0.03, n)
    terminal[2:] -= 0.8 * dv[:-2]
    values[:, columns.index("二级减温调节门阀位")] = valve
    values[:, columns.index("二级减温调节阀设定")] = sp
    values[:, columns.index("二级减温器入口温度")] = tin
    values[:, columns.index("二级减温器出口温度")] = tin - local_drop
    values[:, columns.index("末级过热器出口汽温")] = terminal
    timestamps = np.arange(n, dtype=np.int64) * 3_600_000_000_000
    return Phase35Cache(
        timestamps_ns=timestamps,
        values=values,
        ages_s=np.zeros_like(values),
        columns=columns,
        metadata={
            "side": side,
            "step_seconds": 3600,
            "source": {"sha256": "synthetic"},
        },
    )


def test_repository_config_is_validation_only_and_semantically_conservative():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_ms3r_gate1_config(config)
    parent = ROOT / config["parent_matrix"]["path"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == config["parent_matrix"]["sha256"]
    changed = copy.deepcopy(config)
    changed["data_contract"]["split"] = "test"
    with pytest.raises(Phase35ProtocolError, match="validation-only"):
        validate_ms3r_gate1_config(changed)
    changed = copy.deepcopy(config)
    changed["branch_semantics"]["free_future_action_access"] = True
    with pytest.raises(Phase35ProtocolError, match="must not read future action"):
        validate_ms3r_gate1_config(changed)


def test_cache_contract_rejects_timestamp_misalignment():
    config = _config()
    a, b = _cache("A", 1), _cache("B", 2)
    validate_aligned_caches({"A": a, "B": b}, config)
    b.timestamps_ns = b.timestamps_ns + 1
    with pytest.raises(Phase35ProtocolError, match="align exactly"):
        validate_aligned_caches({"A": a, "B": b}, config)


def test_gate1_recovers_correct_local_path_and_not_wrong_side_or_lead():
    summary, arrays = run_gate1_analysis(
        {"A": _cache("A", 10), "B": _cache("B", 20)}, _config()
    )
    correct = summary["path_diagnostics"]["A"]["correct_local_drop"]["H1"]
    wrong = summary["path_diagnostics"]["A"]["wrong_side_local_drop"]["H1"]
    correct_effect = correct["positive_lag"]["oriented_coefficient"]
    wrong_effect = abs(wrong["positive_lag"]["oriented_coefficient"])
    lead_effect = abs(correct["action_lead_placebo"]["oriented_coefficient"])
    assert correct_effect > 20.0
    assert correct_effect > 3.0 * wrong_effect
    assert correct_effect > 10.0 * lead_effect
    assert (
        correct["positive_lag"]["incremental_r2_diagnostic"]
        > wrong["positive_lag"]["incremental_r2_diagnostic"]
    )
    assert summary["automatic_scientific_pass"] is None
    assert summary["test_accessed"] is False
    assert np.isfinite(arrays["innovation_A"]).sum() > 100
    probe = summary["branch_semantics"]["structural_probe"]
    assert probe["residual_branch_future_action_permutation_max_error"] < 1e-6
    assert probe["response_constant_action_identity_max_error"] < 1e-6
    assert probe["future_action_prefix_leakage_max_error"] < 1e-6
    assert summary["action_information_audit"]["A"]["crossfit_sample_count"] > 100


def test_rank_diagnostics_distinguishes_independent_and_collinear_inputs():
    config = _config()
    rng = np.random.default_rng(5)
    a = rng.normal(size=2000)
    independent = rank_diagnostics({"A": a, "B": rng.normal(size=2000)}, config)
    collinear = rank_diagnostics({"A": a, "B": a.copy()}, config)
    assert independent["differential_to_common_energy_ratio"] > 0.7
    assert independent["condition_number"] < 2.0
    assert collinear["differential_to_common_energy_ratio"] == pytest.approx(0.0)
    assert collinear["condition_number"] is None
    assert collinear["condition_number_is_infinite"] is True
    assert collinear["automatic_rank_pass"] is None
