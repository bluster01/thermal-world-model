from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
AUTHORIZATION = ROOT / "configs/phase3_5/ms2d_order_test_authorization.json"
RUNNER = ROOT / "experiments/phase3_5/ms2d_order_test.py"


def test_ms2d2_test_dry_run_preflights_21_frozen_runs_without_access():
    root_ledger = (
        ROOT
        / "results/phase3_5/ms2d_order/synthetic_test_matrix_access_ledger.json"
    )
    assert not root_ledger.exists()
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2d-d2-test-v1"
    assert payload["run_count"] == 21
    assert payload["archive_member_count"] == 21
    assert payload["validation_screening_pass"] is True
    assert payload["tau_recovery_diagnostic_pass"] is True
    assert payload["no_true_delay_diagnostic_pass"] is False
    assert payload["test_accessed"] is False
    assert not root_ledger.exists()


def test_ms2d2_test_requires_explicit_authorization():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--evaluate-test-matrix"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--allow-synthetic-test" in completed.stderr


def test_ms2d2_test_rejects_content_address_mismatch(tmp_path):
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    authorization["checkpoint_archive"]["sha256"] = "0" * 64
    altered = tmp_path / "authorization.json"
    altered.write_text(json.dumps(authorization), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--authorization",
            str(altered),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "pinned checkpoint archive sha256 mismatch" in completed.stderr


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("gates", "order_aware_response_ci_lower_min", 0.09),
        ("gates", "order_aware_clean_nmae_max", 0.11),
        ("diagnostics", "tau_set_log_mae_max", 0.36),
    ],
)
def test_ms2d2_test_rejects_gate_or_diagnostic_drift(
    tmp_path, section, key, value
):
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    authorization[section][key] = value
    altered = tmp_path / "authorization.json"
    altered.write_text(json.dumps(authorization), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--authorization",
            str(altered),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "differ from the frozen validation matrix" in completed.stderr


def _confirmatory_fixture(*, unpaired: bool = False):
    records = {}
    profiles = [0, 1, 2, 3, 4] * 8
    for seed in (0, 1, 2):
        common = {
            "episode_ids": list(range(40)),
            "profile_ids": profiles,
            "trajectory_design_sha256": f"trajectory-{seed}",
        }
        records[("d2_g2_two_pole", seed)] = {
            **common,
            "clean_effect_mae": [1.0] * 40,
            "clean_effect_scale": [1.0] * 40,
        }
        records[("d2_g3_three_pole", seed)] = {
            **common,
            "trajectory_design_sha256": (
                "unpaired" if unpaired and seed == 0 else common["trajectory_design_sha256"]
            ),
            "clean_effect_mae": [0.07] * 40,
            "clean_effect_scale": [1.0] * 40,
        }
        records[("d2_g3_oracle_structure", seed)] = {
            **common,
            "clean_effect_mae": [0.02] * 40,
            "clean_effect_scale": [1.0] * 40,
        }
    return records


def test_d2_confirmatory_gates_keep_response_and_diagnostics_separate():
    from experiments.phase3_5.summarize_ms2d_order_test import (
        build_confirmatory_gates,
    )

    result = build_confirmatory_gates(
        _confirmatory_fixture(),
        seeds=[0, 1, 2],
        replicates=1000,
        bootstrap_seed=101,
        response_ci_lower_min=0.10,
        oracle_nmae_max=0.05,
        order_aware_nmae_max=0.10,
        tau_diagnostic_pass=True,
        no_true_delay_diagnostic_pass=False,
    )
    assert result["oracle_test"]["all_seeds_pass"] is True
    assert result["order_aware_absolute_test"]["all_seeds_pass"] is True
    assert result["order_aware_response_test"]["all_seeds_pass"] is True
    assert result["no_true_delay_diagnostic_pass"] is False
    assert result["all_confirmatory_gates_pass"] is True


def test_d2_confirmatory_gates_fail_closed_on_unpaired_episodes():
    from experiments.phase3_5.summarize_ms2d_order_test import (
        build_confirmatory_gates,
    )

    with pytest.raises(RuntimeError, match="unpaired"):
        build_confirmatory_gates(
            _confirmatory_fixture(unpaired=True),
            seeds=[0, 1, 2],
            replicates=1000,
            bootstrap_seed=101,
            response_ci_lower_min=0.10,
            oracle_nmae_max=0.05,
            order_aware_nmae_max=0.10,
            tau_diagnostic_pass=True,
            no_true_delay_diagnostic_pass=False,
        )
