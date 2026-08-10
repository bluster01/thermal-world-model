from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.phase3_5.summarize_ms2d_delay_test import (
    build_confirmatory_gates,
    paired_stratified_bootstrap_relative_improvement,
)


ROOT = Path(__file__).resolve().parents[3]
AUTHORIZATION = ROOT / "configs/phase3_5/ms2d_delay_test_authorization.json"
RUNNER = ROOT / "experiments/phase3_5/ms2d_delay_test.py"


def test_ms2d1_test_dry_run_preflights_18_frozen_runs_or_refuses_repeat():
    root_ledger = (
        ROOT
        / "results/phase3_5/ms2d_delay/synthetic_test_matrix_access_ledger.json"
    )
    if root_ledger.exists():
        before = root_ledger.read_bytes()
        ledger = json.loads(before)
        assert ledger["status"] == "completed"
        assert ledger["run_count"] == 18
        refused = subprocess.run(
            [sys.executable, str(RUNNER), "--dry-run"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0
        assert "refusing repeat or partial" in refused.stderr
        assert root_ledger.read_bytes() == before
        return

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2d-d1-test-v1"
    assert payload["run_count"] == 18
    assert payload["archive_member_count"] == 18
    assert payload["test_accessed"] is False
    assert payload["validation_screening_pass"] is True
    assert payload["delay_parameter_diagnostic_pass"] is False
    assert not root_ledger.exists()


def test_ms2d1_test_requires_explicit_authorization():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--evaluate-test-matrix"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--allow-synthetic-test" in completed.stderr


def test_ms2d1_test_rejects_content_address_mismatch(tmp_path):
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    authorization["matrix"]["sha256"] = "0" * 64
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
    assert "pinned matrix sha256 mismatch" in completed.stderr


def test_ms2d1_test_rejects_gate_drift_from_training_matrix(tmp_path):
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    authorization["gates"]["delay_response_ci_lower_min"] = 0.19
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
    assert "test gates differ from the frozen validation matrix" in completed.stderr


def test_paired_bootstrap_is_deterministic_and_uses_episode_pairing():
    profiles = [0, 1, 2, 3, 4] * 8
    baseline = [1.0 + 0.01 * index for index in range(40)]
    candidate = [0.7 * value for value in baseline]
    result = paired_stratified_bootstrap_relative_improvement(
        candidate, baseline, profiles, replicates=1000, seed=17
    )
    assert result["observed"] == pytest.approx(0.30)
    assert result["ci95"] == pytest.approx([0.30, 0.30])
    assert result == paired_stratified_bootstrap_relative_improvement(
        candidate, baseline, profiles, replicates=1000, seed=17
    )


def test_confirmatory_gates_keep_response_and_parameter_recovery_separate():
    records = {}
    profiles = [0, 1, 2, 3, 4] * 8
    for seed in (0, 1, 2):
        common = {
            "episode_ids": list(range(40)),
            "profile_ids": profiles,
            "trajectory_design_sha256": f"trajectory-{seed}",
        }
        records[("d1_g2_no_delay", seed)] = {
            **common,
            "clean_effect_mae": [1.0] * 40,
        }
        records[("d1_g2_learned_delay", seed)] = {
            **common,
            "clean_effect_mae": [0.7] * 40,
        }
        records[("d1_g2_oracle_delay", seed)] = {
            **common,
            "clean_effect_mae": [0.02] * 40,
            "clean_effect_scale": [1.0] * 40,
        }
    result = build_confirmatory_gates(
        records,
        seeds=[0, 1, 2],
        replicates=1000,
        bootstrap_seed=101,
        response_ci_lower_min=0.20,
        oracle_nmae_max=0.05,
        delay_parameter_diagnostic_pass=False,
    )
    assert result["oracle_test"]["all_seeds_pass"] is True
    assert result["delay_response_test"]["all_seeds_pass"] is True
    assert result["delay_parameter_diagnostic_pass"] is False
    assert result["all_confirmatory_gates_pass"] is True


def test_confirmatory_gates_fail_closed_on_unpaired_episodes():
    records = {}
    for candidate in (
        "d1_g2_no_delay",
        "d1_g2_learned_delay",
        "d1_g2_oracle_delay",
    ):
        records[(candidate, 0)] = {
            "episode_ids": [0, 1],
            "profile_ids": [0, 1],
            "trajectory_design_sha256": (
                "different" if candidate == "d1_g2_learned_delay" else "a"
            ),
            "clean_effect_mae": [0.1, 0.1],
            "clean_effect_scale": [1.0, 1.0],
        }
    with pytest.raises(RuntimeError, match="unpaired"):
        build_confirmatory_gates(
            records,
            seeds=[0],
            replicates=1000,
            bootstrap_seed=1,
            response_ci_lower_min=0.20,
            oracle_nmae_max=0.05,
            delay_parameter_diagnostic_pass=False,
        )
