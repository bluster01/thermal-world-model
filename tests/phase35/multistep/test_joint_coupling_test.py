from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.phase3_5.summarize_joint_coupling_test import (
    build_confirmatory_gates,
    paired_stratified_bootstrap_ratio,
    paired_stratified_bootstrap_relative_improvement,
)


ROOT = Path(__file__).resolve().parents[3]
AUTHORIZATION = ROOT / "configs/phase3_5/joint_coupling_test_authorization.json"
RUNNER = ROOT / "experiments/phase3_5/joint_coupling_test.py"


def test_ms2j_test_preflight_is_state_aware_and_refuses_repeat_access():
    root_ledger = (
        ROOT
        / "results/phase3_5/joint_coupling/synthetic_test_matrix_access_ledger.json"
    )
    if root_ledger.exists():
        before = root_ledger.read_bytes()
        ledger = json.loads(before)
        assert ledger["status"] == "completed"
        assert ledger["run_count"] == 27
        assert len(ledger["completed_runs"]) == 27
        refused = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--authorization",
                str(AUTHORIZATION),
                "--dry-run",
            ],
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
        [
            sys.executable,
            str(RUNNER),
            "--authorization",
            str(AUTHORIZATION),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2j-test-v1"
    assert payload["run_count"] == 27
    assert payload["archive_member_count"] == 36
    assert payload["test_accessed"] is False
    assert payload["frozen_validation_status"] == {
        "all_gates_pass": False,
        "joint_module_gate_pass": True,
        "staged_stability_gate_pass": False,
    }
    assert not root_ledger.exists()


def test_ms2j_test_requires_explicit_matrix_authorization():
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--authorization",
            str(AUTHORIZATION),
            "--evaluate-test-matrix",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--allow-synthetic-test" in completed.stderr


def test_ms2j_test_dry_run_rejects_content_address_mismatch(tmp_path):
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


def test_paired_bootstrap_statistics_use_episode_pairing_and_are_deterministic():
    profiles = [0, 1, 2, 3, 4] * 4
    baseline = [1.0 + 0.01 * index for index in range(20)]
    candidate = [0.5 * value for value in baseline]
    improvement = paired_stratified_bootstrap_relative_improvement(
        candidate,
        baseline,
        profiles,
        replicates=1000,
        seed=17,
    )
    assert improvement["observed"] == pytest.approx(0.5)
    assert improvement["ci95"] == pytest.approx([0.5, 0.5])
    assert improvement == paired_stratified_bootstrap_relative_improvement(
        candidate,
        baseline,
        profiles,
        replicates=1000,
        seed=17,
    )

    ratio = paired_stratified_bootstrap_ratio(
        [1.05 * value for value in baseline],
        baseline,
        profiles,
        replicates=1000,
        seed=23,
    )
    assert ratio["observed"] == pytest.approx(1.05)
    assert ratio["ci95"] == pytest.approx([1.05, 1.05])


def test_confirmatory_gates_keep_joint_and_staged_decisions_separate():
    seeds = [0, 1, 2]
    profiles = [0, 1, 2, 3, 4] * 4
    records = {}
    for seed in seeds:
        common = {
            "episode_ids": list(range(20)),
            "profile_ids": profiles,
            "trajectory_design_sha256": f"trajectory-{seed}",
        }
        records[("j_g2_monotone_scheduled_joint", seed)] = {
            **common,
            "clean_effect_mae": [0.5] * 20,
        }
        records[("j_g2_monotone_global", seed)] = {
            **common,
            "clean_effect_mae": [1.0] * 20,
        }
        records[("j_g2_identity_scheduled", seed)] = {
            **common,
            "clean_effect_mae": [1.1] * 20,
        }
        records[("j_g2_monotone_scheduled_staged", seed)] = {
            **common,
            "clean_effect_mae": [0.6] * 20,
        }
        records[("j_g2_monotone_scheduled_staged:stage_a", seed)] = {
            **common,
            "clean_effect_mae": [1.2] * 20,
        }

    result = build_confirmatory_gates(
        records,
        seeds,
        replicates=1000,
        bootstrap_seed=101,
        joint_improvement_min=0.20,
        staged_to_joint_ratio_max=1.10,
        staged_stage_a_improvement_min=0.20,
    )
    assert result["joint_module_test"]["all_seeds_pass"] is True
    assert result["staged_noninferiority_test"]["all_seeds_pass"] is False
    assert result["staged_vs_stage_a_test"]["all_seeds_pass"] is True
    assert result["all_confirmatory_gates_pass"] is False


def test_confirmatory_gates_fail_closed_on_unpaired_episodes():
    records = {
        ("j_g2_monotone_scheduled_joint", 0): {
            "episode_ids": [0, 1],
            "profile_ids": [0, 1],
            "trajectory_design_sha256": "a",
            "clean_effect_mae": [0.1, 0.1],
        },
        ("j_g2_monotone_global", 0): {
            "episode_ids": [0, 1],
            "profile_ids": [0, 1],
            "trajectory_design_sha256": "b",
            "clean_effect_mae": [0.2, 0.2],
        },
        ("j_g2_identity_scheduled", 0): {
            "episode_ids": [0, 1],
            "profile_ids": [0, 1],
            "trajectory_design_sha256": "a",
            "clean_effect_mae": [0.2, 0.2],
        },
        ("j_g2_monotone_scheduled_staged", 0): {
            "episode_ids": [0, 1],
            "profile_ids": [0, 1],
            "trajectory_design_sha256": "a",
            "clean_effect_mae": [0.1, 0.1],
        },
        ("j_g2_monotone_scheduled_staged:stage_a", 0): {
            "episode_ids": [0, 1],
            "profile_ids": [0, 1],
            "trajectory_design_sha256": "a",
            "clean_effect_mae": [0.2, 0.2],
        },
    }
    with pytest.raises(RuntimeError, match="unpaired"):
        build_confirmatory_gates(
            records,
            [0],
            replicates=1000,
            bootstrap_seed=1,
            joint_improvement_min=0.20,
            staged_to_joint_ratio_max=1.10,
            staged_stage_a_improvement_min=0.20,
        )
