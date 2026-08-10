from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tarfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms2d_disturbance_matrix.json"
RUNNER = ROOT / "experiments/phase3_5/ms2d_disturbance.py"


def test_ms2d_d3_matrix_expands_to_21_validation_runs_without_test_access():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--matrix", str(MATRIX), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2d-d3-v1"
    assert payload["run_count"] == 21
    assert payload["test_authorized"] is False
    assert {run["regime_id"] for run in payload["runs"]} == {
        "third_order_r50_context_scheduled_colored_disturbance"
    }


def test_ms2d_d3_freezes_all_reused_protocol_helpers():
    from experiments.phase3_5.ms2d_disturbance import FROZEN_EXECUTION_PATHS

    required = {
        "experiments/phase3_5/ms2d_delay.py",
        "experiments/phase3_5/ms2d_delay_test.py",
        "experiments/phase3_5/ms2d_order.py",
        "experiments/phase3_5/ms2d_order_test.py",
        "experiments/phase3_5/summarize_ms2d_order_test.py",
        "experiments/phase3_5/multistep_mismatch.py",
    }
    assert required <= set(FROZEN_EXECUTION_PATHS)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("synthetic_defaults", "disturbance_std"), 0.02),
        (("synthetic_defaults", "disturbance_tau_seconds"), 100.0),
        (("synthetic_defaults", "seed"), 20260814),
        (("training", "epochs"), 301),
        (("gates", "disturbance_robust_ci_lower_min"), 0.0),
        (("regimes", 0, "question"), "Can a changed question pass?"),
        (("regimes", 0, "candidates", 1, "poles"), 2),
    ],
)
def test_ms2d_d3_matrix_rejects_protocol_mutations(tmp_path, path, value):
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    target = matrix
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(matrix), encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, str(RUNNER), "--matrix", str(changed), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "frozen" in rejected.stderr.lower()


def test_ms2d_d3_cpu_smoke_writes_disturbance_truth_and_validation_episodes(
    tmp_path,
):
    output_root = tmp_path / "ms2d3"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--candidate-id",
            "d3_g3_three_pole",
            "--seed",
            "0",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--smoke",
            "--execute",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    run_dir = Path(payload["output_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (run_dir / "metrics_validation.json").read_text(encoding="utf-8")
    )
    episodes = json.loads(
        (run_dir / "episode_metrics_validation.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol_version"] == "phase3.5-ms2d-d3-v1"
    assert manifest["test_accessed"] is False
    assert manifest["validation_episode_metrics"] == (
        "episode_metrics_validation.json"
    )
    assert metrics["truth"]["truth_regime"] == "disturbed_context_scheduled"
    assert metrics["truth"]["disturbance_std"] == 0.03
    assert metrics["truth"]["disturbance_tau_seconds"] == 120.0
    assert metrics["truth"]["disturbance_realized_lag1_correlation"] > 0.7
    assert len(episodes["episode_ids"]) == 32
    assert len(episodes["colored_disturbance_mae"]) == 32
    assert not (run_dir / "metrics_test.json").exists()


def test_ms2d_d3_validation_refuses_any_premature_test_artifact(tmp_path):
    output_root = tmp_path / "blocked"
    output_root.mkdir()
    (output_root / "summary_test.json").write_text("{}", encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--candidate-id",
            "d3_g3_three_pole",
            "--seed",
            "0",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--smoke",
            "--execute",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "refuses test artifacts" in rejected.stderr


def _confirmatory_fixture(*, unpaired: bool = False):
    records = {}
    profiles = [0, 1, 2, 3, 4] * 8
    for seed in (0, 1, 2):
        common = {
            "episode_ids": list(range(40)),
            "profile_ids": profiles,
            "trajectory_design_sha256": f"trajectory-{seed}",
        }
        records[("d3_g2_two_pole", seed)] = {
            **common,
            "clean_effect_mae": [1.0] * 40,
            "clean_effect_scale": [1.0] * 40,
        }
        records[("d3_g3_three_pole", seed)] = {
            **common,
            "trajectory_design_sha256": (
                "unpaired"
                if unpaired and seed == 0
                else common["trajectory_design_sha256"]
            ),
            "clean_effect_mae": [0.07] * 40,
            "clean_effect_scale": [1.0] * 40,
        }
        records[("d3_g3_oracle_structure", seed)] = {
            **common,
            "clean_effect_mae": [0.02] * 40,
            "clean_effect_scale": [1.0] * 40,
        }
    return records


def test_d3_confirmatory_gates_use_ci_and_keep_diagnostics_nonblocking():
    from experiments.phase3_5.summarize_ms2d_disturbance import (
        build_confirmatory_gates,
    )

    result = build_confirmatory_gates(
        _confirmatory_fixture(),
        seeds=[0, 1, 2],
        replicates=1000,
        bootstrap_seed=101,
        response_ci_lower_min=0.10,
        oracle_nmae_max=0.05,
        robust_nmae_max=0.10,
        tau_diagnostic_pass=False,
        no_true_delay_diagnostic_pass=False,
    )
    assert result["oracle_gate"]["all_seeds_pass"] is True
    assert result["disturbance_robust_absolute_gate"]["all_seeds_pass"] is True
    assert result["disturbance_robust_response_gate"]["all_seeds_pass"] is True
    assert result["tau_recovery_diagnostic_pass"] is False
    assert result["all_primary_gates_pass"] is True


def test_d3_confirmatory_gates_fail_closed_on_unpaired_episodes():
    from experiments.phase3_5.summarize_ms2d_disturbance import (
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
            robust_nmae_max=0.10,
            tau_diagnostic_pass=True,
            no_true_delay_diagnostic_pass=True,
        )


def _write_complete_summary_fixture(output_root: Path) -> None:
    import torch

    from experiments.phase3_5.ms2d_disturbance import (
        FROZEN_EXECUTION_PATHS,
        _build_configs,
    )
    from src.phase35.multistep.staging import environment_payload

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    errors = {
        "d3_g2_two_pole": 0.10,
        "d3_g3_three_pole": 0.07,
        "d3_g3_oracle_structure": 0.02,
        "d3_g2_delay_compensation": 0.08,
        "d3_k4_monotone": 0.20,
        "d3_pi_monotone": 0.12,
        "d3_deeponet": 0.09,
    }
    for candidate in matrix["regimes"][0]["candidates"]:
        candidate_id = candidate["candidate_id"]
        operator_config, training_config, synthetic_spec, _ = _build_configs(
            matrix, matrix["regimes"][0], candidate, False
        )
        for seed in matrix["seeds"]:
            run_dir = output_root / f"ms2d3_{candidate_id}_s{seed}"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoint_best_val.pt"
            checkpoint.write_bytes(f"{candidate_id}-{seed}".encode())
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trajectory_sha = hashlib.sha256(f"d3-seed-{seed}".encode()).hexdigest()
            environment = environment_payload(torch.device("cpu"))
            manifest = {
                "protocol_version": matrix["protocol_version"],
                "evidence_scope": matrix["evidence_scope"],
                "route_id": candidate_id,
                "seed": seed,
                "git_sha": git_sha,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_selector": "validation_effect_mae",
                "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
                "d2_reference_sha256": matrix["d2_reference"]["sha256"],
                "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
                "operator_config": operator_config.to_dict(),
                "training_config": asdict(training_config),
                "synthetic_spec": asdict(
                    replace(
                        synthetic_spec,
                        seed=synthetic_spec.seed + seed * 1_000_003,
                    )
                ),
                "regime_id": matrix["regimes"][0]["regime_id"],
                "candidate_role": candidate["role"],
                "best_epoch": 1,
                "device": "cpu",
                "test_accessed": False,
                "test_authorized": False,
                "validation_episode_metrics": "episode_metrics_validation.json",
                "validation_trajectory_design_sha256": trajectory_sha,
                "environment": environment,
            }
            operator = {"spectral_radius": 0.95}
            if candidate_id in {
                "d3_g3_three_pole",
                "d3_g3_oracle_structure",
            }:
                operator["tau_seconds"] = [42.0, 68.0, 205.0]
            if candidate_id == "d3_g2_delay_compensation":
                operator.update(
                    delay_weights=[0.30, 0.05, 0.15, 0.25, 0.25],
                    expected_delay_seconds=21.0,
                )
            error = errors[candidate_id]
            truth = {
                "truth_regime": "disturbed_context_scheduled",
                "truth_opening_map": "equal_percentage_r50",
                "tau_seconds": [40.0, 70.0, 210.0],
                "input_delay_steps": 0,
                "input_delay_seconds": 0.0,
                "disturbance_std": 0.03,
                "disturbance_tau_seconds": 120.0,
                "disturbance_rho": 0.9200444146293233,
                "disturbance_realized_mean": 0.0,
                "disturbance_realized_std": 0.03,
                "disturbance_realized_lag1_correlation": 0.91,
                "split": "validation",
            }
            metrics = {
                "effect_mae": error,
                "clean_effect_mae": error,
                "clean_effect_scale": 1.0,
                "clean_effect_nmae": error,
                "direction_accuracy_clean_nonzero": 1.0,
                "sample_count": 256,
                "clean_horizon_mae": {
                    "H1": error,
                    "H6": error,
                    "H18": error,
                    "H60": error,
                },
                "truth": truth,
                "structural_diagnostics": {
                    "reference_identity_max_error": 0.0,
                    "future_action_leakage_max_error": 0.0,
                    "post_change_sensitivity_max_c": 0.1,
                    "positive_step_terminal_effect_max_c": -0.1,
                    "finite_effect": True,
                    "finite_state": True,
                    "operator": operator,
                },
            }
            episodes = {
                "episode_ids": list(range(256)),
                "profile_ids": [index % 5 for index in range(256)],
                "profile_names": ["hold", "step", "pulse", "ramp", "multi_step"],
                "trajectory_design_sha256": trajectory_sha,
                "observed_effect_mae": [error] * 256,
                "clean_effect_mae": [error] * 256,
                "clean_effect_scale": [1.0] * 256,
                "colored_disturbance_mae": [0.03] * 256,
                "colored_disturbance_mean": [0.0] * 256,
                "clean_horizon_absolute_error": {
                    "H1": [error] * 256,
                    "H6": [error] * 256,
                    "H18": [error] * 256,
                    "H60": [error] * 256,
                },
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (run_dir / "metrics_validation.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (run_dir / "history.json").write_text(
                '[{"epoch": 1, "validation_effect_mae": 0.1}]', encoding="utf-8"
            )
            (run_dir / "episode_metrics_validation.json").write_text(
                json.dumps(episodes), encoding="utf-8"
            )


def test_ms2d_d3_summary_audits_episodes_and_archives_all_checkpoints(tmp_path):
    output_root = tmp_path / "complete"
    _write_complete_summary_fixture(output_root)
    output = output_root / "summary_validation.json"
    summary_script = ROOT / "experiments/phase3_5/summarize_ms2d_disturbance.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(summary_script),
            "--matrix",
            str(MATRIX),
            "--output-root",
            str(output_root),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["run_count"] == 21
    assert summary["all_artifact_and_structural_gates_pass"] is True
    assert summary["all_primary_gates_pass"] is True
    assert summary["no_true_delay_diagnostic"]["all_seeds_pass"] is False
    archive_path = Path(summary["checkpoint_archive"]["archive_path"])
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    assert archive_path == output_root / "checkpoints_validation.tar"
    with tarfile.open(archive_path, "r") as archive:
        assert len([member for member in archive.getmembers() if member.isfile()]) == 21
