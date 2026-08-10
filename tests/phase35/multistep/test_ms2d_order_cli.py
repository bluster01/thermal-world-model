import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms2d_order_matrix.json"
RUNNER = ROOT / "experiments/phase3_5/ms2d_order.py"
SUMMARY = ROOT / "experiments/phase3_5/summarize_ms2d_order.py"


def test_ms2d_d2_matrix_expands_to_21_validation_runs_without_test_access():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--matrix", str(MATRIX), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2d-d2-v1"
    assert payload["run_count"] == 21
    assert payload["test_authorized"] is False
    assert {run["regime_id"] for run in payload["runs"]} == {
        "third_order_r50_context_scheduled"
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("synthetic_defaults", "tau_seconds"), [40.0, 70.0]),
        (("synthetic_defaults", "input_delay_steps"), 1),
        (("synthetic_defaults", "noise_std"), 0.01),
        (("training", "epochs"), 301),
        (("gates", "order_aware_relative_improvement_min"), 0.0),
        (("regimes", 0, "candidates", 1, "poles"), 2),
        (("regimes", 0, "candidates", 1, "closure_scale"), 0.03),
    ],
)
def test_ms2d_d2_matrix_rejects_protocol_mutations(tmp_path, path, value):
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


def test_ms2d_d2_cpu_smoke_writes_third_order_truth_and_environment(tmp_path):
    output_root = tmp_path / "ms2o"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--candidate-id",
            "d2_g3_three_pole",
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
    assert manifest["protocol_version"] == "phase3.5-ms2d-d2-v1"
    assert manifest["test_accessed"] is False
    assert manifest["evidence_scope"] == (
        "synthetic_order_pressure_validation_not_field_causality"
    )
    assert manifest["operator_config"]["poles"] == 3
    assert manifest["environment"]["device"] == "cpu"
    assert metrics["truth"]["tau_seconds"] == [40.0, 70.0, 210.0]
    assert metrics["truth"]["input_delay_steps"] == 0
    assert len(metrics["structural_diagnostics"]["operator"]["tau_seconds"]) == 3


def _write_complete_fixture(output_root: Path) -> None:
    from experiments.phase3_5.ms2d_order import FROZEN_EXECUTION_PATHS, _build_configs
    from src.phase35.multistep.staging import environment_payload

    import torch

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    nmae = {
        "d2_g2_two_pole": 0.12,
        "d2_g3_three_pole": 0.08,
        "d2_g3_oracle_structure": 0.02,
        "d2_g2_delay_compensation": 0.09,
        "d2_k4_monotone": 0.11,
        "d2_pi_monotone": 0.13,
        "d2_deeponet": 0.10,
    }
    for candidate in matrix["regimes"][0]["candidates"]:
        candidate_id = candidate["candidate_id"]
        route = candidate["route"]
        operator_config, training_config, synthetic_spec, _ = _build_configs(
            matrix, matrix["regimes"][0], candidate, False
        )
        for seed in matrix["seeds"]:
            run_dir = output_root / f"ms2o_{candidate_id}_s{seed}"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoint_best_val.pt"
            checkpoint.write_bytes(f"{candidate_id}-{seed}".encode())
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = {
                "protocol_version": matrix["protocol_version"],
                "evidence_scope": matrix["evidence_scope"],
                "route_id": candidate_id,
                "seed": seed,
                "git_sha": git_sha,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_selector": "validation_effect_mae",
                "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
                "frozen_execution_paths": list(FROZEN_EXECUTION_PATHS),
                "operator_config": operator_config.to_dict(),
                "training_config": asdict(training_config),
                "synthetic_spec": asdict(
                    replace(
                        synthetic_spec,
                        seed=synthetic_spec.seed + seed * 1_000_003,
                    )
                ),
                "regime_id": "third_order_r50_context_scheduled",
                "candidate_role": candidate["role"],
                "best_epoch": 1,
                "device": "cpu",
                "test_accessed": False,
                "test_authorized": False,
                "environment": environment_payload(torch.device("cpu")),
            }
            operator = {"spectral_radius": 0.95}
            if candidate_id in {
                "d2_g3_three_pole",
                "d2_g3_oracle_structure",
            }:
                operator["tau_seconds"] = [42.0, 68.0, 205.0]
            if candidate_id == "d2_g2_delay_compensation":
                operator.update(
                    delay_weights=[0.85, 0.05, 0.04, 0.03, 0.03],
                    expected_delay_seconds=3.4,
                )
            metrics = {
                "effect_mae": 0.02,
                "clean_effect_mae": 0.01,
                "clean_effect_nmae": nmae[candidate_id],
                "direction_accuracy_clean_nonzero": 1.0,
                "truth": {
                    "truth_regime": "context_scheduled",
                    "truth_opening_map": "equal_percentage_r50",
                    "tau_seconds": [40.0, 70.0, 210.0],
                    "input_delay_steps": 0,
                    "input_delay_seconds": 0.0,
                },
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
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (run_dir / "metrics_validation.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (run_dir / "history.json").write_text(
                '[{"epoch": 1, "validation_effect_mae": 0.1}]',
                encoding="utf-8",
            )


def test_ms2d_d2_summary_separates_primary_order_gates_from_diagnostics(tmp_path):
    output_root = tmp_path / "complete"
    _write_complete_fixture(output_root)
    output = output_root / "summary_validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SUMMARY),
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
    assert summary["oracle_gate"]["all_seeds_pass"] is True
    assert summary["order_aware_absolute_gate"]["all_seeds_pass"] is True
    assert summary["order_aware_response_gate"]["all_seeds_pass"] is True
    assert summary["tau_recovery_diagnostic"]["all_seeds_pass"] is True
    assert summary["no_true_delay_diagnostic"]["all_seeds_pass"] is True
    assert summary["all_primary_gates_pass"] is True
    assert summary["test_accessed"] is False

    tampered_path = output_root / "ms2o_d2_g3_three_pole_s0" / "manifest.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["operator_config"]["poles"] = 2
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(SUMMARY),
            "--matrix",
            str(MATRIX),
            "--output-root",
            str(output_root),
            "--output",
            str(output_root / "summary_tampered.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "operator_config" in rejected.stderr


def test_ms2d_d2_validation_refuses_any_premature_test_artifact(tmp_path):
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
            "d2_g3_three_pole",
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
