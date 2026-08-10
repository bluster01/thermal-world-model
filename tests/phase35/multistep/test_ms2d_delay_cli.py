import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms2d_delay_matrix.json"
RUNNER = ROOT / "experiments/phase3_5/ms2d_delay.py"


def test_ms2d_d1_matrix_expands_to_18_validation_runs_without_test_access():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--matrix", str(MATRIX), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2d-d1-v1"
    assert payload["run_count"] == 18
    assert payload["test_authorized"] is False
    assert {run["regime_id"] for run in payload["runs"]} == {
        "pure_delay_r50_context_scheduled"
    }


def test_ms2d_d1_cpu_smoke_writes_delay_truth_and_diagnostics(tmp_path):
    output_root = tmp_path / "ms2d"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--candidate-id",
            "d1_g2_learned_delay",
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
    assert manifest["protocol_version"] == "phase3.5-ms2d-d1-v1"
    assert manifest["test_accessed"] is False
    assert manifest["evidence_scope"] == "synthetic_delay_pressure_validation"
    assert metrics["truth"]["input_delay_steps"] == 2
    operator = metrics["structural_diagnostics"]["operator"]
    assert len(operator["delay_weights"]) == 5
    assert 0 <= operator["expected_delay_seconds"] <= 40


def test_ms2d_summary_applies_oracle_response_and_delay_gates(tmp_path):
    from experiments.phase3_5.ms2d_delay import FROZEN_EXECUTION_PATHS, _build_configs

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output_root = tmp_path / "complete"
    nmae = {
        "d1_g2_no_delay": 0.40,
        "d1_g2_learned_delay": 0.10,
        "d1_g2_oracle_delay": 0.02,
        "d1_k4_monotone": 0.30,
        "d1_pi_monotone": 0.15,
        "d1_deeponet": 0.12,
    }
    for candidate in matrix["regimes"][0]["candidates"]:
        candidate_id = candidate["candidate_id"]
        route = candidate["route"]
        operator_config, training_config, synthetic_spec, _ = _build_configs(
            matrix, matrix["regimes"][0], candidate, False
        )
        for seed in matrix["seeds"]:
            run_dir = output_root / f"ms2d_{candidate_id}_s{seed}"
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
                "regime_id": "pure_delay_r50_context_scheduled",
                "candidate_role": candidate["role"],
                "best_epoch": 1,
                "test_accessed": False,
                "test_authorized": False,
            }
            operator = {"spectral_radius": 0.95}
            if candidate_id == "d1_g2_learned_delay":
                operator.update(
                    delay_weights=[0.02, 0.08, 0.80, 0.08, 0.02],
                    expected_delay_seconds=20.0,
                )
            elif candidate_id == "d1_g2_oracle_delay":
                operator.update(
                    delay_weights=[0, 0, 1, 0, 0],
                    expected_delay_seconds=20.0,
                )
            metrics = {
                "effect_mae": 0.02,
                "clean_effect_mae": 0.01,
                "clean_effect_nmae": nmae[candidate_id],
                "direction_accuracy_clean_nonzero": 1.0,
                "truth": {
                    "truth_regime": "delayed_context_scheduled",
                    "truth_opening_map": "equal_percentage_r50",
                    "input_delay_steps": 2,
                    "input_delay_seconds": 20.0,
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
                '[{"epoch": 1}]', encoding="utf-8"
            )

    output = output_root / "summary_validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/phase3_5/summarize_ms2d_delay.py"),
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
    assert summary["run_count"] == 18
    assert summary["all_artifact_and_structural_gates_pass"] is True
    assert summary["oracle_gate"]["all_seeds_pass"] is True
    assert summary["delay_response_gate"]["all_seeds_pass"] is True
    assert summary["delay_identification_diagnostic"]["all_seeds_within_one_step"] is True
    assert summary["delay_identification_diagnostic"]["all_seeds_pass"] is True
    assert summary["test_accessed"] is False

    tampered_path = output_root / "ms2d_d1_g2_learned_delay_s0" / "manifest.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["operator_config"]["context_scheduled"] = False
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/phase3_5/summarize_ms2d_delay.py"),
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
