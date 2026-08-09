import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_ms2_matrix_expands_to_two_regimes_and_33_validation_runs():
    script = ROOT / "experiments/phase3_5/multistep_mismatch.py"
    matrix = ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--matrix", str(matrix), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2-v1"
    assert payload["run_count"] == 33
    assert {run["regime_id"] for run in payload["runs"]} == {
        "valve_nonlinear_r50",
        "context_scheduled_2p",
    }


def test_ms2_cpu_smoke_writes_clean_truth_metrics_and_checkpoint_hash(tmp_path):
    script = ROOT / "experiments/phase3_5/multistep_mismatch.py"
    matrix = ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"
    output_root = tmp_path / "ms2"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            str(matrix),
            "--candidate-id",
            "c_g2_scheduled",
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
    metrics = json.loads((run_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == "phase3.5-ms2-v1"
    assert len(manifest["checkpoint_sha256"]) == 64
    assert payload["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert metrics["clean_effect_mae"] >= 0
    assert metrics["clean_effect_nmae"] >= 0
    assert metrics["clean_effect_scale"] > 0
    assert metrics["noise_mae"] >= 0
    assert metrics["direction_accuracy_clean_nonzero"] is not None
    assert manifest["test_accessed"] is False


def test_ms2_summary_requires_and_hashes_all_33_validation_checkpoints(tmp_path):
    matrix_path = ROOT / "configs/phase3_5/multistep_mismatch_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output_root = tmp_path / "complete"
    for regime in matrix["regimes"]:
        for candidate in regime["candidates"]:
            candidate_id = candidate["candidate_id"]
            route = candidate["route"]
            for seed in matrix["seeds"]:
                run_dir = output_root / f"ms2_{candidate_id}_s{seed}"
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "checkpoint_best_val.pt"
                checkpoint.write_bytes(f"{candidate_id}-{seed}".encode())
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                nmae = 0.5 if candidate_id in {"v_g2_identity", "c_g2_global"} else 0.3
                manifest = {
                    "protocol_version": "phase3.5-ms2-v1",
                    "route_id": candidate_id,
                    "seed": seed,
                    "git_sha": git_sha,
                    "test_accessed": False,
                    "best_epoch": 1,
                    "elapsed_seconds": 1.0,
                    "checkpoint_sha256": checkpoint_hash,
                }
                metrics = {
                    "effect_mae": 0.02,
                    "clean_effect_mae": 0.01,
                    "clean_effect_nmae": nmae,
                    "direction_accuracy_clean_nonzero": 1.0,
                    "structural_diagnostics": {
                        "reference_identity_max_error": 0.0,
                        "future_action_leakage_max_error": 0.0,
                        "post_change_sensitivity_max_c": 0.1,
                        "positive_step_terminal_effect_max_c": -0.1,
                        "finite_effect": True,
                        "finite_state": True,
                        "operator": {"spectral_radius": 0.95 if route in {"graybox", "koopman"} else None},
                    },
                }
                (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (run_dir / "metrics_validation.json").write_text(json.dumps(metrics), encoding="utf-8")
                (run_dir / "history.json").write_text(json.dumps([{"epoch": 1}]), encoding="utf-8")
    summary_path = output_root / "summary_validation.json"
    script = ROOT / "experiments/phase3_5/summarize_multistep_mismatch.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            str(matrix_path),
            "--output-root",
            str(output_root),
            "--output",
            str(summary_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["run_count"] == 33
    assert summary["all_artifact_and_structural_gates_pass"] is True
    assert all(
        contrast["meets_20pct_screen"]
        for contrast in summary["primary_contrasts"].values()
    )
