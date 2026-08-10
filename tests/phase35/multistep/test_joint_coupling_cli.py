from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/joint_coupling_matrix.json"
RUNNER = ROOT / "experiments/phase3_5/joint_coupling.py"
SUMMARY = ROOT / "experiments/phase3_5/summarize_joint_coupling.py"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def test_ms2j_dry_run_freezes_27_validation_runs():
    completed = _run("--matrix", str(MATRIX), "--dry-run")
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms2j-v1"
    assert payload["run_count"] == 27
    assert len({(run["candidate_id"], run["seed"]) for run in payload["runs"]}) == 27
    assert sum(run["training_mode"] == "staged" for run in payload["runs"]) == 3


def test_ms2j_joint_and_staged_cpu_smoke(tmp_path):
    output_root = tmp_path / "smoke"
    for candidate_id in (
        "j_g2_monotone_scheduled_joint",
        "j_g2_monotone_scheduled_staged",
    ):
        completed = _run(
            "--matrix",
            str(MATRIX),
            "--candidate-id",
            candidate_id,
            "--seed",
            "0",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--execute",
            "--smoke",
        )
        payload = json.loads(completed.stdout)
        assert payload["status"] == "completed"
        run_dir = output_root / f"ms2j_{candidate_id}_s0"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["training_mode"] == (
            "staged" if candidate_id.endswith("staged") else "joint"
        )
        assert manifest["test_accessed"] is False
        assert manifest["environment"]["python"]
        assert manifest["validation_trajectory_design_sha256"]
        assert not (run_dir / "metrics_test.json").exists()
        checkpoint = run_dir / "checkpoint_best_val.pt"
        assert manifest["checkpoint_sha256"] == hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        if manifest["training_mode"] == "staged":
            assert [stage["stage"] for stage in manifest["stage_summaries"]] == [
                "stage_a",
                "stage_b",
                "stage_c",
            ]
            assert set(manifest["stage_checkpoints"]) == {
                "stage_a",
                "stage_b",
                "stage_c",
            }
            summaries = {
                item["stage"]: set(item["trainable_parameters"])
                for item in manifest["stage_summaries"]
            }
            assert {"raw_gain", "raw_tau", "opening_map.raw_slopes"} <= summaries[
                "stage_a"
            ]
            assert not any("schedule" in name for name in summaries["stage_a"])
            assert summaries["stage_b"] == {
                "gain_schedule.weight",
                "tau_schedule.weight",
            }
            assert summaries["stage_c"] == summaries["stage_a"] | summaries[
                "stage_b"
            ]
            assert (run_dir / "metrics_stage_a_validation.json").is_file()
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            assert checkpoint_payload["stage"] == "stage_c"
            assert checkpoint_payload["route_id"] == candidate_id


def test_ms2j_summary_enforces_module_and_staging_gates(tmp_path):
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    output_root = tmp_path / "complete"
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    staged_id = "j_g2_monotone_scheduled_staged"
    joint_id = "j_g2_monotone_scheduled_joint"
    single_ids = {"j_g2_monotone_global", "j_g2_identity_scheduled"}
    for candidate in matrix["regimes"][0]["candidates"]:
        candidate_id = candidate["candidate_id"]
        for seed in matrix["seeds"]:
            run_dir = output_root / f"ms2j_{candidate_id}_s{seed}"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoint_best_val.pt"
            checkpoint.write_bytes(f"{candidate_id}-{seed}".encode())
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            nmae = 0.20
            if candidate_id in single_ids:
                nmae = 0.40
            elif candidate_id == joint_id:
                nmae = 0.10
            elif candidate_id == staged_id:
                nmae = 0.105
            metrics = {
                "effect_mae": 0.02,
                "clean_effect_mae": nmae * 0.05,
                "clean_effect_nmae": nmae,
                "direction_accuracy_clean_nonzero": 1.0,
                "structural_diagnostics": {
                    "reference_identity_max_error": 0.0,
                    "future_action_leakage_max_error": 0.0,
                    "post_change_sensitivity_max_c": 0.1,
                    "positive_step_terminal_effect_max_c": -0.1,
                    "finite_effect": True,
                    "finite_state": True,
                    "operator": {
                        "spectral_radius": 0.95
                        if candidate["route"] in {"graybox", "koopman"}
                        else None
                    },
                },
            }
            manifest = {
                "protocol_version": matrix["protocol_version"],
                "route_id": candidate_id,
                "seed": seed,
                "git_sha": git_sha,
                "training_mode": candidate["training_mode"],
                "checkpoint_sha256": checkpoint_sha,
                "test_accessed": False,
                "environment": {
                    "python": "test",
                    "torch": "test",
                    "cuda_runtime": None,
                    "cuda_available": False,
                    "device": "cpu",
                    "platform": "test",
                },
                "validation_trajectory_design_sha256": f"design-{seed}",
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "metrics_validation.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (run_dir / "history.json").write_text(
                json.dumps([{"epoch": 1}]), encoding="utf-8"
            )
            if candidate_id == staged_id:
                stage_a_metrics = dict(metrics)
                stage_a_metrics["clean_effect_nmae"] = 0.50
                (run_dir / "metrics_stage_a_validation.json").write_text(
                    json.dumps(stage_a_metrics), encoding="utf-8"
                )
                stage_checkpoints = {}
                for stage in ("stage_a", "stage_b", "stage_c"):
                    path = run_dir / f"checkpoint_{stage}.pt"
                    path.write_bytes(f"{candidate_id}-{seed}-{stage}".encode())
                    stage_checkpoints[stage] = {
                        "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                manifest["stage_summaries"] = [
                    {"stage": stage, "optimizer_updates": 1}
                    for stage in ("stage_a", "stage_b", "stage_c")
                ]
                manifest["stage_checkpoints"] = stage_checkpoints
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

    completed = subprocess.run(
        [
            sys.executable,
            str(SUMMARY),
            "--matrix",
            str(MATRIX),
            "--output-root",
            str(output_root),
            "--output",
            str(output_root / "summary_validation.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["run_count"] == 27
    assert summary["all_gates_pass"] is True
    assert summary["joint_module_gate"]["all_seed_improvements_exceed_20pct"] is True
    assert summary["staged_stability_gate"]["all_seeds_pass"] is True
