from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/phase3_5/ms3r_gatec_model_screen.py"
SUMMARIZER = ROOT / "experiments/phase3_5/summarize_ms3r_gatec_model_screen.py"


def _run(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def test_dry_run_prints_frozen_rm0_rm1_matrix_and_budget() -> None:
    completed = _run("--dry-run", "--json")
    payload = json.loads(completed.stdout)
    assert payload["split"] == "validation"
    assert payload["test_allowed"] is False
    assert payload["linux_authorized"] is False
    assert payload["real_training_authorized"] is False
    assert len(payload["rm0_structural_adapters"]) == 5
    assert len(payload["rm1_attribution"]) == 6
    assert len(payload["rm1_operator"]) == 4
    assert payload["rm1_training_run_count"] == 10
    assert payload["budget"]["optimizer_updates_per_run_cap"] == 4000
    assert payload["budget"]["rm1_optimizer_updates_cap"] == 40000


def test_synthetic_smoke_writes_pinned_diagnostic_artifacts(tmp_path: Path) -> None:
    _run(
        "--synthetic-smoke",
        "C1_additive_base",
        "--output-dir",
        str(tmp_path),
        "--json",
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (tmp_path / "synthetic_diagnostics.json").read_text(encoding="utf-8")
    )
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["execution_git_sha"]) == 40
    assert manifest["cache_sha256"] is None
    assert manifest["boundary_mode"] == "forecast_boundary"
    assert manifest["operator_route"] == "a1phys_three_pole"
    assert manifest["test_accessed"] is False
    assert manifest["real_training_executed"] is False
    assert manifest["linux_authorized"] is False
    assert manifest["selector_eligible"] is True
    assert diagnostics["known_truth_relative_gain_error"] < 0.08
    assert "supervisor_decision" not in manifest


def test_real_training_fails_closed_while_matrix_is_not_authorized() -> None:
    completed = _run("--real-run", "C1_additive_base", check=False)
    assert completed.returncode != 0
    assert "not authorized" in (completed.stdout + completed.stderr)


def test_operator_smoke_all_records_four_real_implementations(tmp_path: Path) -> None:
    completed = _run(
        "--operator-smoke-all",
        "--operator-seeds",
        "7",
        "--operator-steps",
        "2",
        "--output-dir",
        str(tmp_path),
        "--json",
    )
    payload = json.loads(completed.stdout)
    assert payload["scope"] == "local_route_specific_known_truth_training"
    assert {item["route"] for item in payload["results"]} == {
        "a1phys_three_pole",
        "stable_koopman_lpv",
        "pi_neural_ode",
        "deeponet_response",
    }
    assert payload["real_training_executed"] is False
    assert payload["automatic_scientific_pass"] is None
    assert "supervisor_decision" not in payload
    assert (tmp_path / "operator_recovery_local.json").is_file()


def test_summarizer_is_diagnostic_only(tmp_path: Path) -> None:
    _run(
        "--synthetic-smoke",
        "C1_additive_base",
        "--output-dir",
        str(tmp_path),
        "--json",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SUMMARIZER),
            "--input-dir",
            str(tmp_path),
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["scope"] == "local_synthetic_diagnostic_only"
    assert payload["run_count"] == 1
    assert payload["automatic_scientific_pass"] is None
    assert "supervisor_decision" not in payload
