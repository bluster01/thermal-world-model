import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.phase35.multistep.training import evaluate_synthetic_test_checkpoint


ROOT = Path(__file__).resolve().parents[3]


def test_multistep_cli_dry_run_and_cpu_smoke(tmp_path):
    script = ROOT / "experiments/phase3_5/multistep_sysid.py"
    matrix = ROOT / "configs/phase3_5/multistep_operator_matrix.json"
    dry = subprocess.run(
        [sys.executable, str(script), "--matrix", str(matrix), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["run_count"] == 18
    assert {run["route_id"] for run in payload["runs"]} >= {
        "graybox_1p", "graybox_2p", "koopman_k2", "koopman_k4", "pi_ode", "causal_deeponet"
    }

    output_root = tmp_path / "runs"
    smoke = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix", str(matrix),
            "--route-id", "graybox_2p",
            "--seed", "0",
            "--output-root", str(output_root),
            "--device", "cpu",
            "--smoke",
            "--execute",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(smoke.stdout)
    run_dir = Path(result["output_dir"])
    assert run_dir.is_dir()
    for name in ("manifest.json", "history.json", "checkpoint_best_val.pt", "metrics_validation.json"):
        assert (run_dir / name).is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_accessed"] is False
    assert manifest["checkpoint_selector"] == "validation_effect_mae"

    metrics = evaluate_synthetic_test_checkpoint(run_dir, test_samples=16, device="cpu", repo_root=ROOT)
    assert metrics["sample_count"] == 16
    assert (run_dir / "metrics_test.json").is_file()
    ledger = json.loads((run_dir / "synthetic_test_access_ledger.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "completed"
    with pytest.raises(RuntimeError, match="repeat synthetic test access"):
        evaluate_synthetic_test_checkpoint(run_dir, test_samples=16, device="cpu", repo_root=ROOT)
