from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.phase3_5.ms3r_rm3_train import _verify_complete, dry_run_payload, execute_matrix


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms3r_rm3_matrix.json"


def test_rm3_train_dry_run_closes_36_plus_12_without_authorizing_linux() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload = dry_run_payload(matrix)
    assert payload["prediction_run_count"] == 36
    assert payload["calibration_unit_count"] == 12
    assert payload["total_run_count"] == 48
    assert payload["calibration_candidates_per_unit"] == [
        "R0_linear_mimo", "R1_a1_scheduled", "R2_a1_common_only"
    ]
    assert payload["matrix_self_authorizing"] is False
    assert payload["registry_authorization_required_for_execute"] is True
    assert payload["automatic_scientific_pass"] is None


def test_execute_refuses_current_unauthorized_registry(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="active and linux_authorized gate"):
        execute_matrix(
            matrix_path=MATRIX, cache_paths={"A": tmp_path / "a", "B": tmp_path / "b"},
            output_root=tmp_path / "out", devices=["cpu"], skip_complete=True, require_clean=False,
        )


def test_complete_detection_rejects_incomplete_directory(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "artifact_ledger.json").write_text("{}", encoding="utf-8")
    assert _verify_complete(tmp_path, ["x.json", "artifact_ledger.json"]) is False


def test_train_cli_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "experiments/phase3_5/ms3r_rm3_train.py"), "--dry-run"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["total_run_count"] == 48
