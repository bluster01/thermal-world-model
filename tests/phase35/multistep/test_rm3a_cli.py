from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.phase3_5.ms3r_rm3a_train import dry_run_payload, execute


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms3r_rm3a_matrix.json"


def test_rm3a_dry_run_expands_only_30_new_runs() -> None:
    payload = dry_run_payload(json.loads(MATRIX.read_text(encoding="utf-8")))
    assert payload["new_candidate_count"] == 5
    assert payload["new_run_count"] == 30
    assert payload["reused_reference_run_count"] == 18
    assert payload["matrix_self_authorizing"] is False
    assert payload["test_authorized"] is False


def test_rm3a_execute_refuses_current_unauthorized_registry(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="active and linux_authorized"):
        execute(MATRIX, tmp_path / "a", tmp_path / "b", tmp_path / "out", ["cpu"])


def test_rm3a_cli_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "experiments/phase3_5/ms3r_rm3a_train.py"), "--dry-run"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["new_run_count"] == 30
