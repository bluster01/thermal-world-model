from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms3_real_adaptation_matrix.json"
RUNNER = ROOT / "experiments/phase3_5/ms3_real_adaptation.py"


def test_ms3_matrix_expands_to_12_validation_runs_and_freezes_cross_pairing():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--matrix", str(MATRIX), "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["protocol_version"] == "phase3.5-ms3-v1"
    assert payload["run_count"] == 12
    assert payload["test_authorized"] is False
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert matrix["data_contract"]["side_mappings"]["A"]["control_loop"] == (
        "A_valve_to_right_B_thermal_train"
    )
    assert matrix["data_contract"]["side_mappings"]["B"]["control_loop"] == (
        "B_valve_to_left_A_thermal_train"
    )


def test_ms3_matrix_rejects_threshold_mutation(tmp_path):
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix["gates"]["min_dynamic_windows"] = 100
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
