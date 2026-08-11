from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.phase35.data import Phase35Cache


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
    assert payload["protocol_version"] == "phase3.5-ms3-v1.1"
    assert payload["run_count"] == 12
    assert payload["test_authorized"] is False
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert matrix["data_contract"]["timestamp_storage_unit"] == "ns"
    assert matrix["data_contract"]["irregular_transition_count"] == 282
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


def test_ms3_cache_validation_rejects_pre_v11_timestamp_units():
    from experiments.phase3_5.ms3_real_adaptation import _validate_cache

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    contract = matrix["data_contract"]
    side_contract = contract["side_mappings"]["A"]
    matrix_sha = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    rows = 3
    cache = Phase35Cache(
        timestamps_ns=contract["grid_start_ns"]
        + np.arange(rows, dtype=np.int64) * 10_000_000_000,
        values=np.zeros((rows, len(contract["history_features"])), dtype=np.float32),
        ages_s=np.zeros((rows, len(contract["history_features"])), dtype=np.float32),
        columns=tuple(contract["history_features"]),
        metadata={
            "side": "A",
            "cross_pairing_frozen": True,
            "control_loop": side_contract["control_loop"],
            "column_map": side_contract["column_map"],
            "source": {"sha256": contract["source_sha256"]},
            "matrix_sha256": matrix_sha,
            "timestamp_storage_unit": "us",
            "grid_start_ns": contract["grid_start_ns"],
            "grid_end_ns": contract["grid_end_ns"],
            "grid_rows": contract["source_rows"],
            "irregular_transition_count": contract["irregular_transition_count"],
            "max_transition_seconds": contract["max_transition_seconds"],
        },
    )
    with pytest.raises(ValueError, match="timeline contract"):
        _validate_cache(cache, "A", matrix, matrix_sha)
    cache.metadata["timestamp_storage_unit"] = "ns"
    with pytest.raises(ValueError, match="cache contents"):
        _validate_cache(cache, "A", matrix, matrix_sha)
