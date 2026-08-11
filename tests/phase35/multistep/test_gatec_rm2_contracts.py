from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.phase35.multistep.gatec_rm2_contracts import (
    partition_rm2_runs,
    rm2_run_specs,
    validate_rm2_matrix,
)
from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_gatec_rm2_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_rm2_matrix_expands_to_closed_54_run_batch() -> None:
    matrix = _matrix()
    validate_rm2_matrix(matrix)
    runs = rm2_run_specs(matrix)
    assert len(runs) == 54
    assert len({run.run_id for run in runs}) == 54
    assert {run.seed for run in runs} == {0, 1, 2}
    assert {run.fold_id for run in runs} == {"F0", "F1"}
    assert {run.group for run in runs} == {"A", "B", "C"}
    assert sum(run.group == "A" for run in runs) == 24
    assert sum(run.group == "B" for run in runs) == 18
    assert sum(run.group == "C" for run in runs) == 12
    assert all(run.validation_fraction[1] <= 0.8 for run in runs)


def test_rm2_static_worker_partition_is_complete_and_disjoint() -> None:
    specs = rm2_run_specs(_matrix())
    partitions = partition_rm2_runs(specs, 4)
    flattened = [run.run_id for partition in partitions for run in partition]
    assert len(flattened) == 54
    assert set(flattened) == {run.run_id for run in specs}
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1


def test_rm2_rejects_test_fold_candidate_and_attempt_drift() -> None:
    changed = copy.deepcopy(_matrix())
    changed["data_contract"]["folds"][1]["validation_fraction"][1] = 0.9
    with pytest.raises(Phase35ProtocolError, match="fold definitions"):
        validate_rm2_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["candidates"][0]["candidate_id"] = changed["candidates"][1]["candidate_id"]
    with pytest.raises(Phase35ProtocolError, match=r"4\+3\+2"):
        validate_rm2_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["execution_contract"]["maximum_attempts_per_run"] = 2
    with pytest.raises(Phase35ProtocolError, match="exactly one attempt"):
        validate_rm2_matrix(changed)
