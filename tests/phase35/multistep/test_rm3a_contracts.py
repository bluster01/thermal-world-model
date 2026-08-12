from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.phase35.multistep.rm3a_contracts import rm3a_run_specs, validate_rm3a_matrix
from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms3r_rm3a_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_rm3a_matrix_closes_30_new_runs_and_parent_pin() -> None:
    matrix = _matrix()
    validate_rm3a_matrix(matrix, repo_root=ROOT)
    runs = rm3a_run_specs(matrix)
    assert len(runs) == len({run.run_id for run in runs}) == 30
    assert {run.candidate_id for run in runs} == {
        "A0_p3_large", "A1_p4_large", "A2_p5_small",
        "A3_p5_local35", "A4_p5_local50",
    }
    assert matrix["execution_contract"]["linux_authorized"] is False
    assert matrix["execution_contract"]["test_authorized"] is False


def test_rm3a_rejects_candidate_and_loss_drift() -> None:
    changed = copy.deepcopy(_matrix())
    changed["new_candidates"][0]["d_model"] = 80
    with pytest.raises(Phase35ProtocolError, match="candidate matrix"):
        validate_rm3a_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["loss_profiles"]["balanced"]["local"] = 0.5
    with pytest.raises(Phase35ProtocolError, match="sum to one"):
        validate_rm3a_matrix(changed)
