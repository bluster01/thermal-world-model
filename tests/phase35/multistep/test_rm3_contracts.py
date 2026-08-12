from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.phase35.multistep.rm3_contracts import (
    rm3_identification_specs,
    rm3_prediction_specs,
    validate_rm3_matrix,
)
from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms3r_rm3_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_rm3_matrix_closes_identification_and_fair_prediction_tables() -> None:
    matrix = _matrix()
    validate_rm3_matrix(matrix)
    assert len(rm3_identification_specs(matrix)) == 3
    assert len(rm3_prediction_specs(matrix)) == 6
    assert matrix["execution_contract"]["linux_authorized"] is False
    assert matrix["execution_contract"]["test_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data_contract", "raw_future_valve_auxiliary_allowed"), True, "raw future-valve"),
        (("execution_contract", "linux_authorized"), True, "linux_authorized"),
        (("execution_contract", "test_authorized"), True, "test_authorized"),
        (("execution_contract", "ms4_authorized"), True, "ms4_authorized"),
    ],
)
def test_rm3_matrix_rejects_scope_expansion(path, value, message) -> None:
    matrix = copy.deepcopy(_matrix())
    matrix[path[0]][path[1]] = value
    with pytest.raises(Phase35ProtocolError, match=message):
        validate_rm3_matrix(matrix)


def test_rm3_logged_future_valve_is_only_an_oracle_upper_bound() -> None:
    matrix = copy.deepcopy(_matrix())
    matrix["prediction_candidates"][1]["future_action_access"] = "logged_future_valve"
    with pytest.raises(Phase35ProtocolError, match="oracle upper bound"):
        validate_rm3_matrix(matrix)
