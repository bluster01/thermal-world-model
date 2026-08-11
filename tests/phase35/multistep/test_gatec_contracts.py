from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_gatec_model_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_repository_gatec_matrix_is_closed_and_validation_only():
    from src.phase35.multistep.gatec_contracts import (
        BOUNDARY_MODES,
        RESPONSE_ROUTES,
        gatec_run_specs,
        validate_gatec_matrix,
    )

    matrix = _matrix()
    validate_gatec_matrix(matrix)
    assert BOUNDARY_MODES == {
        "oracle_boundary",
        "forecast_boundary",
        "scenario_boundary",
    }
    assert RESPONSE_ROUTES == {
        "a1phys_three_pole",
        "stable_koopman_lpv",
        "pi_neural_ode",
        "deeponet_response",
        "none",
    }
    attribution = gatec_run_specs(matrix, "rm1_attribution")
    operators = gatec_run_specs(matrix, "rm1_operator")
    assert len(attribution) == 6
    assert len(operators) == 4
    assert all(run.split == "validation" for run in attribution + operators)
    assert all(run.seed == 0 and run.fold == 0 for run in attribution + operators)
    assert matrix["execution_contract"]["linux_authorized"] is False


def test_gatec_rejects_test_or_oracle_checkpoint_selection():
    from src.phase35.multistep.gatec_contracts import validate_gatec_matrix

    changed = copy.deepcopy(_matrix())
    changed["data_contract"]["split"] = "test"
    with pytest.raises(Phase35ProtocolError, match="validation-only"):
        validate_gatec_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["selector"]["primary_boundary_mode"] = "oracle_boundary"
    with pytest.raises(Phase35ProtocolError, match="oracle boundary"):
        validate_gatec_matrix(changed)


def test_gatec_rejects_future_truth_leakage_and_open_loop_claims():
    from src.phase35.multistep.gatec_contracts import validate_gatec_matrix

    changed = copy.deepcopy(_matrix())
    changed["information_flow"]["forecast_reads_future_tin_truth"] = True
    with pytest.raises(Phase35ProtocolError, match="future Tin"):
        validate_gatec_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["information_flow"]["residual_reads_future_logged_valve"] = True
    with pytest.raises(Phase35ProtocolError, match="future logged valve"):
        validate_gatec_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["claim_contract"]["open_loop_plant_identification"] = True
    with pytest.raises(Phase35ProtocolError, match="open-loop plant"):
        validate_gatec_matrix(changed)


def test_gatec_rejects_matrix_drift_and_unclosed_weights():
    from src.phase35.multistep.gatec_contracts import validate_gatec_matrix

    changed = copy.deepcopy(_matrix())
    changed["rm1_attribution"][0]["candidate_id"] = changed["rm1_attribution"][1]["candidate_id"]
    with pytest.raises(Phase35ProtocolError, match="duplicate candidate"):
        validate_gatec_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["selector"]["weights"]["terminal"] += 0.1
    with pytest.raises(Phase35ProtocolError, match="sum to one"):
        validate_gatec_matrix(changed)
    changed = copy.deepcopy(_matrix())
    changed["rm1_operator"][0]["response_route"] = "unknown"
    with pytest.raises(Phase35ProtocolError, match="response route"):
        validate_gatec_matrix(changed)


def test_gatec_route_state_contract_is_three_bases_per_mode():
    from src.phase35.multistep.gatec_contracts import GateCModelConfig

    with pytest.raises(Phase35ProtocolError, match="six local states"):
        GateCModelConfig(
            window=12,
            horizon=8,
            n_features=15,
            local_state_dim=8,
            response_route="a1phys_three_pole",
            response_scheduling="scheduled",
        ).validate()
