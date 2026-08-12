"""Closed contracts for MS3-R RM3 orthogonal response calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..schema import Phase35ProtocolError


IDENTIFICATION_FIELDS = {"candidate_id", "response_family", "coordinate_mode"}
PREDICTION_FIELDS = {
    "candidate_id",
    "future_action_access",
    "role",
    "output_scope",
    "prefix_causal_action_path",
}


@dataclass(frozen=True)
class RM3IdentificationSpec:
    candidate_id: str
    response_family: str
    coordinate_mode: str


@dataclass(frozen=True)
class RM3PredictionSpec:
    candidate_id: str
    future_action_access: str
    role: str
    output_scope: str
    prefix_causal_action_path: bool


def validate_rm3_matrix(matrix: Mapping[str, Any]) -> None:
    if matrix.get("protocol_version") != "phase3.5-ms3r-rm3-v1":
        raise Phase35ProtocolError("unsupported RM3 protocol version")
    data = matrix.get("data_contract", {})
    if data.get("test_allowed") is not False or int(data.get("horizon_steps", 0)) != 60:
        raise Phase35ProtocolError("RM3 must remain H60 train/validation with test locked")
    if tuple(data.get("primary_response_horizons_steps", ())) != (6, 18):
        raise Phase35ProtocolError("RM3 primary response horizons must be H60/H180")
    if data.get("nuisance_mode") != "expanding_rolling_oof":
        raise Phase35ProtocolError("RM3 nuisance estimates must be expanding rolling OOF")
    if data.get("raw_future_valve_auxiliary_allowed") is not False:
        raise Phase35ProtocolError("RM3 prohibits raw future-valve response auxiliary")
    if data.get("independent_unit") != "UTC_day":
        raise Phase35ProtocolError("RM3 independent unit must be UTC_day")

    identification = matrix.get("identification_candidates", [])
    prediction = matrix.get("prediction_candidates", [])
    if len(identification) != 3 or len(prediction) != 6:
        raise Phase35ProtocolError("RM3 candidate matrices are not closed")
    identifiers: list[str] = []
    for raw in identification:
        if set(raw) != IDENTIFICATION_FIELDS:
            raise Phase35ProtocolError("RM3 identification fields changed")
        if raw["coordinate_mode"] not in {"full_mimo", "common_only"}:
            raise Phase35ProtocolError("RM3 identification coordinate mode is invalid")
        identifiers.append(raw["candidate_id"])
    for raw in prediction:
        if set(raw) != PREDICTION_FIELDS:
            raise Phase35ProtocolError("RM3 prediction fields changed")
        identifiers.append(raw["candidate_id"])
        if raw["output_scope"] not in {"terminal_only", "valve_and_terminal", "full_multitask"}:
            raise Phase35ProtocolError("RM3 prediction output scope is invalid")
        if not isinstance(raw["prefix_causal_action_path"], bool):
            raise Phase35ProtocolError("RM3 prefix-causal flag must be boolean")
    if len(identifiers) != len(set(identifiers)):
        raise Phase35ProtocolError("RM3 candidate IDs must be unique")
    oracle = [item for item in prediction if item["future_action_access"] == "logged_future_valve"]
    if len(oracle) != 1 or oracle[0]["role"] != "oracle_upper_bound":
        raise Phase35ProtocolError("RM3 logged future valve is allowed only for one oracle upper bound")
    envelope = matrix.get("real_matrix_envelope", {})
    if (
        envelope.get("status") != "frozen_but_not_authorized"
        or envelope.get("folds") != ["F0", "F1"]
        or envelope.get("seeds") != [0, 1, 2]
        or int(envelope.get("prediction_run_count", -1)) != 36
        or int(envelope.get("orthogonal_calibration_run_count", -1)) != 12
        or int(envelope.get("total_run_count", -1)) != 48
        or envelope.get("no_composite_ranking_across_output_scopes") is not True
    ):
        raise Phase35ProtocolError("RM3 real matrix envelope is not closed")

    execution = matrix.get("execution_contract", {})
    for key in ("local_real_training_authorized", "linux_authorized", "test_authorized", "ms4_authorized"):
        if execution.get(key) is not False:
            raise Phase35ProtocolError(f"RM3 {key} must remain false")
    if execution.get("local_synthetic_smoke_authorized") is not True:
        raise Phase35ProtocolError("RM3 local synthetic smoke must be explicitly authorized")
    if execution.get("automatic_scientific_pass") is not False:
        raise Phase35ProtocolError("RM3 cannot make an automatic scientific decision")
    for key, value in matrix.get("claim_contract", {}).items():
        if value is not False:
            raise Phase35ProtocolError(f"RM3 forbidden claim enabled: {key}")


def rm3_identification_specs(matrix: Mapping[str, Any]) -> tuple[RM3IdentificationSpec, ...]:
    validate_rm3_matrix(matrix)
    return tuple(RM3IdentificationSpec(**raw) for raw in matrix["identification_candidates"])


def rm3_prediction_specs(matrix: Mapping[str, Any]) -> tuple[RM3PredictionSpec, ...]:
    validate_rm3_matrix(matrix)
    return tuple(RM3PredictionSpec(**raw) for raw in matrix["prediction_candidates"])
