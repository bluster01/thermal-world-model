"""Closed contracts for MS3-R RM3 orthogonal response calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True)
class RM3FoldSpec:
    fold_id: str
    train_fraction: tuple[float, float]
    validation_fraction: tuple[float, float]


@dataclass(frozen=True)
class RM3PredictionRunSpec:
    run_id: str
    candidate_id: str
    future_action_access: str
    role: str
    output_scope: str
    prefix_causal_action_path: bool
    fold_id: str
    seed: int
    train_fraction: tuple[float, float]
    validation_fraction: tuple[float, float]


@dataclass(frozen=True)
class RM3CalibrationSpec:
    calibration_id: str
    fold_id: str
    seed: int
    response_horizon_steps: int
    candidate_ids: tuple[str, ...]
    train_fraction: tuple[float, float]
    validation_fraction: tuple[float, float]


def _fraction_pair(raw: Sequence[Any], name: str) -> tuple[float, float]:
    if len(raw) != 2:
        raise Phase35ProtocolError(f"RM3 {name} fraction must contain two values")
    pair = (float(raw[0]), float(raw[1]))
    if not 0.0 <= pair[0] < pair[1] <= 1.0:
        raise Phase35ProtocolError(f"RM3 {name} fraction is invalid")
    return pair


def validate_rm3_matrix(matrix: Mapping[str, Any]) -> None:
    if matrix.get("protocol_version") != "phase3.5-ms3r-rm3-v1":
        raise Phase35ProtocolError("unsupported RM3 protocol version")
    parent = matrix.get("parent_rm2_audit", {})
    if (
        not isinstance(parent.get("path"), str)
        or not isinstance(parent.get("sha256"), str)
        or len(parent["sha256"]) != 64
        or parent.get("hash_mode") != "utf8_text_normalized_lf"
        or parent.get("required_label")
        != "RM2_COMPLETE_CONDITIONAL_ACTION_PATH_REPRODUCED_OPERATOR_GAIN_NOT_IDENTIFIED"
    ):
        raise Phase35ProtocolError("RM3 parent RM2 audit pin is invalid")
    data = matrix.get("data_contract", {})
    if data.get("test_allowed") is not False or int(data.get("horizon_steps", 0)) != 60:
        raise Phase35ProtocolError("RM3 must remain H60 train/validation with test locked")
    if tuple(data.get("primary_response_horizons_steps", ())) != (6, 18):
        raise Phase35ProtocolError("RM3 primary response horizons must be H60/H180")
    if data.get("response_horizon_semantics") != "full_prefix_trajectory_not_endpoint_only":
        raise Phase35ProtocolError("RM3 response horizons must retain full prefix trajectories")
    if data.get("nuisance_mode") != "expanding_rolling_oof":
        raise Phase35ProtocolError("RM3 nuisance estimates must be expanding rolling OOF")
    if data.get("raw_future_valve_auxiliary_allowed") is not False:
        raise Phase35ProtocolError("RM3 prohibits raw future-valve response auxiliary")
    if data.get("independent_unit") != "UTC_day":
        raise Phase35ProtocolError("RM3 independent unit must be UTC_day")
    expected_folds = {
        "F0": ((0.0, 0.6), (0.6, 0.7)),
        "F1": ((0.0, 0.7), (0.7, 0.8)),
    }
    observed_folds = {
        raw.get("fold_id"): (
            _fraction_pair(raw.get("train_fraction", ()), "train"),
            _fraction_pair(raw.get("validation_fraction", ()), "validation"),
        )
        for raw in data.get("folds", ())
    }
    if observed_folds != expected_folds or float(data.get("pretest_end_fraction", -1)) != 0.8:
        raise Phase35ProtocolError("RM3 rolling folds changed")
    if int(data.get("window_steps", 0)) != 96 or float(data.get("max_age_s", 0)) != 180.0:
        raise Phase35ProtocolError("RM3 real window/age contract changed")
    if not isinstance(data.get("source_sha256"), str) or len(data["source_sha256"]) != 64:
        raise Phase35ProtocolError("RM3 source hash is invalid")

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
    training = matrix.get("training", {})
    positive = (
        "batch_size",
        "optimizer_updates_cap",
        "minimum_updates",
        "evaluation_interval_updates",
        "early_stopping_patience_evaluations",
        "stats_anchor_count",
        "selector_anchor_count",
        "reporting_anchor_count",
        "evaluation_batch_size",
    )
    if any(int(training.get(key, 0)) < 1 for key in positive):
        raise Phase35ProtocolError("RM3 training counts must be positive")
    if int(training["optimizer_updates_cap"]) != int(envelope["optimizer_updates_cap"]):
        raise Phase35ProtocolError("RM3 training cap differs from envelope")
    if int(training["minimum_updates"]) > int(training["optimizer_updates_cap"]):
        raise Phase35ProtocolError("RM3 minimum updates exceed cap")

    execution = matrix.get("execution_contract", {})
    for key in ("local_real_training_authorized", "linux_authorized", "test_authorized", "ms4_authorized"):
        if execution.get(key) is not False:
            raise Phase35ProtocolError(f"RM3 {key} must remain false")
    if execution.get("local_synthetic_smoke_authorized") is not True:
        raise Phase35ProtocolError("RM3 local synthetic smoke must be explicitly authorized")
    if execution.get("automatic_scientific_pass") is not False:
        raise Phase35ProtocolError("RM3 cannot make an automatic scientific decision")
    if int(execution.get("maximum_attempts_per_run", 0)) != 1:
        raise Phase35ProtocolError("RM3 permits exactly one attempt per run")
    if set(execution.get("required_prediction_artifacts", ())) != {
        "manifest.json",
        "checkpoint_best_validation.pt",
        "metrics_validation.json",
        "episodes_validation.npz",
        "artifact_ledger.json",
    }:
        raise Phase35ProtocolError("RM3 prediction artifact contract changed")
    if set(execution.get("required_calibration_artifacts", ())) != {
        "calibration_validation.json",
        "orthogonal_residuals_validation.npz",
        "artifact_ledger.json",
    }:
        raise Phase35ProtocolError("RM3 calibration artifact contract changed")
    if set(execution.get("required_root_artifacts", ())) != {
        "run_manifest.json",
        "matrix_execution_status.json",
        "summary_validation.json",
        "artifact_ledger.json",
    }:
        raise Phase35ProtocolError("RM3 root artifact contract changed")
    for key, value in matrix.get("claim_contract", {}).items():
        if value is not False:
            raise Phase35ProtocolError(f"RM3 forbidden claim enabled: {key}")


def rm3_identification_specs(matrix: Mapping[str, Any]) -> tuple[RM3IdentificationSpec, ...]:
    validate_rm3_matrix(matrix)
    return tuple(RM3IdentificationSpec(**raw) for raw in matrix["identification_candidates"])


def rm3_prediction_specs(matrix: Mapping[str, Any]) -> tuple[RM3PredictionSpec, ...]:
    validate_rm3_matrix(matrix)
    return tuple(RM3PredictionSpec(**raw) for raw in matrix["prediction_candidates"])


def rm3_fold_specs(matrix: Mapping[str, Any]) -> tuple[RM3FoldSpec, ...]:
    validate_rm3_matrix(matrix)
    return tuple(
        RM3FoldSpec(
            fold_id=raw["fold_id"],
            train_fraction=_fraction_pair(raw["train_fraction"], "train"),
            validation_fraction=_fraction_pair(raw["validation_fraction"], "validation"),
        )
        for raw in matrix["data_contract"]["folds"]
    )


def rm3_prediction_run_specs(matrix: Mapping[str, Any]) -> tuple[RM3PredictionRunSpec, ...]:
    predictions = rm3_prediction_specs(matrix)
    folds = rm3_fold_specs(matrix)
    specs = tuple(
        RM3PredictionRunSpec(
            run_id=f"{candidate.candidate_id}_{fold.fold_id}_s{seed}",
            **candidate.__dict__,
            fold_id=fold.fold_id,
            seed=int(seed),
            train_fraction=fold.train_fraction,
            validation_fraction=fold.validation_fraction,
        )
        for candidate in predictions
        for fold in folds
        for seed in matrix["real_matrix_envelope"]["seeds"]
    )
    if len(specs) != int(matrix["real_matrix_envelope"]["prediction_run_count"]):
        raise Phase35ProtocolError("RM3 prediction run expansion changed")
    return specs


def rm3_calibration_specs(matrix: Mapping[str, Any]) -> tuple[RM3CalibrationSpec, ...]:
    candidates = tuple(spec.candidate_id for spec in rm3_identification_specs(matrix))
    folds = rm3_fold_specs(matrix)
    specs = tuple(
        RM3CalibrationSpec(
            calibration_id=f"orthogonal_{fold.fold_id}_s{seed}_h{horizon}",
            fold_id=fold.fold_id,
            seed=int(seed),
            response_horizon_steps=int(horizon),
            candidate_ids=candidates,
            train_fraction=fold.train_fraction,
            validation_fraction=fold.validation_fraction,
        )
        for fold in folds
        for seed in matrix["real_matrix_envelope"]["seeds"]
        for horizon in matrix["data_contract"]["primary_response_horizons_steps"]
    )
    if len(specs) != int(matrix["real_matrix_envelope"]["orthogonal_calibration_run_count"]):
        raise Phase35ProtocolError("RM3 calibration expansion changed")
    return specs
