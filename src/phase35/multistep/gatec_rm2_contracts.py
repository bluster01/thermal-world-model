"""Closed contracts for the real MS3-R Gate C RM2 Linux batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..schema import Phase35ProtocolError
from .gatec_contracts import RESPONSE_ROUTES, RESIDUAL_CAPACITIES, RESPONSE_SCHEDULING


RESPONSE_COORDINATE_MODES = {"full_mimo", "common_only"}
DOWNSTREAM_MODES = {"latent_mimo", "direct_no_latent"}
RM2_GROUP_COUNTS = {"A": 4, "B": 3, "C": 2}


@dataclass(frozen=True)
class RM2FoldSpec:
    fold_id: str
    train_fraction: tuple[float, float]
    validation_fraction: tuple[float, float]


@dataclass(frozen=True)
class RM2RunSpec:
    run_id: str
    group: str
    candidate_id: str
    response_route: str
    residual_capacity: str
    response_scheduling: str
    response_coordinate_mode: str
    downstream_mode: str
    seed: int
    fold_id: str
    train_fraction: tuple[float, float]
    validation_fraction: tuple[float, float]


def _pair(raw: Sequence[Any], label: str) -> tuple[float, float]:
    if len(raw) != 2:
        raise Phase35ProtocolError(f"RM2 {label} must contain two fractions")
    pair = (float(raw[0]), float(raw[1]))
    if not 0.0 <= pair[0] < pair[1] <= 1.0:
        raise Phase35ProtocolError(f"RM2 {label} fractions are invalid")
    return pair


def validate_rm2_matrix(matrix: Mapping[str, Any]) -> None:
    if matrix.get("protocol_version") != "phase3.5-ms3r-gatec-rm2-v1":
        raise Phase35ProtocolError("unsupported Gate C RM2 protocol")
    data = matrix.get("data_contract", {})
    if data.get("test_allowed") is not False or float(data.get("pretest_end_fraction", -1)) != 0.8:
        raise Phase35ProtocolError("Gate C RM2 must preserve the final 20% test lockbox")
    folds = data.get("folds", [])
    expected_folds = {
        "F0": ((0.0, 0.6), (0.6, 0.7)),
        "F1": ((0.0, 0.7), (0.7, 0.8)),
    }
    observed_folds: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for raw in folds:
        fold_id = raw.get("fold_id")
        observed_folds[fold_id] = (
            _pair(raw.get("train_fraction", []), "train"),
            _pair(raw.get("validation_fraction", []), "validation"),
        )
    if observed_folds != expected_folds:
        raise Phase35ProtocolError("Gate C RM2 fold definitions changed")
    if list(matrix.get("seeds", [])) != [0, 1, 2]:
        raise Phase35ProtocolError("Gate C RM2 seeds changed")

    required_candidate_fields = {
        "group",
        "candidate_id",
        "response_route",
        "residual_capacity",
        "response_scheduling",
        "response_coordinate_mode",
        "downstream_mode",
    }
    candidates = matrix.get("candidates", [])
    counts = {group: 0 for group in RM2_GROUP_COUNTS}
    identifiers: list[str] = []
    for candidate in candidates:
        if set(candidate) != required_candidate_fields:
            raise Phase35ProtocolError("Gate C RM2 candidate fields are not closed")
        group = candidate["group"]
        if group not in counts:
            raise Phase35ProtocolError("Gate C RM2 candidate group is invalid")
        counts[group] += 1
        identifiers.append(candidate["candidate_id"])
        if candidate["response_route"] not in RESPONSE_ROUTES:
            raise Phase35ProtocolError("Gate C RM2 response route is invalid")
        if candidate["residual_capacity"] not in RESIDUAL_CAPACITIES:
            raise Phase35ProtocolError("Gate C RM2 residual capacity is invalid")
        if candidate["response_scheduling"] not in RESPONSE_SCHEDULING:
            raise Phase35ProtocolError("Gate C RM2 response scheduling is invalid")
        if candidate["response_coordinate_mode"] not in RESPONSE_COORDINATE_MODES:
            raise Phase35ProtocolError("Gate C RM2 response coordinate mode is invalid")
        if candidate["downstream_mode"] not in DOWNSTREAM_MODES:
            raise Phase35ProtocolError("Gate C RM2 downstream mode is invalid")
        no_response = candidate["response_route"] == "none"
        if no_response != (candidate["response_scheduling"] == "none"):
            raise Phase35ProtocolError("Gate C RM2 response scheduling semantics changed")
        if no_response and candidate["response_coordinate_mode"] != "full_mimo":
            raise Phase35ProtocolError("Gate C RM2 paired-free coordinate mode is invalid")
    if counts != RM2_GROUP_COUNTS or len(identifiers) != 9 or len(set(identifiers)) != 9:
        raise Phase35ProtocolError("Gate C RM2 candidate matrix is not 4+3+2 unique candidates")

    training = matrix.get("training", {})
    positive_counts = (
        "batch_size",
        "optimizer_updates_cap",
        "minimum_updates",
        "evaluation_interval_updates",
        "early_stopping_patience_evaluations",
        "stats_anchor_count",
        "selector_anchor_count",
        "final_anchor_count",
        "evaluation_batch_size",
    )
    if any(int(training.get(key, 0)) < 1 for key in positive_counts):
        raise Phase35ProtocolError("Gate C RM2 training counts must be positive")
    if int(training["minimum_updates"]) > int(training["optimizer_updates_cap"]):
        raise Phase35ProtocolError("Gate C RM2 minimum updates exceed the cap")

    execution = matrix.get("execution_contract", {})
    if execution.get("webhook_execution_authorized") is not True or execution.get("linux_authorized") is not True:
        raise Phase35ProtocolError("Gate C RM2 webhook/Linux execution is not authorized")
    if execution.get("test_authorized") is not False:
        raise Phase35ProtocolError("Gate C RM2 cannot authorize test")
    if int(execution.get("maximum_attempts_per_run", 0)) != 1:
        raise Phase35ProtocolError("Gate C RM2 permits exactly one attempt per run")
    if int(execution.get("expected_run_count", 0)) != 54:
        raise Phase35ProtocolError("Gate C RM2 expected run count changed")
    if set(execution.get("required_root_artifacts", [])) != {
        "run_manifest.json",
        "matrix_execution_status.json",
        "summary_validation.json",
        "checkpoints_validation.tar",
        "artifact_ledger.json",
    }:
        raise Phase35ProtocolError("Gate C RM2 root artifact contract changed")
    if execution.get("automatic_scientific_pass") is not False:
        raise Phase35ProtocolError("Gate C RM2 cannot make an automatic scientific decision")
    if any(value is not False for value in matrix.get("claim_contract", {}).values()):
        raise Phase35ProtocolError("Gate C RM2 claim boundary changed")
    if len(rm2_run_specs(matrix)) != 54:
        raise Phase35ProtocolError("Gate C RM2 expanded run count changed")


def rm2_fold_specs(matrix: Mapping[str, Any]) -> list[RM2FoldSpec]:
    return [
        RM2FoldSpec(
            fold_id=raw["fold_id"],
            train_fraction=_pair(raw["train_fraction"], "train"),
            validation_fraction=_pair(raw["validation_fraction"], "validation"),
        )
        for raw in matrix["data_contract"]["folds"]
    ]


def rm2_run_specs(matrix: Mapping[str, Any]) -> list[RM2RunSpec]:
    folds = rm2_fold_specs(matrix)
    return [
        RM2RunSpec(
            run_id=f"{candidate['candidate_id']}_{fold.fold_id}_s{seed}",
            group=candidate["group"],
            candidate_id=candidate["candidate_id"],
            response_route=candidate["response_route"],
            residual_capacity=candidate["residual_capacity"],
            response_scheduling=candidate["response_scheduling"],
            response_coordinate_mode=candidate["response_coordinate_mode"],
            downstream_mode=candidate["downstream_mode"],
            seed=int(seed),
            fold_id=fold.fold_id,
            train_fraction=fold.train_fraction,
            validation_fraction=fold.validation_fraction,
        )
        for candidate in matrix["candidates"]
        for fold in folds
        for seed in matrix["seeds"]
    ]


def partition_rm2_runs(specs: Sequence[RM2RunSpec], worker_count: int) -> list[list[RM2RunSpec]]:
    if worker_count < 1:
        raise Phase35ProtocolError("Gate C RM2 needs at least one worker")
    partitions: list[list[RM2RunSpec]] = [[] for _ in range(worker_count)]
    for index, spec in enumerate(specs):
        partitions[index % worker_count].append(spec)
    return partitions
