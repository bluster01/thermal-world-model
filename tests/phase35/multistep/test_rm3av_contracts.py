from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.phase35.multistep.rm3av_contracts import (
    RM3AV_CANDIDATE_IDS,
    rm3av_run_specs,
    validate_rm3av_matrix,
)
from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_rm3av_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_rm3av_matrix_closes_32_candidates_and_64_units() -> None:
    matrix = _matrix()
    validate_rm3av_matrix(matrix, repo_root=ROOT)
    specs = rm3av_run_specs(matrix, repo_root=ROOT)

    assert tuple(item["candidate_id"] for item in matrix["candidates"]) == RM3AV_CANDIDATE_IDS
    assert len(RM3AV_CANDIDATE_IDS) == 32
    assert len(specs) == 64
    assert len({spec.run_id for spec in specs}) == 64
    assert {spec.fold_id for spec in specs} == {"F0", "F1"}
    assert {spec.seed for spec in specs} == {0}


def test_rm3av_each_candidate_changes_one_declared_axis() -> None:
    matrix = _matrix()
    validate_rm3av_matrix(matrix, repo_root=ROOT)
    for raw in matrix["candidates"]:
        assert isinstance(raw["intervention"], dict)
        assert set(raw["intervention"]) == {raw["intervention_axis"]}
        assert raw["group"] in {
            "anchor", "head", "supervision", "valve", "mimo", "timing", "init", "optim", "rollout"
        }


def test_rm3av_only_convergence_controls_use_8000_updates() -> None:
    specs = rm3av_run_specs(_matrix(), repo_root=ROOT)
    by_candidate = {spec.candidate_id: spec.optimizer_updates_cap for spec in specs}
    assert {candidate for candidate, cap in by_candidate.items() if cap == 8000} == {"C28", "C29", "C30"}
    assert all(cap in {4000, 8000} for cap in by_candidate.values())


def test_rm3av_matrix_is_never_self_authorizing() -> None:
    matrix = _matrix()
    execution = matrix["execution_contract"]
    assert execution["linux_authorized"] is False
    assert execution["test_authorized"] is False
    assert execution["ms4_authorized"] is False
    assert execution["rm3b_authorized"] is False
    assert execution["automatic_scientific_pass"] is False
    assert execution["maximum_attempts_per_run"] == 1


def test_rm3av_rejects_a_second_intervention_axis() -> None:
    changed = copy.deepcopy(_matrix())
    changed["candidates"][3]["intervention"]["response_mode"] = "off"
    with pytest.raises(Phase35ProtocolError, match="one declared intervention"):
        validate_rm3av_matrix(changed, repo_root=ROOT)


def test_rm3av_rejects_parent_pin_or_unit_drift() -> None:
    changed = copy.deepcopy(_matrix())
    changed["parent_independent_audit"]["sha256"] = "0" * 64
    with pytest.raises(Phase35ProtocolError, match="parent independent audit hash changed"):
        validate_rm3av_matrix(changed, repo_root=ROOT)

    changed = copy.deepcopy(_matrix())
    changed["matrix_envelope"]["training_unit_count"] = 63
    with pytest.raises(Phase35ProtocolError, match="64 training units"):
        validate_rm3av_matrix(changed, repo_root=ROOT)


def test_rm3av_requires_diagnostic_artifact_contract() -> None:
    changed = copy.deepcopy(_matrix())
    changed["execution_contract"]["required_run_artifacts"].remove("diagnostics_validation.json")
    with pytest.raises(Phase35ProtocolError, match="run artifact contract"):
        validate_rm3av_matrix(changed, repo_root=ROOT)


def test_rm3av_rejects_candidate_semantic_or_training_value_drift() -> None:
    changed = copy.deepcopy(_matrix())
    changed["candidates"][19]["intervention"]["response_shape"] = "two_pole"
    with pytest.raises(Phase35ProtocolError, match="candidate semantics"):
        validate_rm3av_matrix(changed, repo_root=ROOT)

    changed = copy.deepcopy(_matrix())
    changed["training"]["diagnostic_anchor_count"] = 511
    with pytest.raises(Phase35ProtocolError, match="training contract"):
        validate_rm3av_matrix(changed, repo_root=ROOT)
