from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from src.phase35.multistep.gatec_data import paired_history_feature_names
from src.phase35.multistep.rm3av_model import RM3AVModelConfig, build_rm3av_model
from src.phase35.multistep.rm3b_contracts import (
    EXPECTED_PAIRS,
    RM3B_CANDIDATE_IDS,
    rm3b_run_specs,
    validate_rm3b_matrix,
)
from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_rm3b_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_rm3b_matrix_closes_11_candidates_8_pairs_and_22_units() -> None:
    matrix = _matrix()
    validate_rm3b_matrix(matrix, repo_root=ROOT)
    specs = rm3b_run_specs(matrix, repo_root=ROOT)
    assert tuple(raw["candidate_id"] for raw in matrix["candidates"]) == RM3B_CANDIDATE_IDS
    assert tuple(tuple(pair) for pair in matrix["pair_contract"]["pairs"]) == EXPECTED_PAIRS
    assert len(specs) == 22
    assert len({spec.run_id for spec in specs}) == 22
    assert {spec.fold_id for spec in specs} == {"F0", "F1"}
    assert {spec.seed for spec in specs} == {0}
    assert {spec.optimizer_updates_cap for spec in specs} == {8000}


def test_rm3b_each_nonanchor_changes_one_predeclared_module() -> None:
    matrix = _matrix()
    candidates = {raw["candidate_id"]: raw for raw in matrix["candidates"]}
    for candidate_id in RM3B_CANDIDATE_IDS[3:]:
        raw = candidates[candidate_id]
        assert raw["pair_anchor_id"] in {"B00", "B01", "B02"}
        assert raw["intervention_axis"] != "anchor"
    assert candidates["B10"]["pair_anchor_id"] == "B00"
    assert candidates["B07"]["intervention_value"] == "diagonal_only"


def test_rm3b_templates_are_executable_real_models() -> None:
    features = paired_history_feature_names()
    history = torch.randn(1, 4, len(features))
    future_sp = torch.randn(1, 60, 2)
    logged_valve = torch.randn(1, 60, 2)
    for spec in rm3b_run_specs(_matrix())[::2]:
        model = build_rm3av_model(
            RM3AVModelConfig(
                candidate_id=spec.template_candidate_id,
                window=4, horizon=60, n_features=len(features),
                d_model=8, latent_dim=4, dropout=0.0,
            ),
            features,
        ).eval()
        output = model(
            history, future_sp,
            logged_future_valve=(
                logged_valve if spec.template_candidate_id in {"C10", "C11", "C12", "C13"} else None
            ),
        )
        assert output["rm3av_candidate_id"] == spec.template_candidate_id
        assert output["terminal_prediction"].shape == (1, 60, 2)
        assert torch.isfinite(output["terminal_prediction"]).all()


def test_rm3b_matrix_is_never_self_authorizing_or_self_promoting() -> None:
    matrix = _matrix()
    execution = matrix["execution_contract"]
    assert execution["linux_authorized"] is False
    assert execution["test_authorized"] is False
    assert execution["rm3b2_authorized"] is False
    assert matrix["pair_contract"]["automatic_pair_verdict"] is False
    assert matrix["matrix_envelope"]["b2_generated_from_results"] is False
    assert all(value is False for value in matrix["claim_contract"].values())


def test_rm3b_rejects_parent_pair_or_candidate_drift() -> None:
    changed = copy.deepcopy(_matrix())
    changed["parent_supervisor_decision"]["sha256"] = "0" * 64
    with pytest.raises(Phase35ProtocolError, match="parent decision hash changed"):
        validate_rm3b_matrix(changed, repo_root=ROOT)

    changed = copy.deepcopy(_matrix())
    changed["pair_contract"]["pairs"][0] = ["B03", "B02"]
    with pytest.raises(Phase35ProtocolError, match="pair map changed"):
        validate_rm3b_matrix(changed, repo_root=ROOT)

    changed = copy.deepcopy(_matrix())
    changed["candidates"][6]["template_candidate_id"] = "C15"
    with pytest.raises(Phase35ProtocolError, match="candidate semantics changed"):
        validate_rm3b_matrix(changed, repo_root=ROOT)
