"""Fail-closed contracts for the RM3-B1 paired composition screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schema import Phase35ProtocolError


RM3B_CANDIDATE_IDS = tuple(f"B{index:02d}" for index in range(11))
EXPECTED_CANDIDATES = (
    ("B00", "prediction_anchor", "C28", None, "anchor", "p3_8000"),
    ("B01", "response_anchor", "C29", None, "anchor", "p4_8000"),
    ("B02", "joint_bypass_anchor", "C30", None, "anchor", "p5_8000"),
    ("B03", "response_calibration", "C09", "B01", "action_shield", "train_oof_projection"),
    ("B04", "response_calibration", "C11", "B01", "response_supervision", "integrated_oof_r_loss"),
    ("B05", "valve_policy", "C14", "B02", "valve_loss", "delta_and_multiscale_roughness"),
    ("B06", "valve_policy", "C16", "B02", "valve_decoder", "structured_pi_plus_gru_residual"),
    ("B07", "response_coordinate", "C18", "B01", "response_coordinate", "diagonal_only"),
    ("B08", "response_shape", "C19", "B01", "response_shape", "one_pole"),
    ("B09", "response_shape_control", "C22", "B01", "response_shape", "linear_ramp"),
    ("B10", "terminal_correction", "C03", "B00", "terminal_bypass", "add_action_invariant"),
)
EXPECTED_PAIRS = (
    ("B03", "B01"), ("B04", "B01"), ("B05", "B02"), ("B06", "B02"),
    ("B07", "B01"), ("B08", "B01"), ("B09", "B01"), ("B10", "B00"),
)
REQUIRED_RUN_ARTIFACTS = {
    "manifest.json", "checkpoint_best_validation.pt", "metrics_validation.json",
    "episodes_validation.npz", "diagnostics_validation.json", "artifact_ledger.json",
}
REQUIRED_ROOT_ARTIFACTS = {
    "run_manifest.json", "matrix_execution_status.json", "summary_validation.json",
    "artifact_ledger.json",
}


@dataclass(frozen=True)
class RM3BRunSpec:
    run_id: str
    candidate_id: str
    role: str
    template_candidate_id: str
    pair_anchor_id: str | None
    intervention_axis: str
    intervention_value: Any
    optimizer_updates_cap: int
    fold_id: str
    seed: int
    train_fraction: tuple[float, float]
    validation_fraction: tuple[float, float]


def _normalized_text_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fraction(raw: Sequence[Any], label: str) -> tuple[float, float]:
    if len(raw) != 2:
        raise Phase35ProtocolError(f"RM3-B1 {label} fraction must have two values")
    pair = (float(raw[0]), float(raw[1]))
    if not 0.0 <= pair[0] < pair[1] <= 1.0:
        raise Phase35ProtocolError(f"RM3-B1 {label} fraction is invalid")
    return pair


def _verify_parent_pin(raw: Mapping[str, Any], *, repo_root: Path, label: str) -> None:
    path = raw.get("path")
    digest = raw.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise Phase35ProtocolError(f"RM3-B1 {label} pin is invalid")
    if raw.get("hash_mode") != "utf8_text_normalized_lf":
        raise Phase35ProtocolError(f"RM3-B1 {label} hash mode changed")
    target = repo_root / path
    if not target.is_file() or _normalized_text_sha(target) != digest:
        raise Phase35ProtocolError(f"RM3-B1 {label} hash changed")


def validate_rm3b_matrix(matrix: Mapping[str, Any], *, repo_root: Path | None = None) -> None:
    if matrix.get("protocol_version") != "phase3.5-ms3r-rm3b1-v1":
        raise Phase35ProtocolError("unsupported RM3-B1 protocol version")
    if repo_root is not None:
        _verify_parent_pin(matrix.get("parent_supervisor_decision", {}), repo_root=repo_root, label="parent decision")
        _verify_parent_pin(matrix.get("parent_design", {}), repo_root=repo_root, label="parent design")
        _verify_parent_pin(matrix.get("parent_rm3av_matrix", {}), repo_root=repo_root, label="parent RM3-AV matrix")

    data = matrix.get("data_contract", {})
    if data.get("test_allowed") is not False or float(data.get("pretest_end_fraction", -1)) != 0.8:
        raise Phase35ProtocolError("RM3-B1 test/pretest boundary changed")
    if (int(data.get("window_steps", 0)), int(data.get("horizon_steps", 0)), int(data.get("step_seconds", 0))) != (96, 60, 10):
        raise Phase35ProtocolError("RM3-B1 temporal contract changed")
    if float(data.get("max_age_s", 0)) != 180.0 or data.get("independent_unit") != "UTC_day":
        raise Phase35ProtocolError("RM3-B1 sampling/independent-unit contract changed")
    expected_folds = {
        "F0": ((0.0, 0.6), (0.6, 0.7)),
        "F1": ((0.0, 0.7), (0.7, 0.8)),
    }
    observed_folds = {
        raw.get("fold_id"): (
            _fraction(raw.get("train_fraction", ()), "train"),
            _fraction(raw.get("validation_fraction", ()), "validation"),
        )
        for raw in data.get("folds", ())
    }
    if observed_folds != expected_folds:
        raise Phase35ProtocolError("RM3-B1 rolling folds changed")

    candidates = matrix.get("candidates", ())
    required_fields = {
        "candidate_id", "role", "template_candidate_id", "pair_anchor_id",
        "intervention_axis", "intervention_value",
    }
    if len(candidates) != 11 or tuple(raw.get("candidate_id") for raw in candidates) != RM3B_CANDIDATE_IDS:
        raise Phase35ProtocolError("RM3-B1 candidate matrix must contain ordered B00-B10")
    if any(set(raw) != required_fields for raw in candidates):
        raise Phase35ProtocolError("RM3-B1 candidate fields changed")
    observed_candidates = tuple(
        (
            raw["candidate_id"], raw["role"], raw["template_candidate_id"],
            raw["pair_anchor_id"], raw["intervention_axis"], raw["intervention_value"],
        )
        for raw in candidates
    )
    if observed_candidates != EXPECTED_CANDIDATES:
        raise Phase35ProtocolError("RM3-B1 candidate semantics changed")

    pair = matrix.get("pair_contract", {})
    if tuple(tuple(value) for value in pair.get("pairs", ())) != EXPECTED_PAIRS:
        raise Phase35ProtocolError("RM3-B1 pair map changed")
    if pair.get("same_direction_required_in_both_folds") is not True or pair.get("shared_module_initialization_hash_required") is not True:
        raise Phase35ProtocolError("RM3-B1 paired evidence contract weakened")
    if pair.get("automatic_pair_verdict") is not False:
        raise Phase35ProtocolError("RM3-B1 cannot assign automatic pair verdicts")
    if pair.get("allowed_verdicts") != ["SUPPORTED", "MIXED", "REJECTED", "NOT_TESTABLE"]:
        raise Phase35ProtocolError("RM3-B1 verdict vocabulary changed")

    envelope = matrix.get("matrix_envelope", {})
    if envelope != {
        "status": "b1_local_verified_external_registry_authorization_required",
        "folds": ["F0", "F1"], "seeds": [0], "candidate_count": 11,
        "training_unit_count": 22, "optimizer_updates_cap": 8000,
        "no_composite_champion": True, "b2_generated_from_results": False,
    }:
        raise Phase35ProtocolError("RM3-B1 envelope must close exactly 22 units")
    if matrix.get("model") != {"d_model": 64, "latent_dim": 32, "dropout": 0.1}:
        raise Phase35ProtocolError("RM3-B1 model capacity contract changed")
    training = matrix.get("training", {})
    if int(training.get("optimizer_updates_cap", 0)) != 8000:
        raise Phase35ProtocolError("RM3-B1 requires 8000 optimizer updates for every candidate")
    if training.get("component_loss_weights") != {"valve": 0.25, "tin": 0.25, "local": 0.25, "terminal": 0.25}:
        raise Phase35ProtocolError("RM3-B1 common four-task selector weights changed")

    diagnostics = matrix.get("diagnostic_contract", {})
    required_modes = {
        "normal", "bypass_off", "bypass_only", "response_off", "predicted_valve",
        "logged_valve", "logged_valve_oracle_tin", "oracle_local", "shuffled",
        "wrong_side", "lead",
    }
    if set(diagnostics.get("required_modes", ())) != required_modes or diagnostics.get("automatic_verdict") is not False:
        raise Phase35ProtocolError("RM3-B1 diagnostics contract changed")

    execution = matrix.get("execution_contract", {})
    for key in ("local_full_training_authorized", "linux_authorized", "test_authorized", "ms4_authorized", "rm3b2_authorized", "automatic_scientific_pass"):
        if execution.get(key) is not False:
            raise Phase35ProtocolError(f"RM3-B1 {key} must remain false in the matrix")
    if execution.get("local_micro_smoke_authorized") is not True or int(execution.get("maximum_attempts_per_run", 0)) != 1:
        raise Phase35ProtocolError("RM3-B1 local smoke/attempt contract changed")
    if set(execution.get("required_run_artifacts", ())) != REQUIRED_RUN_ARTIFACTS:
        raise Phase35ProtocolError("RM3-B1 run artifact contract changed")
    if set(execution.get("required_root_artifacts", ())) != REQUIRED_ROOT_ARTIFACTS:
        raise Phase35ProtocolError("RM3-B1 root artifact contract changed")
    claims = matrix.get("claim_contract", {})
    if not claims or any(value is not False for value in claims.values()):
        raise Phase35ProtocolError("RM3-B1 forbidden claim enabled")


def rm3b_run_specs(matrix: Mapping[str, Any], *, repo_root: Path | None = None) -> tuple[RM3BRunSpec, ...]:
    validate_rm3b_matrix(matrix, repo_root=repo_root)
    folds = {
        raw["fold_id"]: (
            _fraction(raw["train_fraction"], "train"),
            _fraction(raw["validation_fraction"], "validation"),
        )
        for raw in matrix["data_contract"]["folds"]
    }
    specs = tuple(
        RM3BRunSpec(
            run_id=f"{raw['candidate_id']}_{fold_id}_s0",
            candidate_id=raw["candidate_id"], role=raw["role"],
            template_candidate_id=raw["template_candidate_id"],
            pair_anchor_id=raw["pair_anchor_id"],
            intervention_axis=raw["intervention_axis"],
            intervention_value=raw["intervention_value"],
            optimizer_updates_cap=8000, fold_id=fold_id, seed=0,
            train_fraction=fractions[0], validation_fraction=fractions[1],
        )
        for raw in matrix["candidates"]
        for fold_id, fractions in folds.items()
    )
    if len(specs) != 22:
        raise Phase35ProtocolError("RM3-B1 run expansion must produce 22 training units")
    return specs
