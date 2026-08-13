"""Fail-closed contracts for the RM3 independent-audit validation batch."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schema import Phase35ProtocolError


RM3AV_CANDIDATE_IDS = tuple(f"C{index:02d}" for index in range(32))
GROUPS = {"anchor", "head", "supervision", "valve", "mimo", "timing", "init", "optim", "rollout"}
BASE_CANDIDATES = {
    "P3_gatec_paired_free",
    "P4_gatec_a1_scheduled",
    "P5_hybrid_joint_latent",
}
EXPECTED_CANDIDATES = (
    ("C00", "anchor", "P3_gatec_paired_free", "legacy_anchor", "p3_current", 4000),
    ("C01", "anchor", "P4_gatec_a1_scheduled", "legacy_anchor", "p4_current", 4000),
    ("C02", "anchor", "P5_hybrid_joint_latent", "legacy_anchor", "p5_current", 4000),
    ("C03", "head", "P3_gatec_paired_free", "terminal_bypass", "add_action_invariant", 4000),
    ("C04", "head", "P5_hybrid_joint_latent", "terminal_bypass", "off", 4000),
    ("C05", "head", "P5_hybrid_joint_latent", "terminal_path", "bypass_only", 4000),
    ("C06", "head", "P5_hybrid_joint_latent", "response_mode", "off_train_and_inference", 4000),
    ("C07", "head", "P4_gatec_a1_scheduled", "residual_capacity", "small", 4000),
    ("C08", "head", "P4_gatec_a1_scheduled", "residual_capacity", "large", 4000),
    ("C09", "head", "P4_gatec_a1_scheduled", "action_shield", "train_oof_projection", 4000),
    ("C10", "supervision", "P4_gatec_a1_scheduled", "response_supervision", "logged_action_auxiliary", 4000),
    ("C11", "supervision", "P4_gatec_a1_scheduled", "response_supervision", "integrated_oof_r_loss", 4000),
    ("C12", "supervision", "P5_hybrid_joint_latent", "response_supervision", "integrated_oof_r_loss", 4000),
    ("C13", "supervision", "P5_hybrid_joint_latent", "response_supervision", "logged_action_auxiliary", 4000),
    ("C14", "valve", "P5_hybrid_joint_latent", "valve_loss", "delta_and_multiscale_roughness", 4000),
    ("C15", "valve", "P5_hybrid_joint_latent", "valve_decoder", "structured_pi", 4000),
    ("C16", "valve", "P5_hybrid_joint_latent", "valve_decoder", "structured_pi_plus_gru_residual", 4000),
    ("C17", "mimo", "P4_gatec_a1_scheduled", "response_coordinate", "common_only", 4000),
    ("C18", "mimo", "P4_gatec_a1_scheduled", "response_coordinate", "diagonal_only", 4000),
    ("C19", "timing", "P4_gatec_a1_scheduled", "response_shape", "one_pole", 4000),
    ("C20", "timing", "P4_gatec_a1_scheduled", "response_shape", "two_pole", 4000),
    ("C21", "timing", "P4_gatec_a1_scheduled", "response_shape", "power_basis", 4000),
    ("C22", "timing", "P4_gatec_a1_scheduled", "response_shape", "linear_ramp", 4000),
    ("C23", "timing", "P4_gatec_a1_scheduled", "response_timing", "three_pole_bounded_dead_time", 4000),
    ("C24", "timing", "P4_gatec_a1_scheduled", "response_sign", "unconstrained_diagnostic", 4000),
    ("C25", "init", "P3_gatec_paired_free", "initialization", "module_scoped", 4000),
    ("C26", "init", "P4_gatec_a1_scheduled", "initialization", "module_scoped", 4000),
    ("C27", "init", "P5_hybrid_joint_latent", "initialization", "module_scoped", 4000),
    ("C28", "optim", "P3_gatec_paired_free", "optimizer_updates", 8000, 8000),
    ("C29", "optim", "P4_gatec_a1_scheduled", "optimizer_updates", 8000, 8000),
    ("C30", "optim", "P5_hybrid_joint_latent", "optimizer_updates", 8000, 8000),
    ("C31", "rollout", "P5_hybrid_joint_latent", "rollout_loss", "two_window_declared_context", 4000),
)
REQUIRED_RUN_ARTIFACTS = {
    "manifest.json",
    "checkpoint_best_validation.pt",
    "metrics_validation.json",
    "episodes_validation.npz",
    "diagnostics_validation.json",
    "artifact_ledger.json",
}
REQUIRED_ROOT_ARTIFACTS = {
    "run_manifest.json",
    "matrix_execution_status.json",
    "summary_validation.json",
    "artifact_ledger.json",
}


@dataclass(frozen=True)
class RM3AVRunSpec:
    run_id: str
    candidate_id: str
    group: str
    base_candidate_id: str
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
        raise Phase35ProtocolError(f"RM3-AV {label} fraction must have two values")
    pair = (float(raw[0]), float(raw[1]))
    if not 0.0 <= pair[0] < pair[1] <= 1.0:
        raise Phase35ProtocolError(f"RM3-AV {label} fraction is invalid")
    return pair


def _verify_parent_pin(raw: Mapping[str, Any], *, repo_root: Path, label: str) -> None:
    path = raw.get("path")
    digest = raw.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise Phase35ProtocolError(f"RM3-AV {label} pin is invalid")
    if raw.get("hash_mode") != "utf8_text_normalized_lf":
        raise Phase35ProtocolError(f"RM3-AV {label} hash mode changed")
    target = repo_root / path
    if not target.is_file() or _normalized_text_sha(target) != digest:
        raise Phase35ProtocolError(f"RM3-AV {label} hash changed")


def validate_rm3av_matrix(matrix: Mapping[str, Any], *, repo_root: Path | None = None) -> None:
    if matrix.get("protocol_version") != "phase3.5-ms3r-rm3av-v1":
        raise Phase35ProtocolError("unsupported RM3-AV protocol version")
    if repo_root is not None:
        _verify_parent_pin(matrix.get("parent_independent_audit", {}), repo_root=repo_root, label="parent independent audit")
        _verify_parent_pin(matrix.get("parent_design", {}), repo_root=repo_root, label="parent design")
        _verify_parent_pin(matrix.get("parent_rm3_matrix", {}), repo_root=repo_root, label="parent RM3 matrix")

    data = matrix.get("data_contract", {})
    if data.get("test_allowed") is not False or float(data.get("pretest_end_fraction", -1)) != 0.8:
        raise Phase35ProtocolError("RM3-AV test/pretest boundary changed")
    if int(data.get("window_steps", 0)) != 96 or int(data.get("horizon_steps", 0)) != 60:
        raise Phase35ProtocolError("RM3-AV window/horizon changed")
    if int(data.get("step_seconds", 0)) != 10 or float(data.get("max_age_s", 0)) != 180.0:
        raise Phase35ProtocolError("RM3-AV time contract changed")
    if data.get("independent_unit") != "UTC_day":
        raise Phase35ProtocolError("RM3-AV independent unit must be UTC_day")
    expected_folds = {
        "F0": ((0.0, 0.6), (0.6, 0.7)),
        "F1": ((0.0, 0.7), (0.7, 0.8)),
    }
    observed = {
        raw.get("fold_id"): (
            _fraction(raw.get("train_fraction", ()), "train"),
            _fraction(raw.get("validation_fraction", ()), "validation"),
        )
        for raw in data.get("folds", ())
    }
    if observed != expected_folds:
        raise Phase35ProtocolError("RM3-AV rolling folds changed")

    candidates = matrix.get("candidates", ())
    if len(candidates) != 32 or tuple(raw.get("candidate_id") for raw in candidates) != RM3AV_CANDIDATE_IDS:
        raise Phase35ProtocolError("RM3-AV candidate matrix must contain ordered C00-C31")
    for raw in candidates:
        required = {
            "candidate_id", "group", "base_candidate_id", "intervention_axis",
            "intervention", "optimizer_updates_cap",
        }
        if set(raw) != required:
            raise Phase35ProtocolError("RM3-AV candidate fields changed")
        if raw["group"] not in GROUPS or raw["base_candidate_id"] not in BASE_CANDIDATES:
            raise Phase35ProtocolError("RM3-AV candidate group/base is invalid")
        if not isinstance(raw["intervention"], Mapping) or set(raw["intervention"]) != {raw["intervention_axis"]}:
            raise Phase35ProtocolError("RM3-AV candidate must change one declared intervention")
        cap = int(raw["optimizer_updates_cap"])
        if cap not in {4000, 8000}:
            raise Phase35ProtocolError("RM3-AV update cap is invalid")
    observed_candidates = tuple(
        (
            raw["candidate_id"],
            raw["group"],
            raw["base_candidate_id"],
            raw["intervention_axis"],
            raw["intervention"][raw["intervention_axis"]],
            int(raw["optimizer_updates_cap"]),
        )
        for raw in candidates
    )
    if observed_candidates != EXPECTED_CANDIDATES:
        raise Phase35ProtocolError("RM3-AV candidate semantics changed")
    high_cap = {raw["candidate_id"] for raw in candidates if int(raw["optimizer_updates_cap"]) == 8000}
    if high_cap != {"C28", "C29", "C30"}:
        raise Phase35ProtocolError("RM3-AV only C28-C30 may use 8000 updates")

    model = matrix.get("model", {})
    if model != {"d_model": 64, "latent_dim": 32, "dropout": 0.1}:
        raise Phase35ProtocolError("RM3-AV model capacity contract changed")
    training = matrix.get("training", {})
    expected_training = {
        "batch_size": 128,
        "default_optimizer_updates_cap": 4000,
        "evaluation_interval_updates": 100,
        "stats_anchor_count": 16384,
        "selector_anchor_count": 2048,
        "reporting_anchor_count": 4096,
        "diagnostic_anchor_count": 512,
        "evaluation_batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "gradient_clip": 1.0,
        "minimum_score_improvement": 0.0001,
        "component_loss_weights": {"valve": 0.25, "tin": 0.25, "local": 0.25, "terminal": 0.25},
        "logged_action_auxiliary_weight": 0.10,
        "oof_r_loss_weight": 0.10,
        "delta_valve_weight": 0.05,
        "roughness_weight": 0.05,
        "rollout_weight": 0.10,
    }
    if training != expected_training:
        raise Phase35ProtocolError("RM3-AV training contract changed")

    diagnostics = matrix.get("diagnostic_contract", {})
    if diagnostics != {
        "required_modes": [
            "normal", "bypass_off", "bypass_only", "response_off", "predicted_valve",
            "logged_valve", "logged_valve_oracle_tin", "oracle_local", "shuffled",
            "wrong_side", "lead",
        ],
        "response_horizons_steps": [6, 18, 60],
        "finite_difference_valve_points": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        "side_reporting": ["pooled", "A", "B", "common", "differential"],
        "audit_verdicts": ["SUPPORTED", "REFUTED", "MIXED", "NOT_TESTABLE"],
        "automatic_verdict": False,
    }:
        raise Phase35ProtocolError("RM3-AV diagnostic contract changed")

    envelope = matrix.get("matrix_envelope", {})
    if (
        envelope.get("status") != "design_frozen_implementation_not_authorized"
        or envelope.get("folds") != ["F0", "F1"]
        or envelope.get("seeds") != [0]
        or int(envelope.get("candidate_count", -1)) != 32
        or int(envelope.get("training_unit_count", -1)) != 64
        or envelope.get("zero_training_replay_included") is not True
        or envelope.get("no_composite_champion") is not True
    ):
        raise Phase35ProtocolError("RM3-AV envelope must close 64 training units")

    execution = matrix.get("execution_contract", {})
    for key in (
        "local_full_training_authorized", "linux_authorized", "test_authorized",
        "ms4_authorized", "rm3b_authorized", "automatic_scientific_pass",
    ):
        if execution.get(key) is not False:
            raise Phase35ProtocolError(f"RM3-AV {key} must remain false")
    if execution.get("local_micro_smoke_authorized") is not True:
        raise Phase35ProtocolError("RM3-AV local micro smoke must be authorized")
    if int(execution.get("maximum_attempts_per_run", 0)) != 1:
        raise Phase35ProtocolError("RM3-AV permits one attempt per unit")
    if set(execution.get("required_run_artifacts", ())) != REQUIRED_RUN_ARTIFACTS:
        raise Phase35ProtocolError("RM3-AV run artifact contract changed")
    if set(execution.get("required_root_artifacts", ())) != REQUIRED_ROOT_ARTIFACTS:
        raise Phase35ProtocolError("RM3-AV root artifact contract changed")
    claims = matrix.get("claim_contract", {})
    if set(claims) != {
        "model_champion", "causal_identification", "arbitrary_do_valve",
        "independent_side_gain", "true_response_order", "state_closed_simulator",
        "closed_loop_release",
    }:
        raise Phase35ProtocolError("RM3-AV claim contract fields changed")
    for key, value in claims.items():
        if value is not False:
            raise Phase35ProtocolError(f"RM3-AV forbidden claim enabled: {key}")


def rm3av_run_specs(
    matrix: Mapping[str, Any], *, repo_root: Path | None = None
) -> tuple[RM3AVRunSpec, ...]:
    validate_rm3av_matrix(matrix, repo_root=repo_root)
    folds = {
        raw["fold_id"]: (
            _fraction(raw["train_fraction"], "train"),
            _fraction(raw["validation_fraction"], "validation"),
        )
        for raw in matrix["data_contract"]["folds"]
    }
    specs = tuple(
        RM3AVRunSpec(
            run_id=f"{raw['candidate_id']}_{fold_id}_s0",
            candidate_id=raw["candidate_id"],
            group=raw["group"],
            base_candidate_id=raw["base_candidate_id"],
            intervention_axis=raw["intervention_axis"],
            intervention_value=raw["intervention"][raw["intervention_axis"]],
            optimizer_updates_cap=int(raw["optimizer_updates_cap"]),
            fold_id=fold_id,
            seed=0,
            train_fraction=fractions[0],
            validation_fraction=fractions[1],
        )
        for raw in matrix["candidates"]
        for fold_id, fractions in folds.items()
    )
    if len(specs) != 64:
        raise Phase35ProtocolError("RM3-AV run expansion must produce 64 training units")
    return specs
