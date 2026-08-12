"""Closed contracts for the RM3-A capacity and loss trade-off ablation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from ..schema import Phase35ProtocolError


EXPECTED_NEW = {
    "A0_p3_large": ("P3_gatec_paired_free", 77, 32, 120928, "balanced"),
    "A1_p4_large": ("P4_gatec_a1_scheduled", 77, 32, 121103, "balanced"),
    "A2_p5_small": ("P5_hybrid_joint_latent", 52, 32, 83649, "balanced"),
    "A3_p5_local35": ("P5_hybrid_joint_latent", 64, 32, 122301, "local35_terminal35"),
    "A4_p5_local50": ("P5_hybrid_joint_latent", 64, 32, 122301, "local50_terminal30"),
}


@dataclass(frozen=True)
class RM3ARunSpec:
    run_id: str
    candidate_id: str
    base_candidate_id: str
    d_model: int
    latent_dim: int
    state_elements_expected: int
    loss_profile: str
    loss_weights: dict[str, float]
    fold_id: str
    seed: int
    train_fraction: tuple[float, float]
    validation_fraction: tuple[float, float]
    output_scope: str = "full_multitask"
    future_action_access: str = "future_sp"
    role: str = "rm3a_capacity_loss_ablation"
    prefix_causal_action_path: bool = True


def _text_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_rm3a_matrix(matrix: Mapping[str, Any], *, repo_root: Path | None = None) -> None:
    if matrix.get("protocol_version") != "phase3.5-ms3r-rm3a-v1":
        raise Phase35ProtocolError("unsupported RM3-A protocol")
    parent = matrix.get("parent_rm3_audit", {})
    if parent.get("hash_mode") != "utf8_text_normalized_lf" or len(parent.get("sha256", "")) != 64:
        raise Phase35ProtocolError("RM3-A parent audit pin is invalid")
    if repo_root is not None:
        path = repo_root / parent["path"]
        if _text_sha(path) != parent["sha256"]:
            raise Phase35ProtocolError("RM3-A parent audit hash changed")
    data = matrix.get("data_contract", {})
    if data.get("test_allowed") is not False or data.get("window_steps") != 96 or data.get("horizon_steps") != 60:
        raise Phase35ProtocolError("RM3-A data contract changed")
    expected_folds = [
        {"fold_id": "F0", "train_fraction": [0.0, 0.6], "validation_fraction": [0.6, 0.7]},
        {"fold_id": "F1", "train_fraction": [0.0, 0.7], "validation_fraction": [0.7, 0.8]},
    ]
    if data.get("folds") != expected_folds:
        raise Phase35ProtocolError("RM3-A folds changed")
    observed = {
        item["candidate_id"]: (
            item["base_candidate_id"], item["d_model"], item["latent_dim"],
            item["state_elements_expected"], item["loss_profile"],
        )
        for item in matrix.get("new_candidates", ())
    }
    if observed != EXPECTED_NEW:
        raise Phase35ProtocolError("RM3-A new candidate matrix changed")
    profiles = matrix.get("loss_profiles", {})
    if set(profiles) != {"balanced", "local35_terminal35", "local50_terminal30"}:
        raise Phase35ProtocolError("RM3-A loss profiles changed")
    for weights in profiles.values():
        if set(weights) != {"valve", "tin", "local", "terminal"} or abs(sum(weights.values()) - 1.0) > 1e-12:
            raise Phase35ProtocolError("RM3-A loss weights must be closed and sum to one")
    training = matrix.get("training", {})
    if training.get("seeds") != [0, 1, 2] or training.get("new_run_count") != 30:
        raise Phase35ProtocolError("RM3-A run envelope changed")
    if training.get("optimizer_updates_cap") != 4000:
        raise Phase35ProtocolError("RM3-A update cap changed")
    decision = matrix.get("decision_contract", {})
    if decision.get("no_single_composite_champion") is not True or decision.get("automatic_scientific_pass") is not False:
        raise Phase35ProtocolError("RM3-A decision boundary changed")
    execution = matrix.get("execution_contract", {})
    if any(execution.get(key) is not False for key in ("linux_authorized", "test_authorized", "ms4_authorized")):
        raise Phase35ProtocolError("RM3-A execution must remain unauthorized")
    if execution.get("maximum_attempts_per_run") != 1:
        raise Phase35ProtocolError("RM3-A permits exactly one attempt")
    if set(execution.get("required_run_artifacts", ())) != {
        "manifest.json", "checkpoint_best_validation.pt", "metrics_validation.json",
        "episodes_validation.npz", "artifact_ledger.json",
    }:
        raise Phase35ProtocolError("RM3-A run artifact contract changed")
    if set(execution.get("required_root_artifacts", ())) != {
        "run_manifest.json", "matrix_execution_status.json", "summary_validation.json",
        "artifact_ledger.json",
    }:
        raise Phase35ProtocolError("RM3-A root artifact contract changed")
    if len(rm3a_run_specs(matrix)) != 30:
        raise Phase35ProtocolError("RM3-A expanded run count changed")


def rm3a_run_specs(matrix: Mapping[str, Any]) -> tuple[RM3ARunSpec, ...]:
    profiles = matrix["loss_profiles"]
    return tuple(
        RM3ARunSpec(
            run_id=f"{candidate['candidate_id']}_{fold['fold_id']}_s{seed}",
            **candidate,
            loss_weights={key: float(value) for key, value in profiles[candidate["loss_profile"]].items()},
            fold_id=fold["fold_id"],
            seed=int(seed),
            train_fraction=tuple(fold["train_fraction"]),
            validation_fraction=tuple(fold["validation_fraction"]),
        )
        for candidate in matrix["new_candidates"]
        for fold in matrix["data_contract"]["folds"]
        for seed in matrix["training"]["seeds"]
    )
