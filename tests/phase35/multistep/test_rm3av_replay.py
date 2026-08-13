from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import torch

from src.phase35.multistep.rm3av_replay import (
    audit_reference_artifacts,
    audit_rm2_reference_artifacts,
    build_av0_replay,
    build_calibration_corrections,
    blocked_shape_model_audit,
    load_legacy_checkpoint_as_rm3av,
    replay_loaded_model,
)


ROOT = Path(__file__).resolve().parents[3]
RM3 = ROOT / "results/phase3_5/ms3r_rm3/prediction"
RM3A = ROOT / "results/phase3_5/ms3r_rm3a"
RM2 = ROOT / "results/phase3_5/ms3r_gatec_rm2"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_rm2_reference_audit_counts_local_and_archive_checkpoints(
    tmp_path: Path,
) -> None:
    local_payload = b"local-checkpoint"
    archived_payload = b"archived-checkpoint"
    local_run = tmp_path / "local_run"
    archived_run = tmp_path / "archived_run"
    local_run.mkdir()
    archived_run.mkdir()
    (local_run / "checkpoint_best_validation.pt").write_bytes(local_payload)
    (local_run / "artifact_ledger.json").write_text(
        json.dumps({"checkpoint_best_validation.pt": _sha256(local_payload)}),
        encoding="utf-8",
    )
    (archived_run / "artifact_ledger.json").write_text(
        json.dumps({"checkpoint_best_validation.pt": _sha256(archived_payload)}),
        encoding="utf-8",
    )
    archive_path = tmp_path / "checkpoints_validation.tar"
    archived_file = tmp_path / "checkpoint_best_validation.pt"
    archived_file.write_bytes(archived_payload)
    with tarfile.open(archive_path, "w") as archive:
        archive.add(
            archived_file,
            arcname="archived_run/checkpoint_best_validation.pt",
        )
    (tmp_path / "artifact_ledger.json").write_text("{}", encoding="utf-8")

    payload = audit_rm2_reference_artifacts(tmp_path)

    assert payload["run_count"] == 2
    assert payload["checkpoint_count"] == 2
    assert payload["checkpoint_archive_member_count"] == 1
    assert payload["root_hash_error_count"] == 0
    assert payload["run_hash_error_count"] == 0


def test_rm2_reference_audit_closes_all_54_archived_checkpoints() -> None:
    payload = audit_rm2_reference_artifacts(RM2)
    assert payload["run_count"] == 54
    assert payload["checkpoint_count"] == 54
    assert payload["checkpoint_archive_member_count"] == 54
    assert payload["root_hash_error_count"] == 0
    assert payload["run_hash_error_count"] == 0


def test_reference_artifact_audit_closes_all_66_checkpoints() -> None:
    payload = audit_reference_artifacts(RM3, RM3A)
    assert payload["rm3_prediction_run_count"] == 36
    assert payload["rm3a_run_count"] == 30
    assert payload["checkpoint_count"] == 66
    assert payload["hash_error_count"] == 0
    assert payload["test_accessed"] is False


def test_legacy_p4_checkpoint_loads_into_intervention_capable_wrapper() -> None:
    checkpoint_path = RM3 / "P4_gatec_a1_scheduled_F0_s0/checkpoint_best_validation.pt"
    model, metadata = load_legacy_checkpoint_as_rm3av(checkpoint_path)
    assert metadata["legacy_candidate_id"] == "P4_gatec_a1_scheduled"
    assert metadata["rm3av_anchor_candidate_id"] == "C01"
    assert metadata["legacy_parameters_loaded"] is True
    assert model.training is False

    generator = torch.Generator().manual_seed(711)
    history = torch.randn(2, 96, 15, generator=generator)
    future_sp = 540.0 + torch.randn(2, 60, 2, generator=generator)
    logged_valve = 35.0 + torch.randn(2, 60, 2, generator=generator)
    payload = replay_loaded_model(
        model,
        history,
        future_sp,
        logged_valve,
        550.0 + torch.randn(2, 60, 2, generator=generator),
        10.0 + torch.randn(2, 60, 2, generator=generator),
        540.0 + torch.randn(2, 60, 2, generator=generator),
    )
    assert len(payload["mode_metrics"]) == 11
    assert payload["constant_action_identity_max_abs"] == 0.0
    assert payload["prefix_causality_max_abs_before_change"] < 1e-6
    assert payload["finite"] is True
    assert payload["test_accessed"] is False


def test_av0_replay_is_zero_training_and_does_not_overstate_p0_to_p2() -> None:
    payload = build_av0_replay(
        ROOT / "results/phase3_5/ms3r_rm3",
        ROOT / "results/phase3_5/ms3r_rm3a",
        cache_a=None,
        cache_b=None,
    )
    assert payload["reference_artifacts"]["checkpoint_count"] == 66
    assert payload["reference_artifacts"]["rm2"]["checkpoint_count"] == 54
    assert payload["legacy_metrics"]["rm2"]["run_count"] == 54
    assert payload["legacy_episode_diagnostics"]["rm3"]["run_count"] == 36
    assert payload["legacy_episode_diagnostics"]["rm3a"]["run_count"] == 30
    p2 = next(
        row for row in payload["legacy_episode_diagnostics"]["rm3"]["records"]
        if row["candidate_id"] == "P2_m9_future_sp"
    )
    assert set(p2["tasks"]["terminal"]) == {"6", "18", "60"}
    assert p2["exact_history_last_persistence_available"] is False
    assert payload["zero_training"] is True
    assert payload["test_accessed"] is False
    assert payload["automatic_scientific_pass"] is None
    assert payload["state_closure"]["state_closed_simulator"] is False
    assert len(payload["manual_audit_verdicts"]) == 33
    assert payload["functional_replay"]["status"] == "CACHE_REQUIRED_FOR_INPUT_REPLAY"
    assert payload["legacy_output_domain"]["P0_m7_oracle_valve"]["functional_response_replay"] is False
    corrections = payload["calibration_corrections"]
    assert corrections["calibration_unit_count"] == 12
    assert corrections["historical_files_overwritten"] is False
    assert all(record["algorithm"] == "exact_active_set_nnls_v1" for record in corrections["records"])
    assert all(len(record["corrected_payload_sha256"]) == 64 for record in corrections["records"])
    assert all("R1_a1_scheduled" in record["corrected_payload"]["results"] for record in corrections["records"])
    assert max(record["corrected_projection_rmse"] for record in corrections["records"]) < 0.08
    shapes = payload["blocked_shape_model_audits"]
    assert shapes["calibration_unit_count"] == 12
    assert shapes["true_response_order_claim"] is False
    assert set(shapes["cross_calibration_shape_summary"]) == {
        "linear_ramp", "power_basis", "one_pole", "two_pole",
        "three_pole", "three_pole_bounded_dead_time",
    }


def test_blocked_shape_audit_uses_future_date_blocks_without_overlap() -> None:
    source = next((ROOT / "results/phase3_5/ms3r_rm3/calibration").glob(
        "*/orthogonal_residuals_validation.npz"
    ))
    payload = blocked_shape_model_audit(source)
    assert payload["split_count"] == 3
    assert all(record["day_overlap_count"] == 0 for record in payload["records"])
    assert all(max(record["train_days"]) < min(record["held_days"]) for record in payload["records"])
    assert payload["true_response_order_claim"] is False
