from __future__ import annotations

from pathlib import Path

from experiments.phase3_5.audit_ms3r_rm3 import replay


ROOT = Path(__file__).resolve().parents[3]


def test_repository_rm3_replay_is_metric_exact_and_checkpoint_incomplete() -> None:
    payload = replay(ROOT / "results/phase3_5/ms3r_rm3")
    assert payload["execution"]["record_count"] == 48
    assert payload["execution"]["complete_record_count"] == 48
    assert payload["metric_replay"]["prediction_run_count"] == 36
    assert payload["metric_replay"]["terminal_mae_max_absolute_replay_error"] < 3e-8
    assert payload["artifact_integrity"]["available_ledger_entries_verified"] == 132
    assert payload["artifact_integrity"]["non_checkpoint_integrity_errors"] == []
    assert payload["artifact_integrity"]["missing_checkpoint_count"] == 36
    assert payload["artifact_integrity"]["checkpoint_audit_complete"] is False
    assert payload["response_calibration"]["independent_channel_rank_supported_unit_count"] == 12
    corrected = payload["response_calibration"]["corrected_a1_nnls_projection"]
    assert max(item["corrected_nnls_rmse"] for item in corrected) < 0.08
    assert payload["supervisor_status"] == (
        "PROVISIONAL_ARTIFACT_INCOMPLETE_CHECKPOINT_SUPPLEMENT_REQUIRED"
    )
