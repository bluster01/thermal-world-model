from __future__ import annotations

import json
from pathlib import Path

from experiments.phase3_5.audit_ms3r_gatec_rm2 import audit


ROOT = Path(__file__).resolve().parents[3]


def test_repository_rm2_audit_replays_without_cache_or_training() -> None:
    payload = audit(
        ROOT / "results/phase3_5/ms3r_gatec_rm2",
        ROOT / "results/phase3_5/ms3r_gateb_point_closure/supervisor_audit_validation.json",
    )
    integrity = payload["artifact_integrity"]
    decision = payload["supervisor_decision"]
    assert integrity["complete_run_count"] == 54
    assert integrity["per_run_ledger_entries_verified"] == 216
    assert integrity["checkpoint_archive_members_verified"] == 54
    assert integrity["all_sha256_match"] is True
    assert integrity["test_accessed"] is False
    assert integrity["training_executed_locally"] is False
    assert decision["operator_ranking_supported"] is False
    assert decision["linux_authorized_gate"] is None


def test_committed_rm2_audit_matches_replay_decision() -> None:
    expected = json.loads(
        (ROOT / "results/phase3_5/ms3r_gatec_rm2/supervisor_audit_validation.json").read_text(
            encoding="utf-8"
        )
    )
    replayed = audit(
        ROOT / "results/phase3_5/ms3r_gatec_rm2",
        ROOT / "results/phase3_5/ms3r_gateb_point_closure/supervisor_audit_validation.json",
    )
    assert replayed["supervisor_decision"] == expected["supervisor_decision"]
    assert replayed["paired_contrasts"] == expected["paired_contrasts"]
