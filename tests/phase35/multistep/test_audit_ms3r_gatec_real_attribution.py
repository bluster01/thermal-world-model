from __future__ import annotations

from pathlib import Path

import pytest

from experiments.phase3_5.audit_ms3r_gatec_real_attribution import build_audit


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/phase3_5/ms3r_gatec_local_real_rm1a"


def test_real_rm1a_audit_closes_frozen_contrasts() -> None:
    audit = build_audit(RESULTS)
    assert audit["artifact_integrity"]["ledger_entries_verified"] == 8
    assert audit["artifact_integrity"]["test_accessed"] is False
    assert audit["rm0b_a1_reuse"]["max_abs_metric_difference"] == 0.0
    assert audit["rm0b_a1_reuse"]["duplicate_rm1b_rerun_needed"] is False
    capacity = audit["primary_contrasts"]["scheduled_capacity_scan"]
    assert capacity["relative_range"] == pytest.approx(0.0026333946521669115)
    assert capacity["monotonic_disappearance_with_capacity"] is False
    terminal_only = audit["primary_contrasts"]["terminal_only_vs_scheduled_base"]
    assert terminal_only["semantic_failure"] is True
    assert terminal_only["terminal_only_local_to_persistence_ratio"] > 9.0
    decision = audit["supervisor_decision"]
    assert decision["reference_candidate"] == "C3_sched_base"
    assert decision["reference_is_empirical_champion"] is False
    assert decision["operator_ranking_supported"] is False
    assert decision["linux_authorized_gate"] is None
