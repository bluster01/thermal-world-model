from __future__ import annotations

import json
from pathlib import Path

from experiments.phase3_5.ms3r_rm3 import dry_run_payload, synthetic_smoke


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms3r_rm3_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_rm3_dry_run_separates_identification_and_prediction() -> None:
    payload = dry_run_payload(_matrix())
    assert payload["identification_candidates"] == [
        "R0_linear_mimo",
        "R1_a1_scheduled",
        "R2_a1_common_only",
    ]
    assert payload["prediction_candidate_count"] == 6
    assert payload["primary_response_horizons_steps"] == [6, 18]
    assert payload["raw_future_valve_auxiliary_allowed"] is False
    assert payload["local_real_training_authorized"] is False
    assert payload["linux_authorized"] is False
    assert payload["test_authorized"] is False
    assert payload["automatic_scientific_pass"] is None


def test_rm3_synthetic_smoke_recovers_truth_without_real_claim() -> None:
    payload = synthetic_smoke(_matrix())
    assert payload["evaluated_row_count"] == 1200
    assert payload["maximum_absolute_recovery_error"] < 0.03
    assert payload["synthetic_smoke_pass"] is True
    assert payload["real_data_claim"] is None
    assert payload["automatic_scientific_pass"] is None
