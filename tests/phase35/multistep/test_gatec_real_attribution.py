from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase3_5.ms3r_gatec_real_attribution import dry_run, validate_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/phase3_5/ms3r_gatec_local_real_rm1a.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_rm1a_config_is_closed_and_local_only() -> None:
    config = _config()
    validate_config(config)
    payload = dry_run(config)
    assert payload["candidate_ids"] == [
        "C0_paired_free",
        "C1_additive_base",
        "C2_sched_small",
        "C3_sched_base",
        "C4_sched_large",
        "C5_sched_base_terminal_only",
    ]
    assert payload["fraction_denominator"] == 100
    assert payload["linux_authorized"] is False
    assert payload["test_authorized"] is False
    assert payload["automatic_scientific_pass"] is None


def test_rm1a_rejects_candidate_drift() -> None:
    config = copy.deepcopy(_config())
    config["candidates"][0]["residual_capacity"] = "large"
    with pytest.raises(RuntimeError, match="frozen parent matrix"):
        validate_config(config)
