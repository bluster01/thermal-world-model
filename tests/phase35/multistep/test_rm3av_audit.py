from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.phase35.multistep.rm3av_audit import build_av2_audit
from src.phase35.multistep.rm3av_contracts import rm3av_run_specs
from src.phase35.schema import Phase35ProtocolError


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_rm3av_matrix.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def _fake_return(tmp_path: Path) -> tuple[Path, dict]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    required = set(matrix["execution_contract"]["required_run_artifacts"])
    for spec in rm3av_run_specs(matrix, repo_root=ROOT):
        directory = tmp_path / spec.run_id
        directory.mkdir(parents=True)
        initialization = {
            "encoder": "same", "valve_policy": "same", "tin": "same",
            "free_residual": "same", "response": "same", "downstream": "same",
            "bypass": "same",
        }
        _json(directory / "manifest.json", {
            "run_id": spec.run_id,
            "selector_reporting_disjoint": True,
            "selector_reporting_utc_day_disjoint": True,
            "initialization_hashes": initialization,
            "test_accessed": False,
        })
        _json(directory / "metrics_validation.json", {
            "run_id": spec.run_id,
            "metrics": {
                "terminal_mae_c": 1.0, "local_mae_c": 2.0,
                "tin_mae_c": 3.0, "valve_mae": 4.0,
            },
            "optimizer_updates_completed": spec.optimizer_updates_cap,
            "test_accessed": False,
        })
        modes = {
            mode: {"status": "computed", "terminal_mae_c": 1.0, "local_mae_c": 2.0}
            for mode in matrix["diagnostic_contract"]["required_modes"]
        }
        _json(directory / "diagnostics_validation.json", {
            "candidate_id": spec.candidate_id,
            "terminal": {"skill_vs_persistence_pooled": 0.1},
            "local": {"skill_vs_persistence_pooled": 0.2},
            "valve_trajectory": {"persistence_skill": 0.3},
            "response_trajectory": {
                "mean_absolute_effect_by_horizon_steps": {"60": 0.4},
                "explicit_to_total_local_change_ratio": 0.5,
            },
            "mode_records": modes,
            "training_graph": {"response_training_path_reachable": True},
            "manual_audit_verdicts": {f"Q{index:02d}": None for index in range(1, 34)},
            "test_accessed": False,
        })
        (directory / "checkpoint_best_validation.pt").write_bytes(b"synthetic-test-checkpoint")
        (directory / "episodes_validation.npz").write_bytes(b"synthetic-test-episodes")
        ledger = {
            name: _sha(directory / name)
            for name in required - {"artifact_ledger.json"}
        }
        _json(directory / "artifact_ledger.json", ledger)
    _json(tmp_path / "run_manifest.json", {"test_accessed": False})
    _json(tmp_path / "matrix_execution_status.json", {"all_complete": True, "test_accessed": False})
    _json(tmp_path / "summary_validation.json", {"training_unit_count": 64, "test_accessed": False})
    root_ledger = {
        name: _sha(tmp_path / name)
        for name in set(matrix["execution_contract"]["required_root_artifacts"])
        - {"artifact_ledger.json"}
    }
    _json(tmp_path / "artifact_ledger.json", root_ledger)
    return tmp_path, matrix


def test_av2_assembles_all_33_questions_without_automatic_verdicts(tmp_path: Path) -> None:
    output, matrix = _fake_return(tmp_path)
    payload = build_av2_audit(output, matrix, repo_root=ROOT)
    assert payload["artifact_integrity_pass"] is True
    assert payload["training_unit_count"] == 64
    assert len(payload["paired_contrasts"]) == 29
    assert set(payload["question_evidence_index"]) == {
        f"Q{index:02d}" for index in range(1, 34)
    }
    assert all(row["verdict"] is None for row in payload["question_evidence_index"].values())
    assert all(value is None for value in payload["manual_audit_verdicts"].values())
    initialization = payload["initialization_fairness"]
    assert initialization["all_expected_shared_modules_equal"] is True
    assert sum(
        row.get("comparison_role") == "cross_architecture_shared_module_rng_audit"
        for row in initialization["rows"]
    ) == 4
    assert payload["rm3b_authorized"] is False
    assert payload["automatic_scientific_pass"] is None


def test_av2_fails_closed_on_tampered_run_or_linux_verdict(tmp_path: Path) -> None:
    output, matrix = _fake_return(tmp_path)
    path = output / "C00_F0_s0" / "diagnostics_validation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manual_audit_verdicts"]["Q01"] = "SUPPORTED"
    _json(path, payload)
    with pytest.raises(Phase35ProtocolError, match="hash mismatch"):
        build_av2_audit(output, matrix, repo_root=ROOT)
