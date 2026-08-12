from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.phase35.multistep.rm3_contracts import RM3PredictionRunSpec
from src.phase35.multistep.rm3_reporting import file_sha256, summarize_rm3_predictions
from src.phase35.schema import Phase35ProtocolError


REQUIRED = (
    "manifest.json", "checkpoint_best_validation.pt", "metrics_validation.json",
    "episodes_validation.npz", "artifact_ledger.json",
)


def _spec(run_id: str, candidate: str, scope: str) -> RM3PredictionRunSpec:
    return RM3PredictionRunSpec(
        run_id=run_id, candidate_id=candidate, future_action_access="future_sp",
        role="test", output_scope=scope, prefix_causal_action_path=True,
        fold_id="F0", seed=0, train_fraction=(0.0, 0.6), validation_fraction=(0.6, 0.7),
    )


def _write_run(root: Path, spec: RM3PredictionRunSpec, terminal_mae: float) -> None:
    run = root / spec.run_id
    run.mkdir(parents=True)
    (run / "checkpoint_best_validation.pt").write_bytes(b"checkpoint")
    (run / "episodes_validation.npz").write_bytes(b"episodes")
    (run / "manifest.json").write_text(json.dumps({
        "run_id": spec.run_id, "run_spec": {"candidate_id": spec.candidate_id},
        "selector_reporting_disjoint": True, "test_accessed": False,
    }), encoding="utf-8")
    (run / "metrics_validation.json").write_text(json.dumps({
        "run_id": spec.run_id, "output_scope": spec.output_scope,
        "metrics": {"terminal_mae_c": terminal_mae}, "test_accessed": False,
    }), encoding="utf-8")
    names = set(REQUIRED) - {"artifact_ledger.json"}
    (run / "artifact_ledger.json").write_text(
        json.dumps({name: file_sha256(run / name) for name in names}), encoding="utf-8"
    )


def test_reporting_keeps_scope_leaderboards_separate(tmp_path: Path) -> None:
    specs = (
        _spec("p0", "P0", "terminal_only"),
        _spec("p1", "P1", "valve_and_terminal"),
        _spec("p2", "P2", "terminal_only"),
    )
    _write_run(tmp_path, specs[0], 0.7)
    _write_run(tmp_path, specs[1], 0.5)
    _write_run(tmp_path, specs[2], 0.6)
    summary = summarize_rm3_predictions(tmp_path, specs, required_artifacts=REQUIRED)
    assert summary["common_descriptive_metric"] == "terminal_mae_c"
    assert summary["cross_output_scope_composite_ranking"] is None
    assert [row["candidate_id"] for row in summary["scope_qualified_leaderboards"]["terminal_only"]] == ["P2", "P0"]
    assert [row["candidate_id"] for row in summary["scope_qualified_leaderboards"]["valve_and_terminal"]] == ["P1"]


def test_reporting_rejects_tampered_artifact(tmp_path: Path) -> None:
    spec = _spec("p0", "P0", "terminal_only")
    _write_run(tmp_path, spec, 0.7)
    (tmp_path / "p0" / "metrics_validation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Phase35ProtocolError, match="hash mismatch"):
        summarize_rm3_predictions(tmp_path, (spec,), required_artifacts=REQUIRED)
