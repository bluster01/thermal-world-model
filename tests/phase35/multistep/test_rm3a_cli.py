from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import experiments.phase3_5.ms3r_rm3a_train as rm3a_train
from experiments.phase3_5.ms3r_rm3a_train import dry_run_payload


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "configs/phase3_5/ms3r_rm3a_matrix.json"


def test_rm3a_dry_run_expands_only_30_new_runs() -> None:
    payload = dry_run_payload(json.loads(MATRIX.read_text(encoding="utf-8")))
    assert payload["new_candidate_count"] == 5
    assert payload["new_run_count"] == 30
    assert payload["reused_reference_run_count"] == 18
    assert payload["matrix_self_authorizing"] is False
    assert payload["test_authorized"] is False


def test_rm3a_refuses_a_different_authorized_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(
        (ROOT / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    registry["active_gate"] = "ms3_r"
    registry["linux_authorized_gate"] = "ms3_r"
    registry["experiments"]["ms3_r"]["status"] = "ready_for_linux"
    registry["experiments"]["ms3_r"]["decision"]["authorized_batch"] = "RM3-AV0+AV1"
    registry_path = tmp_path / "experiment_registry_rm3av.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(rm3a_train, "REGISTRY", registry_path)
    with pytest.raises(RuntimeError, match="authorized_batch=RM3-A"):
        rm3a_train._verify_registry()


def test_rm3a_registry_accepts_only_its_own_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(
        (ROOT / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    registry["active_gate"] = "ms3_r"
    registry["linux_authorized_gate"] = "ms3_r"
    registry["experiments"]["ms3_r"]["status"] = "ready_for_linux"
    registry["experiments"]["ms3_r"]["decision"]["authorized_batch"] = "RM3-A"
    registry_path = tmp_path / "experiment_registry_rm3a.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(rm3a_train, "REGISTRY", registry_path)
    rm3a_train._verify_registry()


def test_rm3a_cli_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "experiments/phase3_5/ms3r_rm3a_train.py"), "--dry-run"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["new_run_count"] == 30
