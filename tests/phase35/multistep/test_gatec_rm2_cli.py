from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase3_5.ms3r_gatec_rm2 import (
    _dirty_paths,
    _verify_complete_run,
    _verify_registry,
    dry_run_payload,
)
from experiments.phase3_5.summarize_ms3r_gatec_rm2 import summarize
from src.phase35.multistep.gatec_rm2_contracts import rm2_run_specs


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = ROOT / "configs/phase3_5/ms3r_gatec_rm2_matrix.json"


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def test_rm2_dry_run_reports_exact_parallel_batch() -> None:
    payload = dry_run_payload(_matrix())
    assert payload["unique_candidate_count"] == 9
    assert payload["run_count"] == 54
    assert payload["group_run_counts"] == {"A": 24, "B": 18, "C": 12}
    assert payload["seeds"] == [0, 1, 2]
    assert payload["folds"] == ["F0", "F1"]
    assert payload["test_authorized"] is False
    assert payload["automatic_scientific_pass"] is None


def test_rm2_registry_preflight_requires_final_linux_state(tmp_path: Path) -> None:
    registry = json.loads(
        (ROOT / "configs/phase3_5/experiment_registry.json").read_text(encoding="utf-8")
    )
    changed = copy.deepcopy(registry)
    changed["linux_authorized_gate"] = "ms3_r"
    changed["experiments"]["ms3_r"]["status"] = "ready_for_linux"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    _verify_registry(path)
    changed["linux_authorized_gate"] = None
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="linux_authorized_gate"):
        _verify_registry(path)


def test_rm2_clean_check_ignores_only_its_frozen_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.phase3_5.ms3r_gatec_rm2.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                " M src/phase35/multistep/gatec_model.py\n"
                "?? results/phase3_5/ms3r_gatec_rm2/remote_execution/stdout.log\n"
            )
        ),
    )
    dirty = _dirty_paths(allowed_output_root=ROOT / "results/phase3_5/ms3r_gatec_rm2")
    assert dirty == [" M src/phase35/multistep/gatec_model.py"]


def test_rm2_complete_run_check_and_summary_fail_closed(tmp_path: Path) -> None:
    matrix = _matrix()
    required = matrix["execution_contract"]["required_run_artifacts"]
    first = rm2_run_specs(matrix)[0]
    specs = rm2_run_specs(matrix)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"run_ids": [spec.run_id for spec in specs]}), encoding="utf-8"
    )
    (tmp_path / "matrix_execution_status.json").write_text(
        json.dumps({"records": [{"run_id": spec.run_id, "status": "failed"} for spec in specs]}),
        encoding="utf-8",
    )
    run_dir = tmp_path / first.run_id
    ledger = {}
    for name in set(required) - {"artifact_ledger.json"}:
        if name.endswith(".json"):
            if name == "manifest.json":
                value = json.dumps(
                    {
                        "test_accessed": False,
                        "run_spec": {"run_id": first.run_id},
                        "selector_reporting_disjoint": True,
                        "stats_anchor_sha256": "stats",
                        "selector_anchor_sha256": "selector",
                        "final_anchor_sha256": "final",
                    }
                ).encode()
            elif name == "metrics_validation.json":
                value = json.dumps(
                    {
                        "best_update": 1,
                        "optimizer_updates_completed": 2,
                        "metrics": {"finite": True},
                    }
                ).encode()
            else:
                value = b"{}"
        else:
            value = b"artifact"
        ledger[name] = _write(run_dir / name, value)
    (run_dir / "artifact_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    assert _verify_complete_run(run_dir, required) is True
    payload = summarize(matrix, tmp_path)
    assert payload["complete_run_count"] == 1
    assert payload["incomplete_run_count"] == 53
    assert payload["matrix_complete"] is False
    assert payload["automatic_scientific_pass"] is None
