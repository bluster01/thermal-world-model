import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _module():
    from experiments.phase3_5 import experiment_status

    return experiment_status


def test_repository_registry_is_valid_and_has_one_active_gate():
    module = _module()
    registry = module.load_registry(
        ROOT / "configs/phase3_5/experiment_registry.json"
    )
    report = module.validate_registry(registry, ROOT)
    assert report["valid"] is True
    assert report["active_gate"] == "ms3_r"
    assert report["active_status"] == "ready_for_linux"
    assert report["linux_authorized_gate"] == "ms3_r"
    assert report["errors"] == []


def test_registry_rejects_deprecated_e_track_as_active(tmp_path):
    module = _module()
    registry = module.load_registry(
        ROOT / "configs/phase3_5/experiment_registry.json"
    )
    registry["active_gate"] = "legacy_e"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    report = module.validate_registry(module.load_registry(path), ROOT)
    assert report["valid"] is False
    assert any("deprecated" in error for error in report["errors"])


def test_registry_rejects_missing_required_script():
    module = _module()
    registry = module.load_registry(
        ROOT / "configs/phase3_5/experiment_registry.json"
    )
    registry["experiments"]["ms2d_d1"]["scripts"]["runner"] = {
        "path": "experiments/phase3_5/does_not_exist.py",
        "status": "active",
        "required": True,
    }
    report = module.validate_registry(registry, ROOT)
    assert report["valid"] is False
    assert any("does_not_exist.py" in error for error in report["errors"])


def test_status_cli_emits_machine_readable_summary():
    script = ROOT / "experiments/phase3_5/experiment_status.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--check", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["active_gate"] == "ms3_r"
    assert payload["active_status"] == "ready_for_linux"
    assert payload["deprecated_tracks"] == ["legacy_e"]


def test_registry_accepts_test_authorized_active_gate():
    module = _module()
    registry = module.load_registry(
        ROOT / "configs/phase3_5/experiment_registry.json"
    )
    registry["active_gate"] = "ms2d_d1"
    registry["linux_authorized_gate"] = "ms2d_d1"
    registry["experiments"]["ms2d_d1"]["status"] = "test_authorized"
    registry["experiments"]["ms2d_d2"]["status"] = "planned"
    registry["experiments"]["ms2d_d3"]["status"] = "planned"
    registry["experiments"]["ms5"]["status"] = "planned"
    registry["experiments"]["ms3"]["status"] = "planned"
    registry["experiments"]["ms3_r"]["status"] = "planned"
    report = module.validate_registry(registry, ROOT)
    assert report["valid"] is True
    assert report["active_status"] == "test_authorized"


def test_registry_rejects_two_linux_authorized_gates():
    module = _module()
    registry = module.load_registry(
        ROOT / "configs/phase3_5/experiment_registry.json"
    )
    registry["experiments"]["ms3"]["status"] = "ready_for_linux"
    registry["linux_authorized_gate"] = "ms2d_d2"
    registry["experiments"]["ms2d_d2"]["status"] = "ready_for_linux"
    report = module.validate_registry(registry, ROOT)
    assert report["valid"] is False
    assert any("multiple Linux-authorized" in error for error in report["errors"])


def test_registry_rejects_linux_authorization_for_a_non_active_gate():
    module = _module()
    registry = module.load_registry(
        ROOT / "configs/phase3_5/experiment_registry.json"
    )
    registry["experiments"]["ms2d_d1"]["status"] = "local_verified"
    registry["experiments"]["ms3"]["status"] = "ready_for_linux"
    registry["linux_authorized_gate"] = "ms3"
    registry["active_gate"] = "ms5"
    report = module.validate_registry(registry, ROOT)
    assert report["valid"] is False
    assert any("must equal active_gate" in error for error in report["errors"])
