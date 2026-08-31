from __future__ import annotations

import json

import pytest

from experiments.final_wm import matrix_spec as ms
from experiments.final_wm.audit_manifest import expected_run_ids, sha256_file, verify_manifest
from src.final_wm.contracts import FinalWMProtocolError


def test_expected_manifest_run_set_is_full() -> None:
    run_ids = expected_run_ids()
    assert len(run_ids) == 39
    assert {f"o1_steady_seed{s}" for s in ms.SEEDS} <= set(run_ids)
    assert {f"t1_closure_cons_norew_seed{s}" for s in ms.SEEDS} <= set(run_ids)
    assert {f"j1_staged_boundary_from_{s}_seed{s}" for s in ms.SEEDS} <= set(run_ids)


def test_verify_manifest_detects_artifact_tampering(tmp_path) -> None:
    record = tmp_path / "record.npz"
    properties = tmp_path / "properties.npz"
    artifact = tmp_path / "summary.json"
    record.write_bytes(b"record")
    properties.write_bytes(b"properties")
    artifact.write_bytes(b"summary-v1")
    payload = {
        "authoritative": True,
        "matrix_version": ms.MATRIX_VERSION,
        "seeds": list(ms.SEEDS),
        "r1_arm": ms.R1_ARM,
        "test_locked": True,
        "expected_run_ids": expected_run_ids(),
        "inputs": {
            "record": {"path": str(record), "sha256": sha256_file(record)},
            "properties": {"path": str(properties), "sha256": sha256_file(properties)},
        },
        "files": {"summary.json": sha256_file(artifact)},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_manifest(manifest)["verified"] is True
    artifact.write_bytes(b"summary-v2")
    with pytest.raises(FinalWMProtocolError, match="artifact hash mismatch"):
        verify_manifest(manifest)


def test_verify_manifest_prefers_portable_input_path(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    side = bundle / "sideA"
    inputs = bundle / "inputs"
    side.mkdir(parents=True)
    inputs.mkdir()
    record = inputs / "record.npz"
    properties = inputs / "properties.npz"
    artifact = side / "summary.json"
    record.write_bytes(b"record")
    properties.write_bytes(b"properties")
    artifact.write_bytes(b"summary")
    payload = {
        "authoritative": True,
        "matrix_version": ms.MATRIX_VERSION,
        "seeds": list(ms.SEEDS),
        "r1_arm": ms.R1_ARM,
        "test_locked": True,
        "expected_run_ids": expected_run_ids(),
        "inputs": {
            "record": {"path": "/linux/missing/record.npz", "package_relative_path": "../inputs/record.npz", "sha256": sha256_file(record)},
            "properties": {"path": "/linux/missing/properties.npz", "package_relative_path": "../inputs/properties.npz", "sha256": sha256_file(properties)},
        },
        "files": {"summary.json": sha256_file(artifact)},
    }
    manifest = side / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_manifest(manifest)["verified"] is True
