"""Tiny synthetic fixtures for the independent merged-pack verifier."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.unified_merged_20260924.data_contract import file_hash, json_hash, verify_pack


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def pack(tmp_path):
    aliases = ["load", "toy"]
    data = tmp_path / "pack"
    data.mkdir()
    arrays = {"values": np.arange(16, dtype="<f4").reshape(2, 8),
              "valid": np.ones((2, 8), dtype=bool),
              "observed": np.ones((2, 8), dtype=bool),
              "flags": np.zeros((2, 8), dtype="u1"),
              "source_time_ns": np.tile(np.arange(8, dtype="<i8"), (2, 1)),
              "time_ns": np.arange(8, dtype="<i8"),
              "split": np.zeros(8, dtype="i1")}
    for name, value in arrays.items():
        np.save(data / (name + ".npy"), value)
    reader = {"profiles": {"toy": {"history_steps": 2, "forecast_steps": 2}},
              "operating_floor_mw": 0}
    sources = {alias: hashlib.sha256(alias.encode()).hexdigest() for alias in aliases}
    tasks = {"toy": {"targets": ["toy"], "input_channels": aliases}}
    write_json(data / "config.json", reader)
    write_json(data / "normalization.json", {"aliases": aliases, "center": [0, 1], "scale": [1, 2]})
    write_json(data / "tasks.json", tasks)
    write_json(data / "sources.json", [{"alias": alias, "sha256": sha} for alias, sha in sources.items()])
    write_json(data / "manifest.json", {"dataset_id": "synthetic_merged", "grid_rows": 8,
                                         "aliases": aliases, "channels": 2})
    write_json(data / "data_identity.json", {"original_receipt": "preserve"})
    np.savez_compressed(data / "origins_toy.npz", train=np.array([2, 3], dtype="<i8"))
    expected = {"dataset_id": "synthetic_merged", "grid_rows": 8, "aliases": aliases,
                "reader_config": reader, "source_sha256": sources, "tasks_sha256": json_hash(tasks),
                "npy_schema": {name + ".npy": {"shape": list(value.shape), "dtype": value.dtype.str}
                               for name, value in arrays.items()},
                "pack_output_hashes": {path.name: file_hash(path) for path in data.iterdir()
                                       if path.name not in {"manifest.json", "data_identity.json"}}}
    return data, expected


def test_complete_pack_passes_and_preserves_native_receipt(pack):
    data, expected = pack
    before = (data / "data_identity.json").read_bytes()
    result = verify_pack(data, expected)
    assert result["status"] == "PASS" and result["passed"] == result["total"]
    assert result["failed_keys"] == []
    assert json.loads((data / "merged_data_identity.json").read_text()) == result
    assert (data / "data_identity.json").read_bytes() == before


@pytest.mark.parametrize("filename", ["values.npy", "normalization.json", "origins_toy.npz"])
def test_matching_filenames_cannot_hide_changed_bytes(pack, filename):
    data, expected = pack
    path = data / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file:" + filename):
        verify_pack(data, expected)
    report = json.loads((data / "merged_data_identity.json").read_text())
    assert report["status"] == "FAIL" and "file:" + filename in report["failed_keys"]


@pytest.mark.parametrize("change", ["source_sha", "duplicate_source", "profile", "floor", "task", "aliases"])
def test_semantic_contract_rejects_rehashed_metadata(pack, change):
    data, expected = pack
    if change in {"source_sha", "duplicate_source"}:
        filename = "sources.json"
        value = json.loads((data / filename).read_text())
        if change == "source_sha":
            value[0]["sha256"] = "0" * 64
        else:
            value.append(value[0].copy())
    elif change in {"profile", "floor"}:
        filename = "config.json"
        value = json.loads((data / filename).read_text())
        if change == "profile":
            value["profiles"]["toy"]["forecast_steps"] += 1
        else:
            value["operating_floor_mw"] += 1
    elif change == "task":
        filename = "tasks.json"
        value = {"toy": {"targets": ["load"], "input_channels": ["load", "toy"]}}
    else:
        filename = "manifest.json"
        value = json.loads((data / filename).read_text())
        value["aliases"].reverse()
    write_json(data / filename, value)
    if filename in expected["pack_output_hashes"]:
        expected["pack_output_hashes"][filename] = file_hash(data / filename)
    with pytest.raises(ValueError):
        verify_pack(data, expected)
    assert json.loads((data / "merged_data_identity.json").read_text())["failed_keys"]


@pytest.mark.parametrize("kind", ["shape", "dtype", "object"])
def test_rehashed_unsafe_array_layout_is_rejected(pack, kind):
    data, expected = pack
    array = np.zeros((8, 2), dtype="<f4") if kind == "shape" else np.zeros((2, 8), dtype="<f8")
    if kind == "object":
        array = np.full((2, 8), "synthetic", dtype=object)
    np.save(data / "values.npy", array)
    expected["pack_output_hashes"]["values.npy"] = file_hash(data / "values.npy")
    with pytest.raises(ValueError, match="layout:values.npy"):
        verify_pack(data, expected)


def test_missing_declared_file_is_rejected(pack):
    data, expected = pack
    (data / "origins_toy.npz").unlink()
    with pytest.raises(ValueError, match="file:origins_toy.npz"):
        verify_pack(data, expected)


def test_declared_path_cannot_escape_pack(pack):
    data, expected = pack
    outside = data.parent / "outside.json"
    write_json(outside, {"synthetic": True})
    expected["pack_output_hashes"]["../outside.json"] = file_hash(outside)
    with pytest.raises(ValueError, match="file:../outside.json"):
        verify_pack(data, expected)


def test_expected_contract_requires_all_reader_files_and_layouts(pack):
    data, expected = pack
    changed = copy.deepcopy(expected)
    del changed["pack_output_hashes"]["values.npy"]
    with pytest.raises(ValueError, match="contract:reader_files"):
        verify_pack(data, changed)
    changed = copy.deepcopy(expected)
    del changed["npy_schema"]["values.npy"]
    with pytest.raises(ValueError, match="contract:layout_coverage"):
        verify_pack(data, changed)
