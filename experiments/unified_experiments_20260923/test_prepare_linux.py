"""Synthetic-only reconstruction/identity tests; no industrial data is opened.

Run on Linux: python -m pytest experiments/unified_experiments_20260923/test_prepare_linux.py -q
The formal-platform guard is mocked only where a tiny synthetic CLI rebuild is
exercised. This file does not authorize or perform an industrial reconstruction.
"""
from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.unified_experiments_20260923 import prepare_linux as prep


@pytest.fixture(scope="module")
def synthetic_pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_recipe")
    raw = root / "raw"
    raw.mkdir()
    timestamps = [pd.Timestamp("2000-01-01T00:00:00Z")] + list(
        pd.date_range("2000-01-01T00:01:00Z", periods=90, freq="10s"))
    channels = []
    for alias, offset in [("load", 10), ("toy_signal", 1)]:
        path = raw / (alias + ".csv")
        pd.DataFrame({"time": [t.isoformat() for t in timestamps],
                      "valueFloat64": offset + np.arange(91) / 100}).to_csv(path, index=False)
        channels.append({"alias": alias, "role": "synthetic_fixture", "raw_unit": "arbitrary",
                         "minimum": 0, "maximum": None, "statistical_high_rule": False,
                         "filename_sha256": hashlib.sha256(path.name.encode("utf-8")).hexdigest(),
                         "raw_sha256": prep.file_hash(path), "size_bytes": path.stat().st_size})
    recipe = {"schema_version": 1, "grid_start_offset_seconds": 60, "grid_rows": 90,
              "split_offsets_ns_from_grid_start": [300_000_000_000, 600_000_000_000],
              "channels": channels, "config": {
                  "dataset_id": "synthetic_recipe_test", "grid_seconds": 10,
                  "max_age_seconds": 30, "origin_stride_steps": 1,
                  "split_embargo_seconds": 0, "operating_floor_mw": 0,
                  "history_target_valid_fraction": 1, "train_label_valid_fraction": 1,
                  "profiles": {"small": {"history_steps": 4, "forecast_steps": 3}},
                  "historical_exposure": {"toy": "synthetic only"},
                  "tasks": {"toy": {"targets": ["toy_signal"],
                                     "input_channels": ["load", "toy_signal"],
                                     "measured_action_channels": [], "boundary_channels": ["load"],
                                     "future_inputs_default": "none", "routing": {}}}}}
    out = root / "pack"
    out.mkdir()
    cfg = prep.make_private_config(recipe, prep.find_sources(raw, channels), out)
    prep.write_json(out / "private_rebuild_config.json", cfg)
    result = subprocess.run(
        [sys.executable, "-m", "experiments.unified_data_20260923.prepare", "--config",
         str(out / "private_rebuild_config.json"), "--out", str(out)],
        cwd=prep.ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    norm = prep.read_json(out / "normalization.json")
    sources = prep.read_json(out / "sources.json")
    expected = {"dataset_id": "synthetic_recipe_test", "grid_rows": 90,
                "aliases": ["load", "toy_signal"],
                "reader_config": {k: cfg[k] for k in ("profiles", "operating_floor_mw")},
                "npy_sha256": {p.name: prep.file_hash(p) for p in out.glob("*.npy")},
                "origins": {}, "normalization": {
                    k: prep.array_identity(norm[k], dtype) for k, dtype in prep.NORMALIZATION_TYPES.items()},
                "tasks_sha256": prep.json_hash(prep.read_json(out / "tasks.json")),
                "source_sha256": {s["alias"]: s["sha256"] for s in sources},
                "preparation_code_sha256": {
                    name: prep.file_hash(prep.ROOT / "experiments/unified_data_20260923" / name)
                    for name in ("prepare.py", "dataset.py")}}
    for path in out.glob("origins_*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            expected["origins"][path.name] = {k: prep.array_identity(archive[k]) for k in archive.files}
    spec_dir = root / "synthetic_spec"
    spec_dir.mkdir()
    prep.write_json(spec_dir / "data_recipe.json", recipe)
    prep.write_json(spec_dir / "expected_data.json", expected)
    return {"raw": raw, "pack": out, "recipe": recipe, "expected": expected,
            "spec": spec_dir, "cfg": cfg}


def test_calendar_and_values_are_reconstructed(synthetic_pack):
    f = synthetic_pack
    cfg = f["cfg"]
    start = pd.Timestamp("2000-01-01T00:01:00Z").value
    assert pd.Timestamp(cfg["start_utc"]).value == start
    assert pd.Timestamp(cfg["end_utc"]).value == start + 890_000_000_000
    assert pd.Timestamp(cfg["cut1"]).value == start + 300_000_000_000
    assert pd.Timestamp(cfg["cut2"]).value == start + 600_000_000_000
    np.testing.assert_array_equal(np.load(f["pack"] / "time_ns.npy"),
                                  start + np.arange(90) * 10_000_000_000)
    np.testing.assert_allclose(np.load(f["pack"] / "values.npy")[1],
                               1 + np.arange(1, 91) / 100, rtol=1e-7)
    np.testing.assert_array_equal(np.load(f["pack"] / "split.npy"), np.repeat([0, 1, 2], 30))
    report = prep.verify_pack(f["pack"], f["expected"])
    assert report["status"] == "PASS" and report["passed"] == report["total"]


def test_opaque_aliases_are_never_instrument_codes(synthetic_pack):
    out = synthetic_pack["pack"]
    mapping = prep.read_json(out / "source_alias_map.json")
    assert mapping["identity_verified"] is False
    assert mapping["aliases_are_instrument_codes"] is False
    assert {x["opaque_source_alias"] for x in mapping["sources"]} == {"load", "toy_signal"}
    assert pd.read_csv(out / "unresolved_point_mapping.csv").empty
    assert all(not x["code_verified"] and not x["code_candidates"]
               for x in prep.read_json(out / "sources.json"))


def test_source_matching_requires_name_and_bytes(synthetic_pack, tmp_path):
    root = tmp_path / "exports"
    valid = root / "valid"
    shutil.copytree(synthetic_pack["raw"], valid)
    decoy = root / "decoy"
    decoy.mkdir()
    source = valid / "load.csv"
    changed = source.read_bytes().replace(b"10.0", b"90.0", 1)
    assert len(changed) == source.stat().st_size
    (decoy / source.name).write_bytes(changed)
    channels = synthetic_pack["recipe"]["channels"]
    assert prep.find_sources(root, channels)["load"] == source.resolve()
    source.write_bytes(changed)
    with pytest.raises(ValueError, match="Missing byte-verified source"):
        prep.find_sources(root, channels)


@pytest.mark.parametrize("kind", ["values", "normalization", "tasks", "sources", "profile", "floor"])
def test_tampering_fails_identity(synthetic_pack, tmp_path, kind):
    out = tmp_path / "pack"
    shutil.copytree(synthetic_pack["pack"], out)
    if kind == "values":
        values = np.load(out / "values.npy")
        values[0, 0] += 1
        np.save(out / "values.npy", values)
    elif kind == "normalization":
        obj = prep.read_json(out / "normalization.json")
        obj["center"][0] += 1
        prep.write_json(out / "normalization.json", obj)
    elif kind == "tasks":
        obj = prep.read_json(out / "tasks.json")
        obj["toy"]["targets"] = ["load"]
        prep.write_json(out / "tasks.json", obj)
    elif kind == "sources":
        obj = prep.read_json(out / "sources.json")
        obj[0]["sha256"] = "0" * 64
        prep.write_json(out / "sources.json", obj)
    else:
        obj = prep.read_json(out / "config.json")
        if kind == "profile":
            obj["profiles"]["small"]["forecast_steps"] += 1
        else:
            obj["operating_floor_mw"] += 1
        prep.write_json(out / "config.json", obj)
    with pytest.raises(ValueError, match="Data identity mismatch"):
        prep.verify_pack(out, synthetic_pack["expected"])
    assert prep.read_json(out / "data_identity.json")["status"] == "FAIL"


def test_origin_identity_ignores_zip_encoding_but_rejects_changed_members(synthetic_pack, tmp_path):
    out = tmp_path / "pack"
    shutil.copytree(synthetic_pack["pack"], out)
    path = next(out.glob("origins_*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    np.savez(path, **arrays)  # Original preparer uses compressed ZIP members.
    assert prep.verify_pack(out, synthetic_pack["expected"])["status"] == "PASS"
    arrays["train"][0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Data identity mismatch"):
        prep.verify_pack(out, synthetic_pack["expected"])


def test_private_output_rejects_git_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside every Git checkout"):
        prep.require_private_output(tmp_path / "nested" / "new_pack")


def test_synthetic_cli_rebuild_and_verify_only(synthetic_pack, tmp_path, monkeypatch, capsys):
    out = tmp_path / "rebuilt_pack"
    monkeypatch.setattr(prep, "HERE", synthetic_pack["spec"])
    monkeypatch.setattr(prep.platform, "system", lambda: "Linux")
    args = ["prepare_linux", "--raw-root", str(synthetic_pack["raw"]), "--data", str(out)]
    monkeypatch.setattr(sys, "argv", args)
    prep.main()
    assert '"status": "PASS"' in capsys.readouterr().out
    with pytest.raises(FileExistsError, match="new private"):
        prep.main()
    monkeypatch.setattr(prep.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sys, "argv", ["prepare_linux", "--verify-only", "--data", str(out)])
    prep.main()
    assert '"status": "PASS"' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["prepare_linux", "--data", str(tmp_path / "unused")])
    with pytest.raises(RuntimeError, match="requires Linux"):
        prep.main()


def test_frozen_preparation_source_mismatch_stops_before_rebuild(synthetic_pack, tmp_path, monkeypatch):
    spec = tmp_path / "spec"
    shutil.copytree(synthetic_pack["spec"], spec)
    expected = copy.deepcopy(synthetic_pack["expected"])
    expected["preparation_code_sha256"]["prepare.py"] = "0" * 64
    prep.write_json(spec / "expected_data.json", expected)
    monkeypatch.setattr(prep, "HERE", spec)
    monkeypatch.setattr(prep.platform, "system", lambda: "Linux")
    out = tmp_path / "unused_pack"
    monkeypatch.setattr(sys, "argv", ["prepare_linux", "--raw-root", str(synthetic_pack["raw"]),
                                     "--data", str(out)])
    with pytest.raises(ValueError, match="Frozen preparation source mismatch"):
        prep.main()
    assert not out.exists()
