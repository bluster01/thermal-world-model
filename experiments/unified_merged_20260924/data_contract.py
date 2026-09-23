"""Exact, independent identity contract for the accepted Linux merged pack.

This validates declared file bytes plus the layouts/roles consumed by the
reader. It does not reuse the native-export verifier or rewrite its receipt.
Only the private ``merged_data_identity.json`` verification receipt is written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Directory policy is shared; the native pack verifier is never called here.
from experiments.unified_experiments_20260923.prepare_linux import require_private_output


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _inside_file(data, name):
    """Expected pack members are flat filenames; links cannot escape the pack."""
    if not isinstance(name, str) or not name or name in {".", ".."}:
        return None
    if "/" in name or "\\" in name or ":" in name:
        return None
    path = data / name
    return path if path.resolve().parent == data and path.is_file() else None


def verify_pack(data, expected):
    """Return PASS or save failed check names and raise ValueError.

    Expected fields: dataset_id, grid_rows, aliases, pack_output_hashes,
    npy_schema (filename -> dtype/shape), reader_config, source_sha256,
    tasks_sha256 (canonical sorted JSON). All declared files are required.
    """
    data = Path(data).resolve()
    if not data.is_dir():
        raise ValueError("Merged data directory does not exist")
    checks = {}
    hashes = expected.get("pack_output_hashes", {})
    layouts = expected.get("npy_schema", {})
    checks["contract:declared_files"] = isinstance(hashes, dict) and bool(hashes)
    if not isinstance(hashes, dict):
        hashes = {}
    checks["contract:npy_schema"] = isinstance(layouts, dict) and bool(layouts)
    if not isinstance(layouts, dict):
        layouts = {}
    required = {"values.npy", "valid.npy", "observed.npy", "flags.npy",
                "source_time_ns.npy", "time_ns.npy", "split.npy",
                "config.json", "normalization.json", "tasks.json", "sources.json"}
    checks["contract:reader_files"] = required <= set(hashes)
    checks["contract:layout_coverage"] = set(layouts) == {name for name in hashes if name.endswith(".npy")}
    for name, wanted in hashes.items():
        try:
            path = _inside_file(data, name)
            checks["file:" + str(name)] = path is not None and file_hash(path) == wanted
        except (OSError, ValueError):
            checks["file:" + str(name)] = False
    for name, wanted in layouts.items():
        key = "layout:" + str(name)
        try:
            path = _inside_file(data, name)
            if path is None:
                checks[key] = False
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            checks[key] = (not array.dtype.hasobject and list(array.shape) == wanted["shape"]
                           and array.dtype.str == np.dtype(wanted["dtype"]).str)
            del array
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            checks[key] = False
    try:
        manifest_path = _inside_file(data, "manifest.json")
        manifest = _read(manifest_path) if manifest_path else {}
        checks["manifest:dataset_id"] = manifest.get("dataset_id") == expected["dataset_id"]
        checks["manifest:grid_rows"] = manifest.get("grid_rows") == expected["grid_rows"]
        checks["manifest:aliases"] = manifest.get("aliases") == expected["aliases"]
        checks["manifest:channels"] = manifest.get("channels") == len(expected["aliases"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        checks["manifest:read"] = False
    for filename, label in (("config.json", "reader_config"),
                            ("normalization.json", "normalization"),
                            ("tasks.json", "tasks"), ("sources.json", "sources")):
        try:
            path = _inside_file(data, filename)
            if path is None:
                checks[label + ":read"] = False
                continue
            value = _read(path)
            if label == "reader_config":
                fields = expected["reader_config"]
                checks["reader_config:declared_fields"] = {"profiles", "operating_floor_mw"} <= set(fields)
                for key, wanted in fields.items():
                    checks["reader_config:" + key] = value.get(key) == wanted
            elif label == "normalization":
                count = len(expected["aliases"])
                checks["normalization:aliases"] = value["aliases"] == expected["aliases"]
                for key in ("center", "scale"):
                    array = np.asarray(value[key], dtype=np.float64)
                    valid = array.shape == (count,) and np.isfinite(array).all()
                    checks["normalization:" + key] = bool(valid and (key != "scale" or (array > 0).all()))
            elif label == "tasks":
                checks["tasks:roles"] = json_hash(value) == expected["tasks_sha256"]
            else:
                actual = {entry["alias"]: entry["sha256"] for entry in value}
                checks["sources:unique_aliases"] = len(value) == len(actual) == len(expected["aliases"])
                checks["sources:identity"] = actual == expected["source_sha256"]
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            checks[label + ":read"] = False
    failed = [key for key, passed in checks.items() if not passed]
    report = {"schema_version": 1, "dataset_id": expected.get("dataset_id"),
              "source_contract": "accepted_linux_merged_pack_exact_bytes",
              "status": "FAIL" if failed else "PASS", "checks": checks,
              "failed_keys": failed, "passed": sum(bool(x) for x in checks.values()),
              "total": len(checks), "float_tolerance": 0}
    _write(data / "merged_data_identity.json", report)
    if failed:
        raise ValueError("Merged pack identity failed: " + ", ".join(failed))
    return report
