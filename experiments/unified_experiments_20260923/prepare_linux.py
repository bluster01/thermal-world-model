"""Rebuild a private pack from byte-identified exports, or verify an existing pack.

The public recipe contains aliases and hashes, never source names or calendar
timestamps. Actual source names, dates, diagnostics and arrays stay in --data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NORMALIZATION_TYPES = {"center": "<f8", "scale": "<f8", "fit_count": "<i8",
                       "active_from_training_variation": "|b1"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def array_identity(array, dtype=None):
    a = np.ascontiguousarray(array, dtype=dtype)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def verify_pack(data, expected):
    """Strict identity: file bytes, ZIP member arrays, numeric scalers and roles."""
    checks = {}
    for name, wanted in expected["npy_sha256"].items():
        path = data / name
        checks["npy:" + name] = path.is_file() and file_hash(path) == wanted
    for name, members in expected["origins"].items():
        path = data / name
        checks["members:" + name] = False
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                checks["members:" + name] = set(archive.files) == set(members)
                for key, wanted in members.items():
                    checks["origin:" + name + ":" + key] = (
                        key in archive and array_identity(archive[key]) == wanted)
    normalization = read_json(data / "normalization.json")
    for key, dtype in NORMALIZATION_TYPES.items():
        checks["normalization:" + key] = array_identity(normalization[key], dtype) == expected["normalization"][key]
    checks["normalization:aliases"] = normalization["aliases"] == expected["aliases"]
    checks["tasks"] = json_hash(read_json(data / "tasks.json")) == expected["tasks_sha256"]
    reader_config = read_json(data / "config.json")
    for key, wanted in expected["reader_config"].items():
        checks["reader_config:" + key] = reader_config.get(key) == wanted
    manifest = read_json(data / "manifest.json")
    checks["dataset_id"] = manifest["dataset_id"] == expected["dataset_id"]
    checks["grid_rows"] = manifest["grid_rows"] == expected["grid_rows"]
    checks["channel_order"] = manifest["aliases"] == expected["aliases"]
    sources = read_json(data / "sources.json")
    checks["source_identity"] = {x["alias"]: x["sha256"] for x in sources} == expected["source_sha256"]
    report = {"schema_version": 1, "dataset_id": expected["dataset_id"],
              "status": "PASS" if all(checks.values()) else "FAIL",
              "strict_numeric_identity": True, "float_tolerance": 0,
              "checks": checks, "passed": sum(checks.values()), "total": len(checks)}
    write_json(data / "data_identity.json", report)
    if report["status"] != "PASS":
        raise ValueError("Data identity mismatch; inspect private data_identity.json")
    return report


def find_sources(raw_root, channels):
    """Filename SHA selects candidates; full CSV byte SHA resolves duplicates."""
    wanted = {c["filename_sha256"] for c in channels}
    candidates = {}
    for path in raw_root.rglob("*.csv"):
        key = hashlib.sha256(path.name.encode("utf-8")).hexdigest()
        if key in wanted:
            candidates.setdefault(key, []).append(path.resolve())
    resolved, hash_cache = {}, {}
    for c in channels:
        matches = []
        for path in sorted(set(candidates.get(c["filename_sha256"], []))):
            if path.stat().st_size != c["size_bytes"]:
                continue
            if path not in hash_cache:
                hash_cache[path] = file_hash(path)
            if hash_cache[path] == c["raw_sha256"]:
                matches.append(path)
        if not matches:
            raise ValueError("Missing byte-verified source for alias " + c["alias"])
        resolved[c["alias"]] = matches[0]
    return resolved


def calendar_from_load(load_path, recipe):
    """Infer only the UTC day from the byte-verified first load record."""
    row = pd.read_csv(load_path, usecols=["time"], nrows=1)
    day = pd.to_datetime(row["time"], utc=True, errors="raise", format="mixed").iloc[0].floor("D").value
    start = day + recipe["grid_start_offset_seconds"] * 10**9
    offsets = recipe["split_offsets_ns_from_grid_start"]
    return {"start_utc": pd.Timestamp(start, tz="UTC").isoformat(),
            "end_utc": pd.Timestamp(start + (recipe["grid_rows"] - 1) * recipe["config"]["grid_seconds"] * 10**9, tz="UTC").isoformat(),
            "cut1": pd.Timestamp(start + offsets[0], tz="UTC").isoformat(),
            "cut2": pd.Timestamp(start + offsets[1], tz="UTC").isoformat()}


def make_private_config(recipe, paths, data, point_mapping=None):
    cfg = copy.deepcopy(recipe["config"])
    cfg.update(calendar_from_load(paths["load"], recipe))
    # Sources can reside in several export directories; a private link view gives
    # the frozen preparer its single-directory interface without copying CSVs.
    parents = {p.parent for p in paths.values()}
    source_root = next(iter(parents)) if len(parents) == 1 else data / "private_sources"
    if len(parents) != 1:
        source_root.mkdir()
        for p in paths.values():
            (source_root / p.name).symlink_to(p)
    cfg["source_root"] = str(source_root)
    cfg["channels"] = []
    for channel in recipe["channels"]:
        meta = {k: v for k, v in channel.items() if k not in ("filename_sha256", "raw_sha256", "size_bytes")}
        meta["name"] = paths[channel["alias"]].stem
        cfg["channels"].append(meta)
    if point_mapping is None:
        point_mapping = data / "unresolved_point_mapping.csv"
        pd.DataFrame(columns=["点位中文名称", "点位编码"]).to_csv(point_mapping, index=False, encoding="utf-8-sig")
        write_json(data / "source_alias_map.json", {
            "identity_verified": False, "aliases_are_instrument_codes": False,
            "status": "source_bytes_verified_instrument_codes_unresolved",
            "sources": [{"filename": p.name, "opaque_source_alias": a} for a, p in paths.items()]})
    cfg["point_mapping"] = str(point_mapping.resolve())
    return cfg


def require_private_output(data):
    for parent in (data, *data.parents):
        if (parent / ".git").exists():
            raise ValueError("Private data output must be outside every Git checkout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--point-mapping", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    expected = read_json(HERE / "expected_data.json")
    if args.verify_only:
        report = verify_pack(data, expected)
    else:
        if platform.system() != "Linux":
            raise RuntimeError("Formal reconstruction requires Linux")
        if args.raw_root is None or not args.raw_root.is_dir():
            raise ValueError("Provide an existing private --raw-root")
        require_private_output(data)
        if data.exists():
            raise FileExistsError("Use a new private --data directory")
        recipe = read_json(HERE / "data_recipe.json")
        for name, wanted in expected["preparation_code_sha256"].items():
            if file_hash(ROOT / "experiments/unified_data_20260923" / name) != wanted:
                raise ValueError("Frozen preparation source mismatch: " + name)
        paths = find_sources(args.raw_root.resolve(), recipe["channels"])
        data.mkdir(parents=True)
        cfg = make_private_config(recipe, paths, data, args.point_mapping)
        config_path = data / "private_rebuild_config.json"
        write_json(config_path, cfg)
        with (data / "private_prepare.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-m", "experiments.unified_data_20260923.prepare",
                                     "--config", str(config_path), "--out", str(data)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError("Preparation failed; inspect private_prepare.log")
        report = verify_pack(data, expected)
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "total": report["total"]}))


if __name__ == "__main__":
    main()
