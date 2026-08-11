#!/usr/bin/env python3
"""Build frozen A-valve→right and B-valve→left MS3 caches from all_merged_10s."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.phase35.data import Phase35Cache, save_cache, utc_timestamps_to_ns
from src.phase35.schema import MS3_HISTORY_FEATURES


DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms3_real_adaptation_matrix.json"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_matrix(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    contract = matrix["data_contract"]
    if tuple(contract["history_features"]) != tuple(MS3_HISTORY_FEATURES):
        raise ValueError("MS3 matrix history features differ from the code contract")
    if set(contract["side_mappings"]) != {"A", "B"}:
        raise ValueError("MS3 matrix must define both cross-side mappings")
    return matrix


def _write_manifest(path: Path, payload: dict) -> None:
    target = path.with_suffix(".manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-a", required=True)
    parser.add_argument("--output-b", required=True)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = _load_matrix(matrix_path)
    contract = matrix["data_contract"]
    source = Path(args.input).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    observed_sha = sha256_file(source)
    if source.stat().st_size != int(contract["source_size_bytes"]):
        raise ValueError("MS3 source size differs from the frozen data contract")
    if observed_sha != contract["source_sha256"]:
        raise ValueError("MS3 source SHA256 differs from the frozen data contract")

    timestamp_column = contract["timestamp_column"]
    source_columns = {timestamp_column}
    for side_mapping in contract["side_mappings"].values():
        source_columns.update(side_mapping["column_map"].values())
    timestamp_parts: list[np.ndarray] = []
    side_parts: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    row_count = 0
    for chunk in pd.read_csv(
        source,
        usecols=list(source_columns),
        chunksize=args.chunksize,
        low_memory=False,
    ):
        timestamp = utc_timestamps_to_ns(chunk[timestamp_column])
        if np.any(timestamp == np.iinfo(np.int64).min):
            raise ValueError("MS3 source contains invalid timestamps")
        timestamp_parts.append(timestamp)
        for side in ("A", "B"):
            mapping = contract["side_mappings"][side]["column_map"]
            columns = []
            for generic in contract["history_features"]:
                values = pd.to_numeric(
                    chunk[mapping[generic]], errors="coerce"
                ).to_numpy(dtype=np.float32)
                columns.append(values)
            side_parts[side].append(np.stack(columns, axis=1))
        row_count += len(chunk)
        print(json.dumps({"rows_scanned": row_count}), flush=True)

    timestamps = np.concatenate(timestamp_parts)
    differences = np.diff(timestamps)
    if len(timestamps) < 3 or np.any(differences <= 0):
        raise ValueError("MS3 source timestamps must be strictly increasing")
    expected_step_ns = int(contract["step_seconds"] * 1_000_000_000)
    irregular = differences != expected_step_ns
    observed_time_contract = {
        "source_rows": int(row_count),
        "grid_start_ns": int(timestamps[0]),
        "grid_end_ns": int(timestamps[-1]),
        "irregular_transition_count": int(irregular.sum()),
        "max_transition_seconds": float(differences.max() / 1e9),
    }
    expected_time_contract = {
        key: contract[key] for key in observed_time_contract
    }
    if observed_time_contract != expected_time_contract:
        raise ValueError(
            "MS3 source timeline differs from the frozen nanosecond contract: "
            f"observed={observed_time_contract} expected={expected_time_contract}"
        )
    matrix_sha = sha256_file(matrix_path)
    for side, output_value in (("A", args.output_a), ("B", args.output_b)):
        output = Path(output_value).resolve()
        values = np.concatenate(side_parts[side], axis=0).astype(np.float32)
        ages = np.zeros_like(values, dtype=np.float32)
        quality = {}
        for index, column in enumerate(contract["history_features"]):
            finite = np.isfinite(values[:, index])
            quality[column] = {
                "finite_fraction": float(finite.mean()),
                "min": float(values[finite, index].min()) if finite.any() else None,
                "max": float(values[finite, index].max()) if finite.any() else None,
            }
        side_contract = contract["side_mappings"][side]
        metadata = {
            "protocol_version": matrix["protocol_version"],
            "side": side,
            "step_seconds": contract["step_seconds"],
            "timestamp_storage_unit": contract["timestamp_storage_unit"],
            "cross_pairing_frozen": True,
            "control_loop": side_contract["control_loop"],
            "column_map": side_contract["column_map"],
            "source": {
                "path": str(source),
                "size_bytes": int(source.stat().st_size),
                "sha256": observed_sha,
                "rows": int(row_count),
            },
            "matrix_sha256": matrix_sha,
            "grid_start_ns": observed_time_contract["grid_start_ns"],
            "grid_end_ns": observed_time_contract["grid_end_ns"],
            "grid_rows": observed_time_contract["source_rows"],
            "irregular_transition_count": observed_time_contract[
                "irregular_transition_count"
            ],
            "max_transition_seconds": observed_time_contract[
                "max_transition_seconds"
            ],
            "gap_policy": contract["gap_policy"],
            "reconstruction": "premerged_dense_10s_no_additional_fill",
            "age_semantics": "zero_means_dense_merged_row_not_original_tag_age",
            "quality": quality,
        }
        cache = Phase35Cache(
            timestamps_ns=timestamps,
            values=values,
            ages_s=ages,
            columns=tuple(contract["history_features"]),
            metadata=metadata,
        )
        save_cache(cache, output)
        _write_manifest(output, metadata)
        print(
            json.dumps(
                {"side": side, "cache": str(output), "rows": len(timestamps)},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
