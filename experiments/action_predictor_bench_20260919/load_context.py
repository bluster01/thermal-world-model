"""Extract report-window electric load for post-hoc dynamic diagnostics only.

This sidecar never changes a training pack or model input. Future load is used
to describe the recorded operating regime, not as an available forecast input.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SERIES = {
    "load_MW": "机组负荷_GENERATOR_POWER",
    "agc_raw": "AGC指令",
    "load_rate_raw": "机组负荷变化率",
}
AUX_KEYS = ("load", "agc", "load_rate")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class HashingReader:
    """Hash the CSV bytes in the same pass used by pandas' streaming parser."""
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
        return block


def extract(merged, needed, chunksize):
    frames = []
    scanned = 0
    with merged.open("rb") as stream:
        source = HashingReader(stream)
        for chunk in pd.read_csv(
            source, usecols=["time", *SERIES.values()],
            dtype={column: np.float32 for column in SERIES.values()},
            chunksize=chunksize, encoding="utf-8-sig",
        ):
            dates = pd.to_datetime(chunk["time"], utc=True, errors="raise")
            if dates.isna().any():
                raise ValueError("Source has unparsed timestamps")
            epochs = (dates.astype("int64") // 10**9).to_numpy()
            mask = np.isin(epochs, needed)
            if mask.any():
                matched = chunk.loc[mask, list(SERIES.values())].copy()
                matched.index = epochs[mask]
                frames.append(matched)
            scanned += len(chunk)
            print(f"scanned={scanned} matched={sum(len(x) for x in frames)}", flush=True)
        if source.bytes_read != merged.stat().st_size:
            raise RuntimeError("CSV parser did not consume the entire source")
        digest = source.digest.hexdigest()
    if not frames:
        raise ValueError("No requested timestamps found")
    frame = pd.concat(frames).sort_index()
    if frame.index.duplicated().any():
        raise ValueError("Duplicate source timestamps in requested windows")
    return frame, digest, scanned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "results/action_predictor_bench_20260919/focused33_seed11/evaluation_inputs.npz")
    parser.add_argument("--aux", type=Path, default=HERE / "data/hist_bypass_A_33pct_v1.npz")
    parser.add_argument("--merged", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/action_predictor_bench_20260919/ssm_dynamic_20260922")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    started = time.perf_counter()
    destination = args.output / "load_context.npz"
    metadata_path = args.output / "load_context_metadata.json"
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Existing load context preserved; choose another --output")
    aux_meta = json.loads(args.aux.with_suffix(".json").read_text(encoding="utf-8"))
    merged = args.merged or Path(aux_meta["source_merged_table"]["path"])
    with np.load(args.evaluation) as evaluation:
        times = evaluation["times"].copy()
        steps = evaluation["bank"].shape[1]
    epochs = times[:, None] - 630 + 10 * np.arange(steps)[None, :]
    needed = np.unique(epochs)
    frame, source_hash, scanned = extract(merged, needed, args.chunksize)
    if source_hash != aux_meta["source_merged_table"]["sha256"]:
        raise ValueError("Merged CSV identity differs from the bypass-pack source")
    values = frame.reindex(epochs.ravel()).to_numpy(np.float32).reshape(len(times), steps, 3)
    found = np.isin(epochs, frame.index.to_numpy())
    valid = np.isfinite(values)
    with np.load(args.aux) as aux:
        if not np.array_equal(times, aux["evaluation_time"]):
            raise ValueError("Evaluation origins differ from bypass pack")
        names = aux["channel_names"].tolist()
        expected = aux["hist12_evaluation"][:, :, [names.index(key) for key in AUX_KEYS]]
    if not np.array_equal(values[:, :64], expected, equal_nan=True):
        raise ValueError("Extracted histories differ from original bypass values")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, times=times, epochs=epochs, found_time=found, valid=valid,
        channel_names=np.array(list(SERIES)),
        **{key: values[:, :, i] for i, key in enumerate(SERIES)},
    )
    metadata = {
        "purpose": "Post-hoc report-set dynamic-regime stratification; never a model or training input",
        "produced_by": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(__file__),
        "source": {"path": str(merged), "sha256": source_hash, "bytes": merged.stat().st_size, "rows_scanned": scanned},
        "evaluation": {"path": str(args.evaluation), "sha256": sha256(args.evaluation)},
        "auxiliary": {"path": str(args.aux), "sha256": sha256(args.aux)},
        "output_sha256": sha256(destination),
        "columns": SERIES,
        "units": {"load_MW": "MW (canonical-v2 project variable mapping)", "agc_raw": "source raw; MW-like but source engineering unit not independently verified", "load_rate_raw": "source raw discrete control signal; do not interpret as observed MW/min"},
        "alignment": "times = last history epoch; epochs[n,k] = times[n] - 630 + 10*k; first64 history and next512 future; source parser uses utc=True exactly as bypass builder, without claiming source clock timezone",
        "shape": list(values.shape), "history_steps": 64, "future_steps": steps - 64, "dt_seconds": 10,
        "unique_requested_times": int(len(needed)),
        "missing_time_cells": int((~found).sum()),
        "missing_value_cells": {key: int((~valid[:, :, i]).sum()) for i, key in enumerate(SERIES)},
        "history_matches_aux_exactly": True,
        "history_max_abs_difference": 0.0,
        "ranges": {key: [float(np.nanmin(values[:, :, i])), float(np.nanmax(values[:, :, i]))] for i, key in enumerate(SERIES)},
        "seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
