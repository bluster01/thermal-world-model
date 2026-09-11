"""Immutable matched-window manifests for the FMTS experiment.

No function in this module permits SPLIT_TEST.  Start indices identify the
first future sample and are validated again whenever a batch is reconstructed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import (
    SPLIT_TEST,
    CanonicalRecord,
    WindowBatch,
)
from src.final_wm.model import HistoryWindow


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_start_indices(
    record: CanonicalRecord,
    split_id: int,
    n_windows: int,
    history_steps: int,
    horizon: int,
    seed: int,
) -> torch.Tensor:
    if split_id == SPLIT_TEST:
        raise FinalWMProtocolError("locked test manifests require a separate authorization")
    if n_windows < 1 or history_steps < 1 or horizon < 1:
        raise FinalWMProtocolError("manifest dimensions must be positive")
    span = history_steps + horizon
    runs = [(start, end) for start, end in record.split_runs(split_id) if end - start >= span]
    if not runs:
        raise FinalWMProtocolError("no split run can contain the registered window")
    generator = torch.Generator().manual_seed(seed)
    starts: list[int] = []
    for _ in range(n_windows):
        run_start, run_end = runs[int(torch.randint(len(runs), (1,), generator=generator))]
        history_start = int(torch.randint(
            run_start, run_end - span + 1, (1,), generator=generator
        ))
        starts.append(history_start + history_steps)
    return torch.tensor(starts, dtype=torch.int64)


def windows_from_starts(
    record: CanonicalRecord,
    starts: torch.Tensor,
    split_id: int,
    history_steps: int,
    horizon: int,
) -> WindowBatch:
    if split_id == SPLIT_TEST:
        raise FinalWMProtocolError("locked test windows require a separate authorization")
    starts = torch.as_tensor(starts, dtype=torch.int64).reshape(-1)
    hist_idx = starts[:, None] + torch.arange(-history_steps, 0)[None, :]
    fut_idx = starts[:, None] + torch.arange(0, horizon)[None, :]
    if starts.numel() == 0 or int(hist_idx.min()) < 0 or int(fut_idx.max()) >= record.n:
        raise FinalWMProtocolError("manifest index is outside the canonical record")
    joined = torch.cat([hist_idx, fut_idx], dim=1)
    if not bool((record.split[joined] == split_id).all()):
        raise FinalWMProtocolError("manifest window crosses a split or gap")
    return WindowBatch(
        history=HistoryWindow(
            obs=record.obs[hist_idx],
            actions=record.actions[hist_idx],
            boundary=record.boundary[hist_idx],
        ),
        future_boundary=record.boundary[fut_idx],
        future_actions=record.actions[fut_idx],
        future_obs=record.obs[fut_idx],
        day_ids=torch.div(record.timestamps[starts], 86400, rounding_mode="floor"),
    )


def save_manifest_bundle(
    record: CanonicalRecord,
    out_dir: str | Path,
    *,
    train_split: int,
    validation_split: int,
    history_steps: int,
    horizon: int,
    train_windows: int = 20_000,
    validation_windows: int = 256,
    response_windows: int = 64,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train_starts": sample_start_indices(
            record, train_split, train_windows, history_steps, horizon, 15_000
        ).numpy(),
        "validation_starts": sample_start_indices(
            record, validation_split, validation_windows, history_steps, horizon, 30_000
        ).numpy(),
        "response_starts": sample_start_indices(
            record, validation_split, response_windows, history_steps, horizon, 60_000
        ).numpy(),
    }
    npz_path = out_dir / "window_starts.npz"
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "record_path": str(record.path),
        "record_sha256": sha256(record.path),
        "manifest_sha256": sha256(npz_path),
        "history_steps": history_steps,
        "horizon": horizon,
        "locked_test_included": False,
        "arrays": {name: {"count": int(len(value)), "seed": seed} for name, value, seed in (
            ("train_starts", arrays["train_starts"], 15_000),
            ("validation_starts", arrays["validation_starts"], 30_000),
            ("response_starts", arrays["response_starts"], 60_000),
        )},
    }
    (out_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
