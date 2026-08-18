"""D0 pipeline contracts: discovery, mapping validation, gates, windows."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BOUNDARY_ELEMENTS,
    OBSERVATION_ELEMENTS,
    FinalWMProtocolError,
)
from src.final_wm.data import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    CanonicalRecord,
    QualityGateConfig,
    build_canonical,
    discover_dataset,
    load_channel_mapping,
    sample_windows,
)
from src.final_wm.synthetic import synthetic_canonical_arrays


def _write_synthetic_csvs(tmp_path, n: int = 600, stuck: bool = False):
    import pandas as pd

    arrays = synthetic_canonical_arrays(total_steps=n, seed=0)
    times = np.datetime64("2024-01-01") + np.arange(n) * np.timedelta64(10, "s")
    boundary = {n_: arrays["boundary"][:, i] for i, n_ in enumerate(BOUNDARY_ELEMENTS)}
    frame_b = {"time": times, **boundary}
    frame_a = {"time": times, **{n_: arrays["actions"][:, i] * 100.0 for i, n_ in enumerate(ACTION_ELEMENTS)}}
    obs = {n_: arrays["obs"][:, i] for i, n_ in enumerate(OBSERVATION_ELEMENTS)}
    if stuck:
        obs["sh2_outlet_temp"] = np.full(n, 500.0)
    frame_o = {"time": times, **obs}
    import pandas as pd  # noqa: F811

    pd.DataFrame(frame_b).to_csv(tmp_path / "boundary.csv", index=False)
    pd.DataFrame(frame_a).to_csv(tmp_path / "actions.csv", index=False)
    pd.DataFrame(frame_o).to_csv(tmp_path / "temps.csv", index=False)
    mapping = {
        "time_columns": {"boundary.csv": "time", "actions.csv": "time", "temps.csv": "time"},
        "channels": {},
    }
    for name in BOUNDARY_ELEMENTS:
        mapping["channels"][name] = {"file": "boundary.csv", "column": name}
    for name in ACTION_ELEMENTS:
        mapping["channels"][name] = {"file": "actions.csv", "column": name, "unit_scale": 0.01}
    for name in OBSERVATION_ELEMENTS:
        mapping["channels"][name] = {"file": "temps.csv", "column": name}
    (tmp_path / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    return tmp_path / "mapping.json"


def test_discover_dataset_reports_schema(tmp_path) -> None:
    _write_synthetic_csvs(tmp_path)
    report = discover_dataset(tmp_path)
    assert len(report["files"]) == 3
    entry = next(e for e in report["files"] if e["file"] == "boundary.csv")
    assert "steam_flow" in entry["columns"]
    assert entry["numeric_stats"]["steam_flow"]["min"] > 200.0


def test_mapping_validation_fail_closed(tmp_path) -> None:
    mapping_path = _write_synthetic_csvs(tmp_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    del mapping["channels"]["steam_flow"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(FinalWMProtocolError):
        load_channel_mapping(bad)


def test_build_canonical_and_window_sampling(tmp_path) -> None:
    mapping_path = _write_synthetic_csvs(tmp_path)
    gates = QualityGateConfig(min_days=0.0, min_valve_active_ratio=0.0)
    out = tmp_path / "canonical.npz"
    report = build_canonical(tmp_path, mapping_path, out, gates=gates)
    assert report.gap_ratio <= gates.max_gap_ratio
    record = CanonicalRecord(out)
    assert record.boundary.shape[1] == len(BOUNDARY_ELEMENTS)
    assert (record.split == SPLIT_TRAIN).sum() > (record.split == SPLIT_VAL).sum() > 0
    gen = torch.Generator().manual_seed(0)
    batch = sample_windows(record, SPLIT_TRAIN, 8, history_steps=16, horizon=12, generator=gen)
    assert batch.history.obs.shape == (8, 16, 5)
    assert batch.future_obs.shape == (8, 12, 5)
    with pytest.raises(FinalWMProtocolError):
        record.split_runs(SPLIT_TEST)
    meta = json.loads((tmp_path / "canonical_meta.json").read_text(encoding="utf-8"))
    assert meta["test_locked"] is True


def test_quality_gate_rejects_stuck_channel(tmp_path) -> None:
    mapping_path = _write_synthetic_csvs(tmp_path, stuck=True)
    gates = QualityGateConfig(min_days=0.0, min_valve_active_ratio=0.0)
    with pytest.raises(FinalWMProtocolError):
        build_canonical(tmp_path, mapping_path, tmp_path / "out.npz", gates=gates)


def test_dual_canonical_bridge_cross_maps_valves(tmp_path) -> None:
    from src.final_wm.data import import_dual_canonical

    arrays = synthetic_canonical_arrays(total_steps=1500, seed=7)
    n = 1500
    action4 = np.stack(
        [arrays["actions"][:, 0] + 10.0, arrays["actions"][:, 1] + 10.0,  # A-labeled valves
         arrays["actions"][:, 0], arrays["actions"][:, 1]],               # B-labeled valves
        axis=1,
    ).astype(np.float32)
    dual = {
        "date_ns": arrays["timestamps"].astype(np.int64) * 10**9,
        "boundary": arrays["boundary"],
        "action": action4,
        "obsA": arrays["obs"],
        "obsB": arrays["obs"] + 1.0,
        "split_train": np.array([int(n * 0.8)], dtype=np.int64),
    }
    dual_path = tmp_path / "canonical_record.npz"
    np.savez_compressed(dual_path, **dual)
    gates = QualityGateConfig(min_days=0.0, min_valve_active_ratio=0.0)
    written = import_dual_canonical(dual_path, tmp_path / "out", gates=gates)
    assert set(written) == {"A", "B"}
    rec_a = CanonicalRecord(written["A"])
    rec_b = CanonicalRecord(written["B"])
    # side A is driven by the B-labeled valves (columns 2,3 of the dual record)
    assert torch.allclose(rec_a.actions, torch.from_numpy(action4[:, 2:4]), atol=1e-4)
    assert torch.allclose(rec_b.actions, torch.from_numpy(action4[:, 0:2]), atol=1e-4)
    assert rec_a.obs.shape[1] == 5 and rec_b.obs.shape[1] == 5
    assert (rec_a.split == SPLIT_TEST).sum() > 0  # locked test reserve exists
    meta = json.loads((tmp_path / "out" / "canonical_sideA_meta.json").read_text(encoding="utf-8"))
    assert meta["provenance"]["cross_side_valves"] == [2, 3]
    assert meta["test_locked"] is True
