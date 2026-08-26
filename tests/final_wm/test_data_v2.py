"""D0-v2 contracts: mapping validation, gates, builder identity, loader."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BOUNDARY_ELEMENTS,
    OBSERVATION_ELEMENTS,
    FinalWMProtocolError,
)
from src.final_wm.data_v2 import (
    AUX_ELEMENTS,
    BOUNDARY_EXT_ELEMENTS,
    BOUNDARY_FULL_ELEMENTS,
    N_MILLS,
    CanonicalV2Record,
    build_canonical_v2,
    load_channel_mapping_v2,
)


def _fixture(tmp_path, n: int = 400, time_offset_s: int = 0):
    """Write a synthetic all_merged CSV + v1 npz + test mapping."""
    import pandas as pd

    rng = np.random.default_rng(0)
    t_sec = np.arange(n) * 10
    times = np.datetime64("2024-01-01") + t_sec.astype("timedelta64[s]")

    steam_kgs = 350.0 + 50.0 * np.sin(np.arange(n) / 40.0)
    valve_v1a = 40.0 + 15.0 * np.sin(np.arange(n) / 22.0)
    valve_v1b = 45.0 + 15.0 * np.cos(np.arange(n) / 22.0)
    valve_v2a = 30.0 + 10.0 * np.sin(np.arange(n) / 17.0)
    valve_v2b = 35.0 + 10.0 * np.cos(np.arange(n) / 17.0)
    cols: dict[str, np.ndarray] = {
        "v1a_col": valve_v1a, "v1b_col": valve_v1b,
        "v2a_col": valve_v2a, "v2b_col": valve_v2b,
        "time": times,
        "fuel_col": 250.0 + rng.normal(0, 20, n).cumsum() * 0.01 + 10 * np.sin(np.arange(n) / 30.0),
        "o2_col": 4.0 + 0.5 * np.sin(np.arange(n) / 25.0),
        "air_col": 900.0 + 50.0 * np.sin(np.arange(n) / 35.0),
        "rh_gas_a": 700.0 + 30.0 * np.sin(np.arange(n) / 50.0),
        "rh_gas_b": 705.0 + 30.0 * np.cos(np.arange(n) / 50.0),
        "steam_col_th": steam_kgs * 3.6,  # t/h; alignment applies 1/3.6
        "sep_p_col": 17.0 + 0.3 * np.sin(np.arange(n) / 60.0),
        "att1_out_l": 500.0 + 5.0 * np.sin(np.arange(n) / 20.0),
    }
    for k in range(1, N_MILLS + 1):
        cols[f"feeder{k}"] = np.full(n, 5.0 if k <= 3 else 0.0)
        cols[f"mill_t{k}"] = 600.0 + 10.0 * k + np.sin(np.arange(n) / (15.0 + k))
        cols[f"mill_f{k}"] = np.where(np.arange(n) % 2 == 0, 10.0 if k == 1 else 0.0,
                                      10.0 if k == 1 else 0.0)
    frame = pd.DataFrame(cols)
    src = tmp_path / "all_merged.csv"
    frame.to_csv(src, index=False)

    boundary = np.zeros((n, len(BOUNDARY_ELEMENTS)), dtype=np.float32)
    boundary[:, 0] = steam_kgs  # steam_flow kg/s
    boundary[:, 2] = cols["sep_p_col"]  # separator_pressure MPa
    actions = np.full((n, len(ACTION_ELEMENTS)), 0.5, dtype=np.float32)
    actions[:, 0] = valve_v1a / 100.0          # side-A wiring: valve1 = stage-1 A
    actions[:, 1] = valve_v2b / 100.0          # side-A wiring: valve2 = stage-2 B
    obs = np.zeros((n, len(OBSERVATION_ELEMENTS)), dtype=np.float32)
    obs[:, 1] = cols["att1_out_l"]  # sh1_outlet_temp
    obs += np.arange(n, dtype=np.float32)[:, None] * 0.001  # de-stuck all channels
    for j in (1, 3, 4, 5, 6):
        boundary[:, j] += np.arange(n, dtype=np.float32) * 0.001
    split = np.zeros(n, dtype=np.int8)
    v1 = tmp_path / "canonical_sideX.npz"
    np.savez_compressed(
        v1, boundary=boundary, actions=actions, obs=obs,
        valid=np.ones(n, dtype=bool),
        timestamps=(t_sec + time_offset_s
                    + np.datetime64("2024-01-01").astype("datetime64[s]").astype(np.int64)),
        split=split,
    )

    mapping = {
        "version": 2,
        "source": {"file": "all_merged.csv", "time_column": "time"},
        "boundary_ext": {
            "fuel_corrected": {"column": "fuel_col", "range": [0.0, 600.0], "stuck_max": 0.05},
            "mill_count_on": {"derived": "mill_on_count", "range": [0.0, 8.0], "stuck_max": None},
            "mill_gas_temp_wavg": {"derived": "mill_gas_wavg", "range": [0.0, 900.0], "stuck_max": 0.05},
            "flue_o2": {"column": "o2_col", "range": [0.0, 15.0], "stuck_max": 0.05},
            "secondary_air_total": {"column": "air_col", "range": [0.0, 2000.0], "stuck_max": 0.05},
            "rh_gas_in_temp_a": {"column": "rh_gas_a", "range": [0.0, 900.0], "stuck_max": 0.05},
            "rh_gas_in_temp_b": {"column": "rh_gas_b", "range": [0.0, 900.0], "stuck_max": 0.05},
        },
        "aux": {},
        "actions": {
            "A": {"valve1": {"column": "v1a_col", "unit_scale": 0.01, "range": [0.0, 1.0]},
                  "valve2": {"column": "v2b_col", "unit_scale": 0.01, "range": [0.0, 1.0]}},
            "B": {"valve1": {"column": "v1b_col", "unit_scale": 0.01, "range": [0.0, 1.0]},
                  "valve2": {"column": "v2a_col", "unit_scale": 0.01, "range": [0.0, 1.0]}},
            "continuity": {"valve2_min_corr": 0.999, "valve2_max_mae": 0.02},
        },
        "mill_on": {
            "threshold_tph": 2.0,
            "feeder_columns": [f"feeder{k}" for k in range(1, N_MILLS + 1)],
            "mill_gas_temp_columns": [f"mill_t{k}" for k in range(1, N_MILLS + 1)],
            "mill_gas_flow_columns": [f"mill_f{k}" for k in range(1, N_MILLS + 1)],
        },
        "alignment_checks": {
            "steam_flow": {"column": "steam_col_th", "unit_scale": 1.0 / 3.6,
                           "min_corr": 0.999, "max_mae": 0.5},
            "separator_pressure": {"column": "sep_p_col", "min_corr": 0.99, "max_mae": 0.05},
            "sh1_outlet_temp": {"candidates": ["att1_out_l", "fuel_col"],
                                "min_corr": 0.99, "max_mae": 0.5},
        },
    }
    for name in AUX_ELEMENTS:
        mapping["aux"][name] = {"column": "att1_out_l", "range": [0.0, 800.0],
                                "stuck_max": 0.05}
    mpath = tmp_path / "mapping_v2.json"
    mpath.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return v1, mpath


def test_production_mapping_validates() -> None:
    mapping = load_channel_mapping_v2("configs/final_wm/channel_mapping_v2.json")
    assert set(mapping["boundary_ext"]) == set(BOUNDARY_EXT_ELEMENTS)
    assert set(mapping["aux"]) == set(AUX_ELEMENTS)


def test_mapping_v2_fail_closed(tmp_path) -> None:
    _, mpath = _fixture(tmp_path)
    mapping = json.loads(mpath.read_text(encoding="utf-8"))
    del mapping["boundary_ext"]["flue_o2"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(FinalWMProtocolError, match="boundary_ext"):
        load_channel_mapping_v2(bad)

    mapping = json.loads(mpath.read_text(encoding="utf-8"))
    mapping["boundary_ext"]["flue_o2"]["column"] = None
    mapping["boundary_ext"]["flue_o2"]["derived"] = "mill_gas_wavg"
    bad.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(FinalWMProtocolError, match="exclusive"):
        load_channel_mapping_v2(bad)

    mapping = json.loads(mpath.read_text(encoding="utf-8"))
    mapping["alignment_checks"] = {}
    bad.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(FinalWMProtocolError, match="alignment"):
        load_channel_mapping_v2(bad)

    mapping = json.loads(mpath.read_text(encoding="utf-8"))
    del mapping["actions"]
    bad.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(FinalWMProtocolError, match="actions"):
        load_channel_mapping_v2(bad)

    mapping = json.loads(mpath.read_text(encoding="utf-8"))
    del mapping["actions"]["B"]["valve1"]
    bad.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(FinalWMProtocolError, match="actions.B"):
        load_channel_mapping_v2(bad)


def test_build_happy_path(tmp_path) -> None:
    v1, mpath = _fixture(tmp_path)
    out = tmp_path / "canonical_sideX_v2.npz"
    meta = build_canonical_v2(v1, tmp_path, mpath, out, side="A")
    v1_arrays = np.load(v1)
    v2_arrays = np.load(out)
    for key in ("boundary", "obs", "valid", "timestamps", "split"):
        np.testing.assert_array_equal(v2_arrays[key], v1_arrays[key])
    # v2.1: actions rebuilt per corrected wiring; side-A fixture wiring matches,
    # so actions are still byte-equal here, and valve2 continuity must be perfect.
    np.testing.assert_array_equal(v2_arrays["actions"], v1_arrays["actions"])
    assert meta["version"] == "2.1"
    assert meta["side"] == "A"
    assert meta["actions_continuity"]["valve2"]["corr_with_v1"] == pytest.approx(1.0)
    assert meta["actions_continuity"]["valve2"]["mae_vs_v1"] == pytest.approx(0.0, abs=1e-4)
    n = v1_arrays["boundary"].shape[0]
    assert v2_arrays["boundary_ext"].shape == (n, len(BOUNDARY_EXT_ELEMENTS))
    assert v2_arrays["aux"].shape == (n, len(AUX_ELEMENTS))
    assert v2_arrays["mill_on"].shape == (n, N_MILLS)

    mill_on = v2_arrays["mill_on"]
    assert mill_on[:, :3].sum() == 3 * n
    assert mill_on[:, 3:].sum() == 0
    ext = v2_arrays["boundary_ext"]
    i_count = BOUNDARY_EXT_ELEMENTS.index("mill_count_on")
    np.testing.assert_array_equal(ext[:, i_count], 3.0)
    i_wavg = BOUNDARY_EXT_ELEMENTS.index("mill_gas_temp_wavg")
    np.testing.assert_allclose(ext[:, i_wavg], 610.0 + np.sin(np.arange(n) / 16.0), atol=1e-3)

    # candidates auto-select picks the correlated column
    chosen = {r["channel"]: r["column"] for r in meta["alignment"]}
    assert chosen["sh1_outlet_temp"] == "att1_out_l"
    assert all(r["passed"] for r in meta["alignment"])
    assert (tmp_path / "canonical_sideX_v2_meta.json").exists()

    record = CanonicalV2Record(out)
    assert record.boundary_full().shape == (n, len(BOUNDARY_FULL_ELEMENTS))
    assert record.aux_index("superheat_sep") == AUX_ELEMENTS.index("superheat_sep")


def test_side_b_wiring(tmp_path) -> None:
    v1, mpath = _fixture(tmp_path)
    out = tmp_path / "out_b.npz"
    # side B rewires both valves; valve2 continuity against v1 (built with the
    # side-A pattern) must therefore FAIL -- proving the gate actually bites.
    with pytest.raises(FinalWMProtocolError, match="continuity gate"):
        build_canonical_v2(v1, tmp_path, mpath, out, side="B")


def test_actions_continuity_fail_closed(tmp_path) -> None:
    import pandas as pd

    v1, mpath = _fixture(tmp_path)
    frame = pd.read_csv(tmp_path / "all_merged.csv")
    rng = np.random.default_rng(2)
    frame["v2b_col"] = rng.uniform(20, 50, len(frame))
    frame.to_csv(tmp_path / "all_merged.csv", index=False)
    with pytest.raises(FinalWMProtocolError, match="continuity gate"):
        build_canonical_v2(v1, tmp_path, mpath, tmp_path / "out.npz", side="A")


def test_actions_range_gate(tmp_path) -> None:
    import pandas as pd

    v1, mpath = _fixture(tmp_path)
    frame = pd.read_csv(tmp_path / "all_merged.csv")
    frame["v1a_col"] = 150.0  # 1.5 fraction after scale -> out of [0,1]
    frame.to_csv(tmp_path / "all_merged.csv", index=False)
    with pytest.raises(FinalWMProtocolError, match="actions.valve1 range_violation"):
        build_canonical_v2(v1, tmp_path, mpath, tmp_path / "out.npz", side="A")


def test_grid_containment_breach(tmp_path) -> None:
    v1, mpath = _fixture(tmp_path, time_offset_s=5)
    with pytest.raises(FinalWMProtocolError, match="containment"):
        build_canonical_v2(v1, tmp_path, mpath, tmp_path / "out.npz", side="A")


def test_alignment_breach_fail_closed(tmp_path) -> None:
    v1, mpath = _fixture(tmp_path)
    mapping = json.loads(mpath.read_text(encoding="utf-8"))
    mapping["alignment_checks"]["steam_flow"]["min_corr"] = 0.999
    # corrupt the source column -> correlation collapses
    import pandas as pd

    frame = pd.read_csv(tmp_path / "all_merged.csv")
    rng = np.random.default_rng(1)
    frame["steam_col_th"] = rng.normal(1200, 300, len(frame))
    frame.to_csv(tmp_path / "all_merged.csv", index=False)
    with pytest.raises(FinalWMProtocolError, match="alignment check failed"):
        build_canonical_v2(v1, tmp_path, mpath, tmp_path / "out.npz", side="A")


def test_quality_gates_fail_closed(tmp_path) -> None:
    v1, mpath = _fixture(tmp_path)
    import pandas as pd

    # range breach: aux channel driven out of [0, 800]
    frame = pd.read_csv(tmp_path / "all_merged.csv")
    frame["att1_out_l"] = 900.0 + np.arange(len(frame)) * 0.001
    frame.to_csv(tmp_path / "all_merged.csv", index=False)
    with pytest.raises(FinalWMProtocolError, match="range_violation|alignment"):
        build_canonical_v2(v1, tmp_path, mpath, tmp_path / "out.npz", side="A")

    # stuck breach on a dynamic ext channel (alignment cols restored by fresh fixture)
    v1, mpath = _fixture(tmp_path)
    frame = pd.read_csv(tmp_path / "all_merged.csv")
    frame["o2_col"] = 4.0
    frame.to_csv(tmp_path / "all_merged.csv", index=False)
    with pytest.raises(FinalWMProtocolError, match="stuck_ratio"):
        build_canonical_v2(v1, tmp_path, mpath, tmp_path / "out.npz", side="A")


def _prepend_rows(v1_path, n_pre: int):
    """Return a new v1 npz with n_pre duplicated rows prepended (earlier times)."""
    arrays = np.load(v1_path)
    out = {}
    for key in ("boundary", "actions", "obs", "valid", "split"):
        out[key] = np.concatenate([arrays[key][:1].repeat(n_pre, 0), arrays[key]])
    out["timestamps"] = np.concatenate(
        [arrays["timestamps"][0] - (n_pre - np.arange(n_pre)) * 10,
         arrays["timestamps"]])
    return out


def test_edge_trim_within_bound(tmp_path) -> None:
    v1, mpath = _fixture(tmp_path)
    padded = tmp_path / "v1_pad.npz"
    np.savez_compressed(padded, **_prepend_rows(v1, 12))
    out = tmp_path / "out.npz"
    meta = build_canonical_v2(padded, tmp_path, mpath, out, side="A")
    assert meta["edge_trim"] == {"leading": 12, "trailing": 0}
    v1_arrays = np.load(v1)
    v2_arrays = np.load(out)
    np.testing.assert_array_equal(v2_arrays["boundary"], v1_arrays["boundary"])
    np.testing.assert_array_equal(v2_arrays["timestamps"], v1_arrays["timestamps"])


def test_edge_trim_beyond_bound_fails(tmp_path) -> None:
    v1, mpath = _fixture(tmp_path)
    padded = tmp_path / "v1_pad.npz"
    np.savez_compressed(padded, **_prepend_rows(v1, 100))
    with pytest.raises(FinalWMProtocolError, match="edge trim"):
        build_canonical_v2(padded, tmp_path, mpath, tmp_path / "out.npz", side="A")


def test_loader_refuses_v1(tmp_path) -> None:
    v1, _ = _fixture(tmp_path)
    with pytest.raises(FinalWMProtocolError, match="not a v2"):
        CanonicalV2Record(v1)
