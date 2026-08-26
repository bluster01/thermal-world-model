"""D0-v2 pipeline: canonical record v2 built from all_merged_10s.csv.

Design (spec: docs/plans/2026-08-25-v06-canonical-v2-spec.md):

- v1 registries stay frozen.  The v2 record carries the six v1 keys
  (boundary/actions/obs/valid/timestamps/split) byte-verbatim from the v1
  npz, plus three extension keys: ``boundary_ext`` (7), ``aux`` (15),
  ``mill_on`` (8, uint8).
- Fail-closed gates: (a) grid containment -- every v1 timestamp must land
  exactly on the all_merged 10 s grid; (b) numeric alignment checks of v1
  channels against all_merged columns (raw values, thresholds in the
  mapping; left/right candidates auto-selected by correlation); (c) new
  channel quality gates (coverage / stuck / physical range).
- v2 records remain loadable by the v1 ``CanonicalRecord`` (extension keys
  are simply ignored there); ``CanonicalV2Record`` refuses non-v2 files.

Amendment v0.6 Phase 1 is data-only: nothing here feeds a model yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from src.final_wm.contracts import (
    BOUNDARY_ELEMENTS,
    FinalWMProtocolError,
    OBSERVATION_ELEMENTS,
)
from src.final_wm.data import CanonicalRecord

CANONICAL_V2_VERSION = 2
CANONICAL_V2_REVISION = "2.1"  # 2.1 = actions rebuilt per corrected valve wiring

# Registry order is contractual; the mapping must match these exactly.
BOUNDARY_EXT_ELEMENTS = (
    "fuel_corrected",       # corrected total fuel, t/h
    "mill_count_on",        # mills with feeder coal > threshold
    "mill_gas_temp_wavg",   # flow-weighted mill furnace-gas temperature, degC
    "flue_o2",              # flue gas O2, %
    "secondary_air_total",  # total secondary air, t/h
    "rh_gas_in_temp_a",     # vertical LT-reheater inlet flue gas temp (A), degC
    "rh_gas_in_temp_b",     # vertical LT-reheater inlet flue gas temp (B), degC
)
AUX_ELEMENTS = (
    "att1_in_temp_l", "att1_in_temp_r",
    "att1_out_temp_l", "att1_out_temp_r",
    "att2_in_temp_l", "att2_in_temp_r",
    "att2_out_temp_l", "att2_out_temp_r",
    "spray_flow_sh_total", "spray_flow_rh_total",
    "superheat_sep",
    "rh_steam_in_temp_l", "rh_steam_in_temp_r",
    "rh_steam_out_temp_l", "rh_steam_out_temp_r",
)
N_MILLS = 8

BOUNDARY_FULL_ELEMENTS = tuple(BOUNDARY_ELEMENTS) + BOUNDARY_EXT_ELEMENTS


# ---------------------------------------------------------------------------
# Mapping validation
# ---------------------------------------------------------------------------

def load_channel_mapping_v2(path: str | Path) -> dict:
    """Validate a v2 mapping against the v2 registries (fail-closed)."""
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(mapping.get("version", 0)) != CANONICAL_V2_VERSION:
        raise FinalWMProtocolError("v2 mapping must declare version=2")
    src = mapping.get("source", {})
    if "file" not in src or "time_column" not in src:
        raise FinalWMProtocolError("v2 mapping lacks source.file/time_column")
    for section, registry in (("boundary_ext", BOUNDARY_EXT_ELEMENTS),
                              ("aux", AUX_ELEMENTS)):
        block = mapping.get(section, {})
        missing = set(registry) - set(block)
        extra = set(block) - set(registry)
        if missing or extra:
            raise FinalWMProtocolError(
                f"v2 mapping {section} mismatch; missing={sorted(missing)} extra={sorted(extra)}")
        for name, spec in block.items():
            if "derived" in spec and "column" in spec:
                raise FinalWMProtocolError(f"{section}.{name}: column and derived are exclusive")
            if "derived" not in spec and "column" not in spec:
                raise FinalWMProtocolError(f"{section}.{name} lacks column/derived")
            rng = spec.get("range")
            if rng is None or float(rng[0]) >= float(rng[1]):
                raise FinalWMProtocolError(f"{section}.{name} lacks a valid range gate")
    mill = mapping.get("mill_on", {})
    for key in ("feeder_columns", "mill_gas_temp_columns", "mill_gas_flow_columns"):
        if len(mill.get(key, [])) != N_MILLS:
            raise FinalWMProtocolError(f"mill_on.{key} must list {N_MILLS} columns")
    if float(mill.get("threshold_tph", 0.0)) <= 0.0:
        raise FinalWMProtocolError("mill_on.threshold_tph must be positive")
    checks = mapping.get("alignment_checks", {})
    if not checks:
        raise FinalWMProtocolError("v2 mapping must carry alignment checks")
    for name, spec in checks.items():
        if name not in set(BOUNDARY_ELEMENTS) | set(OBSERVATION_ELEMENTS):
            raise FinalWMProtocolError(f"alignment check targets unknown v1 channel: {name}")
        if ("column" in spec) == ("candidates" in spec):
            raise FinalWMProtocolError(f"alignment.{name}: exactly one of column/candidates")
        if float(spec.get("min_corr", 0.0)) <= 0.0 or float(spec.get("max_mae", 0.0)) <= 0.0:
            raise FinalWMProtocolError(f"alignment.{name} lacks min_corr/max_mae")
    actions = mapping.get("actions")
    if not isinstance(actions, dict):
        raise FinalWMProtocolError("v2.1 mapping must carry an actions section")
    for side in ("A", "B"):
        block = actions.get(side)
        if not isinstance(block, dict):
            raise FinalWMProtocolError(f"actions.{side} missing")
        for valve in ("valve1", "valve2"):
            spec = block.get(valve, {})
            if "column" not in spec or float(spec.get("unit_scale", 0.0)) <= 0.0:
                raise FinalWMProtocolError(f"actions.{side}.{valve} lacks column/unit_scale")
            rng = spec.get("range")
            if rng is None or float(rng[0]) >= float(rng[1]):
                raise FinalWMProtocolError(f"actions.{side}.{valve} lacks a valid range")
    cont = actions.get("continuity", {})
    if float(cont.get("valve2_min_corr", 0.0)) <= 0.0 or float(cont.get("valve2_max_mae", 0.0)) <= 0.0:
        raise FinalWMProtocolError("actions.continuity lacks valve2_min_corr/valve2_max_mae")
    return mapping


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V2QualityGateConfig:
    dt_seconds: float = 10.0
    interp_limit: int = 3              # 30 s, consistent with v1
    min_coverage: float = 0.99
    stuck_minutes: float = 30.0
    max_range_violation: float = 0.001
    # Grid containment: interior off-grid rows always fail.  Leading/trailing
    # off-grid runs (v1 record edges outside the source coverage) are trimmed
    # up to this many rows per edge and recorded in meta.
    max_edge_trim: int = 60


class AlignmentResult(NamedTuple):
    channel: str
    column: str
    corr: float
    mae: float
    passed: bool


def _stuck_ratio(values: np.ndarray, min_run: int) -> float:
    if len(values) < min_run:
        return 0.0
    change = np.abs(np.diff(values)) > 1e-12
    idx = np.arange(len(values))
    last_change = np.maximum.accumulate(np.where(np.concatenate(([True], change)), idx, 0))
    run_length = idx - last_change + 1
    return float((run_length >= min_run).mean())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_columns(mapping: dict) -> list[str]:
    cols: set[str] = set()
    for section in ("boundary_ext", "aux"):
        for spec in mapping[section].values():
            if "column" in spec:
                cols.add(spec["column"])
    mill = mapping["mill_on"]
    cols.update(mill["feeder_columns"])
    cols.update(mill["mill_gas_temp_columns"])
    cols.update(mill["mill_gas_flow_columns"])
    for side in ("A", "B"):
        for valve in ("valve1", "valve2"):
            cols.add(mapping["actions"][side][valve]["column"])
    for spec in mapping["alignment_checks"].values():
        if "column" in spec:
            cols.add(spec["column"])
        else:
            cols.update(spec["candidates"])
    return sorted(cols)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_canonical_v2(
    v1_path: str | Path,
    data_root: str | Path,
    mapping_path: str | Path,
    out_path: str | Path,
    *,
    side: str,
    gates: V2QualityGateConfig = V2QualityGateConfig(),
) -> dict:
    """Build one side's canonical v2 record; raises (fail-closed) on any breach."""
    import pandas as pd

    if side not in ("A", "B"):
        raise FinalWMProtocolError(f"side must be A or B, got: {side!r}")
    v1_path = Path(v1_path)
    data_root = Path(data_root)
    mapping = load_channel_mapping_v2(mapping_path)
    v1 = np.load(v1_path)
    for key in ("boundary", "actions", "obs", "valid", "timestamps", "split"):
        if key not in v1:
            raise FinalWMProtocolError(f"v1 record lacks key: {key}")
    v1_timestamps = v1["timestamps"].astype(np.int64)
    v1_grid = pd.to_datetime(v1_timestamps, unit="s")
    n = len(v1_timestamps)

    src_rel = mapping["source"]["file"]
    time_col = mapping["source"]["time_column"]
    src_path = data_root / src_rel
    if not src_path.exists():
        raise FinalWMProtocolError(f"all_merged source missing: {src_path}")
    wanted = set(_required_columns(mapping)) | {time_col}
    frame = pd.read_csv(src_path, usecols=lambda c: c in wanted)
    missing_cols = (wanted - {time_col}) - set(frame.columns)
    if missing_cols:
        raise FinalWMProtocolError(f"all_merged lacks mapped columns: {sorted(missing_cols)}")
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame = frame.set_index(time_col).sort_index()

    # Gate 1: grid containment -- v1 grid must be an exact subset of the
    # source 10 s grid (allowing the v1-consistent interpolation first).
    dt = pd.Timedelta(seconds=gates.dt_seconds)
    full_grid = pd.date_range(frame.index.min(), frame.index.max(), freq=dt)
    aligned = frame.reindex(frame.index.union(full_grid)).sort_index()
    aligned = aligned.interpolate(method="time", limit=gates.interp_limit).reindex(full_grid)
    epoch = (full_grid.as_unit("ns").asi8 // 10**9).astype(np.int64)
    lookup = pd.Series(np.arange(len(full_grid)), index=epoch)
    pos = lookup.reindex(v1_timestamps)
    trim_lo = trim_hi = 0
    if pos.isna().any():
        on_grid = pos.notna().to_numpy()
        interior_bad = int((~on_grid).sum() - np.argmax(on_grid) - np.argmax(on_grid[::-1]))
        # off-grid rows must form a leading and/or trailing contiguous run
        first_ok = int(np.argmax(on_grid))
        last_ok = int(len(on_grid) - np.argmax(on_grid[::-1]))
        if interior_bad > 0:
            raise FinalWMProtocolError(
                f"grid containment breached: {interior_bad} interior v1 timestamps "
                "off the source 10 s grid")
        trim_lo, trim_hi = first_ok, len(on_grid) - last_ok
        if trim_lo > gates.max_edge_trim or trim_hi > gates.max_edge_trim:
            raise FinalWMProtocolError(
                f"edge trim exceeds bound: leading={trim_lo} trailing={trim_hi} "
                f"> {gates.max_edge_trim}")
        v1_timestamps = v1_timestamps[first_ok:last_ok]
        v1_grid = v1_grid[first_ok:last_ok]
        pos = lookup.reindex(v1_timestamps)
        n = len(v1_timestamps)
    rows = aligned.iloc[pos.to_numpy(dtype=np.int64)]

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(rows[name], errors="coerce").to_numpy(dtype=np.float64)

    # Gate 2: numeric alignment of v1 channels against source columns (raw).
    sl = slice(trim_lo, len(v1["timestamps"]) - trim_hi if trim_hi else None)
    v1_series: dict[str, np.ndarray] = {
        name: v1["boundary"][sl, i].astype(np.float64) for i, name in enumerate(BOUNDARY_ELEMENTS)
    }
    v1_series.update({
        name: v1["obs"][sl, i].astype(np.float64) for i, name in enumerate(OBSERVATION_ELEMENTS)
    })
    alignment: list[AlignmentResult] = []
    for name, spec in mapping["alignment_checks"].items():
        target = v1_series[name]
        candidates = [spec["column"]] if "column" in spec else list(spec["candidates"])
        scale = float(spec.get("unit_scale", 1.0))
        best: AlignmentResult | None = None
        for cand in candidates:
            cand_values = col(cand) * scale
            mask = np.isfinite(target) & np.isfinite(cand_values)
            if mask.mean() < gates.min_coverage:
                res = AlignmentResult(name, cand, 0.0, float("inf"), False)
            else:
                t, c = target[mask], cand_values[mask]
                res = AlignmentResult(name, cand, _corr(t, c), float(np.abs(t - c).mean()), False)
            passed = res.corr >= float(spec["min_corr"]) and res.mae <= float(spec["max_mae"])
            res = res._replace(passed=passed)
            if best is None or res.corr > best.corr:
                best = res
        assert best is not None
        alignment.append(best)
        if not best.passed:
            raise FinalWMProtocolError(
                f"alignment check failed: {name} vs {best.column} "
                f"(corr={best.corr:.4f} < {spec['min_corr']} or mae={best.mae:.4f} > {spec['max_mae']})")

    # Derived channels (raw, before clip).
    mill = mapping["mill_on"]
    thresh = float(mill["threshold_tph"])
    feeders = np.stack([col(c) for c in mill["feeder_columns"]], axis=1)
    mill_on = (np.nan_to_num(feeders, nan=0.0) > thresh).astype(np.uint8)
    mill_count = mill_on.sum(axis=1).astype(np.float64)
    gas_t = np.stack([col(c) for c in mill["mill_gas_temp_columns"]], axis=1)
    gas_f = np.clip(np.nan_to_num(
        np.stack([col(c) for c in mill["mill_gas_flow_columns"]], axis=1), nan=0.0), 0.0, None)
    w_sum = gas_f.sum(axis=1)
    wavg = np.where(
        w_sum > 1e-9,
        (np.nan_to_num(gas_t, nan=0.0) * gas_f).sum(axis=1) / np.maximum(w_sum, 1e-9),
        np.nan_to_num(gas_t, nan=0.0).mean(axis=1),
    )
    derived = {"mill_on_count": mill_count, "mill_gas_wavg": wavg}

    def channel_values(spec: dict) -> np.ndarray:
        if "derived" in spec:
            kind = spec["derived"]
            if kind not in derived:
                raise FinalWMProtocolError(f"unknown derived kind: {kind}")
            return derived[kind].copy()
        return col(spec["column"])

    # Assemble ext/aux, apply clip, then Gate 3: quality gates.
    min_run = int(gates.stuck_minutes * 60.0 / gates.dt_seconds)
    quality: dict[str, dict] = {}
    ext = np.stack([channel_values(mapping["boundary_ext"][k]) for k in BOUNDARY_EXT_ELEMENTS], axis=1)
    aux = np.stack([channel_values(mapping["aux"][k]) for k in AUX_ELEMENTS], axis=1)

    # v2.1: actions rebuilt per corrected wiring (stage-1 same-side, stage-2
    # cross).  Continuity gate: new valve2 must reproduce the old v1 valve2
    # (v1 stage-2 wiring was already correct); new valve1 differs by design
    # and its correlation with the old (mis-wired) channel is provenance only.
    act_map = mapping["actions"][side]
    cont = mapping["actions"]["continuity"]
    new_actions = np.stack(
        [col(act_map[v]["column"]) * float(act_map[v]["unit_scale"]) for v in ("valve1", "valve2")],
        axis=1,
    )
    v1_actions = v1["actions"][sl].astype(np.float64)
    continuity: dict[str, dict] = {}
    for j, valve in enumerate(("valve1", "valve2")):
        mask = np.isfinite(new_actions[:, j]) & np.isfinite(v1_actions[:, j])
        if mask.mean() < gates.min_coverage:
            raise FinalWMProtocolError(f"actions.{valve} coverage {mask.mean():.4f} < {gates.min_coverage}")
        c = _corr(new_actions[mask, j], v1_actions[mask, j])
        mae = float(np.abs(new_actions[mask, j] - v1_actions[mask, j]).mean())
        continuity[valve] = {"corr_with_v1": c, "mae_vs_v1": mae,
                             "source_column": act_map[valve]["column"]}
    if not (continuity["valve2"]["corr_with_v1"] >= float(cont["valve2_min_corr"])
            and continuity["valve2"]["mae_vs_v1"] <= float(cont["valve2_max_mae"])):
        raise FinalWMProtocolError(
            "actions continuity gate breached on valve2: "
            f"corr={continuity['valve2']['corr_with_v1']:.4f} mae={continuity['valve2']['mae_vs_v1']:.5f}")
    for j, valve in enumerate(("valve1", "valve2")):
        lo, hi = float(act_map[valve]["range"][0]), float(act_map[valve]["range"][1])
        finite = np.isfinite(new_actions[:, j])
        viol = float(((new_actions[finite, j] < lo) | (new_actions[finite, j] > hi)).mean())
        quality[f"actions.{valve}"] = {"coverage": float(finite.mean()),
                                       "range_violation": viol, "stuck_ratio": None}
        if viol > gates.max_range_violation:
            raise FinalWMProtocolError(f"actions.{valve} range_violation {viol:.5f}")
    new_actions = np.clip(np.nan_to_num(new_actions, nan=0.0), 0.0, 1.0).astype(np.float32)

    breaches: list[str] = []
    for block_name, registry, arr in (("boundary_ext", BOUNDARY_EXT_ELEMENTS, ext),
                                      ("aux", AUX_ELEMENTS, aux)):
        for j, name in enumerate(registry):
            spec = mapping[block_name][name]
            values = arr[:, j]
            coverage = float(np.isfinite(values).mean())
            if "clip" in spec:
                lo, hi = float(spec["clip"][0]), float(spec["clip"][1])
                values = np.clip(values, lo, hi)
                arr[:, j] = values
            lo, hi = float(spec["range"][0]), float(spec["range"][1])
            finite = np.isfinite(values)
            range_violation = float(((values[finite] < lo) | (values[finite] > hi)).mean()) \
                if finite.any() else 1.0
            stuck_max = spec.get("stuck_max")
            stuck = None if stuck_max is None else _stuck_ratio(
                np.nan_to_num(values, nan=0.0), min_run)
            quality[f"{block_name}.{name}"] = {
                "coverage": coverage, "range_violation": range_violation, "stuck_ratio": stuck,
            }
            if coverage < gates.min_coverage:
                breaches.append(f"{block_name}.{name} coverage {coverage:.4f} < {gates.min_coverage}")
            if range_violation > gates.max_range_violation:
                breaches.append(f"{block_name}.{name} range_violation {range_violation:.5f}")
            if stuck is not None and stuck > float(stuck_max):
                breaches.append(f"{block_name}.{name} stuck_ratio {stuck:.4f} > {stuck_max}")
    if breaches:
        raise FinalWMProtocolError("v2 quality gates breached: " + "; ".join(breaches))

    ext = np.nan_to_num(ext, nan=0.0).astype(np.float32)
    aux = np.nan_to_num(aux, nan=0.0).astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        boundary=v1["boundary"][sl], actions=new_actions, obs=v1["obs"][sl],
        valid=v1["valid"][sl], timestamps=v1["timestamps"][sl], split=v1["split"][sl],
        boundary_ext=ext, aux=aux, mill_on=mill_on,
    )
    meta = {
        "version": CANONICAL_V2_REVISION,
        "side": side,
        "actions_continuity": continuity,
        "known_defect_fix": "results/final_wm/known_defect_v1_valve1_20260826.md",
        "n_samples": n,
        "boundary_ext_elements": list(BOUNDARY_EXT_ELEMENTS),
        "aux_elements": list(AUX_ELEMENTS),
        "n_mills": N_MILLS,
        "quality": quality,
        "alignment": [r._asdict() for r in alignment],
        "provenance": {
            "v1_record_path": str(v1_path),
            "v1_record_sha256": _sha256(v1_path),
            "source_path": str(src_path),
            "source_sha256": _sha256(src_path),
            "mapping_path": str(mapping_path),
            "mapping_sha256": _sha256(Path(mapping_path)),
            "mill_on_threshold_tph": thresh,
        },
        "v1_keys_verbatim": ["boundary", "obs", "valid", "timestamps", "split"],
        "v1_actions_replaced": True,
        "edge_trim": {"leading": trim_lo, "trailing": trim_hi},
    }
    (out_path.parent / (out_path.stem + "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class CanonicalV2Record(CanonicalRecord):
    """v2 record: v1 view plus boundary_ext / aux / mill_on.

    ``boundary_full`` concatenates boundary + boundary_ext (registry order
    BOUNDARY_FULL_ELEMENTS) -- the Phase 2-A1 model input view.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        arrays = np.load(path)
        for key in ("boundary_ext", "aux", "mill_on"):
            if key not in arrays:
                raise FinalWMProtocolError(f"not a v2 canonical record (lacks {key}): {path}")
        self.boundary_ext = torch.from_numpy(arrays["boundary_ext"].astype(np.float32))
        self.aux = torch.from_numpy(arrays["aux"].astype(np.float32))
        self.mill_on = torch.from_numpy(arrays["mill_on"].astype(np.uint8))
        if self.boundary_ext.shape != (self.n, len(BOUNDARY_EXT_ELEMENTS)):
            raise FinalWMProtocolError("v2 boundary_ext width mismatch")
        if self.aux.shape != (self.n, len(AUX_ELEMENTS)):
            raise FinalWMProtocolError("v2 aux width mismatch")
        if self.mill_on.shape != (self.n, N_MILLS):
            raise FinalWMProtocolError("v2 mill_on width mismatch")

    def boundary_full(self) -> torch.Tensor:
        return torch.cat([self.boundary, self.boundary_ext], dim=1)

    def aux_index(self, name: str) -> int:
        if name not in AUX_ELEMENTS:
            raise FinalWMProtocolError(f"unknown aux element: {name}")
        return AUX_ELEMENTS.index(name)
