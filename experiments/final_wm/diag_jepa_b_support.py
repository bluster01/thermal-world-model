"""Diagnose direction gate support-domain behavior under the original-trajectory semantics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import experiments.final_wm.run_jepa_b as R

if __name__ == "__main__":
    matrix = json.loads(Path("configs/final_wm/jepa_b_series_v1.json").read_text())
    out_root = Path(matrix["result_root"])
    record = R.JepaBRecord(Path(matrix["record"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = matrix["evaluation"]["direction"]
    indices = R._fixed_indices(
        record, R.SPLIT_VAL, matrix["data_contract"]["history_steps"], 1,
        256, matrix["evaluation"]["paired_seed"],
    )
    # check raw counterfactual support fractions on the CONTROL model (c0)
    # for the ORIGINAL-trajectory base vs the old constant base — without
    # direction_gate, just probe the support masks.
    from src.final_wm.jepa import fit_privileged_normalizer
    from src.final_wm.properties import load_grid_properties

    normalizer = fit_privileged_normalizer(record)
    properties = load_grid_properties(Path(matrix["properties"]))
    R._git_commit = lambda: "a0495d9ddfaa95449c0a1d97b835890bfedfa3c1"
    model = R._load_arm(
        "c0", matrix, R._sha256(Path("configs/final_wm/jepa_b_series_v1.json")),
        normalizer, properties, out_root, device,
    )
    horizon, valve, delta = 18, 1, cfg["delta_valve"]
    for start in range(0, min(len(indices), 64), 32):
        chunk = indices[start:start + 32]
        raw = R.sample_jepa_windows(
            record, R.SPLIT_VAL, len(chunk), matrix["data_contract"]["history_steps"],
            1, torch.Generator().manual_seed(0), fixed_indices=chunk,
        )
        batch = R._device_batch(raw, device)
        boundary = batch.future_boundary[:, :horizon]
        base_act = batch.future_actions[:, :horizon]
        step_act = base_act.clone()
        step_act[..., valve] = (step_act[..., valve] + delta).clamp(max=1.0)
        base = model.counterfactual(
            batch.history, base_act, boundary_mode="oracle",
            true_future_boundary=boundary, allow_extrapolation=True,
        )
        step = model.counterfactual(
            batch.history, step_act, boundary_mode="oracle",
            true_future_boundary=boundary, allow_extrapolation=True,
        )
        print(
            f"chunk@{start}: base_support={base.in_support.float().mean().item():.3f} "
            f"step_support={step.in_support.float().mean().item():.3f} "
            f"| valve2 mean range={step_act[..., 1].mean().item():.3f}", flush=True
        )
