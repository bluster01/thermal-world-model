"""Single-window debug: is the step action actually different? Is temps delta really ~0?"""
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
    from src.final_wm.jepa import fit_privileged_normalizer
    from src.final_wm.properties import load_grid_properties

    normalizer = fit_privileged_normalizer(record)
    properties = load_grid_properties(Path(matrix["properties"]))
    R._git_commit = lambda: "a0495d9ddfaa95449c0a1d97b835890bfedfa3c1"
    model = R._load_arm(
        "c0", matrix, R._sha256(Path("configs/final_wm/jepa_b_series_v1.json")),
        normalizer, properties, out_root, device,
    )
    idx = R._fixed_indices(
        record, R.SPLIT_VAL, matrix["data_contract"]["history_steps"], 1,
        256, matrix["evaluation"]["paired_seed"],
    )[:1]
    raw = R.sample_jepa_windows(
        record, R.SPLIT_VAL, 1, matrix["data_contract"]["history_steps"], 1,
        torch.Generator().manual_seed(0), fixed_indices=idx,
    )
    batch = R._device_batch(raw, device)
    horizon = 18
    boundary = batch.future_boundary[:, :horizon]
    base_act = batch.future_actions[:, :horizon]
    step_act = base_act.clone()
    step_act[..., 0] = (step_act[..., 0] + 0.05).clamp(max=1.0)
    print("valve1 base:", base_act[0, :, 0].tolist())
    print("valve1 step:", step_act[0, :, 0].tolist())
    print("valve2 base:", base_act[0, :, 1].tolist()[:6], "...")
    base = model.counterfactual(
        batch.history, base_act, boundary_mode="oracle",
        true_future_boundary=boundary, allow_extrapolation=True,
    )
    step = model.counterfactual(
        batch.history, step_act, boundary_mode="oracle",
        true_future_boundary=boundary, allow_extrapolation=True,
    )
    d = (step.temps_mu[:, -3:, -1] - base.temps_mu[:, -3:, -1])
    print("temps delta (tail3, last channel):", d.tolist())
    print("state dims:", base.temps_mu.shape, "| support:", base.in_support, step.in_support)
