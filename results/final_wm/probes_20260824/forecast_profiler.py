"""Profile one forecast: where does the 340ms/batch forward actually go?"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.data import SPLIT_TRAIN, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative_norew",
                epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
model = build_world_model(spec, props).to(DEVICE)
model.load_state_dict(torch.load(
    ROOT / "results/final_wm/probes_20260824/retrain_probe"
    "/armC_anchor_s1const_seed0/checkpoints/t1_closure_cons_norew_seed0.pt",
    map_location=DEVICE, weights_only=False)["state_dict"])
model.eval()

gen = torch.Generator().manual_seed(7)
batch = sample_windows(record, SPLIT_TRAIN, 32, 96, 18, gen)
history = batch.history.__class__(obs=batch.history.obs.to(DEVICE),
                                  actions=batch.history.actions.to(DEVICE),
                                  boundary=batch.history.boundary.to(DEVICE))
fa = batch.future_actions.to(DEVICE)
fbnd = batch.future_boundary.to(DEVICE)

with torch.no_grad():
    for _ in range(3):
        model.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile, record_function
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        with record_function("forecast_total"):
            model.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
    # python-side op count
    from torch.profiler import profile as prof2
    with prof2(with_stack=False) as p2:
        model.forecast(history, fa, boundary_mode="oracle", true_future_boundary=fbnd)
    print(f"\ntotal cpu ops dispatched: {p2.events().count()} ")
