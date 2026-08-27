"""R1 direction re-adjudication on the corrected record (2026-08-26).

Obligation: every frozen direction verdict (R1/F3, the norew arm's sole
certificate) was adjudicated on the OLD wrong-side-v1 record. The record has now
changed (v2.1 wiring) and the resulting model sits in a different basin
(best@86 / 107 epochs vs best@51 / 72). Today's earlier direction numbers are
VOID because those checkpoints were trained with AnalyticThermoProperties.

This re-runs the repo protocol (evaluation.step_response_direction: +0.05 valve,
60 steps, terminal 10-step mean, frac_negative = 1.0 means correct/cooling) on
the properly trained grid-properties checkpoints, and adds a load-stratified
view. Eval-only, no training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.contracts import BOUNDARY_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.evaluation import step_response_direction
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "direction_v2"
OUT.mkdir(parents=True, exist_ok=True)
IDX_FLOW = BOUNDARY_ELEMENTS.index("steam_flow")
torch.backends.cuda.matmul.allow_tf32 = True

props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)

CASES = {
    # tag: (record npz, checkpoint, anchored?)
    "corrected_unanchored": (
        P / "v1fix_probe/canonical_sideA_v1fixed.npz",
        P / "v1fix_probe/v1fix_unanchored/checkpoints/t1_closure_cons_norew_seed0.pt",
        False),
    "corrected_anchored": (
        P / "v1fix_probe/canonical_sideA_v1fixed.npz",
        P / "v1fix_probe/v1fix_anchored/checkpoints/t1_closure_cons_norew_seed0.pt",
        True),
    "old_record_armA": (
        ROOT / "artifacts/final_wm/canonical_sideA.npz",
        P / "retrain_probe/armA_budget/checkpoints/t1_closure_cons_norew_seed0.pt",
        False),
    "old_record_armC": (
        ROOT / "artifacts/final_wm/canonical_sideA.npz",
        P / "retrain_probe/armC_anchor_s1const_seed0/checkpoints/t1_closure_cons_norew_seed0.pt",
        True),
}


def load(ckpt, anchored):
    kw = dict(boundary_mode="oracle", initial_state_mode="hybrid",
              closure_mode="conservative_norew", epochs=120, patience=20,
              batch_size=32, batches_per_epoch=200)
    if anchored:
        kw["init_checkpoint"] = str(
            P / "retrain_probe/anchor_assets/anchor_init_s1constants_seed0.pt")
    spec = ms._base("t1", "closure_cons_norew", 0, **kw)
    m = build_world_model(spec, props).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                 weights_only=False)["state_dict"])
    return m.eval()


@torch.no_grad()
def by_load(model, record, valve_index, n_win=256, rollout=60, delta_v=0.05):
    gen = torch.Generator().manual_seed(0)
    b = sample_windows(record, SPLIT_VAL, n_win, 96, 1, gen)
    b0 = b.future_boundary[:, 0].to(DEVICE)
    a0 = b.future_actions[:, 0].to(DEVICE)
    o0 = b.history.obs[:, -1].to(DEVICE)
    s0 = model.transition.initial_steady_state(b0, a0, o0)
    bseq = b0.unsqueeze(1).repeat(1, rollout, 1)
    base = a0.unsqueeze(1).repeat(1, rollout, 1)
    st = base.clone()
    st[:, :, valve_index] = (st[:, :, valve_index] + delta_v).clamp(max=1.0)
    _, tb = model.transition.integrate(s0, bseq, base)
    _, ts = model.transition.integrate(s0, bseq, st)
    d = (ts[:, -10:, 4] - tb[:, -10:, 4]).mean(dim=1).cpu().numpy()
    return d, b0[:, IDX_FLOW].cpu().numpy()


report = {}
print("=" * 78)
print("R1 direction, grid-properties checkpoints (frac_neg = 1.0 is correct)")
print("=" * 78)
for tag, (rec_path, ckpt, anchored) in CASES.items():
    if not ckpt.exists():
        print(f"  {tag:22s} MISSING {ckpt.name}")
        continue
    record = CanonicalRecord(rec_path)
    model = load(ckpt, anchored)
    entry = {}
    for vi, vn in ((0, "v1"), (1, "v2")):
        r = step_response_direction(model, record, SPLIT_VAL, n_windows=32,
                                    rollout_steps=60, valve_index=vi,
                                    delta_v=0.05, seed=0, device=DEVICE)
        entry[vn] = r
        flag = "OK" if r["frac_negative"] == 1.0 else (
            "WRONG-SIGN" if r["frac_negative"] < 0.5 else "MIXED")
        print(f"  {tag:22s} {vn}: mean_delta={r['mean_delta_c']:+8.4f} degC  "
              f"frac_neg={r['frac_negative']:.3f}  [{flag}]")
    report[tag] = entry

print("\n" + "=" * 78)
print("load-stratified direction (Qk: mean_delta / frac_negative)")
print("=" * 78)
edges = None
for tag, (rec_path, ckpt, anchored) in CASES.items():
    if not ckpt.exists():
        continue
    record = CanonicalRecord(rec_path)
    model = load(ckpt, anchored)
    for vi, vn in ((0, "v1"), (1, "v2")):
        d, flow = by_load(model, record, vi)
        e = np.quantile(flow, [0, .2, .4, .6, .8, 1.0]) if edges is None else edges
        row = f"  {tag:22s} {vn}: "
        for i in range(5):
            m = (flow >= e[i]) & (flow <= e[i+1] if i == 4 else flow < e[i+1])
            row += (f" Q{i+1} {d[m].mean():+7.4f}/{(d[m] < 0).mean():.2f}"
                    if m.sum() else f" Q{i+1} n/a")
        print(row)
        report.setdefault(tag, {}).setdefault(f"{vn}_by_load", 
            [float(d[(flow >= e[i]) & (flow <= e[i+1] if i == 4 else flow < e[i+1])].mean())
             for i in range(5)])

(OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
print("\ndone")
