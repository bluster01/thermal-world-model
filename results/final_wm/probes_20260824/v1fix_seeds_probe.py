"""Corrected-record seed replication (2026-08-26).

seed0 on the corrected v2.1 record gave H18 0.484 unanchored (vs 0.723 for the
same seed on the old record, -33%), with nearly flat load bins (1.13x spread vs
1.69x). But the old record's 120/20 three-seed MEAN was 0.546 while its seed0 was
0.723 -- seed0 happens to be the worst seed there -- so the corrected record's
mean is still unknown and the -33% is a same-seed statement only.

This arm runs seeds 1 and 2, corrected record, unanchored, everything else
identical, so the three-seed mean and spread become comparable to the old
record's. Grid properties are asserted before training and verified from the
ledger afterwards (the 2026-08-26 analytic-properties defect).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.final_wm.analysis import STEAM_FLOW_INDEX, WindowErrors, binning_stats
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from experiments.final_wm import matrix_spec as ms
from probe_guard import assert_grid, verify_ledger_properties

DEVICE = "cuda"
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "v1fix_seeds"
OUT.mkdir(parents=True, exist_ok=True)
REC = P / "v1fix_probe/canonical_sideA_v1fixed.npz"
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(REC)
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)


@torch.no_grad()
def eval_h18(model, n_win=256, seed_eval=50_000):
    model.eval()
    gen = torch.Generator().manual_seed(seed_eval)
    errs, loads, done = [], [], 0
    while done < n_win:
        bsz = min(32, n_win - done)
        b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
        hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                   actions=b.history.actions.to(DEVICE),
                                   boundary=b.history.boundary.to(DEVICE))
        r = model.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                           true_future_boundary=b.future_boundary.to(DEVICE))
        errs.append((b.future_obs.to(DEVICE) - r.temps_mu).abs().cpu())
        loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX])
        done += bsz
    we = WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads),
                      day_ids=torch.zeros(done, dtype=torch.long))
    ch4 = binning_stats(we)["H18"]["final_outlet_temp"]
    return float(np.mean(ch4["bin_means"])), ch4["bin_means"]


if __name__ == "__main__":
    report = {}
    for seed in (1, 2):
        spec = ms._base("t1", "closure_cons_norew", seed, boundary_mode="oracle",
                        initial_state_mode="hybrid", closure_mode="conservative_norew",
                        epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
        arm_dir = OUT / f"seed{seed}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[v1fix_seed{seed}] training (corrected record, unanchored)", flush=True)
        assert_grid(props)
        final = train_arm(spec, record, arm_dir, device=DEVICE, properties=props)
        verify_ledger_properties(arm_dir)
        model = build_world_model(spec, props).to(DEVICE)
        model.load_state_dict(torch.load(
            arm_dir / "checkpoints" / f"{final['run_id']}.pt",
            map_location=DEVICE, weights_only=False)["state_dict"])
        mae, bins = eval_h18(model)
        report[f"seed{seed}"] = {
            "best_val_nll": final["best_val_nll"], "best_epoch": final["best_epoch"],
            "epochs_run": final["epochs_run"], "overall_h18_mae": mae,
            "bins_q1q5": bins, "bin_spread": max(bins) / min(bins)}
        print(f"[v1fix_seed{seed}] H18={mae:.3f} bins={[round(x,3) for x in bins]} "
              f"spread={max(bins)/min(bins):.2f}x val={final['best_val_nll']:.4f}"
              f"@{final['best_epoch']}", flush=True)
        (OUT / "report.json").write_text(json.dumps(report, indent=2))

    seed0 = 0.4840  # v1fix_probe unanchored, same record/spec
    all_mae = [seed0] + [report[k]["overall_h18_mae"] for k in sorted(report)]
    print(f"\n[corrected record, unanchored] three-seed H18: "
          f"{[round(x,3) for x in all_mae]}  mean={np.mean(all_mae):.3f} "
          f"spread={max(all_mae) - min(all_mae):.3f}")
    print("  old-record reference: seed0 0.723, three-seed mean 0.546 "
          "(120/20 budget); anchored three-seed mean 0.454")
    (OUT / "report.json").write_text(json.dumps(
        {**report, "summary": {"three_seed_h18": all_mae,
                               "mean": float(np.mean(all_mae)),
                               "spread": float(max(all_mae) - min(all_mae)),
                               "old_record_mean_120_20": 0.546,
                               "old_record_anchored_mean": 0.454}}, indent=2))
    print("done")
