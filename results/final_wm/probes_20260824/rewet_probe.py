"""Rewet-intact probe on the CORRECTED record (2026-08-26).

Question: is the corrected-v1 collapse (val 1.60 / probe 6.31, both arms)
caused by the norew rewet ablation, adjudicated on the OLD wrong-v1 record?
Arm: corrected record (v1=一级A) + closure_cons INTACT + s1-constant anchor
(aW1/aW2 start pinned at -30 = ablated; training may re-learn rewet).
Budget 120/20, seed0, oracle. Probe-side acceleration per PROBE_PROTOCOL.md:
train_arm(compile_substep=True) [aot_eager, bit-identical] + P1 (src, auto).
Comparison: v1fix_unanchored 6.31/val1.601, v1fix_anchored val1.610,
old-record intact (v05 table) 0.580.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import STEAM_FLOW_INDEX, WindowErrors, binning_stats
from src.final_wm.contracts import OBSERVATION_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
OUT = ROOT / "results/final_wm/probes_20260824/rewet_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN = 256
EVAL_SEED = 50_000
torch.backends.cuda.matmul.allow_tf32 = True

RECORD = ROOT / "results/final_wm/probes_20260824/v1fix_probe/canonical_sideA_v1fixed.npz"
ANCHOR = (ROOT / "results/final_wm/probes_20260824/retrain_probe/anchor_assets"
          / "anchor_init_s1constants_seed0.pt")
record = CanonicalRecord(RECORD)
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)


def eval_probe_set(model, tag):
    model.eval()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    preds, acts, errs, loads = [], [], [], []
    done = 0
    with torch.no_grad():
        while done < N_WIN:
            bsz = min(32, N_WIN - done)
            b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
            hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                       actions=b.history.actions.to(DEVICE),
                                       boundary=b.history.boundary.to(DEVICE))
            r = model.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                               true_future_boundary=b.future_boundary.to(DEVICE))
            preds.append(r.temps_mu[:, :, CH].cpu())
            acts.append(b.future_obs[:, :, CH].cpu())
            errs.append((b.future_obs.to(DEVICE) - r.temps_mu).abs().cpu())
            loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX])
            done += bsz
    we = WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads),
                      day_ids=torch.zeros(done, dtype=torch.long))
    bins = binning_stats(we)
    ch4 = bins["H18"]["final_outlet_temp"]
    np.savez_compressed(OUT / f"preds_{tag}.npz",
                        pred=torch.cat(preds).numpy(), actual=torch.cat(acts).numpy(),
                        load=we.load.numpy())
    return {"overall_h18_mae": float(np.mean(ch4["bin_means"])), "bins_q1q5": ch4["bin_means"]}


if __name__ == "__main__":
    spec = ms._base("t1", "closure_cons", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative",
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200,
                    init_checkpoint=str(ANCHOR))
    print(f"[rewet_intact_corrected] training (anchor={ANCHOR.name}, "
          f"compile_substep=True)", flush=True)
    final = train_arm(spec, record, OUT, device=DEVICE, compile_substep=True)
    model = build_world_model(spec, props).to(DEVICE)
    model.load_state_dict(torch.load(OUT / "checkpoints" / f"{final['run_id']}.pt",
                                     map_location=DEVICE, weights_only=False)["state_dict"])
    ev = eval_probe_set(model, "rewet_intact_corrected")
    print(f"[rewet_intact_corrected] H18 ch4 overall={ev['overall_h18_mae']:.3f} "
          f"bins={[round(x, 3) for x in ev['bins_q1q5']]} | "
          f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']} "
          f"stop={final['stop_reason']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(
        {"train": {k: final[k] for k in ("best_val_nll", "best_epoch", "epochs_run",
                                         "stop_reason", "flags")},
         "eval": ev}, indent=2))
    print("done")
