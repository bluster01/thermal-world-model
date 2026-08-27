"""Common-mode stage-1 action probe (2026-08-26).

Motivation (two_side_coupling_diag.py): stage-1 is 65-85% common-mode; the
corrected own-side valve (1A) is near-closed and partially stuck exactly in
the high-load bins where the corrected-record arm exploded (Q4/Q5 12.7/10.3);
cross-side correlations match or exceed own-side (Gate B H600 finding
replicated). Hypothesis: the identifiable stage-1 input is the COMMON mode,
not either single-side valve.

Arm: valve1 := (一级A + 一级B)/2, valve2 := 二级B (own-side, unchanged, already
correct in v1). Everything else (obs/boundary/future temps/split) untouched.
closure_cons_norew + s1 anchor + 120/20, per PROBE_PROTOCOL.md
(compile_substep=True, P1 auto).

Comparison line (all seed0, 256-window seed50k, oracle, H18 final_outlet):
  old record  (v1=一级B, wrong side)          0.723   val 1.259
  corrected   (v1=一级A) unanchored           6.313   val 1.601
  corrected   (v1=一级A) anchored             --      val 1.610 (killed)
  old record  + anchor (armC)                 0.478   val 1.099
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
OUT = ROOT / "results/final_wm/probes_20260824/common_mode_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN, EVAL_SEED = 256, 50_000
torch.backends.cuda.matmul.allow_tf32 = True

A = ROOT / "artifacts/final_wm"
ANCHOR = (ROOT / "results/final_wm/probes_20260824/retrain_probe/anchor_assets"
          / "anchor_init_s1constants_seed0.pt")
REC = OUT / "canonical_sideA_s1common.npz"

if not REC.exists():
    a_old = np.load(A / "canonical_sideA.npz")          # v1: actions=(一级B, 二级B)
    a_new = np.load(A / "canonical_sideA_v2.npz")       # v2.1: actions=(一级A, 二级B)
    b_new = np.load(A / "canonical_sideB_v2.npz")       # v2.1: actions=(一级B, 二级A)
    TRIM = 12                                           # v2 trims 12 leading rows
    s1a = a_new["actions"][:, 0]
    s1b = b_new["actions"][:, 0]
    s2b = a_new["actions"][:, 1]
    assert np.corrcoef(a_old["actions"][TRIM:, 0], s1b)[0, 1] > 0.9999, "provenance check"
    acts = np.stack([(s1a + s1b) / 2.0, s2b], axis=1).astype(np.float32)
    keep = {k: a_old[k][TRIM:] for k in ("boundary", "obs", "valid", "timestamps", "split")}
    np.savez_compressed(REC, actions=acts, **keep)
    print(f"[record] s1common built n={len(acts)}  "
          f"corr(common, 1A)={np.corrcoef(acts[:, 0], s1a)[0, 1]:.3f}  "
          f"corr(common, 1B)={np.corrcoef(acts[:, 0], s1b)[0, 1]:.3f}  "
          f"mean={acts[:, 0].mean():.4f} std={acts[:, 0].std():.4f}", flush=True)

record = CanonicalRecord(REC)
props = load_grid_properties(A / "iapws_surrogate.npz", device=DEVICE)


def eval_probe_set(model, tag):
    model.eval()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    preds, acts_, errs, loads = [], [], [], []
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
            acts_.append(b.future_obs[:, :, CH].cpu())
            errs.append((b.future_obs.to(DEVICE) - r.temps_mu).abs().cpu())
            loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX])
            done += bsz
    we = WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads),
                      day_ids=torch.zeros(done, dtype=torch.long))
    ch4 = binning_stats(we)["H18"]["final_outlet_temp"]
    np.savez_compressed(OUT / f"preds_{tag}.npz", pred=torch.cat(preds).numpy(),
                        actual=torch.cat(acts_).numpy(), load=we.load.numpy())
    return {"overall_h18_mae": float(np.mean(ch4["bin_means"])), "bins_q1q5": ch4["bin_means"]}


if __name__ == "__main__":
    spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative_norew",
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200,
                    init_checkpoint=str(ANCHOR))
    print("[s1common_anchored] training (compile_substep=True)", flush=True)
    final = train_arm(spec, record, OUT, device=DEVICE, properties=props,
                      compile_substep=True)
    model = build_world_model(spec, props).to(DEVICE)
    model.load_state_dict(torch.load(OUT / "checkpoints" / f"{final['run_id']}.pt",
                                     map_location=DEVICE, weights_only=False)["state_dict"])
    ev = eval_probe_set(model, "s1common_anchored")
    print(f"[s1common_anchored] H18 ch4 overall={ev['overall_h18_mae']:.3f} "
          f"bins={[round(x, 3) for x in ev['bins_q1q5']]} | "
          f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']} "
          f"stop={final['stop_reason']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(
        {"train": {k: final[k] for k in ("best_val_nll", "best_epoch", "epochs_run",
                                         "stop_reason", "flags")}, "eval": ev}, indent=2))
    print("done")
