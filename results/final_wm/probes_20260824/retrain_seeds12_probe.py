"""Retrain seeds 1/2 with the armA budget (epochs=120, patience=20).

Continuation of retrain_seed0_probe.py (2026-08-25). armA config won on
seed0 (1.047 -> 0.724). Now get the full 3-seed retrained mean for a
potential budget-amendment proposal. Exploratory: probe out dirs only,
frozen checkpoints untouched, no verdict blocks.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import WindowErrors, binning_stats, STEAM_FLOW_INDEX
from src.final_wm.contracts import OBSERVATION_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN = 256
EVAL_SEED = 50_000
OUT = ROOT / "results/final_wm/probes_20260824/retrain_probe"
record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
torch.backends.cuda.matmul.allow_tf32 = True

OVERRIDES = dict(epochs=120, patience=20, batch_size=32, batches_per_epoch=200)


def eval_probe_set(model, tag, out_dir):
    model.eval()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    errs, loads, days, preds, acts = [], [], [], [], []
    done = 0
    with torch.no_grad():
        while done < N_WIN:
            bsz = min(32, N_WIN - done)
            b = sample_windows(record, SPLIT_VAL, bsz, ms.HISTORY_STEPS, ms.HORIZON, gen)
            hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                       actions=b.history.actions.to(DEVICE),
                                       boundary=b.history.boundary.to(DEVICE))
            r = model.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                               true_future_boundary=b.future_boundary.to(DEVICE))
            preds.append(r.temps_mu[:, :, CH].cpu())
            acts.append(b.future_obs[:, :, CH].cpu())
            errs.append((b.future_obs.to(DEVICE) - r.temps_mu).abs().cpu())
            loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX])
            days.append(b.day_ids)
            done += bsz
    we = WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads),
                      day_ids=torch.cat(days))
    bins = binning_stats(we)
    ch4 = bins["H18"]["final_outlet_temp"]
    np.savez_compressed(out_dir / f"preds_{tag}.npz",
                        pred=torch.cat(preds).numpy(),
                        actual=torch.cat(acts).numpy(),
                        load=we.load.numpy(), days=we.day_ids.numpy())
    return {"overall_h18_mae": float(np.mean(ch4["bin_means"])),
            "bins_q1q5": ch4["bin_means"]}


report = {}
for seed in (1, 2):
    tag = f"armA_budget_seed{seed}"
    out_dir = OUT / tag
    spec = ms._base("t1", "closure_cons_norew", seed, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative_norew",
                    **OVERRIDES)
    t0 = time.time()
    print(f"[{tag}] training", flush=True)
    final = train_arm(spec, record, out_dir, device=DEVICE, properties=props)
    print(f"[{tag}] stop={final['stop_reason']} epochs_run={final['epochs_run']} "
          f"best_epoch={final['best_epoch']} best_val_nll={final['best_val_nll']:.4f} "
          f"wall={final['wall_seconds']/3600:.2f}h", flush=True)
    model = build_world_model(spec, props).to(DEVICE)
    ckpt = out_dir / "checkpoints" / f"{spec.unit}_{spec.arm}_seed{spec.seed}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["state_dict"])
    ev = eval_probe_set(model, tag, out_dir)
    print(f"[{tag}] H18 ch4 overall={ev['overall_h18_mae']:.3f} "
          f"bins={[round(x, 3) for x in ev['bins_q1q5']]}", flush=True)
    report[tag] = {"train": {k: final[k] for k in
                   ("stop_reason", "epochs_run", "best_epoch", "best_val_nll",
                    "wall_seconds")}, "eval": ev}
    (OUT / "report_seeds12.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")

print(json.dumps(report, indent=2))
