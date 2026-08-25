"""Retrain probe: is t1_closure_cons_norew seed0 undertrained? (2026-08-25)

Evidence chain (ledger, artifacts/final_wm/ledger.jsonl):
- seed0: stop=patience, epochs_run=41, best_epoch=30, best_val_nll=1.536
- seed1: stop=cap,     epochs_run=60, best_epoch=56, best_val_nll=1.240
- seed2: stop=cap,     epochs_run=60, best_epoch=51, best_val_nll=1.272
=> all three seeds were still descending at stop; seed0 was cut earliest by
patience on a noisy val curve. User hypothesis: seed0's H18 MAE (1.046 on
the 256-window probe set) is a training-budget artifact, not a bad basin.

Two levers, seed0 only, each isolated in its own out dir (frozen checkpoints
in artifacts/final_wm untouched; no verdict blocks; exploratory):

  armA "budget": epochs=120, patience=20, batch=32  (patience/budget up)
  armB "batch" : epochs=60,  patience=10, batch=64  (batch size up)

Eval = canonical probe set: sideA val, 256 windows seed 50_000, oracle
boundary, H18. Compare against production seed0 (overall 1.046, bins
0.874/0.997/1.244/1.201/0.914) and the seed1/seed2 band (0.597/0.652).
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
OUT.mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)

ARMS = {
    "armA_budget": dict(epochs=120, patience=20, batch_size=32, batches_per_epoch=200),
    "armB_batch": dict(epochs=60, patience=10, batch_size=64, batches_per_epoch=200),
}


def eval_probe_set(model, tag: str, out_dir: Path) -> dict:
    """Canonical probe eval: 256 windows seed 50k, oracle, H18. Returns
    overall ch4 MAE + load-quintile bins (same recipe as binning_stats)."""
    model.eval()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    errs, loads, days, preds, acts = [], [], [], [], []
    done = 0
    with torch.no_grad():
        while done < N_WIN:
            bsz = min(32, N_WIN - done)
            b = sample_windows(record, SPLIT_VAL, bsz, ms.HISTORY_STEPS, ms.HORIZON, gen)
            hist = b.history.__class__(
                obs=b.history.obs.to(DEVICE),
                actions=b.history.actions.to(DEVICE),
                boundary=b.history.boundary.to(DEVICE),
            )
            r = model.forecast(hist, b.future_actions.to(DEVICE),
                               boundary_mode="oracle",
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
    ch4_h18 = bins["H18"]["final_outlet_temp"]
    overall = float(np.mean(ch4_h18["bin_means"]))
    preds_t = torch.cat(preds).numpy()
    acts_t = torch.cat(acts).numpy()
    np.savez_compressed(out_dir / f"preds_{tag}.npz", pred=preds_t, actual=acts_t,
                        load=we.load.numpy(), days=we.day_ids.numpy())
    return {
        "overall_h18_mae": overall,
        "bins_q1q5": ch4_h18["bin_means"],
        "between_ratio": ch4_h18["between_ratio"],
        "sh1_inlet_h18": bins["H18"]["sh1_inlet_temp"]["bin_means"],
        "sh1_outlet_h18": bins["H18"]["sh1_outlet_temp"]["bin_means"],
    }


report = {"production_seed0": {"overall_h18_mae": 1.046,
                               "bins_q1q5": [0.874, 0.997, 1.244, 1.201, 0.914]},
          "seed1_band": 0.597, "seed2_band": 0.652}

for tag, overrides in ARMS.items():
    out_dir = OUT / tag
    spec = ms._base("t1", "closure_cons_norew", 0,
                    boundary_mode="oracle", initial_state_mode="hybrid",
                    closure_mode="conservative_norew", **overrides)
    t0 = time.time()
    print(f"[{tag}] training spec={overrides}", flush=True)
    final = train_arm(spec, record, out_dir, device=DEVICE, properties=props)
    print(f"[{tag}] trained: stop={final['stop_reason']} epochs_run={final['epochs_run']} "
          f"best_epoch={final['best_epoch']} best_val_nll={final['best_val_nll']:.4f} "
          f"wall={final['wall_seconds']/3600:.2f}h", flush=True)
    model = build_world_model(spec, props).to(DEVICE)
    ckpt = out_dir / "checkpoints" / f"{spec.unit}_{spec.arm}_seed{spec.seed}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["state_dict"])
    ev = eval_probe_set(model, tag, out_dir)
    print(f"[{tag}] H18 ch4: overall={ev['overall_h18_mae']:.3f} "
          f"bins={[round(x, 3) for x in ev['bins_q1q5']]}", flush=True)
    report[tag] = {"spec": overrides, "train": {k: final[k] for k in
                   ("stop_reason", "epochs_run", "best_epoch", "best_val_nll",
                    "wall_seconds")}, "eval": ev}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
