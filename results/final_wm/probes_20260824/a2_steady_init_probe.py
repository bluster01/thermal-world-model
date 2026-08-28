"""A2 — pure-physics initial-state ablation (2026-08-28, user-ordered).

PREREG (PREREG_load_scheduling_20260826.md s2-A2): quantifies what the NN
observer actually buys for H18. The arm trains the SAME model with
initial_state_mode='steady' (pure initial_steady_state(), no learned
posterior correction); everything else identical to the corrected-record
unanchored baseline (0.4840 / val 1.1096@86, hybrid init).

Honesty check, not a gain hunt: if steady ties hybrid, the paper must say the
NN state correction has no accuracy gain and its value lives only in closure
(+6%), joint identification and probabilistic output.

Baseline pair (same protocol, seed0, corrected v2.1, grid properties):
  hybrid init (v1fix_unanchored) : H18 0.4840, bins .459/.455/.503/.513/.489
  steady init (this arm)         : TBD
Prior from the O1 matrix (v0.2 budget, OLD wrong-side record): learned MIXED,
hybrid REJECTED -- to be retested under 120/20 + v2.1.
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
from src.final_wm.evaluation import step_response_direction
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "a2_steady_init_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN = 256
EVAL_SEED = 50_000
REC = P / "v1fix_probe/canonical_sideA_v1fixed.npz"
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(REC)
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz",
                             device=DEVICE)


def eval_probe_set(model, tag):
    model.eval()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    errs, loads, done = [], [], 0
    with torch.no_grad():
        while done < N_WIN:
            bsz = min(32, N_WIN - done)
            b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
            hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                       actions=b.history.actions.to(DEVICE),
                                       boundary=b.history.boundary.to(DEVICE))
            r = model.forecast(hist, b.future_actions.to(DEVICE),
                               boundary_mode="oracle",
                               true_future_boundary=b.future_boundary.to(DEVICE))
            errs.append((b.future_obs.to(DEVICE) - r.temps_mu).abs().cpu())
            loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX])
            done += bsz
    we = WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads),
                      day_ids=torch.zeros(done, dtype=torch.long))
    ch4 = binning_stats(we)["H18"]["final_outlet_temp"]
    return {"overall_h18_mae": float(np.mean(ch4["bin_means"])),
            "bins_q1q5": ch4["bin_means"]}


if __name__ == "__main__":
    spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                    initial_state_mode="steady", closure_mode="conservative_norew",
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
    sys.path.insert(0, str(P))
    from probe_guard import assert_grid, verify_ledger_properties
    assert_grid(props)
    print("[a2_steady] training (pure physical steady init; everything else = "
          "corrected-record unanchored baseline)", flush=True)
    final = train_arm(spec, record, OUT, device=DEVICE, properties=props)
    verify_ledger_properties(OUT)
    model = build_world_model(spec, props).to(DEVICE)
    model.load_state_dict(torch.load(
        OUT / "checkpoints" / f"{final['run_id']}.pt", map_location=DEVICE,
        weights_only=False)["state_dict"])
    ev = eval_probe_set(model, "a2_steady")
    dirs = {}
    for vi, vn in ((0, "v1"), (1, "v2")):
        dirs[vn] = step_response_direction(model, record, SPLIT_VAL, n_windows=32,
                                           rollout_steps=60, valve_index=vi,
                                           delta_v=0.05, seed=0, device=DEVICE)
    print(f"[a2_steady] H18 ch4 overall={ev['overall_h18_mae']:.3f} "
          f"bins={[round(x, 3) for x in ev['bins_q1q5']]} | "
          f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']} | "
          f"dir v1 frac={dirs['v1']['frac_negative']:.3f} "
          f"v2 frac={dirs['v2']['frac_negative']:.3f}", flush=True)
    (OUT / "report.json").write_text(json.dumps(
        {"arm": "a2_steady_init",
         "train": {k: final[k] for k in ("best_val_nll", "best_epoch",
                                         "epochs_run", "stop_reason")},
         "eval": ev, "direction": dirs}, indent=2))
    print("done")
