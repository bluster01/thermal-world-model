"""Corrected-v1 probe: quantify the wrong-sided stage-1 valve impact.

2026-08-25. Plant truth: stage-1 SAME-side (A->left), stage-2 CROSS.
Canonical sideA (left temps) has v1=一级B (wrong), v2=二级B (correct).
Corrected record: v1=一级A (dual action col 0), v2=二级B (col 3, unchanged).
Only the action[:,0] column changes; obs/boundary/future_obs/splits identical
-> H18 ch4 MAE is directly comparable against armA seed0 (0.723) and the
anchored seed0 (0.478).

Arms (seed0, 120/20, oracle, conservative_norew, real train_arm):
  v1fix_unanchored : corrected record, fresh init
  v1fix_anchored   : corrected record, s1-constants anchor init
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

from src.final_wm.analysis import STEAM_FLOW_INDEX, WindowErrors, binning_stats
from src.final_wm.contracts import OBSERVATION_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.training import build_world_model, train_arm
from src.final_wm.properties import load_grid_properties
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
OUT = ROOT / "results/final_wm/probes_20260824/v1fix_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN = 256
EVAL_SEED = 50_000
torch.backends.cuda.matmul.allow_tf32 = True

SIDEA = ROOT / "artifacts/final_wm/canonical_sideA.npz"
DUAL = ROOT / "results/final_wm/d0/canonical_record.npz"

# ---- build corrected record ----
sideA = np.load(SIDEA)
dual = np.load(DUAL)
n = len(sideA["timestamps"])
assert len(dual["action"]) == n, "row count mismatch"
# self-check: current sideA v1 = dual col 2 (一级B), v2 = dual col 3 (二级B)
old_v1, old_v2 = sideA["actions"][:, 0], sideA["actions"][:, 1]
assert np.allclose(old_v2, dual["action"][:, 3], atol=1e-6), "sideA v2 != dual col3"
assert np.allclose(old_v1, dual["action"][:, 2], atol=1e-6), "sideA v1 != dual col2 (bridge changed?)"
corr_old = float(np.corrcoef(old_v1, dual["action"][:, 0])[0, 1])
print(f"[record] sideA n={n}, corr(old v1=一级B, new v1=一级A)={corr_old:.3f}", flush=True)
new_actions = sideA["actions"].copy()
new_actions[:, 0] = dual["action"][:, 0].astype(np.float32)   # 一级A
corr_new_own = float(np.corrcoef(new_actions[:, 0], sideA["obs"][:, 1])[0, 1])
corr_old_own = float(np.corrcoef(old_v1, sideA["obs"][:, 1])[0, 1])
print(f"[record] v1 vs own-side sh1_outlet: old={corr_old_own:+.3f} new={corr_new_own:+.3f}", flush=True)
corrected_path = OUT / "canonical_sideA_v1fixed.npz"
np.savez_compressed(corrected_path,
                    boundary=sideA["boundary"], actions=new_actions, obs=sideA["obs"],
                    valid=sideA["valid"], timestamps=sideA["timestamps"], split=sideA["split"])
record = CanonicalRecord(corrected_path)


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
    base_kw = dict(boundary_mode="oracle", initial_state_mode="hybrid",
                   closure_mode="conservative_norew", epochs=120, patience=20,
                   batch_size=32, batches_per_epoch=200)
    report = {}
    # NOTE 2026-08-26: props MUST be loaded before train_arm and passed in.
    # Omitting properties= made train_arm fall back to AnalyticThermoProperties
    # while evaluation used the IAPWS grid -> train/eval physics mismatch and
    # garbage MAE (this arm's 6.313 is void).
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz",
                                 device=DEVICE)
    for tag, anchor in (("v1fix_unanchored", None),
                        ("v1fix_anchored", ROOT / "results/final_wm/probes_20260824/retrain_probe"
                         "/anchor_assets/anchor_init_s1constants_seed0.pt")):
        kw = dict(base_kw)
        if anchor is not None:
            kw["init_checkpoint"] = str(anchor)
        spec = ms._base("t1", "closure_cons_norew", 0, **kw)
        arm_dir = OUT / tag
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{tag}] training (init_checkpoint={kw.get('init_checkpoint')})", flush=True)
        sys.path.insert(0, str(Path(__file__).parent))
        from probe_guard import assert_grid, verify_ledger_properties
        assert_grid(props)
        final = train_arm(spec, record, arm_dir, device=DEVICE, properties=props)
        verify_ledger_properties(arm_dir)
        model = build_world_model(spec, props).to(DEVICE)
        model.load_state_dict(torch.load(
            arm_dir / "checkpoints" / f"{final['run_id']}.pt", map_location=DEVICE,
            weights_only=False)["state_dict"])
        ev = eval_probe_set(model, tag)
        print(f"[{tag}] H18 ch4 overall={ev['overall_h18_mae']:.3f} "
              f"bins={[round(x, 3) for x in ev['bins_q1q5']]} | "
              f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']}", flush=True)
        report[tag] = {"train": {k: final[k] for k in
                                 ("best_val_nll", "best_epoch", "epochs_run", "stop_reason")},
                       "eval": ev}
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
