"""Live-gradient rewet arm (2026-08-26): the ACTUAL rewet test.

Why this arm exists: rewet_probe was intended to test whether the v0.4 norew
verdict was a wrong-side-v1 artifact, but it was void -- anchoring from a norew
checkpoint imported aW1/aW2 raw = -30, where softplus(-30) ~ 9.4e-14 and its
gradient is equally tiny, so the "intact" arm could never leave zero rewet
(verified: the trained checkpoint still has aW1=aW2=-30.0 exactly, and the
captured q_w1/q_w2 are identically 0 in every load bin).

This arm re-initialises aW1/aW2 into a live gradient region (raw=-2 ->
softplus ~ 0.127 -> a_w ~ 19 kW/K, about 13% of the 150 kW/K prior) with
requires_grad=True, on the corrected record, everything else identical to
rewet_probe. Outcomes and what they mean:
  - MAE improves and aW grows  -> the term acts as a fitting knob (bad for the
    physics claim, since mixpoint superheat is +80..+199 degC everywhere:
    mixpoint_twophase_test.py) -> report as a fudge-factor finding
  - aW decays back toward zero -> norew is doubly confirmed, now on a live
    gradient path rather than a dead one
Comparison: corrected-record norew unanchored 6.313 / val 1.601; the void
"intact" arm 4.410 / val 1.436 (whose gain must be attributed to the anchor,
not to rewet).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
OUT = P / "rewet_live_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
REC = P / "v1fix_probe/canonical_sideA_v1fixed.npz"
ANCHOR = P / "retrain_probe/anchor_assets/anchor_init_s1constants_seed0.pt"
AW_INIT_RAW = -2.0
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(REC)
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)


def revive_rewet(model):
    """Put aW1/aW2 back on a live gradient path after the anchor pinned them."""
    raw = model.transition.raw
    for name in ("aW1", "aW2"):
        raw[name].data.fill_(AW_INIT_RAW)
        raw[name].requires_grad_(True)
    a1 = float(model.transition.val("aW1"))
    print(f"[revive] aW raw={AW_INIT_RAW} -> a_w1={a1:.3f} kW/K "
          f"(prior 150), grad_live={raw['aW1'].requires_grad}, "
          f"softplus'={float(torch.sigmoid(torch.tensor(AW_INIT_RAW))):.4f}", flush=True)
    return model


if __name__ == "__main__":
    spec = ms._base("t1", "closure_cons", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative",
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200,
                    init_checkpoint=str(ANCHOR))
    import src.final_wm.training as T
    orig = T.build_world_model
    # train_arm builds, then loads the anchor; revive AFTER the load by wrapping
    # the builder and re-reviving through a forward pre-hook on the first call.
    state = {"revived": False}

    def builder(sp, pr, **kw):
        m = orig(sp, pr, **kw)

        def pre_hook(module, args):
            if not state["revived"]:
                revive_rewet(m)
                state["revived"] = True
        m.register_forward_pre_hook(pre_hook)
        return m

    T.build_world_model = builder
    try:
        final = train_arm(spec, record, OUT, device=DEVICE, compile_substep=False)
    finally:
        T.build_world_model = orig

    model = orig(spec, props).to(DEVICE)
    model.load_state_dict(torch.load(OUT / "checkpoints" / f"{final['run_id']}.pt",
                                     map_location=DEVICE,
                                     weights_only=False)["state_dict"])
    model.eval()
    aw1 = float(model.transition.val("aW1"))
    aw2 = float(model.transition.val("aW2"))
    raw1 = float(model.transition.raw["aW1"])
    print(f"[learned rewet] aW1={aw1:.3f} aW2={aw2:.3f} kW/K "
          f"(raw {raw1:.3f}, init {AW_INIT_RAW}, prior 150)", flush=True)

    gen = torch.Generator().manual_seed(50_000)
    errs, loads, done = [], [], 0
    with torch.no_grad():
        while done < 256:
            bsz = min(32, 256 - done)
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
    dirs = {}
    for vi, vn in ((0, "v1"), (1, "v2")):
        dirs[vn] = step_response_direction(model, record, SPLIT_VAL, n_windows=32,
                                           rollout_steps=60, valve_index=vi,
                                           delta_v=0.05, seed=0, device=DEVICE)
    print(f"[rewet_live] H18 ch4 overall={np.mean(ch4['bin_means']):.3f} "
          f"bins={[round(x, 3) for x in ch4['bin_means']]} | "
          f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']} | "
          f"dir v1 frac_neg={dirs['v1']['frac_negative']:.3f} "
          f"({dirs['v1']['mean_delta_c']:+.4f}) "
          f"v2 frac_neg={dirs['v2']['frac_negative']:.3f} "
          f"({dirs['v2']['mean_delta_c']:+.4f})", flush=True)
    (OUT / "report.json").write_text(json.dumps(
        {"train": {k: final[k] for k in ("best_val_nll", "best_epoch",
                                         "epochs_run", "stop_reason")},
         "eval": {"overall_h18_mae": float(np.mean(ch4["bin_means"])),
                  "bins_q1q5": ch4["bin_means"]},
         "rewet": {"aW1_kW_per_K": aw1, "aW2_kW_per_K": aw2,
                   "raw_init": AW_INIT_RAW, "raw_final": raw1},
         "direction": dirs}, indent=2))
    print("done")
