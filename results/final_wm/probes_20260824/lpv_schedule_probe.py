"""LPV (flow-scheduled) gray-box prototype (2026-08-26).

Implements the user's architecture critique probe-side: transition parameters
become functions of the operating condition instead of global scalars, with the
schedule FORM fixed by physics and only the magnitude learnable.

  tau_mix1/2, tauB : transport   -> theta * s^(-alpha_tau)   (plug flow, 1/mdot)
  UA0-2            : convection  -> theta * s^(+alpha_ua)    (Dittus-Boelter 0.8)
  Cm, M            : NOT scheduled (metal property / inventory)

s = mdot / mdot_ref, clamped [0.1, 3.0]; alpha = alpha_max * sigmoid(raw), so the
sign is pinned by physics and only the magnitude is learned (cannot invert).

Injection point: override _substep (transition.py:399) which already receives
d_flow together with ua/tau_b/tau_mix1/tau_mix2 -> no src change, no copy of
step(). Sanity gate: with alpha forced to 0 the subclass must be bit-identical
to the parent (same pattern that validated the encoder probe).

Usage:
  python lpv_schedule_probe.py --sanity       # identity check only
  python lpv_schedule_probe.py --train        # train the scheduled arm
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import STEAM_FLOW_INDEX, WindowErrors, binning_stats
from src.final_wm.contracts import OBSERVATION_ELEMENTS
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from src.final_wm.transition import Fan2020UDETransition
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results/final_wm/probes_20260824/lpv_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN, EVAL_SEED = 256, 50_000
FLOW_REF = 300.0          # kg/s, near the training-set median
ALPHA_TAU_MAX = 2.5       # physics bound (measured exponents 0.5-2.2)
ALPHA_UA_MAX = 1.2        # Dittus-Boelter 0.8 with headroom
torch.backends.cuda.matmul.allow_tf32 = True


class FlowScheduledTransition(Fan2020UDETransition):
    """Flow-scheduled transport lags and heat-transfer coefficients.

    Schedule sign is pinned by physics; only the magnitude is learnable.
    alpha_scale=0.0 reproduces the parent exactly (sanity gate).
    """

    def __init__(self, *args, alpha_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_scale = float(alpha_scale)
        # Nested-model init: raw=-4 -> sigmoid~0.018 -> alpha~0.045, i.e. the
        # schedule starts essentially OFF so the arm begins at the global-constant
        # baseline and can only grow the schedule if the data pays for it.
        self.alpha_tau_raw = nn.Parameter(torch.tensor(-4.0))
        self.alpha_ua_raw = nn.Parameter(torch.tensor(-4.0))

    def _alphas(self):
        a_tau = ALPHA_TAU_MAX * torch.sigmoid(self.alpha_tau_raw) * self.alpha_scale
        a_ua = ALPHA_UA_MAX * torch.sigmoid(self.alpha_ua_raw) * self.alpha_scale
        return a_tau, a_ua

    def _substep(self, h, tm, rb, m1, m2, lag1, lag2, dsw1, dsw2, d_flow, u_b,
                 p_stack, p0, p1, h_spray, h_sep, m_cap, ua, cm, k_t,
                 tau_b, tau_evap, tau_mix1, tau_mix2, dt_sub, h_lo, h_hi,
                 steam_power, metal_power):
        if self.alpha_scale != 0.0:
            a_tau, a_ua = self._alphas()
            s = (d_flow / FLOW_REF).clamp(min=0.1, max=3.0)
            # transport lags shrink with flow (mass conservation)
            inv = s.pow(-a_tau)
            # Stability guard: the explicit lag update is stable for
            # tau > dt_sub/2 (= 1 s here). Worst case with alpha=2.5 at max flow
            # gives tau_mix 80 -> 17.5 s (17x margin), so this floor should never
            # bind -- it is cheap insurance against a pathological alpha/flow.
            floor = 3.0 * dt_sub
            tau_b = (tau_b * inv).clamp(min=floor)
            tau_mix1 = (tau_mix1 * inv).clamp(min=floor)
            tau_mix2 = (tau_mix2 * inv).clamp(min=floor)
            # convective UA grows with flow (Dittus-Boelter)
            ua = ua * s.pow(a_ua).unsqueeze(-1)
        return super()._substep(h, tm, rb, m1, m2, lag1, lag2, dsw1, dsw2,
                                d_flow, u_b, p_stack, p0, p1, h_spray, h_sep,
                                m_cap, ua, cm, k_t, tau_b, tau_evap, tau_mix1,
                                tau_mix2, dt_sub, h_lo, h_hi,
                                steam_power, metal_power)


def _promote(model, alpha_scale: float):
    """Swap the transition module in-place for the scheduled subclass.

    Rebinds __class__ so the subclass _substep hook takes effect without
    reconstructing the module (all buffers/params/priors preserved verbatim),
    then registers the two schedule-magnitude parameters.
    """
    tr = model.transition
    dev = next(tr.parameters()).device
    tr.__class__ = FlowScheduledTransition
    tr.alpha_scale = float(alpha_scale)
    if "alpha_tau_raw" not in tr._parameters:
        tr.register_parameter("alpha_tau_raw",
                              nn.Parameter(torch.tensor(-4.0, device=dev)))
        tr.register_parameter("alpha_ua_raw",
                              nn.Parameter(torch.tensor(-4.0, device=dev)))
    return model


def build(spec, props, alpha_scale: float):
    model = build_world_model(spec, props)
    return _promote(model, alpha_scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()

    record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz",
                                 device=DEVICE)
    spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative_norew",
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200)

    if args.sanity:
        torch.manual_seed(0)
        base = build_world_model(spec, props).to(DEVICE).eval()
        sd = {k: v.clone() for k, v in base.state_dict().items()}
        sched = build(spec, props, alpha_scale=0.0).to(DEVICE).eval()
        missing = sched.load_state_dict(sd, strict=False)
        print(f"load_state_dict: missing={list(missing.missing_keys)} "
              f"unexpected={list(missing.unexpected_keys)}")
        gen = torch.Generator().manual_seed(7)
        b = sample_windows(record, SPLIT_TRAIN, 8, 96, 18, gen)
        hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                   actions=b.history.actions.to(DEVICE),
                                   boundary=b.history.boundary.to(DEVICE))
        fa, fb = b.future_actions.to(DEVICE), b.future_boundary.to(DEVICE)
        with torch.no_grad():
            r0 = base.forecast(hist, fa, boundary_mode="oracle", true_future_boundary=fb)
            r1 = sched.forecast(hist, fa, boundary_mode="oracle", true_future_boundary=fb)
        d = (r0.temps_mu - r1.temps_mu).abs().max().item()
        print(f"[sanity alpha=0] max_abs_diff = {d:.3e}  -> "
              f"{'IDENTICAL' if d == 0.0 else 'MISMATCH'}")
        # and show the schedule actually bites when enabled
        sched2 = build(spec, props, alpha_scale=1.0).to(DEVICE).eval()
        sched2.load_state_dict(sd, strict=False)
        with torch.no_grad():
            r2 = sched2.forecast(hist, fa, boundary_mode="oracle", true_future_boundary=fb)
        print(f"[alpha=1 active] max_abs_diff vs base = "
              f"{(r0.temps_mu - r2.temps_mu).abs().max().item():.3f} degC "
              f"(must be > 0 for the schedule to bite)")
        a_tau, a_ua = sched2.transition._alphas()
        print(f"  initial alphas: alpha_tau={a_tau.item():.3f} alpha_ua={a_ua.item():.3f}")
        flows = b.future_boundary[:, :, STEAM_FLOW_INDEX]
        s = (flows / FLOW_REF).clamp(0.1, 3.0)
        print(f"  flow range in batch {flows.min():.0f}-{flows.max():.0f} kg/s -> "
              f"tau factor {s.pow(-a_tau.item()).min():.2f}-{s.pow(-a_tau.item()).max():.2f}, "
              f"UA factor {s.pow(a_ua.item()).min():.2f}-{s.pow(a_ua.item()).max():.2f}")
        return

    if args.train:
        print("[lpv_scheduled] training (compile_substep disabled: subclass hook)",
              flush=True)
        # train_arm builds the model internally; patch the builder so the arm
        # gets the scheduled transition.
        import src.final_wm.training as T
        orig = T.build_world_model
        T.build_world_model = lambda sp, pr, **kw: _promote(orig(sp, pr, **kw), 1.0)
        try:
            final = train_arm(spec, record, OUT, device=DEVICE, compile_substep=False)
        finally:
            T.build_world_model = orig
        model = build(spec, props, 1.0).to(DEVICE)
        model.load_state_dict(torch.load(OUT / "checkpoints" / f"{final['run_id']}.pt",
                                         map_location=DEVICE,
                                         weights_only=False)["state_dict"], strict=False)
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
        a_tau, a_ua = model.transition._alphas()
        print(f"[lpv_scheduled] H18 ch4 overall={np.mean(ch4['bin_means']):.3f} "
              f"bins={[round(x, 3) for x in ch4['bin_means']]} | "
              f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']} | "
              f"learned alpha_tau={a_tau.item():.3f} alpha_ua={a_ua.item():.3f}",
              flush=True)
        (OUT / "report.json").write_text(json.dumps(
            {"train": {k: final[k] for k in ("best_val_nll", "best_epoch",
                                             "epochs_run", "stop_reason")},
             "eval": {"overall_h18_mae": float(np.mean(ch4["bin_means"])),
                      "bins_q1q5": ch4["bin_means"]},
             "alphas": {"alpha_tau": float(a_tau.item()),
                        "alpha_ua": float(a_ua.item())}}, indent=2))
        print("done")


if __name__ == "__main__":
    main()
