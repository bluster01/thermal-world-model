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


def pin_tau_to_prior(model):
    """Freeze the transport lags at their PHYSICAL prior so the flow schedule
    carries ALL load dependence.

    Arm 1 (free tau + schedule) failed at H18 7.788 vs the 0.723 baseline. The
    diagnosis: the baseline's learned tau (tau_mix1 365-757 s vs an 80 s prior)
    is ALREADY a load-averaged compensation for the missing schedule, so
    multiplying it by s^-alpha double-counts (alpha_tau grew to 1.421, s spans
    0.52-1.83 -> a 5.5x tau swing) and the high-load bins blew up (Q4/Q5
    15.6/13.3). Pinning raw at softplus_inverse(1) makes val(name) == prior
    exactly, leaving the delay structure with just two free parameters.
    """
    from src.final_wm.transition import _softplus_inverse
    raw = model.transition.raw
    pinned = {}
    for name in ("tau_mix1", "tau_mix2", "tauB"):
        raw[name].data.fill_(_softplus_inverse(1.0))
        raw[name].requires_grad_(False)
        pinned[name] = float(model.transition.val(name))
    print(f"[pin] tau frozen at physical prior: {pinned} "
          f"(only alpha_tau/alpha_ua remain learnable for the delay structure)",
          flush=True)
    return model


def fix_alphas_to_physics(model, a_tau: float = 0.0, a_ua: float = 0.8):
    """Freeze the schedule exponents at the physics values (no free parameters).

    Motivation (tau_dependence_diag.py): the MEASURED Q1/Q5 lag ratio 2.33 is
    reproduced to 3% by metal thermal inertia Cm/UA with UA ~ mdot^0.8
    (Dittus-Boelter, predicts 2.26), while plug-flow transport rho/mdot predicts
    only 1.21 (48% off) because density rises 32 -> 74 kg/m3 with load and nearly
    cancels the flow effect. So the load-dependent lag is likely a heat-transfer
    effect, and the learned alpha_tau = 1.35 is a PROXY for it. This arm removes
    the proxy: alpha_tau = 0 (no transport scheduling at all) and alpha_ua = 0.8
    (pure Dittus-Boelter), both frozen.
    """
    tr = model.transition
    # invert alpha = alpha_max * sigmoid(raw) -> raw = logit(alpha / alpha_max)
    def inv(alpha, amax):
        r = min(max(alpha / amax, 1e-6), 1 - 1e-6)
        return float(np.log(r / (1 - r)))
    tr.alpha_tau_raw.data.fill_(inv(1e-6, ALPHA_TAU_MAX) if a_tau == 0 else inv(a_tau, ALPHA_TAU_MAX))
    tr.alpha_ua_raw.data.fill_(inv(a_ua, ALPHA_UA_MAX))
    tr.alpha_tau_raw.requires_grad_(False)
    tr.alpha_ua_raw.requires_grad_(False)
    got_tau, got_ua = tr._alphas()
    print(f"[physics] alphas FROZEN: alpha_tau={got_tau.item():.4f} (target {a_tau}), "
          f"alpha_ua={got_ua.item():.4f} (target {a_ua}); "
          f"grad_tau={tr.alpha_tau_raw.requires_grad} grad_ua={tr.alpha_ua_raw.requires_grad}",
          flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--pin-prior", action="store_true",
                    help="freeze tau_mix1/tau_mix2/tauB at their PHYSICAL PRIOR "
                         "and let the flow schedule carry ALL load dependence "
                         "(fixes the double-compensation failure of arm 1)")
    ap.add_argument("--record", choices=("old", "corrected"), default="corrected",
                    help="old = canonical_sideA.npz (baseline 0.723); corrected = "
                         "v2.1 wiring (baseline 0.484, already flat load bins)")
    ap.add_argument("--ua-physics", action="store_true",
                    help="discriminating arm: FIX alpha_tau=0 and alpha_ua=0.8 "
                         "(Dittus-Boelter), zero free schedule parameters. Tests "
                         "whether the measured load-dependent lag is metal thermal "
                         "inertia Cm/UA (predicts the Q1/Q5 ratio to 3%) rather "
                         "than plug-flow transport (rho/mdot is off by 48%).")
    args = ap.parse_args()

    if args.record == "corrected":
        rec_path = OUT.parent / "v1fix_probe/canonical_sideA_v1fixed.npz"
        baseline = 0.484
    else:
        rec_path = ROOT / "artifacts/final_wm/canonical_sideA.npz"
        baseline = 0.723
    print(f"[record] {args.record} -> {rec_path.name} (unscheduled baseline "
          f"H18 {baseline})", flush=True)

    record = CanonicalRecord(rec_path)
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
        tag = f"lpv_{'uaphys' if args.ua_physics else ('pinned' if args.pin_prior else 'free')}_{args.record}"
        out_dir = OUT.parent / f"{tag}_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{tag}] training (compile_substep disabled: subclass hook; "
              f"pin_prior={args.pin_prior})", flush=True)
        sys.path.insert(0, str(Path(__file__).parent))
        from probe_guard import assert_grid, verify_ledger_properties
        assert_grid(props)
        # train_arm builds the model internally; patch the builder so the arm
        # gets the scheduled transition.
        import src.final_wm.training as T
        orig = T.build_world_model

        def builder(sp, pr, **kw):
            m = _promote(orig(sp, pr, **kw), 1.0)
            if args.pin_prior:
                pin_tau_to_prior(m)
            if args.ua_physics:
                fix_alphas_to_physics(m, a_tau=0.0, a_ua=0.8)
            return m

        T.build_world_model = builder
        try:
            final = train_arm(spec, record, out_dir, device=DEVICE, properties=props,
                              compile_substep=False)
        finally:
            T.build_world_model = orig
        verify_ledger_properties(out_dir)
        model = build(spec, props, 1.0).to(DEVICE)
        model.load_state_dict(torch.load(out_dir / "checkpoints" / f"{final['run_id']}.pt",
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
        print(f"[{tag}] H18 ch4 overall={np.mean(ch4['bin_means']):.3f} "
              f"bins={[round(x, 3) for x in ch4['bin_means']]} | "
              f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']} | "
              f"learned alpha_tau={a_tau.item():.3f} alpha_ua={a_ua.item():.3f}",
              flush=True)
        (out_dir / "report.json").write_text(json.dumps(
            {"arm": tag, "pin_prior": bool(args.pin_prior),
             "train": {k: final[k] for k in ("best_val_nll", "best_epoch",
                                             "epochs_run", "stop_reason")},
             "eval": {"overall_h18_mae": float(np.mean(ch4["bin_means"])),
                      "bins_q1q5": ch4["bin_means"]},
             "alphas": {"alpha_tau": float(a_tau.item()),
                        "alpha_ua": float(a_ua.item())}}, indent=2))
        print("done")


if __name__ == "__main__":
    main()
