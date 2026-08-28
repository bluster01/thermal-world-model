"""A4 — load-scheduled fuel->metal gain k (2026-08-28, user-ordered).

PREREG s2-A4: Fan2021 Eq.24 schedules k1 (fuel->metal heating gain) as a load
polynomial. Ours: multiply the existing wet/dry pressure-blended k (B,3) by a
BOUNDED load factor exp(tanh(w*(s-1))*ln(k_max)), k_max=1.5 (+-50% band, does
not violate physics). w is the single learnable parameter, starts at 0
(factor=1, schedule off -- nested-init discipline). Sign is free (data may
choose k increasing OR decreasing with load).

Pre-registered support: single water-coal ratio a=4.485 gave energy-mismatch
bin means swinging -43.1 (Q1) .. +107.2 (Q5) t/h -> systematic load-correlated
heat-input bias exists.

Injection: _substep receives k_t (B,3) at position 19 and d_flow at 9; scale
k_t by factor (B,) -> (B,3). No src change; identity gate at w=0.
"""
from __future__ import annotations

import argparse
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
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "a4_k_schedule_probe"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
N_WIN, EVAL_SEED = 256, 50_000
FLOW_REF = 300.0
K_MAX = 1.5
torch.backends.cuda.matmul.allow_tf32 = True


class KScheduledTransition(Fan2020UDETransition):
    """k_t *= exp(tanh(w*(s-1))*ln(k_max)): bounded +-50% load schedule on the
    fuel->metal gain. w=0 reproduces the parent exactly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_w_raw = nn.Parameter(torch.tensor(0.0))

    def _k_factor(self, d_flow: torch.Tensor) -> torch.Tensor:
        s = (d_flow / FLOW_REF).clamp(min=0.1, max=3.0)
        return torch.exp(torch.tanh(self.k_w_raw * (s - 1.0)) * float(np.log(K_MAX)))

    def _substep(self, h, tm, rb, m1, m2, lag1, lag2, dsw1, dsw2, d_flow, u_b,
                 p_stack, p0, p1, h_spray, h_sep, m_cap, ua, cm, k_t,
                 tau_b, tau_evap, tau_mix1, tau_mix2, dt_sub, h_lo, h_hi,
                 steam_power, metal_power):
        k_t = k_t * self._k_factor(d_flow).unsqueeze(-1)
        return super()._substep(h, tm, rb, m1, m2, lag1, lag2, dsw1, dsw2,
                                d_flow, u_b, p_stack, p0, p1, h_spray, h_sep,
                                m_cap, ua, cm, k_t, tau_b, tau_evap, tau_mix1,
                                tau_mix2, dt_sub, h_lo, h_hi,
                                steam_power, metal_power)


def _promote(model):
    tr = model.transition
    dev = next(tr.parameters()).device
    tr.__class__ = KScheduledTransition
    if "k_w_raw" not in tr._parameters:
        tr.register_parameter("k_w_raw", nn.Parameter(torch.tensor(0.0, device=dev)))
    return model


def build(spec, props):
    return _promote(build_world_model(spec, props))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()

    rec_path = P / "v1fix_probe/canonical_sideA_v1fixed.npz"
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
        sched = build(spec, props).to(DEVICE).eval()
        sched.load_state_dict(sd, strict=False)
        gen = torch.Generator().manual_seed(7)
        b = sample_windows(record, SPLIT_TRAIN, 8, 96, 18, gen)
        hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                   actions=b.history.actions.to(DEVICE),
                                   boundary=b.history.boundary.to(DEVICE))
        fa, fb = b.future_actions.to(DEVICE), b.future_boundary.to(DEVICE)
        with torch.no_grad():
            r0 = base.forecast(hist, fa, boundary_mode="oracle",
                               true_future_boundary=fb)
            r1 = sched.forecast(hist, fa, boundary_mode="oracle",
                                true_future_boundary=fb)
        print(f"[sanity w=0] max_abs_diff = {(r0.temps_mu - r1.temps_mu).abs().max().item():.3e}"
              f" -> {'IDENTICAL' if (r0.temps_mu == r1.temps_mu).all() else 'MISMATCH'}")
        sched.transition.k_w_raw.data.fill_(2.0)
        with torch.no_grad():
            r2 = sched.forecast(hist, fa, boundary_mode="oracle",
                                true_future_boundary=fb)
        print(f"[w=2 active] max_abs_diff vs base = "
              f"{(r0.temps_mu - r2.temps_mu).abs().max().item():.3f} degC (must be > 0)")
        flows = b.future_boundary[:, :, STEAM_FLOW_INDEX]
        s = (flows / FLOW_REF).clamp(0.1, 3.0)
        f = torch.exp(torch.tanh(torch.tensor(2.0) * (s - 1.0)) * np.log(K_MAX))
        print(f"  flow {flows.min():.0f}-{flows.max():.0f} kg/s -> k factor "
              f"{f.min():.2f}-{f.max():.2f} (band [1/1.5, 1.5])")
        return

    if args.train:
        sys.path.insert(0, str(P))
        from probe_guard import assert_grid, verify_ledger_properties
        assert_grid(props)
        import src.final_wm.training as T
        orig = T.build_world_model

        def builder(sp, pr, **kw):
            return _promote(orig(sp, pr, **kw))

        T.build_world_model = builder
        try:
            final = train_arm(spec, record, OUT, device=DEVICE, properties=props,
                              compile_substep=False)
        finally:
            T.build_world_model = orig
        verify_ledger_properties(OUT)
        model = build(spec, props).to(DEVICE)
        model.load_state_dict(torch.load(
            OUT / "checkpoints" / f"{final['run_id']}.pt", map_location=DEVICE,
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
        w = float(model.transition.k_w_raw)
        flows = torch.cat(loads)
        s = (flows / FLOW_REF).clamp(0.1, 3.0)
        f = torch.exp(torch.tanh(torch.tensor(w) * (s - 1.0)) * np.log(K_MAX))
        print(f"[a4_k] H18 ch4 overall={np.mean(ch4['bin_means']):.3f} "
              f"bins={[round(x, 3) for x in ch4['bin_means']]} | "
              f"best_val={final['best_val_nll']:.3f}@{final['best_epoch']} | "
              f"w={w:+.3f} k_factor_range=[{f.min():.3f},{f.max():.3f}]", flush=True)
        (OUT / "report.json").write_text(json.dumps(
            {"arm": "a4_k_schedule",
             "train": {k: final[k] for k in ("best_val_nll", "best_epoch",
                                             "epochs_run", "stop_reason")},
             "eval": {"overall_h18_mae": float(np.mean(ch4["bin_means"])),
                      "bins_q1q5": ch4["bin_means"]},
             "k_schedule": {"w": w, "k_max": K_MAX,
                            "factor_min": float(f.min()),
                            "factor_max": float(f.max())}}, indent=2))
        print("done")


if __name__ == "__main__":
    main()
