"""R1 direction gate, v0.3 criteria (2026-08-27).

WHY THIS EXISTS
---------------
My 2026-08-26 direction probe (`direction_v2_test.py`) gated on
`frac_negative == 1.000`.  That is the SUPERSEDED v0.2 rule.  The frozen matrix
v0.3 (effective 2026-08-20, doc section 5.1) rules that criterion a calibration
error -- the real plant only achieves 0.68 (up, n=22) / 0.75 (down, n=48)
correct-direction fraction over 60 steps -- and replaces it with:

    mean terminal dT < 0
    AND day-block bootstrap CI entirely below 0
    AND correct-direction fraction >= 0.60
    AND report BOTH H18 and H60

External audit also flagged (point 3) that the step probe bypasses
`counterfactual()` and never reports `in_support`, and that support boxes were
built by flattening the whole batch so one trajectory could borrow another
trajectory's action range.  Protocol line 131: "counterfactual metrics are only
computed inside the action support domain; out-of-support steps must be reported
with in_support=False".

So this probe additionally builds the support box PER WINDOW from that window's
own history actions (margin = support_margin = 0.05) and reports the in-support
fraction, plus the gate recomputed on the in-support subset only.

Eval-only: no training, uses the grid-properties checkpoints already on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.final_wm.contracts import action_support_from_history          # noqa: E402
from src.final_wm.data import CanonicalRecord, sample_windows           # noqa: E402
from src.final_wm.properties import load_grid_properties                # noqa: E402
from src.final_wm.training import build_world_model                     # noqa: E402
from experiments.final_wm import matrix_spec as ms                      # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent / "direction_v03"
OUT.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 256          # same count as the probe MAE metric
SEED = 50_000            # same seed family as the probe MAE metric
DELTA_V = 0.05           # protocol step size
SUPPORT_MARGIN = 0.05    # contracts.py default
N_BOOT = 1000

# grid-properties arms only -- the analytic-properties arms are void
ARMS = {
    "baseline_unsched": ("v1fix_probe/v1fix_unanchored", "canonical_sideA_v1fixed.npz"),
    "lpv_free":         ("lpv_free_corrected_probe",     "canonical_sideA_v1fixed.npz"),
    "lpv_pinned":       ("lpv_pinned_corrected_probe",   "canonical_sideA_v1fixed.npz"),
    "a1_uaphys":        ("lpv_uaphys_corrected_probe",   "canonical_sideA_v1fixed.npz"),
    "armA_oldrec":      ("retrain_probe/armA_budget",    None),
}
CKPT = "t1_closure_cons_norew_seed0.pt"


def day_block_ci(values: torch.Tensor, day_ids: torch.Tensor, *, n_boot=N_BOOT, seed=0):
    """One-sample UTC-day block bootstrap CI on the mean of `values`.

    Same resampling scheme as evaluation.relative_improvement_ci: collapse to a
    per-day mean, then resample DAYS with replacement.
    """
    days = torch.unique(day_ids)
    if len(days) < 2:
        return None
    by_day = torch.stack([values[day_ids == d].mean() for d in days])
    gen = torch.Generator().manual_seed(seed)
    boots = torch.tensor([
        float(by_day[torch.randint(len(by_day), (len(by_day),), generator=gen)].mean())
        for _ in range(n_boot)
    ])
    return {
        "point": float(by_day.mean()),
        "ci_lo": float(torch.quantile(boots, 0.025)),
        "ci_hi": float(torch.quantile(boots, 0.975)),
        "n_days": int(len(days)),
    }


@torch.no_grad()
def direction_gate(model, record, *, valve_index: int, horizon: int, tail: int):
    gen = torch.Generator().manual_seed(SEED)
    batch = sample_windows(record, 1, N_WINDOWS, 96, 1, gen)
    b0 = batch.future_boundary[:, 0].to(DEVICE)
    a0 = batch.future_actions[:, 0].to(DEVICE)
    o0 = batch.history.obs[:, -1].to(DEVICE)
    state0 = model.transition.initial_steady_state(b0, a0, o0)

    bseq = b0.unsqueeze(1).repeat(1, horizon, 1)
    base = a0.unsqueeze(1).repeat(1, horizon, 1)
    step = base.clone()
    step[:, :, valve_index] = (step[:, :, valve_index] + DELTA_V).clamp(max=1.0)

    _s, t_base = model.transition.integrate(state0, bseq, base)
    _s, t_step = model.transition.integrate(state0, bseq, step)
    d = (t_step[:, -tail:, 4] - t_base[:, -tail:, 4]).mean(dim=1).cpu()

    # PER-WINDOW support box from that window's own history actions
    hist_act = batch.history.actions              # (B, 96, 2)
    in_support = []
    for i in range(hist_act.shape[0]):
        sup = action_support_from_history(hist_act[i], SUPPORT_MARGIN)
        in_support.append(bool(sup.contains(step[i, 0].cpu())))
    in_support = torch.tensor(in_support)

    def gate(mask):
        v, dd = d[mask], batch.day_ids[mask]
        if v.numel() == 0:
            return None
        ci = day_block_ci(v, dd)
        frac = float((v < 0).float().mean())
        mean = float(v.mean())
        passed = (mean < 0) and (ci is not None and ci["ci_hi"] < 0) and (frac >= 0.60)
        return {"n": int(v.numel()), "mean_delta_c": mean, "frac_negative": frac,
                "ci": ci, "gate_pass": bool(passed)}

    return {
        "all_windows": gate(torch.ones_like(in_support, dtype=torch.bool)),
        "in_support_only": gate(in_support),
        "in_support_frac": float(in_support.float().mean()),
    }


def main():
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
    P = Path(__file__).resolve().parent
    report = {"criteria": "v0.3: mean<0 AND day-block CI entirely<0 AND frac>=0.60; H18+H60",
              "real_plant_reference": {"up": 0.68, "down": 0.75, "source": "matrix v0.3 s5.1"},
              "arms": {}}
    for name, (rel, rec_name) in ARMS.items():
        ck = P / rel / "checkpoints" / CKPT
        if not ck.exists():
            cands = list((P / rel / "checkpoints").glob("*.pt")) if (P / rel / "checkpoints").exists() else []
            if not cands:
                print(f"[skip] {name}: no checkpoint under {rel}", flush=True)
                continue
            ck = cands[0]
        rec_path = (P / "v1fix_probe" / rec_name) if rec_name else (ROOT / "artifacts/final_wm/canonical_sideA.npz")
        if not rec_path.exists():
            print(f"[skip] {name}: record missing {rec_path}", flush=True)
            continue
        record = CanonicalRecord(rec_path)
        spec = ms._base("t1", "closure_cons_norew", 0)
        model = build_world_model(spec, props).to(DEVICE)
        sd = torch.load(ck, map_location=DEVICE, weights_only=False)["state_dict"]
        if "transition.alpha_tau_raw" in sd:
            # LPV arms carry the two schedule-magnitude parameters; promote the
            # transition to the scheduled subclass so the state dict matches AND
            # the scheduled _substep hook is actually active during the rollout.
            import importlib.util
            spec_l = importlib.util.spec_from_file_location(
                "lpvp", P / "lpv_schedule_probe.py")
            lpv = importlib.util.module_from_spec(spec_l)
            spec_l.loader.exec_module(lpv)
            model = lpv._promote(model, 1.0)
        model.load_state_dict(sd)
        entry = {}
        for valve, vname in ((0, "v1"), (1, "v2")):
            for horizon, tail, hname in ((18, 3, "H18"), (60, 10, "H60")):
                entry[f"{vname}_{hname}"] = direction_gate(
                    model, record, valve_index=valve, horizon=horizon, tail=tail)
        report["arms"][name] = entry
        print(f"[{name}]", flush=True)
        for k, v in entry.items():
            a, s = v["all_windows"], v["in_support_only"]
            ci = a["ci"]
            print("  %-8s mean=%+.4f frac=%.3f CI=[%+.4f,%+.4f] n_days=%d  GATE=%s | in_supp=%.1f%% (frac=%.3f GATE=%s)"
                  % (k, a["mean_delta_c"], a["frac_negative"], ci["ci_lo"], ci["ci_hi"], ci["n_days"],
                     "PASS" if a["gate_pass"] else "FAIL",
                     100 * v["in_support_frac"],
                     s["frac_negative"] if s else float("nan"),
                     ("PASS" if s["gate_pass"] else "FAIL") if s else "n/a"), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\nwritten", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
