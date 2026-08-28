"""Three-arm direction re-run (2026-08-28, user-ordered; the "可以跑吧" batch).

Arms = the ORIGINAL three groups, re-checked now that the aW resurrection
issue is fixed (training.py calls model.forecast(...), not model(...), so the
anchor's forward_pre_hook never fired and anchored arms stayed at aW=-30;
verified revive: raw -30 -> -2.0, a_w=19.04 kW/K, grad 0.119):

  1. phy_con         t1_closure_cons_seed0       old record, grid props,
                     closure 'conservative'      aW learned (raw~0.54 -> aW=1.0)
  2. phy_con_norew   t1_closure_cons_norew_seed0 old record, grid props,
                     'conservative_norew'        aW=-30 by design (ablation)
  3. rewet_live_grid (corrected v2.1 record, grid props, anchored + revived
                     aW raw=-2, live gradient)   trained by rewet_live_probe.py
                     into rewet_live_grid_probe/ (the sibling dir carrying the
                     Analytic-props ledger is VOID, 08-26 defect)

Protocol = repo step_response_direction (+0.05 valve, 60 steps, terminal
10-step mean) + v0.3 gate statistics (mean<0 AND day-block CI entirely <0
AND frac>=0.60; H18/H60 both reported) + per-window action-support box +
load stratification Q1-Q5 (prior: low-load amplitude ~2x high-load).

Eval-only: frozen grid-properties checkpoints, no training, no src changes.
The gate/traversal machinery is a deliberate sibling-copy of
direction_gate_v03.py (independent reimplementation = the check).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.final_wm.contracts import BOUNDARY_ELEMENTS, action_support_from_history  # noqa: E402
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows        # noqa: E402
from src.final_wm.properties import load_grid_properties                       # noqa: E402
from src.final_wm.training import build_world_model                            # noqa: E402
from experiments.final_wm import matrix_spec as ms                             # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = Path(__file__).resolve().parent
OUT = P / "direction_three_arms"
OUT.mkdir(parents=True, exist_ok=True)
IDX_FLOW = BOUNDARY_ELEMENTS.index("steam_flow")

N_WINDOWS = 256        # same count as the probe MAE metric
SEED = 50_000          # same seed family as the probe MAE metric
DELTA_V = 0.05         # protocol step size
SUPPORT_MARGIN = 0.05  # contracts.py default
N_BOOT = 1000

# arm -> (record npz, checkpoint, closure arm tag, closure_mode)
ARMS = {
    "phy_con": (
        ROOT / "artifacts/final_wm/canonical_sideA.npz",
        ROOT / "artifacts/final_wm/checkpoints/t1_closure_cons_seed0.pt",
        "closure_cons", "conservative"),
    "phy_con_norew": (
        ROOT / "artifacts/final_wm/canonical_sideA.npz",
        ROOT / "artifacts/final_wm/checkpoints/t1_closure_cons_norew_seed0.pt",
        "closure_cons_norew", "conservative_norew"),
    "rewet_live_grid": (
        P / "v1fix_probe/canonical_sideA_v1fixed.npz",
        P / "rewet_live_grid_probe/checkpoints/t1_closure_cons_seed0.pt",
        "closure_cons", "conservative"),
    "norew_corrected": (   # corrected-record norew baseline for the arm-3 comparison
        P / "v1fix_probe/canonical_sideA_v1fixed.npz",
        P / "v1fix_probe/v1fix_unanchored/checkpoints/t1_closure_cons_norew_seed0.pt",
        "closure_cons_norew", "conservative_norew"),
}


def day_block_ci(values: torch.Tensor, day_ids: torch.Tensor, *, n_boot=N_BOOT, seed=0):
    """One-sample UTC-day block bootstrap CI on the mean (same scheme as
    evaluation.relative_improvement_ci / direction_gate_v03)."""
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
def direction_probe(model, record, *, valve_index: int, horizon: int, tail: int):
    """One (valve, horizon) cell: gate stats + per-window d/flow/day/support."""
    gen = torch.Generator().manual_seed(SEED)
    batch = sample_windows(record, SPLIT_VAL, N_WINDOWS, 96, 1, gen)
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

    hist_act = batch.history.actions
    in_support = torch.zeros(N_WINDOWS, dtype=torch.bool)
    for i in range(N_WINDOWS):
        sup = action_support_from_history(hist_act[i], SUPPORT_MARGIN)
        in_support[i] = bool(sup.contains(step[i, 0].cpu()))

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
        "all_windows": gate(torch.ones(N_WINDOWS, dtype=torch.bool)),
        "in_support_only": gate(in_support),
        "in_support_frac": float(in_support.float().mean()),
        "_per_window": {"d": d.numpy(), "flow": b0[:, IDX_FLOW].cpu().numpy()},
    }


def load_bins(d, flow):
    e = np.quantile(flow, [0, .2, .4, .6, .8, 1.0])
    out = []
    for i in range(5):
        m = (flow >= e[i]) & (flow <= e[i + 1] if i == 4 else flow < e[i + 1])
        out.append({
            "q": i + 1,
            "n": int(m.sum()),
            "mean_delta_c": float(d[m].mean()) if m.sum() else None,
            "frac_negative": float((d[m] < 0).mean()) if m.sum() else None,
        })
    return out


def main():
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz",
                                 device=DEVICE)
    report = {
        "protocol": ("repo step_response_direction (+0.05 valve, 60 steps, terminal 10-step mean) "
                     "+ v0.3 gate (mean<0 AND day-block CI<0 AND frac>=0.60), H18/H60, "
                     "per-window support box, Q1-Q5 load bins; 256 windows seed 50k"),
        "real_plant_reference": {"up": 0.68, "down": 0.75, "source": "matrix v0.3 s5.1"},
        "arms": {},
    }
    for name, (rec_path, ckpt, arm, cm) in ARMS.items():
        if not ckpt.exists():
            print(f"[skip] {name}: checkpoint not yet trained ({ckpt.name})", flush=True)
            continue
        record = CanonicalRecord(rec_path)
        spec = ms._base("t1", arm, 0, boundary_mode="oracle",
                        initial_state_mode="hybrid", closure_mode=cm,
                        epochs=120, patience=20)
        model = build_world_model(spec, props).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                         weights_only=False)["state_dict"])
        model.eval()
        raw = model.transition.raw
        aw = {n: float(getattr(raw, n)) for n in ("aW1", "aW2") if hasattr(raw, n)}
        aw_val = {n: float(model.transition.val(n)) for n in ("aW1", "aW2")
                  if hasattr(raw, n)}
        print(f"[{name}] aW raw={aw} -> kW/K {aw_val}", flush=True)

        entry: dict = {"aw_raw": aw, "aw_kW_per_K": aw_val}
        for valve, vname in ((0, "v1"), (1, "v2")):
            for horizon, tail, hname in ((18, 3, "H18"), (60, 10, "H60")):
                cell = direction_probe(model, record, valve_index=valve,
                                       horizon=horizon, tail=tail)
                key = f"{vname}_{hname}"
                entry[key] = {k: v for k, v in cell.items()
                              if not k.startswith("_")}
                a, s = cell["all_windows"], cell["in_support_only"]
                ci = a["ci"]
                print("  %-8s mean=%+.4f frac=%.3f CI=[%+.4f,%+.4f] n_days=%d GATE=%s"
                      " | in_supp=%.1f%% (frac=%.3f GATE=%s)"
                      % (key, a["mean_delta_c"], a["frac_negative"],
                         ci["ci_lo"], ci["ci_hi"], ci["n_days"],
                         "PASS" if a["gate_pass"] else "FAIL",
                         100 * cell["in_support_frac"],
                         s["frac_negative"] if s else float("nan"),
                         ("PASS" if s["gate_pass"] else "FAIL") if s else "n/a"),
                      flush=True)
                if hname == "H60":
                    bins = load_bins(*cell["_per_window"].values())
                    entry[f"{vname}_H60_by_load"] = bins
                    row = "    " + " ".join(
                        f"Q{b['q']} {b['mean_delta_c']:+.4f}/{b['frac_negative']:.2f}"
                        if b["mean_delta_c"] is not None else f"Q{b['q']} n/a"
                        for b in bins)
                    print(row, flush=True)
        report["arms"][name] = entry
        del model
        torch.cuda.empty_cache()

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\nwritten", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
