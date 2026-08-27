"""Direction probe against the FACTUAL trajectory (2026-08-27).

WHY (user correction, and it is correct)
----------------------------------------
The frozen-boundary step probe and the plant event study measure DIFFERENT
quantities:

  plant 0.68/0.75  =  P(dT<0 | the valve moved)          OBSERVATIONAL
                      analysis.py:73 only excludes contamination from the OTHER
                      VALVE; load / fuel / pressure / feedwater move freely, and
                      in closed loop the valve moves BECAUSE temperature drifted
                      -> 25-32% "wrong sign" events are common-cause dominated.

  our step probe    =  P(dT<0 | do(valve + 0.05), boundary held fixed)   INTERVENTIONAL
                      ceteris paribus; spray always cools by energy balance.

Requiring the interventional fraction to match the observational one is a
category error -- exactly the confounding this project exists to avoid.  So the
v0.3 rationale ("100% is stricter than the real object's observable behaviour")
does not license relaxing an interventional gate, and my earlier reading of
frac_negative = 1.000 as "suspicious over-determinism" is RETRACTED.

WHAT THIS PROBE DOES INSTEAD
----------------------------
Compare against the ORIGINAL trajectory, with the boundary on its FACTUAL path:

  factual   : integrate(state0, TRUE boundary_seq, TRUE actions)
  cf        : integrate(state0, TRUE boundary_seq, TRUE actions + step on one valve)
  direction : cf - factual   (isolates the valve, boundary follows what really happened)

Also reported:
  * fidelity of the factual model rollout vs the RECORDED observations (sanity:
    if the factual rollout is wrong, the counterfactual contrast is meaningless);
  * EFFECTIVE step size after clamping to [0,1] -- windows already at the valve
    limit receive no intervention and must not be counted as evidence;
  * per-window action-support check (own history only, margin 0.05);
  * day-block bootstrap CI, H18 and H60.
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
P = Path(__file__).resolve().parent
OUT = P / "direction_factual"
OUT.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 256
SEED = 50_000
DELTA_V = 0.05
SUPPORT_MARGIN = 0.05
N_BOOT = 1000
MIN_EFFECTIVE = 0.04     # below this the clamp ate the intervention

ARMS = {
    "baseline_unsched": "v1fix_probe/v1fix_unanchored",
    "lpv_free":         "lpv_free_corrected_probe",
    "a1_uaphys":        "lpv_uaphys_corrected_probe",
}
CKPT = "t1_closure_cons_norew_seed0.pt"
REC = P / "v1fix_probe" / "canonical_sideA_v1fixed.npz"


def day_block_ci(values, day_ids, *, n_boot=N_BOOT, seed=0):
    days = torch.unique(day_ids)
    if len(days) < 2:
        return None
    by_day = torch.stack([values[day_ids == d].mean() for d in days])
    gen = torch.Generator().manual_seed(seed)
    boots = torch.tensor([
        float(by_day[torch.randint(len(by_day), (len(by_day),), generator=gen)].mean())
        for _ in range(n_boot)
    ])
    return {"point": float(by_day.mean()),
            "ci_lo": float(torch.quantile(boots, 0.025)),
            "ci_hi": float(torch.quantile(boots, 0.975)),
            "n_days": int(len(days))}


@torch.no_grad()
def run(model, record, *, valve_index, horizon, tail):
    gen = torch.Generator().manual_seed(SEED)
    batch = sample_windows(record, 1, N_WINDOWS, 96, horizon, gen)
    bnd = batch.future_boundary.to(DEVICE)          # (B,H,7) TRUE path
    act = batch.future_actions.to(DEVICE)           # (B,H,2) TRUE actions
    o0 = batch.history.obs[:, -1].to(DEVICE)
    state0 = model.transition.initial_steady_state(bnd[:, 0], act[:, 0], o0)

    step = act.clone()
    stepped = (step[:, :, valve_index] + DELTA_V).clamp(min=0.0, max=1.0)
    effective = (stepped - act[:, :, valve_index]).mean(dim=1)      # (B,)
    step[:, :, valve_index] = stepped

    _s, t_fact = model.transition.integrate(state0, bnd, act)
    _s, t_cf = model.transition.integrate(state0, bnd, step)

    d = (t_cf[:, -tail:, 4] - t_fact[:, -tail:, 4]).mean(dim=1).cpu()
    eff = effective.cpu()

    # fidelity of the factual rollout against the RECORDED observations
    fid = (t_fact[:, :, 4].cpu() - batch.future_obs[:, :, 4]).abs().mean(dim=1)

    hist_act = batch.history.actions
    in_sup = torch.tensor([
        bool(action_support_from_history(hist_act[i], SUPPORT_MARGIN).contains(step[i, 0].cpu()))
        for i in range(hist_act.shape[0])
    ])
    usable = in_sup & (eff >= MIN_EFFECTIVE)

    def gate(mask, label):
        v, dd = d[mask], batch.day_ids[mask]
        if v.numel() < 2:
            return {"label": label, "n": int(v.numel()), "note": "too few windows"}
        ci = day_block_ci(v, dd)
        frac = float((v < 0).float().mean())
        mean = float(v.mean())
        return {"label": label, "n": int(v.numel()), "mean_delta_c": mean,
                "frac_negative": frac, "ci": ci,
                "gate_pass_v03": bool(mean < 0 and ci and ci["ci_hi"] < 0 and frac >= 0.60)}

    return {
        "all": gate(torch.ones_like(usable), "all windows"),
        "usable": gate(usable, "in-support AND effective step"),
        "in_support_frac": float(in_sup.float().mean()),
        "effective_step_mean": float(eff.mean()),
        "clamped_out_frac": float((eff < MIN_EFFECTIVE).float().mean()),
        "factual_fidelity_mae_c": float(fid.mean()),
    }


def main():
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
    record = CanonicalRecord(REC)
    report = {"design": "counterfactual vs FACTUAL trajectory, TRUE boundary path",
              "note": "interventional quantity; plant 0.68/0.75 is observational and NOT comparable",
              "arms": {}}
    for name, rel in ARMS.items():
        ck = P / rel / "checkpoints" / CKPT
        if not ck.exists():
            print(f"[skip] {name}", flush=True)
            continue
        spec = ms._base("t1", "closure_cons_norew", 0)
        model = build_world_model(spec, props).to(DEVICE)
        sd = torch.load(ck, map_location=DEVICE, weights_only=False)["state_dict"]
        if "transition.alpha_tau_raw" in sd:
            import importlib.util
            sl = importlib.util.spec_from_file_location("lpvp", P / "lpv_schedule_probe.py")
            lpv = importlib.util.module_from_spec(sl)
            sl.loader.exec_module(lpv)
            model = lpv._promote(model, 1.0)
        model.load_state_dict(sd)
        model.eval()
        entry = {}
        for valve, vn in ((0, "v1"), (1, "v2")):
            for horizon, tail, hn in ((18, 3, "H18"), (60, 10, "H60")):
                entry[f"{vn}_{hn}"] = run(model, record, valve_index=valve,
                                          horizon=horizon, tail=tail)
        report["arms"][name] = entry
        print(f"\n[{name}]  factual-rollout fidelity (final ch, H60) = %.3f degC"
              % entry["v2_H60"]["factual_fidelity_mae_c"], flush=True)
        for k, v in entry.items():
            a, u = v["all"], v["usable"]
            print("  %-8s ALL n=%3d mean=%+.4f frac=%.3f CI=[%+.4f,%+.4f] | USABLE n=%3d mean=%+.4f frac=%.3f %s"
                  % (k, a["n"], a["mean_delta_c"], a["frac_negative"],
                     a["ci"]["ci_lo"], a["ci"]["ci_hi"],
                     u["n"], u.get("mean_delta_c", float("nan")), u.get("frac_negative", float("nan")),
                     "PASS" if u.get("gate_pass_v03") else "FAIL"), flush=True)
            print("           in_support=%.1f%%  eff_step=%.4f  clamped_out=%.1f%%"
                  % (100 * v["in_support_frac"], v["effective_step_mean"],
                     100 * v["clamped_out_frac"]), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\nwritten", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
