"""Direction probe on the THREE historical matrix arms, corrected protocol (2026-08-27).

User request: rerun the direction check on
  physics_only          -- pure physics, no closure
  closure_cons          -- physics + conservative closure, REWETTING INTACT
  closure_cons_norew    -- same closure, rewetting ablated (aW = 0)
with the fixed protocol, i.e.

  * counterfactual vs the FACTUAL trajectory (not two synthetic rollouts),
  * boundary follows its TRUE path over the whole horizon (no freezing at t0),
  * v0.3 gate: mean < 0 AND day-block bootstrap CI entirely < 0 AND frac >= 0.60,
  * H18 and H60 both reported,
  * per-window action support (own history only, margin 0.05),
  * effective step size after clamping reported,
  * ALL THREE SEEDS (these arms have seeds 0/1/2 on disk, unlike the probe arms).

Historical context: the v0.2 R1 unit FAILED on direction with frac_negative
0.19-0.34, and ablating the rewetting term moved correct-direction 0.12 -> 0.94.
This probe re-measures that on the corrected protocol.

Arms trained on the OLD canonical record (grid properties, matrix run), so this
uses artifacts/final_wm/canonical_sideA.npz -- NOT the corrected v2.1 record.
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
OUT = P / "direction_historical"
OUT.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 256
SEED = 50_000
DELTA_V = 0.05
SUPPORT_MARGIN = 0.05
N_BOOT = 1000
MIN_EFFECTIVE = 0.04

# (arm label, closure_mode passed to matrix_spec, checkpoint stem)
ARMS = [
    ("physics_only",       "physics_only",       "t1_physics_only"),
    ("closure_cons_REWET", "conservative",       "t1_closure_cons"),
    ("closure_cons_norew", "conservative_norew", "t1_closure_cons_norew"),
]
SEEDS = (0, 1, 2)
CKPT_DIR = ROOT / "artifacts/final_wm/checkpoints"
REC = ROOT / "artifacts/final_wm/canonical_sideA.npz"


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
    bnd = batch.future_boundary.to(DEVICE)
    act = batch.future_actions.to(DEVICE)
    o0 = batch.history.obs[:, -1].to(DEVICE)
    state0 = model.transition.initial_steady_state(bnd[:, 0], act[:, 0], o0)

    step = act.clone()
    stepped = (step[:, :, valve_index] + DELTA_V).clamp(min=0.0, max=1.0)
    eff = (stepped - act[:, :, valve_index]).mean(dim=1).cpu()
    step[:, :, valve_index] = stepped

    _s, t_fact = model.transition.integrate(state0, bnd, act)
    _s, t_cf = model.transition.integrate(state0, bnd, step)
    d = (t_cf[:, -tail:, 4] - t_fact[:, -tail:, 4]).mean(dim=1).cpu()
    fid = (t_fact[:, :, 4].cpu() - batch.future_obs[:, :, 4]).abs().mean(dim=1)

    hist_act = batch.history.actions
    in_sup = torch.tensor([
        bool(action_support_from_history(hist_act[i], SUPPORT_MARGIN).contains(step[i, 0].cpu()))
        for i in range(hist_act.shape[0])
    ])
    usable = in_sup & (eff >= MIN_EFFECTIVE)
    v, dd = d[usable], batch.day_ids[usable]
    ci = day_block_ci(v, dd) if v.numel() >= 2 else None
    frac = float((v < 0).float().mean()) if v.numel() else float("nan")
    mean = float(v.mean()) if v.numel() else float("nan")
    return {"n": int(v.numel()), "mean_delta_c": mean, "frac_negative": frac, "ci": ci,
            "gate_pass_v03": bool(v.numel() and mean < 0 and ci and ci["ci_hi"] < 0 and frac >= 0.60),
            "in_support_frac": float(in_sup.float().mean()),
            "clamped_out_frac": float((eff < MIN_EFFECTIVE).float().mean()),
            "factual_fidelity_mae_c": float(fid.mean())}


def main():
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
    record = CanonicalRecord(REC)
    report = {"record": str(REC), "protocol": "CF vs factual, TRUE boundary path, v0.3 gate",
              "history": "v0.2 R1 failed direction at frac 0.19-0.34; aW=0 moved 0.12->0.94",
              "arms": {}}
    for label, closure_mode, stem in ARMS:
        report["arms"][label] = {}
        print(f"\n=== {label} ({closure_mode}) ===", flush=True)
        for seed in SEEDS:
            ck = CKPT_DIR / f"{stem}_seed{seed}.pt"
            if not ck.exists():
                print(f"  seed{seed}: checkpoint missing", flush=True)
                continue
            spec = ms._base("t1", closure_mode, seed)
            model = build_world_model(spec, props).to(DEVICE)
            payload = torch.load(ck, map_location=DEVICE, weights_only=False)
            sd = payload.get("state_dict", payload)
            try:
                model.load_state_dict(sd)
            except RuntimeError as exc:
                print(f"  seed{seed}: load failed -> {str(exc)[:120]}", flush=True)
                continue
            model.eval()
            entry = {}
            for valve, vn in ((0, "v1"), (1, "v2")):
                for horizon, tail, hn in ((18, 3, "H18"), (60, 10, "H60")):
                    entry[f"{vn}_{hn}"] = run(model, record, valve_index=valve,
                                              horizon=horizon, tail=tail)
            report["arms"][label][f"seed{seed}"] = entry
            f = entry["v2_H60"]
            print("  seed%d fid=%.2fdegC | v2_H60 mean=%+.4f frac=%.3f CI=[%+.4f,%+.4f] %s"
                  % (seed, f["factual_fidelity_mae_c"], f["mean_delta_c"], f["frac_negative"],
                     f["ci"]["ci_lo"], f["ci"]["ci_hi"], "PASS" if f["gate_pass_v03"] else "FAIL"),
                  flush=True)
            for k in ("v1_H18", "v1_H60", "v2_H18"):
                e = entry[k]
                print("         %-7s mean=%+.4f frac=%.3f %s"
                      % (k, e["mean_delta_c"], e["frac_negative"],
                         "PASS" if e["gate_pass_v03"] else "FAIL"), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\nwritten", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
