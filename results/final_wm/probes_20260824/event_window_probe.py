"""Model response measured ON the plant's actual v2 step events (2026-08-27).

Protocol kept identical to direction_factual_probe.py (the "probe standard"):
  * initial state from initial_steady_state at the event's t-1,
  * boundary follows the TRUE future path (no freezing),
  * counterfactual = the EVENT NEVER HAPPENED: v2 held at its pre-event value
    for the whole horizon (the factual sequence already contains the plant's
    own move, so this is the analogue of the DiD's matched controls),
  * terminal response = mean of the last 10 steps of the final channel (H60),
    H18 (last 3 steps) reported in parallel.

The windows are exactly the plant-DiD events (corrected v2.1 record, val split):
up n=22, down n=47.  Reproducing those counts is the self-check gate BEFORE any
model is run -- if they do not match, the extraction is wrong and nothing else
in this script means anything.

Comparison target (observational DiD, from probes_20260828/corrected_record/):
  up   H60 mean -1.416 degC (placebo p=0.000), frac 0.727
  down H60 mean +1.191 degC (placebo p=0.000), frac 0.723

No ratio / multiplier is computed here -- the two columns are reported side by
side on IDENTICAL windows and the reader compares.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.final_wm.data import SPLIT_VAL, CanonicalRecord                 # noqa: E402
from src.final_wm.properties import load_grid_properties                # noqa: E402
from src.final_wm.training import build_world_model                     # noqa: E402
from experiments.final_wm import matrix_spec as ms                      # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = Path(__file__).resolve().parent
OUT = P / "event_window_probe"
OUT.mkdir(parents=True, exist_ok=True)

MIN_STEP = 0.04
HORIZON = 60
REC = P / "v1fix_probe" / "canonical_sideA_v1fixed.npz"

ARMS = {
    "baseline_unsched": P / "v1fix_probe" / "v1fix_unanchored",
    "norew_oldrecord": ROOT / "artifacts" / "final_wm" / "checkpoints",
}
CKPT = "t1_closure_cons_norew_seed0.pt"


def find_v2_events(record: CanonicalRecord, split_id: int = SPLIT_VAL):
    """Re-implement the event filter used by the plant DiD study.

    min |dv| >= 0.04, no other valve step >= 0.04 within HORIZON steps before
    (t-1) or after t, window [t-1, t+HORIZON] inside the run.  Returns lists of
    absolute start indices grouped by direction.
    """
    a = record.actions.numpy()
    up, down = [], []
    for start, end in record.split_runs(split_id):
        v = a[start:end, 1]
        step = np.abs(np.diff(a[start:end, :], axis=0)).max(axis=1)
        for t_rel in range(1, end - start - HORIZON):
            dv = v[t_rel] - v[t_rel - 1]
            if abs(dv) < MIN_STEP:
                continue
            if step[t_rel:t_rel + HORIZON].max(initial=0.0) >= MIN_STEP:
                continue
            if step[max(0, t_rel - 1 - HORIZON):t_rel - 1].max(initial=0.0) >= MIN_STEP:
                continue
            (up if dv > 0 else down).append((start + t_rel, dv))
    return up, down


@torch.no_grad()
def model_response(model, record, events, *, horizon=HORIZON, tail=10):
    """Counterfactual = the EVENT NEVER HAPPENED (v2 held at its pre-event value).

    The factual action sequence already contains the plant's own move; the
    matched-control comparison in the DiD is approximated in the model by
    holding v2 constant at the pre-event value for the whole horizon, with the
    boundary still on the event's own true path.  Reported dT = T_factual -
    T_no_event (negative for openings that cool, positive for closures).
    """
    a = record.actions.numpy()
    b = record.boundary.numpy()
    o = record.obs.numpy()
    deltas = []
    for t, dv in events:
        b0 = torch.tensor(b[t - 1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a0 = torch.tensor(a[t - 1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        o0 = torch.tensor(o[t - 1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        state0 = model.transition.initial_steady_state(b0, a0, o0)

        bseq = torch.tensor(b[t:t + horizon], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        act = torch.tensor(a[t:t + horizon], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        no_event = act.clone()
        no_event[:, :, 1] = float(a[t - 1, 1])      # hold v2 at pre-event value

        _s, t_fact = model.transition.integrate(state0, bseq, act)
        _s, t_no = model.transition.integrate(state0, bseq, no_event)
        deltas.append(float((t_fact[:, -tail:, 4] - t_no[:, -tail:, 4]).mean()))
    return np.array(deltas, dtype=np.float64)


def summarize(name, d):
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"name": name, "n": 0}
    return {"name": name, "n": int(len(d)),
            "mean_delta_c": float(d.mean()), "median": float(np.median(d)),
            "min": float(d.min()), "max": float(d.max()),
            "frac_negative": float((d < 0).mean())}


def main():
    record = CanonicalRecord(REC)
    up, down = find_v2_events(record)
    print(f"[self-check] up n={len(up)} (expect 22), down n={len(down)} (expect 47)", flush=True)
    if len(up) != 22 or len(down) != 47:
        print("!! event extraction does NOT match the plant DiD -- aborting", flush=True)
        sys.exit(1)
    dvs_up = np.array([d for _, d in up])
    dvs_down = np.array([d for _, d in down])
    print(f"[self-check] up |dv| mean={dvs_up.mean():.4f} (plant 0.0634), "
          f"down |dv| mean={dvs_down.mean():.4f} (plant 0.0648)", flush=True)

    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
    report = {"record": str(REC), "events": {"up_n": len(up), "down_n": len(down)},
              "plant_did_reference": {
                  "up_h60_mean_c": -1.416, "up_frac": 0.727,
                  "down_h60_mean_c": 1.191, "down_frac": 0.723,
                  "note": "observational DiD, probes_20260828/corrected_record; parallel-trends assumption untestable"},
              "arms": {}}

    for name, ckdir in ARMS.items():
        ck = ckdir / "checkpoints" / CKPT if name != "norew_oldrecord" else ckdir / CKPT
        if not ck.exists():
            print(f"[skip] {name}", flush=True)
            continue
        spec = ms._base("t1", "closure_cons_norew" if name == "norew_oldrecord" else "closure_cons_norew", 0)
        model = build_world_model(spec, props).to(DEVICE)
        model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=False)["state_dict"])
        model.eval()
        entry = {}
        for label, evs in (("up", up), ("down", down)):
            d60 = model_response(model, record, evs, horizon=60, tail=10)
            d18 = model_response(model, record, evs, horizon=18, tail=3)
            entry[f"{label}_h60"] = summarize(f"{name} {label} H60", d60)
            entry[f"{label}_h18"] = summarize(f"{name} {label} H18", d18)
        report["arms"][name] = entry
        print(f"\n[{name}]", flush=True)
        for k in ("up_h60", "down_h60", "up_h18", "down_h18"):
            v = entry[k]
            if v["n"] == 0:
                print(f"  {k}: n=0", flush=True)
                continue
            print(f"  {k}: n={v['n']:2d} mean={v['mean_delta_c']:+.3f} "
                  f"median={v['median']:+.3f} [{v['min']:+.3f},{v['max']:+.3f}] "
                  f"frac_neg={v['frac_negative']:.3f}", flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\nwritten", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
