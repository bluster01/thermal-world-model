"""Object-side valve-effect direction: before-after vs DiD (2026-08-28).

WHY THIS EXISTS
---------------
Frozen matrix v0.3 §5.1 recalibrated the R1 direction gate to
"correct-direction fraction >= 0.60", taking the margin below a plant-side
reference band of 0.68 (up, n=22) / 0.75 (down, n=48).  That band comes from
`analysis.valve_step_events`, which estimates the valve effect as a pure
BEFORE-AFTER difference:

    dT(h) = obs[t-1+h] - obs[t-1]

The model-side gate (`evaluation.step_response_direction`) instead computes a
pure COUNTERFACTUAL difference: same initial state, boundary pinned, only the
valve perturbed, two integrations subtracted.  The two are NOT comparable:

  * before-after carries every common cause active in [t-1, t-1+h] (load ramp,
    coal feed, air, sootblowing, SP moves) -- the event filter only excludes
    OTHER VALVE steps, never boundary movement; and
  * it carries closed-loop simultaneity: the valve moves BECAUSE temperature is
    already drifting, so opening spray during an upward ramp can leave observed
    dT > 0 even when the spray genuinely cooled relative to no-action.

WHAT THIS PROBE REPORTS
-----------------------
  D1  pre-event trend over the 30 steps before each event (simultaneity probe).
  D2  before-after replication (must reproduce the 0.68/0.75 band).
  D3  DiD: match k no-action control windows on (load, temperature, pre-event
      trend) and report dT_treated - mean_controls dT.  A trend-only-matched
      variant decomposes simultaneity from other common causes.
  D4  inference on D3, against the estimator's OWN placebo null: relabel random
      no-action windows as pseudo-treated, rerun the matcher N_PLACEBO times,
      and locate the real effect in that null.  Wilson CIs are reported as
      DESCRIPTIVE only; the hypothesis "spray cools" is directional, so the
      inferential test is ONE-SIDED (a two-sided CI-contains-0.5 reading
      understates the evidence -- this superseded an earlier draft of this
      probe that concluded, wrongly, that n=22 carried no information).

Read-only: validation split, no model, no training, no GPU.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import (  # noqa: E402
    EVENT_HORIZONS,
    FINAL_OBS_INDEX,
    STEAM_FLOW_INDEX,
    event_study_summary,
    valve_step_events,
)
from src.final_wm.data import SPLIT_VAL, CanonicalRecord  # noqa: E402

RECORD = ROOT / "artifacts/final_wm/canonical_sideA.npz"
OUT = Path(__file__).resolve().parent

MIN_STEP = 0.04          # protocol event threshold
HORIZON = 60             # protocol contamination window / max horizon
TREND_STEPS = 30         # pre-event slope window (300 s)
K_CONTROLS = 5           # nearest-neighbour controls per treated event
N_PLACEBO = 200          # placebo relabellings for the randomisation null
Z = 1.96
HORIZONS = tuple(h for h in EVENT_HORIZONS if h <= HORIZON)
JUDGED = {18: "H18", 60: "H60"}   # H1/H6 excluded: SNR below the noise floor

record = CanonicalRecord(RECORD)
obs = record.obs.numpy()
act = record.actions.numpy()
bnd = record.boundary.numpy()

T = obs[:, FINAL_OBS_INDEX]
L = bnd[:, STEAM_FLOW_INDEX]


# ---------------------------------------------------------------------------
# events and controls (traversal mirrors analysis.valve_step_events)
# ---------------------------------------------------------------------------
def collect(valve_index: int):
    treated = {"up": [], "down": []}
    controls = []
    for start, end in record.split_runs(SPLIT_VAL):
        v = act[start:end, valve_index]
        step = np.abs(np.diff(act[start:end, :], axis=0)).max(axis=1)  # any valve
        for t_rel in range(1, end - start - HORIZON):
            t = start + t_rel
            if t_rel - 1 - TREND_STEPS < 0:      # need history for the slope
                continue
            dv = v[t_rel] - v[t_rel - 1]
            isolated = (
                step[t_rel : t_rel + HORIZON].max(initial=0.0) < MIN_STEP
                and step[max(0, t_rel - 1 - HORIZON) : t_rel - 1].max(initial=0.0) < MIN_STEP
            )
            base = T[t - 1]
            deltas = np.array([T[t - 1 + h] - base for h in HORIZONS])
            feat = np.array([
                L[t - 1],
                base,
                (base - T[t - 1 - TREND_STEPS]) / TREND_STEPS,   # °C per step
            ])
            if abs(dv) >= MIN_STEP:
                if isolated:
                    treated["up" if dv > 0 else "down"].append((t, deltas, feat, dv))
            elif isolated and step[t_rel - 1] < MIN_STEP:
                # Control = pure no-action window: no valve step anywhere in
                # [t-1-HORIZON, t+HORIZON].  `isolated` leaves slot t_rel-1
                # unchecked (for a TREATED event that slot IS the event), so it
                # must be closed explicitly here -- otherwise the OTHER valve
                # may jump at exactly the treatment position and still qualify.
                controls.append((t, deltas, feat))
    return treated, controls


# ---------------------------------------------------------------------------
# estimator and inference
# ---------------------------------------------------------------------------
COLS_FULL = np.array([0, 1, 2])
COLS_TREND = np.array([2])


def did(treated, controls, cols=COLS_FULL):
    """Match K_CONTROLS nearest no-action controls per event; return DiD deltas."""
    cf = np.stack([c[2] for c in controls])
    cd = np.stack([c[1] for c in controls])
    scale = np.maximum(cf.std(axis=0), 1e-9)
    rows, dists = [], []
    for item in treated:
        d_tr, f_tr = item[1], item[2]
        dist = np.abs((cf[:, cols] - f_tr[cols]) / scale[cols]).sum(axis=1)
        pick = np.argsort(dist)[:K_CONTROLS]
        rows.append(d_tr - cd[pick].mean(axis=0))
        dists.append(dist[pick].mean())
    return np.stack(rows), float(np.mean(dists))


def frac_correct(deltas, direction):
    ok = deltas < 0 if direction == "up" else deltas > 0
    return ok.astype(np.float64).mean(axis=0)


def wilson(k: int, n: int, z: float = Z):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def one_sided_p(k: int, n: int):
    """Score test of frac > 0.5 -- the directional hypothesis 'spray cools'."""
    z = (k / n - 0.5) / math.sqrt(0.25 / n)
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def placebo_null(n_treated: int, controls, direction: str, *, seed: int = 0):
    """Relabel random no-action windows as pseudo-treated and rerun the matcher."""
    rng = np.random.default_rng(seed)
    means, fracs = [], []
    for _ in range(N_PLACEBO):
        idx = rng.choice(len(controls), n_treated, replace=False)
        keep = np.ones(len(controls), bool)
        keep[idx] = False
        d, _ = did([controls[i] for i in idx],
                   [c for j, c in enumerate(controls) if keep[j]])
        means.append(d.mean(axis=0))
        fracs.append(frac_correct(d, direction))
    return np.stack(means), np.stack(fracs)


def fmt(a):
    return np.array2string(a, precision=3, floatmode="fixed")


# ---------------------------------------------------------------------------
report = {
    "record": str(RECORD),
    "split": "val",
    "horizons_steps": list(HORIZONS),
    "horizons_seconds": [h * 10 for h in HORIZONS],
    "judged_horizons": list(JUDGED.values()),
    "protocol": {
        "min_step": MIN_STEP, "contamination_horizon": HORIZON,
        "trend_steps": TREND_STEPS, "k_controls": K_CONTROLS,
        "n_placebo": N_PLACEBO,
        "inference": "one-sided score test (directional); Wilson CI descriptive only",
    },
    "valves": {},
}

for valve_index, vname in ((1, "v2"), (0, "v1")):
    treated, controls = collect(valve_index)
    ref = event_study_summary(valve_step_events(record, SPLIT_VAL, valve_index,
                                               min_step=MIN_STEP, horizon=HORIZON))
    entry = {"n_controls": len(controls), "upstream_reference": {
        d: {"n": ref[d]["n"], "frac_correct": ref[d].get("frac_correct")}
        for d in ("up", "down")}}

    print(f"\n{'='*78}\n{vname} (valve_index={valve_index})   no-action controls: {len(controls)}\n{'='*78}")
    for direction in ("up", "down"):
        tl = treated[direction]
        if not tl:
            print(f"  {direction}: no events")
            continue
        n = len(tl)
        d_ba = np.stack([x[1] for x in tl])
        trend = np.stack([x[2] for x in tl])[:, 2] * 60.0        # °C per minute
        blk = {
            "n": n,
            "pre_trend_c_per_min": {
                "mean": float(trend.mean()), "median": float(np.median(trend)),
                "frac_positive": float((trend > 0).mean()),
            },
            "before_after": {
                "mean_delta_c": d_ba.mean(axis=0).tolist(),
                "frac_correct": frac_correct(d_ba, direction).tolist(),
            },
        }
        print(f"\n  [{direction}]  n={n}")
        print(f"    D1 pre-event trend : {trend.mean():+.3f} °C/min "
              f"(median {np.median(trend):+.3f}, frac>0 = {(trend > 0).mean():.2f})")
        print(f"    D2 before-after    : mean {fmt(d_ba.mean(axis=0))} "
              f"| frac {fmt(frac_correct(d_ba, direction))}")

        d_did, dist_full = did(tl, controls, COLS_FULL)      # primary estimate
        d_trend, dist_trend = did(tl, controls, COLS_TREND)  # simultaneity-only
        for label, d, mdist, key in (
            ("D3 DiD load+T+trend", d_did, dist_full, "did_full"),
            ("D3 DiD trend-only  ", d_trend, dist_trend, "did_trend_only"),
        ):
            blk[key] = {
                "mean_delta_c": d.mean(axis=0).tolist(),
                "frac_correct": frac_correct(d, direction).tolist(),
                "mean_match_distance": mdist,
            }
            print(f"    {label}: mean {fmt(d.mean(axis=0))} "
                  f"| frac {fmt(frac_correct(d, direction))} | match_dist {mdist:.3f}")

        # D4 -- inference on the primary (load+T+trend) DiD estimate
        null_m, null_f = placebo_null(n, controls, direction)
        inf = {}
        print(f"    D4 inference vs {N_PLACEBO}-draw placebo null (primary DiD):")
        for hi, h in enumerate(HORIZONS):
            if h not in JUDGED:
                continue
            real_m = float(d_did.mean(axis=0)[hi])
            real_f = float(frac_correct(d_did, direction)[hi])
            k = int(round(real_f * n))
            lo, up = wilson(k, n)
            z1, p1 = one_sided_p(k, n)
            nm, nf = null_m[:, hi], null_f[:, hi]
            # signed one-sided placebo p: how often the null is as extreme
            p_m = float((nm <= real_m).mean() if direction == "up" else (nm >= real_m).mean())
            p_f = float((nf >= real_f).mean())
            inf[JUDGED[h]] = {
                "mean_delta_c": real_m, "frac_correct": real_f,
                "wilson_ci_descriptive": [lo, up], "wilson_excludes_half": not (lo <= 0.5 <= up),
                "one_sided_z": z1, "one_sided_p": p1,
                "placebo_mean": {"null_mean": float(nm.mean()), "null_sd": float(nm.std()),
                                 "p_one_sided": p_m},
                "placebo_frac": {"null_mean": float(nf.mean()), "null_sd": float(nf.std()),
                                 "p_one_sided": p_f},
            }
            print(f"      {JUDGED[h]}  mean {real_m:+.3f} °C  vs null {nm.mean():+.3f}±{nm.std():.3f}"
                  f"  p={p_m:.3f}")
            print(f"           frac {real_f:.3f} ({k}/{n})  vs null {nf.mean():.3f}±{nf.std():.3f}"
                  f"  p={p_f:.3f}  | Wilson [{lo:.3f},{up:.3f}] "
                  f"{'excludes' if not (lo <= 0.5 <= up) else 'contains'} 0.5"
                  f"  | one-sided z={z1:+.2f} p={p1:.3f}")
        blk["inference"] = inf
        entry[direction] = blk
    report["valves"][vname] = entry

(OUT / "plant_did_event_study.json").write_text(json.dumps(report, indent=2))
print(f"\nhorizons (steps) = {HORIZONS} -> seconds = {[h*10 for h in HORIZONS]}")
print("written", OUT / "plant_did_event_study.json")
