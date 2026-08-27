"""Ad-hoc verification of the 2026-08-28 object-side DiD probe.

NOT a test suite -- four independent falsification checks on
`plant_did_event_study.py`:

  C1  event-set fidelity: an INDEPENDENT re-traversal must reproduce the repo's
      tested `valve_step_events` / `event_study_summary` counts and
      before-after correct-direction fractions exactly.
  C2  control purity: every window entering the DiD control pool must have NO
      valve step >= min_step anywhere in [t-1-H, t+H] -- INCLUDING the event
      slot itself.  The `isolated` predicate leaves index t_rel-1 unchecked
      (the canonical function does so deliberately, because for a TREATED event
      that slot IS the event).  Reused for controls that is a leak.
  C3  placebo / randomization inference: relabel random no-action windows as
      pseudo-treated and rerun the matcher.  A valid estimator must return
      mean dT ~ 0 and frac_correct ~ 0.5; the real effect must sit outside that
      null.  Real values are read from the probe's JSON, never hardcoded.
  C4  Wilson CI endpoints must satisfy the score equation |z(p)| = 1.96, and
      the two-sided-CI vs one-sided-test framing is contrasted explicitly.

The traversal, matcher and placebo null are deliberately REIMPLEMENTED here
rather than imported: independent reimplementation is the verification method,
so the duplication with the probe is intentional, not a DRY violation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.analysis import (  # noqa: E402
    EVENT_HORIZONS, FINAL_OBS_INDEX, STEAM_FLOW_INDEX,
    event_study_summary, valve_step_events,
)
from src.final_wm.data import SPLIT_VAL, CanonicalRecord  # noqa: E402

MIN_STEP, HORIZON, TREND_STEPS, K = 0.04, 60, 30, 5
HZ = tuple(h for h in EVENT_HORIZONS if h <= HORIZON)
PROBE = ROOT / "results/final_wm/probes_20260828/plant_did_event_study.json"

rec = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
obs, act, bnd = rec.obs.numpy(), rec.actions.numpy(), rec.boundary.numpy()
T, L = obs[:, FINAL_OBS_INDEX], bnd[:, STEAM_FLOW_INDEX]
probe = json.loads(PROBE.read_text())
fails = []


def traverse(vi: int, *, strict_controls: bool):
    """Independent re-traversal.  strict_controls closes the t_rel-1 slot."""
    treated, controls = {"up": [], "down": []}, []
    for start, end in rec.split_runs(SPLIT_VAL):
        v = act[start:end, vi]
        step = np.abs(np.diff(act[start:end, :], axis=0)).max(axis=1)
        for tr in range(1, end - start - HORIZON):
            if tr - 1 - TREND_STEPS < 0:
                continue
            t, dv = start + tr, v[tr] - v[tr - 1]
            post = step[tr : tr + HORIZON].max(initial=0.0)
            pre = step[max(0, tr - 1 - HORIZON) : tr - 1].max(initial=0.0)
            if post >= MIN_STEP or pre >= MIN_STEP:
                continue
            d = np.array([T[t - 1 + h] - T[t - 1] for h in HZ])
            f = np.array([L[t - 1], T[t - 1],
                          (T[t - 1] - T[t - 1 - TREND_STEPS]) / TREND_STEPS])
            if abs(dv) >= MIN_STEP:
                treated["up" if dv > 0 else "down"].append((t, d, f))
            elif not strict_controls or step[tr - 1] < MIN_STEP:
                controls.append((t, d, f))
    return treated, controls


def did(tl, controls, *, cols=np.array([0, 1, 2])):
    cf = np.stack([c[2] for c in controls])
    cd = np.stack([c[1] for c in controls])
    sc = np.maximum(cf.std(axis=0), 1e-9)
    out = []
    for _t, d, f in tl:
        dist = np.abs((cf[:, cols] - f[cols]) / sc[cols]).sum(axis=1)
        out.append(d - cd[np.argsort(dist)[:K]].mean(axis=0))
    return np.stack(out)


def frac(d, direction):
    return ((d < 0) if direction == "up" else (d > 0)).mean(axis=0)


print("=" * 78)
print("C1  event-set fidelity: independent re-traversal vs repo valve_step_events")
print("=" * 78)
for vi, vn in ((1, "v2"), (0, "v1")):
    mine, _ = traverse(vi, strict_controls=True)
    ref = event_study_summary(valve_step_events(rec, SPLIT_VAL, vi,
                                                min_step=MIN_STEP, horizon=HORIZON))
    for dr in ("up", "down"):
        n_mine, n_ref = len(mine[dr]), ref[dr]["n"]
        ok_n = n_mine == n_ref
        if not ok_n:
            fails.append(f"C1 {vn} {dr}: n {n_mine} != {n_ref}")
        line = f"  {vn} {dr:4s}  n mine={n_mine:3d} repo={n_ref:3d} {'OK' if ok_n else 'MISMATCH'}"
        if n_mine and n_ref:
            fm = frac(np.stack([x[1] for x in mine[dr]]), dr)
            fr = np.array(ref[dr]["frac_correct"])
            ok_f = np.allclose(fm, fr, atol=1e-12)
            if not ok_f:
                fails.append(f"C1 {vn} {dr}: frac {fm} != {fr}")
            line += f" | frac match {'OK' if ok_f else 'MISMATCH'} {np.array2string(fm, precision=3)}"
            # also cross-check the JSON the probe wrote
            js = probe["valves"][vn][dr]["before_after"]["frac_correct"]
            if not np.allclose(fm, js, atol=1e-12):
                fails.append(f"C1 {vn} {dr}: probe JSON frac {js} != {fm}")
        print(line)

print()
print("=" * 78)
print("C2  control purity: valve step inside [t-1-H, t+H] including slot t_rel-1")
print("=" * 78)
for vi, vn in ((1, "v2"), (0, "v1")):
    _, loose = traverse(vi, strict_controls=False)
    _, strict = traverse(vi, strict_controls=True)
    leak = len(loose) - len(strict)
    print(f"  {vn}: loose pool={len(loose)}  strict pool={len(strict)}  "
          f"leak closed={leak} ({100*leak/max(len(loose),1):.3f}%)")
    js_n = probe["valves"][vn]["n_controls"]
    if js_n == len(strict):
        print(f"       probe JSON n_controls={js_n} == strict pool  OK (fix landed)")
    elif js_n == len(loose):
        fails.append(f"C2 {vn}: probe JSON still on LOOSE pool ({js_n}), fix not re-run")
    else:
        fails.append(f"C2 {vn}: probe JSON n_controls {js_n} matches neither pool")

print()
print("=" * 78)
print("C3  randomization inference: 200 placebo draws vs the probe's claimed effect")
print("=" * 78)
_, ctrl = traverse(1, strict_controls=True)
# Real values are READ FROM THE PROBE JSON, not hardcoded, so the verifier
# cannot silently pass against stale numbers if the probe changes.
REAL = {}
for dr in ("up", "down"):
    inf = probe["valves"]["v2"][dr]["inference"]["H60"]
    REAL[dr] = (probe["valves"]["v2"][dr]["n"], inf["mean_delta_c"], inf["frac_correct"])
    print(f"  probe JSON v2 {dr} H60: n={REAL[dr][0]} mean={REAL[dr][1]:+.3f} frac={REAL[dr][2]:.3f}")
rng = np.random.default_rng(0)
N_PLACEBO = 200
for dr, (n_fake, real_mean, real_frac) in REAL.items():
    null_mean, null_frac = [], []
    for _ in range(N_PLACEBO):
        idx = rng.choice(len(ctrl), n_fake, replace=False)
        keep = np.ones(len(ctrl), bool)
        keep[idx] = False
        d = did([ctrl[i] for i in idx], [c for j, c in enumerate(ctrl) if keep[j]])
        null_mean.append(d.mean(axis=0)[-1])
        null_frac.append(frac(d, dr)[-1])
    nm, nf = np.array(null_mean), np.array(null_frac)
    p_mean = float((np.abs(nm) >= abs(real_mean)).mean())
    p_frac = float((nf >= real_frac).mean())
    print(f"  {dr} (n={n_fake}) H60 placebo null over {N_PLACEBO} draws:")
    print(f"    mean : null {nm.mean():+.3f} +- {nm.std():.3f}  "
          f"[{nm.min():+.3f},{nm.max():+.3f}]  real {real_mean:+.3f}  p(|null|>=|real|)={p_mean:.3f}")
    print(f"    frac : null {nf.mean():.3f} +- {nf.std():.3f}  "
          f"[{nf.min():.3f},{nf.max():.3f}]  real {real_frac:.3f}  p(null>=real)={p_frac:.3f}")
    if abs(nm.mean()) > 0.15:
        fails.append(f"C3 {dr}: placebo mean biased away from 0 ({nm.mean():+.3f})")
    if not (0.35 <= nf.mean() <= 0.65):
        fails.append(f"C3 {dr}: placebo frac biased away from 0.5 ({nf.mean():.3f})")
    if p_mean > 0.05:
        fails.append(f"C3 {dr}: real H60 mean NOT separable from placebo null (p={p_mean:.3f})")
    if p_frac > 0.05:
        fails.append(f"C3 {dr}: real H60 frac NOT separable from placebo null (p={p_frac:.3f})")
    # cross-check: the probe computes its own placebo null; ours is an
    # INDEPENDENT reimplementation, so the two p-values must broadly agree.
    probe_p = probe["valves"]["v2"][dr]["inference"]["H60"]["placebo_frac"]["p_one_sided"]
    if abs(probe_p - p_frac) > 0.10:
        fails.append(f"C3 {dr}: probe placebo p {probe_p:.3f} vs independent {p_frac:.3f} diverge")
    else:
        print(f"    probe's own placebo p={probe_p:.3f} vs independent {p_frac:.3f}  OK")

print()
print("=" * 78)
print("C4  Wilson CI: endpoints must satisfy the score equation |z(p)| = 1.96")
print("=" * 78)
import math  # noqa: E402


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score(phat, p0, n):
    """Wilson/score statistic -- variance evaluated AT THE NULL p0."""
    return (phat - p0) / math.sqrt(p0 * (1 - p0) / n)


try:
    from statsmodels.stats.proportion import proportion_confint
    HAVE_SM = True
except ImportError:
    HAVE_SM = False
    print("  (statsmodels unavailable -- using the score-equation root check instead,")
    print("   which is independent of the closed form rather than circular)")

for k, n in ((15, 22), (16, 22), (27, 48), (36, 48), (5, 6)):
    lo, hi = wilson(k, n)
    s_lo, s_hi = score(k / n, lo, n), score(k / n, hi, n)
    ok = abs(abs(s_lo) - 1.96) < 1e-6 and abs(abs(s_hi) - 1.96) < 1e-6
    if not ok:
        fails.append(f"C4 k={k},n={n}: score at endpoints ({s_lo:+.6f},{s_hi:+.6f}) != +-1.96")
    extra = ""
    if HAVE_SM:
        ref = proportion_confint(k, n, alpha=0.05, method="wilson")
        if not np.allclose((lo, hi), ref, atol=5e-4):
            fails.append(f"C4 k={k},n={n}: {(lo,hi)} != statsmodels {ref}")
        extra = f"  sm [{ref[0]:.4f},{ref[1]:.4f}]"
    print(f"  k={k:3d} n={n:3d}  [{lo:.4f},{hi:.4f}]  score({s_lo:+.4f},{s_hi:+.4f})  "
          f"{'OK' if ok else 'MISMATCH'}{extra}")

# The framing check that matters: two-sided CI-contains-0.5 vs one-sided
# directional test.  The physical hypothesis (spray cools) IS directional.
print("\n  two-sided 'is 0.5 inside the Wilson CI' vs one-sided directional test:")
for k, n, lbl in ((16, 22, "v2 up H60 DiD 0.727"), (34, 48, "v2 down H60 DiD 0.708")):
    lo, hi = wilson(k, n)
    z1 = score(k / n, 0.5, n)
    p1 = 0.5 * math.erfc(z1 / math.sqrt(2))
    print(f"    {lbl}: CI [{lo:.3f},{hi:.3f}] contains 0.5 = "
          f"{'YES -> two-sided n.s.' if lo <= 0.5 <= hi else 'no'};  "
          f"one-sided z={z1:+.2f} p={p1:.3f}")

print()
print("=" * 78)
if fails:
    print(f"RESULT: {len(fails)} finding(s)")
    for f in fails:
        print(f"  - {f}")
else:
    print("RESULT: all four checks clean")
