"""Rewet direction + load-stratification test (2026-08-26).

Two user points this answers:
  (a) v0.4 rejected rewet on DIRECTION (F3: intact rewet flipped the v1
      downstream sign), not accuracy -- so the accuracy-based "partial
      vindication" from rewet_probe does not address the actual criterion.
  (b) physical hypothesis: rewetting should only exist at LOW load (low mass
      flux -> droplets/film survive); the rewet arm's biggest gains were at
      HIGH load (Q4/Q5), which under that hypothesis is the term acting as a
      free fudge factor in the wrong regime.

Tests (all evaluation-only on frozen checkpoints, reusing the repo protocol
evaluation.step_response_direction -- the same one R1/F3 used):
  T1 direction, intact vs norew, both valves, on the CORRECTED record
  T2 direction stratified by load (does the sign depend on regime?)
  T3 the model's own rewet power q_w1/q_w2 vs load (where does it spend it?)
  T4 data-side feasibility: superheat margin at the mixing points vs load
     (can liquid survive there at all?)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))

from src.final_wm.contracts import BOUNDARY_ELEMENTS, OBSERVATION_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.evaluation import step_response_direction
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "rewet_direction"
OUT.mkdir(parents=True, exist_ok=True)
IDX_FLOW = BOUNDARY_ELEMENTS.index("steam_flow")
REC = P / "v1fix_probe/canonical_sideA_v1fixed.npz"
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(REC)
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)

ARMS = {
    "intact_rewet": (P / "rewet_probe/checkpoints/t1_closure_cons_seed0.pt",
                     "closure_cons", "conservative"),
    "norew": (P / "v1fix_probe/checkpoints/t1_closure_cons_norew_seed0.pt",
              "closure_cons_norew", "conservative_norew"),
}


def load_arm(ckpt, arm, closure_mode):
    spec = ms._base("t1", arm, 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode=closure_mode,
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
    m = build_world_model(spec, props).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                 weights_only=False)["state_dict"])
    return m.eval()


models = {}
for tag, (ckpt, arm, cm) in ARMS.items():
    if not ckpt.exists():
        print(f"!! missing checkpoint for {tag}: {ckpt}")
        continue
    models[tag] = load_arm(ckpt, arm, cm)
    print(f"loaded {tag}: {ckpt.name}")

print("\n" + "=" * 72)
print("T1  step-response direction (repo protocol: +0.05 valve, 60 steps,")
print("    terminal 10-step mean; frac_negative = 1.0 is correct/cooling)")
print("=" * 72)
res_t1 = {}
for tag, m in models.items():
    for vi, vname in ((0, "v1 stage-1"), (1, "v2 stage-2")):
        r = step_response_direction(m, record, SPLIT_VAL, n_windows=32,
                                    rollout_steps=60, valve_index=vi,
                                    delta_v=0.05, seed=0, device=DEVICE)
        res_t1[f"{tag}/{vname}"] = r
        flag = "OK" if r["frac_negative"] == 1.0 else (
            "WRONG-SIGN" if r["frac_negative"] < 0.5 else "MIXED")
        print(f"  {tag:14s} {vname:11s}: mean_delta={r['mean_delta_c']:+7.4f} degC  "
              f"frac_neg={r['frac_negative']:.3f}  [{flag}]")

print("\n" + "=" * 72)
print("T2  direction stratified by load (does the sign depend on regime?)")
print("=" * 72)


@torch.no_grad()
def direction_by_load(model, valve_index, n_win=256, rollout=60, delta_v=0.05):
    gen = torch.Generator().manual_seed(0)
    b = sample_windows(record, SPLIT_VAL, n_win, 96, 1, gen)
    b0 = b.future_boundary[:, 0].to(DEVICE)
    a0 = b.future_actions[:, 0].to(DEVICE)
    o0 = b.history.obs[:, -1].to(DEVICE)
    s0 = model.transition.initial_steady_state(b0, a0, o0)
    bseq = b0.unsqueeze(1).repeat(1, rollout, 1)
    base = a0.unsqueeze(1).repeat(1, rollout, 1)
    stepped = base.clone()
    stepped[:, :, valve_index] = (stepped[:, :, valve_index] + delta_v).clamp(max=1.0)
    _, t_base = model.transition.integrate(s0, bseq, base)
    _, t_step = model.transition.integrate(s0, bseq, stepped)
    d = (t_step[:, -10:, 4] - t_base[:, -10:, 4]).mean(dim=1).cpu().numpy()
    flow = b0[:, IDX_FLOW].cpu().numpy()
    v_now = a0[:, valve_index].cpu().numpy()
    return d, flow, v_now


edges = None
for tag, m in models.items():
    for vi, vname in ((0, "v1"), (1, "v2")):
        d, flow, v_now = direction_by_load(m, vi)
        if edges is None:
            edges = np.quantile(flow, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        row = f"  {tag:14s} {vname}: "
        for i in range(5):
            msk = (flow >= edges[i]) & (flow <= edges[i+1] if i == 4 else flow < edges[i+1])
            if msk.sum() == 0:
                row += "  Q%d n/a" % (i+1)
                continue
            row += (f"  Q{i+1}[{msk.sum():3d}] {d[msk].mean():+6.3f}/"
                    f"{(d[msk] < 0).mean():.2f}")
        print(row)
print(f"  (format: Qk[n] mean_delta/frac_negative; load bins from "
      f"{edges[0]:.0f} to {edges[-1]:.0f} kg/s)")

print("\n" + "=" * 72)
print("T3  where does the model SPEND rewet power? q_w1/q_w2 vs load")
print("=" * 72)
if "intact_rewet" in models:
    m = models["intact_rewet"]
    captured = []
    orig = m.transition._rewetting_powers

    def spy(tm, m1, m2, p0, p1, h2, h_spray):
        q1, q2 = orig(tm, m1, m2, p0, p1, h2, h_spray)
        captured.append((q1.detach().float().cpu(), q2.detach().float().cpu()))
        return q1, q2

    m.transition._rewetting_powers = spy
    with torch.no_grad():
        gen = torch.Generator().manual_seed(50_000)
        flows_all, q1_all, q2_all = [], [], []
        done = 0
        while done < 128:
            bsz = min(32, 128 - done)
            b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
            captured.clear()
            hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                      actions=b.history.actions.to(DEVICE),
                                      boundary=b.history.boundary.to(DEVICE))
            m.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                       true_future_boundary=b.future_boundary.to(DEVICE))
            if captured:
                q1 = torch.stack([c[0] for c in captured]).mean(0)
                q2 = torch.stack([c[1] for c in captured]).mean(0)
                q1_all.append(q1)
                q2_all.append(q2)
                flows_all.append(b.future_boundary[:, 0, IDX_FLOW])
            done += bsz
    m.transition._rewetting_powers = orig
    if q1_all:
        q1 = torch.cat(q1_all).numpy().reshape(-1)
        q2 = torch.cat(q2_all).numpy().reshape(-1)
        fl = torch.cat(flows_all).numpy()
        n = min(len(q1), len(fl))
        e2 = np.quantile(fl, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        print("  rewet power magnitude by load bin (mean |q_w|, kW):")
        for i in range(5):
            msk = (fl >= e2[i]) & (fl <= e2[i+1] if i == 4 else fl < e2[i+1])
            if msk.sum() == 0:
                continue
            print(f"    Q{i+1} {e2[i]:5.0f}-{e2[i+1]:5.0f} kg/s (n={msk.sum():3d}): "
                  f"|q_w1|={np.abs(q1[:len(fl)][msk]).mean():9.3f}  "
                  f"|q_w2|={np.abs(q2[:len(fl)][msk]).mean():9.3f}")
        print("  -> physical expectation (user): rewet should concentrate at LOW load")

print("\n" + "=" * 72)
print("T4  data-side feasibility: can liquid survive at the mixing points?")
print("    superheat margin = measured temp - saturation temp at local pressure")
print("=" * 72)
raw = np.load(REC)
obs = raw["obs"]
bnd = raw["boundary"]
flow = bnd[:, IDX_FLOW]
pm = bnd[:, BOUNDARY_ELEMENTS.index("separator_pressure")]
p_out = bnd[:, BOUNDARY_ELEMENTS.index("outlet_pressure")]
e3 = np.quantile(flow, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
with torch.no_grad():
    pt = torch.tensor(pm, dtype=torch.float32, device=DEVICE)
    tsat = props.saturation_temperature(pt).cpu().numpy()
print("  (saturation temp from the model's own IAPWS surrogate at separator pressure)")
for i in range(5):
    msk = (flow >= e3[i]) & (flow <= e3[i+1] if i == 4 else flow < e3[i+1])
    for cname in ("sh1_outlet_temp", "sh2_inlet_temp"):
        t = obs[:, OBSERVATION_ELEMENTS.index(cname)][msk]
        sh = t - tsat[msk]
        if cname == "sh1_outlet_temp":
            print(f"    Q{i+1} {e3[i]:5.0f}-{e3[i+1]:5.0f} kg/s: "
                  f"p_sep={pm[msk].mean():5.2f} MPa Tsat={tsat[msk].mean():6.1f} "
                  f"| {cname} superheat {sh.mean():6.1f} degC "
                  f"(p5={np.percentile(sh,5):6.1f})", end="")
        else:
            print(f"  | {cname} superheat {sh.mean():6.1f} degC "
                  f"(p5={np.percentile(sh,5):6.1f})")

json.dump({"t1_direction": res_t1}, open(OUT / "report.json", "w"), indent=2)
print("\ndone")
