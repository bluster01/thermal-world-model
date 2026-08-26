"""In-flight delay content test (2026-08-26), evaluation-only, no training.

Hypothesis: the observer emits lumped current states with no transport-delay
line, so cooling injected BEFORE t=0 that arrives DURING the horizon cannot be
represented. Prediction: windows with high pre-window valve activity have much
larger error than quiet windows, and the gap grows with horizon step.

Model: armC anchored seed0 (best arm, H18 0.478) on the ORIGINAL record --
frozen checkpoint, zero training. Stratify the 256-window probe set by valve
movement in the 300 s (30 steps) immediately before t=0.
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
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms

DEVICE = "cuda"
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
IDX_FLOW = BOUNDARY_ELEMENTS.index("steam_flow")
CKPT = (ROOT / "results/final_wm/probes_20260824/retrain_probe"
        / "armC_anchor_s1const_seed0/checkpoints/t1_closure_cons_norew_seed0.pt")
OUT = ROOT / "results/final_wm/probes_20260824/inflight_test"
OUT.mkdir(parents=True, exist_ok=True)
PRE = 30          # 300 s pre-window activity measure
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative_norew",
                epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
model = build_world_model(spec, props).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE,
                                 weights_only=False)["state_dict"])
model.eval()
print(f"loaded {CKPT.name}")

gen = torch.Generator().manual_seed(50_000)
err_steps, act_pre, act_in, loads, quiet_temp, persists = [], [], [], [], [], []
done = 0
with torch.no_grad():
    while done < 256:
        bsz = min(32, 256 - done)
        b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
        hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                   actions=b.history.actions.to(DEVICE),
                                   boundary=b.history.boundary.to(DEVICE))
        r = model.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                           true_future_boundary=b.future_boundary.to(DEVICE))
        e = (b.future_obs[:, :, CH].to(DEVICE) - r.temps_mu[:, :, CH]).abs().cpu()
        err_steps.append(e)
        ha = b.history.actions[:, -PRE:, :]                 # pre-window activity
        act_pre.append(ha.diff(dim=1).abs().sum(dim=1).sum(dim=1))
        fa = b.future_actions
        act_in.append(fa.diff(dim=1).abs().sum(dim=1).sum(dim=1))
        loads.append(b.future_boundary[:, 0, IDX_FLOW])
        quiet_temp.append((b.future_obs[:, :, CH].max(dim=1).values
                           - b.future_obs[:, :, CH].min(dim=1).values))
        # persistence baseline: hold the last observed temperature
        last = b.history.obs[:, -1, CH].unsqueeze(1)
        persists.append((b.future_obs[:, :, CH] - last).abs().mean(dim=1))
        done += bsz

err = torch.cat(err_steps).numpy()               # (256, 18)
pre = torch.cat(act_pre).numpy()
inw = torch.cat(act_in).numpy()
load = torch.cat(loads).numpy()
span = torch.cat(quiet_temp).numpy()
persist = torch.cat(persists).numpy()
mae = err.mean(1)
print(f"\n256 windows: MAE mean={mae.mean():.3f} median={np.median(mae):.3f}")
print(f"pre-window valve travel (300 s): min={pre.min():.3f} med={np.median(pre):.3f} "
      f"max={pre.max():.3f}")

print("\n[1] MAE by PRE-window valve travel tercile (t<0, the in-flight content)")
q = np.quantile(pre, [0, 1/3, 2/3, 1.0])
for i in range(3):
    m = (pre >= q[i]) & (pre <= q[i+1] if i == 2 else pre < q[i+1])
    print(f"  T{i+1} travel {q[i]:.3f}-{q[i+1]:.3f}: n={m.sum():3d} "
          f"MAE={mae[m].mean():6.3f}  H1={err[m, 0].mean():5.3f} "
          f"H6={err[m, 5].mean():5.3f} H12={err[m, 11].mean():5.3f} "
          f"H18={err[m, 17].mean():6.3f}  actual_span={span[m].mean():5.2f}")

print("\n[2] MAE by IN-window valve travel tercile (t>=0, control -- observable to model)")
q2 = np.quantile(inw, [0, 1/3, 2/3, 1.0])
for i in range(3):
    m = (inw >= q2[i]) & (inw <= q2[i+1] if i == 2 else inw < q2[i+1])
    print(f"  T{i+1} travel {q2[i]:.3f}-{q2[i+1]:.3f}: n={m.sum():3d} "
          f"MAE={mae[m].mean():6.3f}  H18={err[m, 17].mean():6.3f}  "
          f"actual_span={span[m].mean():5.2f}")

print("\n[3] 2x2: pre-window activity x load (median split)")
ld_med, pre_med = np.median(load), np.median(pre)
for lo_hi, lm in (("low-load ", load < ld_med), ("high-load", load >= ld_med)):
    for p_hi, pm in (("quiet-pre", pre < pre_med), ("busy-pre ", pre >= pre_med)):
        m = lm & pm
        if m.sum() < 5:
            continue
        print(f"  {lo_hi} {p_hi}: n={m.sum():3d} MAE={mae[m].mean():6.3f} "
              f"H18={err[m, 17].mean():6.3f} span={span[m].mean():5.2f}")

print("\n[4] error growth profile by pre-window activity (per-step MAE)")
m_lo, m_hi = pre < np.quantile(pre, 1/3), pre >= np.quantile(pre, 2/3)
print("  step:  " + "".join(f"{s+1:6d}" for s in range(0, 18, 2)))
print("  quiet: " + "".join(f"{err[m_lo, s].mean():6.2f}" for s in range(0, 18, 2)))
print("  busy:  " + "".join(f"{err[m_hi, s].mean():6.2f}" for s in range(0, 18, 2)))
ratio = [err[m_hi, s].mean() / max(err[m_lo, s].mean(), 1e-9) for s in range(0, 18, 2)]
print("  ratio: " + "".join(f"{x:6.2f}" for x in ratio))

# ---- CONFOUND CONTROL: busy-pre windows also move more. Normalize and match. ----
np.savez_compressed(OUT / "window_stats.npz", err=err, pre=pre, inw=inw,
                    load=load, span=span, persist=persist, mae=mae)
print("\n[5] CONFOUND CONTROL -- persistence-baseline skill (1 - MAE/MAE_persistence)")
print("    (persistence = hold the last observed temperature; a model that only")
print("     smooths cannot beat it on moving windows)")
skill = 1.0 - mae / np.maximum(persist, 1e-9)
for i in range(3):
    m = (pre >= q[i]) & (pre <= q[i+1] if i == 2 else pre < q[i+1])
    print(f"  pre-T{i+1}: n={m.sum():3d} MAE={mae[m].mean():6.3f} "
          f"persist={persist[m].mean():6.3f} skill={skill[m].mean():+.3f} "
          f"MAE/span={np.mean(mae[m]/np.maximum(span[m],1e-9)):.3f}")

print("\n[6] SPAN-MATCHED strata: within each actual-movement quartile,")
print("    compare quiet-pre vs busy-pre (controls window difficulty)")
sq = np.quantile(span, [0, 0.25, 0.5, 0.75, 1.0])
pre_med = np.median(pre)
for i in range(4):
    sm = (span >= sq[i]) & (span <= sq[i+1] if i == 3 else span < sq[i+1])
    a = sm & (pre < pre_med)
    b_ = sm & (pre >= pre_med)
    if a.sum() < 8 or b_.sum() < 8:
        print(f"  span {sq[i]:.2f}-{sq[i+1]:.2f}: too few (quiet {a.sum()}, busy {b_.sum()})")
        continue
    print(f"  span {sq[i]:.2f}-{sq[i+1]:.2f}: quiet n={a.sum():3d} MAE={mae[a].mean():6.3f} "
          f"skill={skill[a].mean():+.3f}  |  busy n={b_.sum():3d} MAE={mae[b_].mean():6.3f} "
          f"skill={skill[b_].mean():+.3f}  |  dMAE={mae[b_].mean()-mae[a].mean():+.3f}")


json.dump({"mae_mean": float(mae.mean()),
           "pre_tercile_mae": [float(mae[(pre >= q[i]) & (pre <= q[i+1] if i == 2
                                          else pre < q[i+1])].mean()) for i in range(3)],
           "busy_quiet_ratio_by_step": [float(x) for x in ratio]},
          open(OUT / "report.json", "w"), indent=2)
print("\ndone")
