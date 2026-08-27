"""Fast recovery check (2026-08-26): did accuracy come back with grid properties?

Two cheap stages instead of a 2.5 h full arm:

  A) EVAL-PATH validation (~2 min, no training): run the frozen armA_budget
     checkpoint (the 0.723 reference) through this probe file's own evaluation
     code. If it does not reproduce ~0.723, the evaluation path is broken too and
     no re-run can be trusted.

  B) SHORT-TRAIN validation (~15-20 min): 12 epochs on the SAME old record, same
     spec as armA, grid properties, unanchored, norew -- then compare the val_nll
     trajectory against armA's own recorded ep1-12 curve. Tracking armA means the
     harness is fixed; tracking the void analytic arms means something else is
     still wrong.

Reference curves are read from the arms' ledgers, so nothing is hard-coded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/bluster/projectA/thermal-world-model")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.final_wm.analysis import STEAM_FLOW_INDEX, WindowErrors, binning_stats
from src.final_wm.contracts import OBSERVATION_ELEMENTS
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from experiments.final_wm import matrix_spec as ms
from probe_guard import assert_grid, verify_ledger_properties

DEVICE = "cuda"
P = ROOT / "results/final_wm/probes_20260824"
OUT = P / "recovery_check"
OUT.mkdir(parents=True, exist_ok=True)
CH = OBSERVATION_ELEMENTS.index("final_outlet_temp")
ARMA = P / "retrain_probe/armA_budget"
SHORT_EPOCHS = 12
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(ROOT / "artifacts/final_wm/canonical_sideA.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)


def val_curve(arm_dir):
    rows = [json.loads(l) for l in (Path(arm_dir) / "ledger.jsonl").read_text().splitlines() if l.strip()]
    return {r["epoch"]: r["val_nll"] for r in rows if "val_nll" in r and "epoch" in r}


@torch.no_grad()
def eval_h18(model, n_win=256, seed=50_000):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    errs, loads, done = [], [], 0
    while done < n_win:
        bsz = min(32, n_win - done)
        b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
        hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                   actions=b.history.actions.to(DEVICE),
                                   boundary=b.history.boundary.to(DEVICE))
        r = model.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                           true_future_boundary=b.future_boundary.to(DEVICE))
        errs.append((b.future_obs.to(DEVICE) - r.temps_mu).abs().cpu())
        loads.append(b.future_boundary[:, 0, STEAM_FLOW_INDEX])
        done += bsz
    we = WindowErrors(abs_err=torch.cat(errs), load=torch.cat(loads),
                      day_ids=torch.zeros(done, dtype=torch.long))
    ch4 = binning_stats(we)["H18"]["final_outlet_temp"]
    return float(np.mean(ch4["bin_means"])), ch4["bin_means"]


print("=" * 72)
print("STAGE A  evaluation-path validation on the frozen armA checkpoint")
print("=" * 72)
spec_ref = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative_norew",
                    epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
m = build_world_model(spec_ref, props).to(DEVICE)
ck = ARMA / "checkpoints" / "t1_closure_cons_norew_seed0.pt"
m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=False)["state_dict"])
mae, bins = eval_h18(m)
ref = 0.723
print(f"  armA frozen checkpoint -> H18 ch4 = {mae:.3f}  bins={[round(x,3) for x in bins]}")
print(f"  reference (previous session) = {ref:.3f}   delta = {mae - ref:+.3f}")
print(f"  VERDICT: {'eval path OK' if abs(mae - ref) < 0.05 else 'EVAL PATH MISMATCH'}")

print("\n" + "=" * 72)
print(f"STAGE B  short {SHORT_EPOCHS}-epoch train with grid properties, old record")
print("=" * 72)
spec = ms._base("t1", "closure_cons_norew", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative_norew",
                epochs=SHORT_EPOCHS, patience=SHORT_EPOCHS + 5, batch_size=32,
                batches_per_epoch=200)
assert_grid(props)
final = train_arm(spec, record, OUT, device=DEVICE, properties=props)
verify_ledger_properties(OUT)

new = val_curve(OUT)
a = val_curve(ARMA)
void_arms = {"v1fix(analytic)": val_curve(P / "v1fix_probe"),
             "lpv(analytic)": val_curve(P / "lpv_probe")}
print("\n  val_nll trajectory, matched epochs:")
hdr = "  epoch |   NEW(grid) |  armA(grid) | " + " | ".join(f"{k:16s}" for k in void_arms)
print(hdr)
for ep in sorted(new)[:SHORT_EPOCHS]:
    row = f"  {ep:5d} | {new[ep]:11.3f} | "
    row += f"{a.get(ep, float('nan')):11.3f} | "
    row += " | ".join(f"{d.get(ep, float('nan')):16.3f}" for d in void_arms.values())
    print(row)

eps = [e for e in sorted(new)[:SHORT_EPOCHS] if e in a]
if eps:
    dn = np.mean([abs(new[e] - a[e]) for e in eps])
    dv = np.mean([abs(new[e] - void_arms["v1fix(analytic)"].get(e, np.nan)) for e in eps
                  if e in void_arms["v1fix(analytic)"]])
    print(f"\n  mean |new - armA|  = {dn:.3f}")
    print(f"  mean |new - void|  = {dv:.3f}")
    print(f"  VERDICT: {'harness RESTORED (tracks armA)' if dn < dv else 'STILL BROKEN (tracks the void arms)'}")

model = build_world_model(spec, props).to(DEVICE)
model.load_state_dict(torch.load(OUT / "checkpoints" / f"{final['run_id']}.pt",
                                 map_location=DEVICE, weights_only=False)["state_dict"])
mae_s, bins_s = eval_h18(model)
print(f"\n  short-arm H18 = {mae_s:.3f} bins={[round(x,3) for x in bins_s]} "
      f"(armA at 120/20 = 0.723; a 12-epoch arm should be worse but same order)")
(OUT / "report.json").write_text(json.dumps(
    {"stage_a_frozen_armA_eval": {"h18": mae, "bins": bins, "reference": ref},
     "stage_b_short_train": {"h18": mae_s, "bins": bins_s,
                             "best_val_nll": final["best_val_nll"],
                             "epochs": SHORT_EPOCHS}}, indent=2))
print("\ndone")
