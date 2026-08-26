"""Mixing-point two-phase feasibility (2026-08-26).

The user's hypothesis: rewetting can only exist at LOW load. Testing it at the
superheater OUTLET (rewet_direction_test T4) is the wrong location -- droplets
would survive (or not) right after the spray injection, where the mixed
temperature is lowest. This computes the mixed-state superheat margin at both
desuperheater mixing points per load bin, using the model's own mix enthalpy
(aux hm1/hm2) and the IAPWS surrogate, plus a data-side bulk cross-check from
the measured total spray flow.
"""
from __future__ import annotations

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = ROOT / "results/final_wm/probes_20260824"
IDX_FLOW = BOUNDARY_ELEMENTS.index("steam_flow")
torch.backends.cuda.matmul.allow_tf32 = True

record = CanonicalRecord(P / "v1fix_probe/canonical_sideA_v1fixed.npz")
props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz", device=DEVICE)
spec = ms._base("t1", "closure_cons", 0, boundary_mode="oracle",
                initial_state_mode="hybrid", closure_mode="conservative",
                epochs=120, patience=20, batch_size=32, batches_per_epoch=200)
model = build_world_model(spec, props).to(DEVICE)
model.load_state_dict(torch.load(P / "rewet_probe/checkpoints/t1_closure_cons_seed0.pt",
                                 map_location=DEVICE, weights_only=False)["state_dict"])
model.eval()

# Capture the mix enthalpies and local pressures the transition actually uses.
grab = []
orig_mix = model.transition._mix_enthalpies


def spy(h, lags, d_flow, h_spray):
    hm1, hm2 = orig_mix(h, lags, d_flow, h_spray)
    grab.append((hm1.detach().float(), hm2.detach().float()))
    return hm1, hm2


model.transition._mix_enthalpies = spy
rows = []
with torch.no_grad():
    gen = torch.Generator().manual_seed(50_000)
    done = 0
    while done < 128:
        bsz = min(32, 128 - done)
        b = sample_windows(record, SPLIT_VAL, bsz, 96, 18, gen)
        grab.clear()
        hist = b.history.__class__(obs=b.history.obs.to(DEVICE),
                                   actions=b.history.actions.to(DEVICE),
                                   boundary=b.history.boundary.to(DEVICE))
        model.forecast(hist, b.future_actions.to(DEVICE), boundary_mode="oracle",
                       true_future_boundary=b.future_boundary.to(DEVICE))
        if grab:
            hm1 = torch.stack([g[0] for g in grab]).mean(0)      # (B,)
            hm2 = torch.stack([g[1] for g in grab]).mean(0)
            bd0 = b.future_boundary[:, 0].to(DEVICE)
            pm = bd0[:, BOUNDARY_ELEMENTS.index("separator_pressure")]
            p_out = bd0[:, BOUNDARY_ELEMENTS.index("outlet_pressure")]
            p0, p1, p2 = model.transition._pressures(pm, p_out)
            t1 = props.temperature_of_ph(p0, hm1)
            t2 = props.temperature_of_ph(p1, hm2)
            ts0 = props.saturation_temperature(p0)
            ts1 = props.saturation_temperature(p1)
            rows.append(torch.stack([
                bd0[:, IDX_FLOW], p0, p1, t1, t2, ts0, ts1,
                t1 - ts0, t2 - ts1], dim=1).cpu())
        done += bsz
model.transition._mix_enthalpies = orig_mix

M = torch.cat(rows).numpy()
flow, p0, p1, t1, t2, ts0, ts1, sh1, sh2 = [M[:, i] for i in range(9)]
e = np.quantile(flow, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
print("Mixing-point state after spray injection (model's own mix enthalpy):")
print("  bin  flow kg/s     p_mix1   T_mix1  Tsat1  superheat1 |  T_mix2  Tsat2  superheat2")
for i in range(5):
    m = (flow >= e[i]) & (flow <= e[i+1] if i == 4 else flow < e[i+1])
    if m.sum() == 0:
        continue
    print(f"  Q{i+1}  {e[i]:5.0f}-{e[i+1]:5.0f} ({m.sum():3d})  "
          f"{p0[m].mean():6.2f}  {t1[m].mean():7.1f} {ts0[m].mean():6.1f} "
          f"{sh1[m].mean():+8.1f}   | {t2[m].mean():7.1f} {ts1[m].mean():6.1f} "
          f"{sh2[m].mean():+8.1f}")
print(f"\n  min superheat over all windows: stage1 {sh1.min():+.1f} degC, "
      f"stage2 {sh2.min():+.1f} degC")
print("  (<=0 would mean the mixed state is at/below saturation -> liquid can")
print("   survive and wall rewetting is physically possible)")

print("\nData-side bulk cross-check (measured total spray flow, adiabatic mix):")
raw = np.load(P / "v1fix_probe/canonical_sideA_v1fixed.npz")
bnd = raw["boundary"]
obs = raw["obs"]
fl = bnd[:, IDX_FLOW]
pm_all = bnd[:, BOUNDARY_ELEMENTS.index("separator_pressure")]
tfw = bnd[:, BOUNDARY_ELEMENTS.index("feedwater_temperature")]
w_spray = bnd[:, BOUNDARY_ELEMENTS.index("spray_flow_total")]
t_in = obs[:, OBSERVATION_ELEMENTS.index("sh1_inlet_temp")]
with torch.no_grad():
    pt = torch.tensor(pm_all, dtype=torch.float32, device=DEVICE)
    tt = torch.tensor(t_in, dtype=torch.float32, device=DEVICE)
    h_steam = props.enthalpy_of_pt(pt, tt) if hasattr(props, "enthalpy_of_pt") else None
    h_water = props.liquid_enthalpy(torch.tensor(tfw, dtype=torch.float32, device=DEVICE))
    tsat_all = props.saturation_temperature(pt).cpu().numpy()
print(f"  spray/steam mass ratio by load bin (measured):")
e2 = np.quantile(fl, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
for i in range(5):
    m = (fl >= e2[i]) & (fl <= e2[i+1] if i == 4 else fl < e2[i+1])
    ratio = w_spray[m] / np.maximum(fl[m], 1e-6)
    print(f"    Q{i+1} {e2[i]:5.0f}-{e2[i+1]:5.0f}: spray/steam={ratio.mean()*100:5.2f}% "
          f"(p95 {np.percentile(ratio,95)*100:5.2f}%)  "
          f"T_sh1_in={t_in[m].mean():6.1f} Tsat={tsat_all[m].mean():6.1f} "
          f"margin={t_in[m].mean()-tsat_all[m].mean():+6.1f}")
print("\ndone")
