"""Repair-batch-1 diagnostic: decompose the H1 (10 s) error per channel.

Questions (matrix amendment v0.3 item 1, target signature sh1_in H1 bin means
5.3-13.3 C, 38x persistence-abandonment):
  (a) is the t=0 anchor itself off?       g(x0, b0) - obs_0
  (b) how much appears after ONE 10 s step with true boundary/action?
  (c) how does that compare to persistence (obs_0 as the H1 forecast)?

Usage:
  python -m experiments.final_wm.h1_anchor_decomposition \
      --checkpoint artifacts/final_wm/checkpoints/t1_closure_cons_seed0.pt \
      --arm closure_cons --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.training import build_world_model
from src.final_wm.properties import load_grid_properties, AnalyticThermoProperties
from experiments.final_wm import matrix_spec as ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default="artifacts/final_wm/canonical_sideA.npz")
    ap.add_argument("--properties-npz", default="artifacts/final_wm/iapws_surrogate.npz")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", default="closure_cons")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/final_wm/h1_anchor_decomposition_seed0.json")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    props = load_grid_properties(args.properties_npz) if args.properties_npz else AnalyticThermoProperties()
    spec = next(s for s in ms.t1_specs((args.seed,)) if s.arm == args.arm)
    model = build_world_model(spec, props).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["state_dict"])
    model.eval()

    record = CanonicalRecord(args.record)
    gen = torch.Generator().manual_seed(60_000 + args.seed)
    batch = sample_windows(record, SPLIT_VAL, args.n_windows, ms.HISTORY_STEPS, 1, gen)
    b0 = batch.future_boundary[:, 0].to(device)
    a0 = batch.future_actions[:, 0].to(device)
    obs0 = batch.history.obs[:, -1].to(device)
    obs1 = batch.future_obs[:, 0].to(device)

    with torch.no_grad():
        tr = model.transition
        x0 = tr.initial_steady_state(b0, a0, obs0)
        g0 = tr.output_temperatures(x0, b0)
        history = batch.history.__class__(
            obs=batch.history.obs.to(device),
            actions=batch.history.actions.to(device),
            boundary=batch.history.boundary.to(device),
        )
        result = model.forecast(
            history,
            batch.future_actions[:, :1].to(device),
            boundary_mode="oracle",
            true_future_boundary=b0.unsqueeze(1),
        )
        g1 = result.temps_mu[:, 0]

    names = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "final"]
    report = {"checkpoint": args.checkpoint, "n_windows": args.n_windows,
              "init_mode": spec.initial_state_mode, "channels": {}}
    for c, name in enumerate(names):
        anchor_err = (g0[:, c] - obs0[:, c]).cpu()
        h1_err = (g1[:, c] - obs1[:, c]).cpu()
        pers_err = (obs0[:, c] - obs1[:, c]).cpu()
        report["channels"][name] = {
            "anchor_mae_c": float(anchor_err.abs().mean()),
            "anchor_bias_c": float(anchor_err.mean()),
            "h1_mae_c": float(h1_err.abs().mean()),
            "h1_bias_c": float(h1_err.mean()),
            "persistence_mae_c": float(pers_err.abs().mean()),
            "h1_over_persistence": float(h1_err.abs().mean() / pers_err.abs().mean().clamp_min(1e-9)),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
