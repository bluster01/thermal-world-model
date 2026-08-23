"""v0.4 comparison report: accuracy per channel + per-valve response (2026-08-23).

Answers two questions with committed artifacts only:
1. Accuracy on the main steam temperature (final outlet, channel 4): MAE and
   MAPE at H1/H6/H18 for physics_only / closure_cons / closure_cons_norew on
   the production stack (fresh checkpoints, 256 val windows, same protocol as
   run_matrix._train_and_eval: seed 50_000+seed).
2. Response per desuperheater valve: step_response_direction for v1
   (valve_index=0) and v2 (valve_index=1) at 60-step transient and 240-step
   steady horizons, same seeds as the R1 gate (80_000/85_000+seed) so the v2
   numbers must reproduce the committed r1_report_* values -- a built-in
   replay-consistency check.

Evaluation-only: loads checkpoints, no training, no verdict mutation.
Writes results/final_wm/v04_comparison_report_20260823.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.final_wm import matrix_spec as ms
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.evaluation import step_response_direction
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model

RECORD = "artifacts/final_wm/canonical_sideA.npz"
PROPS = "artifacts/final_wm/iapws_surrogate.npz"
OUT_JSON = Path("results/final_wm/v04_comparison_report_20260823.json")
ARMS = ("physics_only", "closure_cons", "closure_cons_norew")
SEEDS = (0, 1, 2)
N_EVAL = 256
MAIN_STEAM_CH = 4  # final outlet = 主汽温


@torch.no_grad()
def per_channel_mae(model, record, spec, seed, device):
    """Replicates evaluate_windows protocol but keeps per-channel errors."""
    model.eval()
    gen = torch.Generator().manual_seed(50_000 + seed)
    abs_err, targets = [], []
    done = 0
    while done < N_EVAL:
        bsz = min(32, N_EVAL - done)
        batch = sample_windows(record, SPLIT_VAL, bsz, ms.HISTORY_STEPS, ms.HORIZON, gen)
        history = batch.history.__class__(
            obs=batch.history.obs.to(device),
            actions=batch.history.actions.to(device),
            boundary=batch.history.boundary.to(device),
        )
        result = model.forecast(
            history, batch.future_actions.to(device),
            boundary_mode=spec.boundary_mode,
            true_future_boundary=batch.future_boundary.to(device)
            if spec.boundary_mode == "oracle" else None,
        )
        target = batch.future_obs.to(device)
        abs_err.append((target - result.temps_mu).abs().cpu())  # (B, H, 5)
        targets.append(target.cpu())
        done += bsz
    err = torch.cat(abs_err)       # (N, H, 5)
    tgt = torch.cat(targets)       # (N, H, 5)
    out = {}
    for h in (1, 3, 6, 12, 18):
        e = err[:, :h, MAIN_STEAM_CH]
        t = tgt[:, :h, MAIN_STEAM_CH]
        out[f"H{h}"] = {
            "mae_c": float(e.mean()),
            "mape_pct": float(((e / t.clamp_min(1e-6)).mean()) * 100.0),
            "mean_target_c": float(t.mean()),
        }
        out[f"H{h}_allch_mae_c"] = float(err[:, :h, :].mean())
        out[f"H{h}_per_channel_mae_c"] = [float(err[:, :h, c].mean()) for c in range(err.shape[2])]
    # per-step main-steam curve (phase1 baselines report step-17 point MAE)
    e_last = err[:, :, MAIN_STEAM_CH]  # (N, H)
    out["main_steam_step_curve_mae"] = [float(e_last[:, i].mean()) for i in range(e_last.shape[1])]
    out["main_steam_step17_mae"] = float(e_last[:, -1].mean())
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = load_grid_properties(PROPS)
    record = CanonicalRecord(RECORD)

    report = {"record": RECORD, "stack": "e95bb88/4b03ebf production", "arms": {}}
    for arm in ARMS:
        for seed in SEEDS:
            ckpt = Path(f"artifacts/final_wm/checkpoints/t1_{arm}_seed{seed}.pt")
            spec = next(s for s in ms.t1_specs((seed,)) if s.arm == arm)
            model = build_world_model(spec, props).to(device)
            model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
            entry = {"accuracy": per_channel_mae(model, record, spec, seed, device)}
            for valve, tag in ((0, "v1"), (1, "v2")):
                for steps, base in ((60, 80_000), (240, 85_000)):
                    r = step_response_direction(
                        model, record, SPLIT_VAL, n_windows=32,
                        history_steps=ms.HISTORY_STEPS, rollout_steps=steps,
                        valve_index=valve, seed=base + seed, device=device)
                    entry[f"{tag}_{steps}step"] = r
            report["arms"].setdefault(arm, {})[f"seed{seed}"] = entry
            acc = entry["accuracy"]
            print(f"[{arm} seed{seed}] main-steam MAE H1={acc['H1']['mae_c']:.3f}C "
                  f"H6={acc['H6']['mae_c']:.3f}C H18={acc['H18']['mae_c']:.3f}C "
                  f"MAPE18={acc['H18']['mape_pct']:.3f}% | "
                  f"v1_60s fracneg={entry['v1_60step']['frac_negative']:.3f} "
                  f"v2_60s fracneg={entry['v2_60step']['frac_negative']:.3f}")
            del model
            torch.cuda.empty_cache()

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[report] written {OUT_JSON}")


if __name__ == "__main__":
    main()
