"""Amendment v0.4 decision experiment (audit F3, 2026-08-22).

Matched-budget local A/B: closure_cons (intact rewetting) vs
closure_cons_norew (aW frozen ~0).  Local decision tier: epochs=10,
patience=4, same batches/lr as the full tier; quick tier (2 epochs) was
verified uninformative for this decision (both arms best_val ~10800).

Pre-registered decision rule (repair1_rerun_audit_20260822.md F3):
adopt norew iff ALL of
  (i)   falsification reproduces under training: intact v1-step downstream
        response is wrong-signed (sh2_in or final > 0 at 60 s) while norew
        is correct-signed (both < 0);
  (ii)  v2 direction is not degraded: frac_negative(norew) >= frac_negative(intact);
  (iii) parsimony guard: val NLL(norew) <= val NLL(intact) + 0.05.
Otherwise keep repair-3 as-is and register the v1 anomaly as a structural
limitation.  Full-tier adjudication (3 seeds) follows on the execution side
for whichever arm this selects.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from experiments.final_wm import matrix_spec as ms
from src.final_wm.data import SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.evaluation import evaluate_windows, step_response_direction
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import train_arm

RECORD = "artifacts/final_wm/canonical_sideA.npz"
PROPS = "artifacts/final_wm/iapws_surrogate.npz"
OUT = Path("artifacts/final_wm_v04_decision")
REPORT = Path("results/final_wm/v04_rewet_decision_20260822.json")
DECISION_BUDGET = dict(epochs=10, patience=4, eval_windows=64)
NLL_SLACK = 0.05

OBS_NAMES = ["sh1_in", "sh1_out", "sh2_in", "sh2_out", "final"]


@torch.no_grad()
def v1_step_per_channel(model, record, device, n_windows=64, steps=6, delta_v=0.05, seed=60_000):
    """Per-channel gain (degC per full opening) for a v1 +5% step at 60 s."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, SPLIT_VAL, n_windows, ms.HISTORY_STEPS, 1, gen)
    b0 = batch.future_boundary[:, 0].to(device)
    a0 = batch.future_actions[:, 0].to(device)
    obs0 = batch.history.obs[:, -1].to(device)
    s0 = model.transition.initial_steady_state(b0, a0, obs0)
    bs = b0.unsqueeze(1).repeat(1, steps, 1)
    base = a0.unsqueeze(1).repeat(1, steps, 1)
    step = base.clone()
    step[:, :, 0] = (step[:, :, 0] + delta_v).clamp(max=1.0)
    _x, tb = model.transition.integrate(s0, bs, base)
    _y, ts = model.transition.integrate(s0, bs, step)
    gains = ((ts[:, -1] - tb[:, -1]).mean(dim=0) / delta_v).cpu()
    return {name: float(gains[i]) for i, name in enumerate(OBS_NAMES)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = load_grid_properties(PROPS)
    record = CanonicalRecord(RECORD)
    OUT.mkdir(parents=True, exist_ok=True)

    results = {}
    for arm, closure_mode in (
        ("closure_cons", "conservative"),
        ("closure_cons_norew", "conservative_norew"),
    ):
        spec = ms._base(
            "t1", arm, 0, boundary_mode="oracle", initial_state_mode="hybrid",
            closure_mode=closure_mode, latent_dim=0, **DECISION_BUDGET,
        )
        final = train_arm(spec, record, OUT, device=device, properties=props)
        from src.final_wm.training import build_world_model

        model = build_world_model(spec, props).to(device)
        ckpt = OUT / "checkpoints" / f"{final['run_id']}.pt"
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
        )
        metrics = evaluate_windows(
            model, record, SPLIT_VAL, n_windows=64, batch_size=32,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
            boundary_mode="oracle", seed=50_000, device=device,
        )
        probe = dict(record=record, split_id=SPLIT_VAL, n_windows=64,
                     history_steps=ms.HISTORY_STEPS, seed=50_000, device=device)
        results[arm] = {
            "best_val_nll": float(final["best_val_nll"]),
            "eval_h18_mae": float(metrics.mae[:, -1].mean()),
            "v1_step_60s_per_channel": v1_step_per_channel(model, record, device),
            "v1_direction_600s": step_response_direction(model, rollout_steps=60, valve_index=0, **probe),
            "v2_direction_600s": step_response_direction(model, rollout_steps=60, valve_index=1, **probe),
        }
        print(f"[v04] {arm}: " + json.dumps(results[arm], default=str)[:400], flush=True)

    intact = results["closure_cons"]
    norew = results["closure_cons_norew"]
    crit_i = (
        (intact["v1_step_60s_per_channel"]["sh2_in"] > 0
         or intact["v1_step_60s_per_channel"]["final"] > 0)
        and norew["v1_step_60s_per_channel"]["sh2_in"] < 0
        and norew["v1_step_60s_per_channel"]["final"] < 0
    )
    crit_ii = (norew["v2_direction_600s"]["frac_negative"]
               >= intact["v2_direction_600s"]["frac_negative"])
    crit_iii = norew["best_val_nll"] <= intact["best_val_nll"] + NLL_SLACK
    verdict = {
        "budget": DECISION_BUDGET,
        "nll_slack": NLL_SLACK,
        "criteria": {
            "i_falsification_reproduced": bool(crit_i),
            "ii_v2_not_degraded": bool(crit_ii),
            "iii_parsimony": bool(crit_iii),
        },
        "adopt_norew": bool(crit_i and crit_ii and crit_iii),
        "arms": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict["criteria"], indent=1))
    print("adopt_norew:", verdict["adopt_norew"])


if __name__ == "__main__":
    main()
