"""Z-conditioned parameter scheduling probe (A then B, user-ordered 2026-08-29).

USER DESIGN (with assistant bottleneck): schedule transition parameters from a
2-dim linear projection z of the FUSED HYBRID INITIAL STATE (anchor + masked NN
posterior correction).  The NN correction is computed from the 96-step history
window by the observer GRU, so z carries the encoder's window information --
the user's claim is that this beats the single-scalar load conditioner (A4).

ARMS
  a (run first):  z conditions k only.   k = k(pm) * exp(tanh(v_k . z) * ln K_MAX)
  b (after a):    same shared z conditions k + transport lags + UA.
                  tau_* *= exp(tanh(v_tau . z) * ln TAU_MAX)   (floor 3*dt_sub)
                  ua    *= exp(tanh(v_ua  . z) * ln UA_MAX)

CONTROL: the A5 matched filtered control (t1_a5_filtered_control_seed0)
checkpoint is REUSED as the decision denominator (user-ordered) -- no retrain.
The arm trains on the SAME 7-channel filtered view (BaseRecordView) so the
causal contrast is data- and architecture-identical except for the z factor.

IDENTITY GATE: v = 0 -> factor = 1 -> forecast bit-identical to the control
model.  Training starts at v = 1e-3 (small but nonzero so gradients flow),
mirroring the LPV probe's "starts at baseline, grows only if the data pays".

GATES (identical to A5): accuracy <= -5% vs control, spread <= +10%, direction
v0.3 both valves H18/H60.  Single seed 0 = EXPLORATORY, cannot create a route
verdict.  All src/ untouched -- probe-side subclass rebinding only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.final_wm import matrix_spec as ms
from src.final_wm.analysis import STEAM_FLOW_INDEX
from src.final_wm.contracts import BOUNDARY_ELEMENTS
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL, sample_windows
from src.final_wm.model import HistoryWindow
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import build_world_model, train_arm
from src.final_wm.transition import Fan2020UDETransition
from src.final_wm.water_coal import WaterCoalRecord

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
from a5_water_coal_probe import (  # noqa: E402  (exact protocol reuse)
    BaseRecordView,
    _base_history,
    _error_summary,
    direction_metrics,
    direction_passed,
    probe_spec as a5_probe_spec,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = P / "zcond_schedule_probe"
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_RECORD = ROOT / "artifacts/final_wm/canonical_sideA_v2.npz"
A5_OUT = P / "a5_water_coal_probe"
CONTROL_CKPT = A5_OUT / "control" / "checkpoints" / "t1_a5_filtered_control_seed0.pt"

N_WIN = 256
EVAL_SEED = 50_000
Z_DIM = 2
K_MAX = 1.5        # same band as A4 (k factor in [1/1.5, 1.5])
TAU_MAX = 2.0      # B only: lag factor in [0.5, 2]
UA_MAX = 1.5       # B only
V_INIT = 1e-3
W_INIT = 0.01
A5_CONTROL_H18 = 0.4261806488037109  # stored in the A5 report; re-eval must match


def probe_spec(arm: str):
    return a5_probe_spec(arm)  # identical budget/protocol to A5


class ZSchedTransition(Fan2020UDETransition):
    """Bottleneck-z conditioned k / tau / UA (probe-side subclass).

    `_z` is set at the top of `integrate` from the (B, dim) state_0, so every
    parameter evaluation inside the rollout sees the window's own z.  Outside
    `integrate` (e.g. the steady anchor in `initial_steady_state`) `_z` is
    None and the base behaviour is returned untouched.
    """

    def integrate(self, state_0, boundary_seq, action_seq, *,
                  closure=None, noise=None):
        self._z = self.z_proj((state_0 - self.z_center) / self.z_scale)
        try:
            return super().integrate(state_0, boundary_seq, action_seq,
                                     closure=closure, noise=noise)
        finally:
            # State hygiene: `initial_steady_state` (the NEXT window's anchor)
            # also calls k_of; a stale `_z` would graft the previous batch's
            # freed autograd graph onto the new forward and blow up backward.
            self._z_last = None if self._z is None else self._z.detach()
            self._z = None

    def k_of(self, pm):
        k = super().k_of(pm)
        if self._z is None or self.v_k is None:
            return k
        f = torch.exp(torch.tanh((self.v_k * self._z).sum(-1)) * math.log(self.k_max))
        return k * f.unsqueeze(-1)

    def _substep(self, h, tm, rb, m1, m2, lag1, lag2, dsw1, dsw2, d_flow, u_b,
                 p_stack, p0, p1, h_spray, h_sep, m_cap, ua, cm, k_t,
                 tau_b, tau_evap, tau_mix1, tau_mix2, dt_sub, h_lo, h_hi,
                 steam_power, metal_power):
        if self._z is not None and self.v_tau is not None:
            f_tau = torch.exp(torch.tanh((self.v_tau * self._z).sum(-1)) * math.log(self.tau_max))
            f_ua = torch.exp(torch.tanh((self.v_ua * self._z).sum(-1)) * math.log(self.ua_max))
            floor = 3.0 * dt_sub
            tau_b = (tau_b * f_tau).clamp(min=floor)
            tau_mix1 = (tau_mix1 * f_tau).clamp(min=floor)
            tau_mix2 = (tau_mix2 * f_tau).clamp(min=floor)
            ua = ua * f_ua.unsqueeze(-1)
        return super()._substep(h, tm, rb, m1, m2, lag1, lag2, dsw1, dsw2,
                                d_flow, u_b, p_stack, p0, p1, h_spray, h_sep,
                                m_cap, ua, cm, k_t, tau_b, tau_evap, tau_mix1,
                                tau_mix2, dt_sub, h_lo, h_hi,
                                steam_power, metal_power)


def promote_zcond(model, center: torch.Tensor, scale: torch.Tensor,
                  groups: tuple[str, ...], v_init: float = V_INIT) -> nn.Module:
    """Rebind the transition class in place (LPV-probe pattern) and register
    the z bottleneck.  No src change, all base params/buffers preserved."""
    tr = model.transition
    dev = next(tr.parameters()).device
    tr.__class__ = ZSchedTransition
    tr._z = None
    tr._z_last = None
    tr.k_max = float(K_MAX)
    tr.tau_max = float(TAU_MAX)
    tr.ua_max = float(UA_MAX)
    tr.register_buffer("z_center", center.to(dev))
    tr.register_buffer("z_scale", scale.to(dev))
    z_proj = nn.Linear(center.shape[0], Z_DIM, bias=False).to(dev)
    nn.init.normal_(z_proj.weight, std=W_INIT)
    tr.add_module("z_proj", z_proj)
    init = lambda g: float(v_init) if g in groups else 0.0
    for name in ("v_k", "v_tau", "v_ua"):
        tr.register_parameter(
            name, nn.Parameter(torch.full((Z_DIM,), init(name), device=dev)))
    return model


@torch.no_grad()
def z_stats(model, view) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim mean/std of the fused hybrid initial state over train windows."""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    states = []
    for _ in range(8):  # 256 train windows
        batch = sample_windows(view, SPLIT_TRAIN, 32, 96, 18, gen)
        history = HistoryWindow(
            batch.history.obs.to(DEVICE),
            batch.history.actions.to(DEVICE),
            batch.history.boundary.to(DEVICE),
        )
        states.append(model._initial_state(history).cpu())
    s = torch.cat(states)
    center = s.mean(dim=0)
    scale = s.std(dim=0).clamp(min=1e-3)
    return center, scale


def identity_gate(view, props, state_dict, center, scale,
                  groups: tuple[str, ...]) -> dict:
    """v=0 must reproduce the control model bit-for-bit."""
    torch.manual_seed(0)
    spec = probe_spec("zcond_gate")
    base = build_world_model(spec, props).to(DEVICE).eval()
    base.load_state_dict(state_dict, strict=False)
    arm = build_world_model(spec, props).to(DEVICE).eval()
    arm.load_state_dict(state_dict, strict=False)
    promote_zcond(arm, center, scale, groups, v_init=0.0).eval()
    batch = sample_windows(view, SPLIT_TRAIN, 8, 96, 18,
                           torch.Generator().manual_seed(7))
    history = HistoryWindow(
        batch.history.obs.to(DEVICE),
        batch.history.actions.to(DEVICE),
        batch.history.boundary.to(DEVICE),
    )
    actions = batch.future_actions.to(DEVICE)
    future = batch.future_boundary.to(DEVICE)
    with torch.no_grad():
        r0 = base.forecast(history, actions, boundary_mode="oracle",
                           true_future_boundary=future)
        r1 = arm.forecast(history, actions, boundary_mode="oracle",
                          true_future_boundary=future)
    max_diff = float((r0.temps_mu - r1.temps_mu).abs().max())
    return {"exact": bool(torch.equal(r0.temps_mu, r1.temps_mu)),
            "max_abs_temp_diff_c": max_diff}


def data_diagnostics(record) -> dict:
    return {
        "operating_fraction": float(record.operating_mask.float().mean()),
        "n_total": int(record.n),
        "n_operating": int(record.operating_mask.sum()),
    }


@torch.no_grad()
def paired_eval_z(arm_model, control_model, view) -> dict:
    """Paired 256-window eval on the 7-channel filtered view; captures z."""
    gen = torch.Generator().manual_seed(EVAL_SEED)
    err_arm, err_ctl, loads, days, zs, futs = [], [], [], [], [], []
    done = 0
    while done < N_WIN:
        bsz = min(32, N_WIN - done)
        batch = sample_windows(view, SPLIT_VAL, bsz, 96, 18, gen)
        history = HistoryWindow(
            batch.history.obs.to(DEVICE),
            batch.history.actions.to(DEVICE),
            batch.history.boundary.to(DEVICE),
        )
        result = arm_model.forecast(
            history, batch.future_actions.to(DEVICE), boundary_mode="oracle",
            true_future_boundary=batch.future_boundary.to(DEVICE))
        result_ctl = control_model.forecast(
            _base_history(history), batch.future_actions.to(DEVICE),
            boundary_mode="oracle",
            true_future_boundary=batch.future_boundary.to(DEVICE))
        target = batch.future_obs.to(DEVICE)
        err_arm.append((target - result.temps_mu).abs().cpu())
        err_ctl.append((target - result_ctl.temps_mu).abs().cpu())
        loads.append(batch.future_boundary[:, 0, STEAM_FLOW_INDEX])
        days.append(batch.day_ids)
        zs.append(arm_model.transition._z_last.detach().cpu())
        futs.append(batch.future_boundary.cpu())
        done += bsz
    abs_arm = torch.cat(err_arm)
    abs_ctl = torch.cat(err_ctl)
    load = torch.cat(loads)
    day_ids = torch.cat(days)
    z = torch.cat(zs)
    fb = torch.cat(futs)
    arm_s = _error_summary(abs_arm, load, day_ids)
    ctl_s = _error_summary(abs_ctl, load, day_ids)
    return {
        "arm": arm_s,
        "matched_filtered_control": ctl_s,
        "relative_h18_change": (arm_s["overall_h18_mae"] - ctl_s["overall_h18_mae"])
        / ctl_s["overall_h18_mae"],
        "spread_change": (arm_s["spread_ratio"] - ctl_s["spread_ratio"])
        / ctl_s["spread_ratio"],
        "_z": z,
        "_load": load,
        "_dload": fb[:, -1, STEAM_FLOW_INDEX] - fb[:, 0, STEAM_FLOW_INDEX],
    }


def z_diagnostics(model, eval_pack: dict) -> dict:
    v = model.transition.v_k.detach().cpu()  # z/load captured on CPU
    z = eval_pack["_z"]
    load = eval_pack["_load"].float()
    dload = eval_pack["_dload"].float()
    eff = torch.tanh((v * z).sum(-1))
    factor = torch.exp(eff * math.log(K_MAX))
    corr = lambda a, b: float(np.corrcoef(a.numpy(), b.numpy())[0, 1])
    # ridge R^2 of each z dim on load (standardized)
    def r2(y, x):
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        beta = (x * y).sum() / (x * x).sum()
        return float(1.0 - ((y - beta * x) ** 2).sum() / (y * y).sum())
    return {
        "z_dim": Z_DIM,
        "k_max_band": K_MAX,
        "factor_p01_p50_p99": [float(torch.quantile(factor, q))
                               for q in (0.01, 0.50, 0.99)],
        "corr_effz_load": corr(eff, load),
        "corr_effz_dload": corr(eff, dload),
        "corr_z0_load": corr(z[:, 0], load),
        "corr_z1_load": corr(z[:, 1], load),
        "r2_z0_on_load": r2(z[:, 0], load),
        "r2_z1_on_load": r2(z[:, 1], load),
        "v_k_norm": float(v.norm()),
        "z_proj_w_norm": float(model.transition.z_proj.weight.norm()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--arm", choices=("a", "b"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    groups = ("k",) if args.arm == "a" else ("k", "tau", "ua")
    arm_name = "zcond_k" if args.arm == "a" else "zcond_all"

    record = WaterCoalRecord(args.record)
    view = BaseRecordView(record)
    props = load_grid_properties(ROOT / "artifacts/final_wm/iapws_surrogate.npz",
                                 device=DEVICE)

    if not CONTROL_CKPT.exists():
        raise RuntimeError(f"control checkpoint missing: {CONTROL_CKPT}")
    state_dict = torch.load(CONTROL_CKPT, map_location="cpu",
                            weights_only=False)["state_dict"]
    control = build_world_model(probe_spec("a5_filtered_control"), props).to(DEVICE)
    control.load_state_dict(state_dict)
    control.eval()
    center, scale = z_stats(control, view)

    diagnostics = data_diagnostics(record)
    identity = identity_gate(view, props, state_dict, center, scale, groups)
    print(f"[ZCOND identity] arm={args.arm} groups={groups} "
          f"exact={identity['exact']} max_diff={identity['max_abs_temp_diff_c']:.3e}")
    print(f"[ZCOND data] operating={diagnostics['operating_fraction']:.3%} "
          f"n={diagnostics['n_operating']:,}/{diagnostics['n_total']:,}")
    if not identity["exact"]:
        raise RuntimeError("z-conditioned identity gate failed; refusing to train")

    if args.sanity:
        print(json.dumps({"identity": identity, "data": diagnostics,
                          "groups": groups}, ensure_ascii=False, indent=2))
        return

    sys.path.insert(0, str(P))
    from probe_guard import assert_grid, verify_ledger_properties
    assert_grid(props)
    import src.final_wm.training as training

    arm_out = OUT / f"arm_{args.arm}"

    if args.eval:
        # Reuse the trained arm: replay the post-train eval stage only.
        raw = [json.loads(l) for l in
               (arm_out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]
        final = raw[-1]
        assert final["run_id"] == f"t1_{arm_name}_seed0", final["run_id"]
    else:
        original_builder = training.build_world_model

        def builder(spec, properties=None):
            return promote_zcond(original_builder(spec, properties),
                                 center, scale, groups)

        training.build_world_model = builder
        try:
            final = train_arm(
                probe_spec(arm_name), view, arm_out,
                device=DEVICE, properties=props, compile_substep=False,
            )
        finally:
            training.build_world_model = original_builder
        verify_ledger_properties(arm_out)

    arm_model = promote_zcond(build_world_model(probe_spec(arm_name), props),
                              center, scale, groups).to(DEVICE)
    ckpt = arm_out / "checkpoints" / f"{final['run_id']}.pt"
    arm_model.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                         weights_only=False)["state_dict"])
    arm_model.eval()

    evaluation = paired_eval_z(arm_model, control, view)
    arm_eval = evaluation["arm"]
    ctl_eval = evaluation["matched_filtered_control"]
    directions = direction_metrics(arm_model, view)  # 7-channel arm -> 7-channel view
    zdiag = z_diagnostics(arm_model, evaluation)

    accuracy_pass = arm_eval["overall_h18_mae"] <= ctl_eval["overall_h18_mae"] * 0.95
    spread_pass = arm_eval["spread_ratio"] <= ctl_eval["spread_ratio"] * 1.10
    direction_pass = direction_passed(directions)
    if arm_eval["overall_h18_mae"] >= ctl_eval["overall_h18_mae"] * 1.05 or not direction_pass:
        outcome = "REJECT_EXPLORATORY_SEED0"
    elif accuracy_pass and spread_pass:
        outcome = "PROMOTE_TO_FIXED_SEEDS_1_2"
    else:
        outcome = "INCONCLUSIVE_EXPLORATORY_SEED0"

    a5_report = json.loads((A5_OUT / "report.json").read_text(encoding="utf-8"))
    report = {
        "arm": f"{arm_name}_{'_'.join(groups)}",
        "status": outcome,
        "single_seed_exploratory": True,
        "user_design": "z = 2-dim bottleneck of fused hybrid initial state; "
                       "physical-form bounded factors; A then B (user-ordered)",
        "record": str(args.record),
        "identity_gate": identity,
        "data": diagnostics,
        "train": {
            arm_name: {k: final[k] for k in (
                "run_id", "commit", "best_val_nll", "best_epoch", "epochs_run",
                "stop_reason", "converged", "flags", "timing")},
            "reused_control": {
                "source": str(CONTROL_CKPT),
                "a5_train_record": a5_report["train"]["matched_filtered_control"],
                "a5_report_h18": A5_CONTROL_H18,
                "re_evaluated_h18": ctl_eval["overall_h18_mae"],
                "max_abs_diff": abs(ctl_eval["overall_h18_mae"] - A5_CONTROL_H18),
            },
        },
        "evaluation": {k: v for k, v in evaluation.items()
                       if not k.startswith("_")},
        "z_diagnostics": zdiag,
        "direction_v03": directions,
        "preregistered_gates": {
            "accuracy_at_least_5pct": bool(accuracy_pass),
            "spread_no_more_than_10pct_worse": bool(spread_pass),
            "both_valves_h18_h60_direction": bool(direction_pass),
            "all_pass": bool(accuracy_pass and spread_pass and direction_pass),
        },
        "baseline": {
            "decision_comparator": "reused A5 matched filtered control",
            "protocol": "A5 probe protocol, seed0 exploratory",
        },
    }
    (OUT / f"report_{args.arm}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ZCOND result] arm={args.arm} {outcome} "
          f"H18={arm_eval['overall_h18_mae']:.4f} vs ctl={ctl_eval['overall_h18_mae']:.4f} "
          f"spread={arm_eval['spread_ratio']:.3f} vs {ctl_eval['spread_ratio']:.3f} "
          f"corr(effz,load)={zdiag['corr_effz_load']:+.3f}")
    print(f"written {OUT / f'report_{args.arm}.json'}")


if __name__ == "__main__":
    main()
