"""Unified Linux execution entry for the final world model matrix.

Phases (each idempotent, artifacts appended read-only):

    python experiments/final_wm/run_matrix.py --phase discover --data-root /home/bluster/Desktop/AI --out artifacts/final_wm
    python experiments/final_wm/run_matrix.py --phase build    --data-root ... --mapping configs/final_wm/channel_mapping.json --out artifacts/final_wm
    python experiments/final_wm/run_matrix.py --phase dsyn     --out artifacts/final_wm
    python experiments/final_wm/run_matrix.py --phase matrix   --record artifacts/final_wm/canonical.npz --out artifacts/final_wm

`--quick` is a dry-run/smoke mode (tiny sizes, no verdicts).  Execution
must not edit code or thresholds; failures are reported as-is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import (
    SPLIT_VAL,
    CanonicalRecord,
    build_canonical,
    discover_dataset,
    import_dual_canonical,
)
from src.final_wm.diagnostics import leakage_probe
from src.final_wm.evaluation import (
    WindowMetrics,
    boundary_forecast_metrics,
    calibration_coverage,
    constant_condition_stability,
    constraint_checks,
    counterfactual_fidelity_synthetic,
    day_block_mean_ci,
    evaluate_windows,
    horizon_summary,
    paired_difference_ci,
    persistence_boundary_metrics,
    relative_improvement_ci,
    residual_quantiles,
    state_continuity_metrics,
    step_response_direction,
)
from src.final_wm.properties import AnalyticThermoProperties, load_grid_properties
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.training import build_world_model, config_fingerprint, train_arm

from experiments.final_wm import matrix_spec as ms


def closure_blindness_check(model, device) -> dict:
    """Runtime action/W blindness check for the closure (R1).

    The W channel (spray_flow_total, excluded from the closure whitelist) must
    not move closure output.  Extracted as a top-level function so the import
    path is covered by tests (the original inline version referenced a
    non-existent CHANNEL_INDEX and crashed the first Linux run).
    """
    from src.final_wm.contracts import BOUNDARY_ELEMENTS, CLOSURE_BOUNDARY_CHANNELS

    if "spray_flow_total" in CLOSURE_BOUNDARY_CHANNELS:
        raise FinalWMProtocolError("closure whitelist must exclude spray_flow_total")
    batch_state = torch.zeros(2, model.layout.dim, device=device)
    boundary = torch.zeros(2, len(BOUNDARY_ELEMENTS), device=device)
    w_idx = BOUNDARY_ELEMENTS.index("spray_flow_total")
    boundary[:, w_idx] = 5.0
    with torch.no_grad():
        out_w = model.closure(batch_state, boundary)
        boundary[:, w_idx] = 0.0
        out_wo = model.closure(batch_state, boundary)
    blind = (
        torch.allclose(out_w.steam_power, out_wo.steam_power)
        and torch.allclose(out_w.metal_power, out_wo.metal_power)
    )
    if out_w.latent_step is not None or out_wo.latent_step is not None:
        blind = blind and (
            out_w.latent_step is not None
            and out_wo.latent_step is not None
            and torch.allclose(out_w.latent_step, out_wo.latent_step)
        )
    return {"runtime_blind_ok": bool(blind)}


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _properties(path: str | None):
    if path:
        return load_grid_properties(path)
    return AnalyticThermoProperties()


def _np_default(o):
    """JSON default hook: convert numpy scalars (additive, no behavior change for serializable payloads)."""
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_np_default), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase: discover / build
# ---------------------------------------------------------------------------

def phase_discover(args) -> None:
    report = discover_dataset(args.data_root)
    out = Path(args.out)
    _write_json(out / "d0_discovery_report.json", report)
    print(f"[d0] discovery report written: {out / 'd0_discovery_report.json'} "
          f"({len(report['files'])} files)")


def phase_build(args) -> None:
    if not args.mapping:
        raise FinalWMProtocolError("--mapping is required for the build phase")
    out = Path(args.out)
    record_path = args.record or str(out / "canonical.npz")
    report = build_canonical(args.data_root, args.mapping, record_path)
    _write_json(out / "d0_quality_report.json", {
        "gap_ratio": report.gap_ratio,
        "stuck_ratio": report.stuck_ratio,
        "valve_active_ratio": report.valve_active_ratio,
        "days": report.days,
    })
    print(f"[d0] canonical record built: {record_path} (gates passed)")


def phase_split_sides(args) -> None:
    """Bridge the D0 dual-side record into per-side registry-schema records."""
    if not args.record:
        raise FinalWMProtocolError("--record (dual canonical npz) required for split-sides")
    written = import_dual_canonical(args.record, args.out)
    for side, path in written.items():
        print(f"[d0] side {side} record: {path} (gates passed)")
    _write_json(Path(args.out) / "split_sides_report.json", {"records": written})


# ---------------------------------------------------------------------------
# Phase: D-SYN same-type solvability gate
# ---------------------------------------------------------------------------

def run_dsyn(args) -> dict:
    device = _device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = (0,) if args.quick else ms.SEEDS
    total_steps = 2_000 if args.quick else 20_000
    results = []
    for seed in seeds:
        torch.manual_seed(10_000 + seed)
        teacher_spec = ms._base("dsyn", "teacher", seed, closure_mode="conservative")
        teacher = build_world_model(teacher_spec, AnalyticThermoProperties()).to(device)
        with torch.no_grad():  # deterministic same-type perturbation away from priors
            gen = torch.Generator().manual_seed(20_000 + seed)
            for name, p in teacher.transition.named_parameters():
                if name.startswith("raw_"):
                    p.add_(0.15 * torch.randn(p.shape, generator=gen).to(device))
        arrays = synthetic_canonical_arrays(total_steps, seed=30_000 + seed, teacher=teacher.transition)
        record_path = out / f"dsyn_record_seed{seed}.npz"
        np.savez_compressed(record_path, **arrays)
        record = CanonicalRecord(record_path)

        skeleton = build_world_model(
            ms._base("dsyn", "skeleton", seed), AnalyticThermoProperties()
        ).to(device)
        skeleton_val = float(evaluate_windows(
            skeleton, record, SPLIT_VAL, n_windows=32 if args.quick else 128, batch_size=16,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON, boundary_mode="oracle",
            seed=seed, device=device,
        ).nll.mean())

        student_spec = ms._base(
            "dsyn", "student", seed, boundary_mode="oracle",
            initial_state_mode="learned", closure_mode="conservative",
        )
        if args.quick:
            student_spec = ms.quicken(student_spec)
        final = train_arm(student_spec, record, out, device=device, properties=AnalyticThermoProperties(), compile_substep=_use_compile(args))
        # Matrix rule: student must cut the skeleton's validation NLL by >=30%
        # of its magnitude (sign-agnostic gap criterion).
        improvement = skeleton_val - final["best_val_nll"]
        passed = bool(improvement >= 0.3 * abs(skeleton_val))
        # CF-1 (checklist credential B4): counterfactual delta-trajectory
        # fidelity vs the known teacher, evidence-only for now.
        student = build_world_model(student_spec, AnalyticThermoProperties()).to(device)
        student.load_state_dict(torch.load(
            out / "checkpoints" / f"{final['run_id']}.pt",
            map_location=device, weights_only=False)["state_dict"])
        cf1 = counterfactual_fidelity_synthetic(
            student, teacher.transition, record, SPLIT_VAL,
            n_windows=16 if args.quick else 64,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
            seed=95_000 + seed, device=device,
        )
        results.append({
            "seed": seed,
            "skeleton_val_nll": skeleton_val,
            "student_val_nll": final["best_val_nll"],
            "improvement": improvement,
            "pass": passed,
            "cf1": cf1,
        })
        print(f"[dsyn] seed={seed} skeleton={skeleton_val:.3f} student={final['best_val_nll']:.3f} pass={passed}")
    verdict = {
        "unit": "dsyn",
        "per_seed": results,
        "passes": sum(r["pass"] for r in results),
        "verdict": "PASS" if sum(r["pass"] for r in results) >= ms.MIN_SEED_PASSES or args.quick else "FAIL",
        "quick": bool(args.quick),
    }
    # Quick smoke runs must not clobber the full-size verdict artifact
    # (Hermes rerun failure report 2026-08-20 §6).
    name = "dsyn_verdict_quick.json" if args.quick else "dsyn_verdict.json"
    _write_json(out / name, verdict)
    return verdict


# ---------------------------------------------------------------------------
# Phase: matrix units
# ---------------------------------------------------------------------------

def _save_metrics(out: Path, run_id: str, metrics, spec, final) -> Path:
    path = out / "metrics" / f"{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metrics": {
            "nll": metrics.nll, "mae": metrics.mae, "crps": metrics.crps,
            "day_ids": metrics.day_ids,
        },
        "final": final,
        "fingerprint": config_fingerprint(spec),
    }, path)
    return path


def _save_eval_metrics(out: Path, run_id: str, metrics) -> Path:
    """Flat storage for secondary evaluation metrics (B1 boundary, J1 staged).
    These are audit artifacts, not resume keys."""
    path = out / "metrics" / f"{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "nll": metrics.nll, "mae": metrics.mae, "crps": metrics.crps, "day_ids": metrics.day_ids,
    }, path)
    return path


def _try_resume(spec, out) -> tuple[dict, WindowMetrics] | None:
    """Skip-if-artifacts-exist resume: a run is reused iff its checkpoint and
    metrics exist AND the stored code fingerprint matches.

    2026-08-22 audit fix: the legacy flat metrics format (no fingerprint)
    used to authorize resume via a spec-only ledger match; a code-only
    repair (batch 1 changed the observer architecture, not any spec field)
    then silently re-emitted pre-repair O1 verdicts as fresh.  Legacy blobs
    therefore never resume -- they retrain."""
    run_id = f"{spec.unit}_{spec.arm}_seed{spec.seed}"
    ckpt = Path(out) / "checkpoints" / f"{run_id}.pt"
    mpath = Path(out) / "metrics" / f"{run_id}.pt"
    if not (ckpt.exists() and mpath.exists()):
        return None
    blob = torch.load(mpath, map_location="cpu", weights_only=False)
    if "metrics" not in blob:  # legacy flat format: no fingerprint, no resume
        return None
    if blob.get("fingerprint") != config_fingerprint(spec):
        return None
    metrics = WindowMetrics(**blob["metrics"])
    final = dict(blob.get("final") or {}, run_id=run_id, resumed=True)
    final.setdefault("best_val_nll", float("nan"))
    return final, metrics


def _use_compile(args) -> bool:
    """CLI flag; test Namespaces may not carry the attribute."""
    return bool(getattr(args, "compile", False))


def _train_and_eval(spec, record, out, device, properties, quick, use_compile=False):
    if quick:
        spec = ms.quicken(spec)
    resumed = _try_resume(spec, out)
    if resumed is not None:
        final, metrics = resumed
        print(f"[{spec.unit}] {spec.arm} seed={spec.seed} RESUMED (artifacts match spec) "
              f"eval={horizon_summary(metrics)}")
        return final, metrics
    final = train_arm(spec, record, out, device=device, properties=properties,
                      compile_substep=use_compile)
    run_id = final["run_id"]
    ckpt = out / "checkpoints" / f"{run_id}.pt"
    model = build_world_model(spec, properties).to(device)
    if use_compile:
        model.transition._substep = torch.compile(model.transition._substep, dynamic=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    n_eval = 32 if quick else 256
    metrics = evaluate_windows(
        model, record, SPLIT_VAL, n_windows=n_eval, batch_size=32,
        history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
        boundary_mode=spec.boundary_mode, seed=50_000 + spec.seed, device=device,
    )
    _save_metrics(out, run_id, metrics, spec, final)
    print(f"[{spec.unit}] {spec.arm} seed={spec.seed} best_val={final['best_val_nll']:.3f} "
          f"eval={horizon_summary(metrics)}")
    return final, metrics


def _load_checkpoint_model(spec, out, device, properties):
    run_id = f"{spec.unit}_{spec.arm}_seed{spec.seed}"
    model = build_world_model(spec, properties).to(device)
    model.load_state_dict(torch.load(
        Path(out) / "checkpoints" / f"{run_id}.pt",
        map_location=device,
        weights_only=False,
    )["state_dict"])
    return model


def _seed_relative_passes(pairs, threshold, metric, horizon):
    """pairs: list of (baseline_metrics, arm_metrics) per seed."""
    passes, details = 0, []
    for base, arm in pairs:
        ci = relative_improvement_ci(
            base, arm, horizon=horizon, metric=metric
        )
        ok = ci.point >= threshold and ci.ci_lo > 0.0
        passes += int(ok)
        details.append({"point": ci.point, "ci_lo": ci.ci_lo, "ci_hi": ci.ci_hi,
                        "n_days": ci.n_days, "pass": bool(ok)})
    return passes, details


def _seed_delta_passes(pairs, horizon):
    """Formal NLL gate: paired ``arm - baseline`` CI must be below zero."""
    passes, details = 0, []
    for base, arm in pairs:
        ci = paired_difference_ci(base, arm, horizon=horizon, metric="nll")
        ok = ci.ci_hi < 0.0
        passes += int(ok)
        details.append({"point": ci.point, "ci_lo": ci.ci_lo, "ci_hi": ci.ci_hi,
                        "n_days": ci.n_days, "pass": bool(ok)})
    return passes, details


def _practical_effects(pairs, horizon):
    """Relative effect sizes for positive-scale metrics; never a NLL gate."""
    return [
        {
            metric: relative_improvement_ci(
                base, arm, horizon=horizon, metric=metric
            )._asdict()
            for metric in ("crps", "mae")
        }
        for base, arm in pairs
    ]


def _verdict(passes, n_seeds):
    if passes >= ms.MIN_SEED_PASSES:
        return "SUPPORTED"
    if passes == 0:
        return "REJECTED"
    return "MIXED"


def _adjudicate(unit, proposed_verdict, evidence, *, quick, seeds, arm_filter):
    """Apply the executable evidence contract before exposing a verdict."""
    required = ms.REQUIRED_EVIDENCE[unit]
    missing = [name for name in required if name not in evidence or evidence[name] is None]
    reasons = []
    if tuple(seeds) != tuple(ms.SEEDS):
        reasons.append("partial_seed_set")
    if arm_filter is not None:
        reasons.append("arm_filtered_execution")
    if missing:
        reasons.append("missing_required_evidence")
    if quick:
        verdict = "SMOKE"
        status = "SMOKE"
    elif reasons:
        verdict = "INCOMPLETE"
        status = "INCOMPLETE"
    else:
        verdict = proposed_verdict
        status = "COMPLETE"
    return {
        "verdict": verdict,
        "status": status,
        "required_evidence": list(required),
        "missing_evidence": missing,
        "incomplete_reasons": reasons,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# auditpack: protocolized evidence-chain analyses (audit 2026-08-20 §P3)
# ---------------------------------------------------------------------------

def run_auditpack(args) -> dict:
    """Record-only analyses always run; model-based probes run when
    --checkpoint is given (model rebuilt from the matching T1 spec)."""
    from src.final_wm.analysis import (
        binning_stats,
        error_floor_anchors,
        event_study_summary,
        mixing_cooling_reference,
        persistence_increment_mae,
        position_binned_gain,
        rewetting_ablation,
        spray_sensitivity,
        valve_step_events,
        window_abs_errors,
    )

    device = _device(args.device)
    record = CanonicalRecord(args.record)
    out = Path(args.out)
    report: dict = {"record": str(args.record), "matrix_version": ms.MATRIX_VERSION,
                    "arm": args.arm}
    sensitivity = spray_sensitivity(record, SPLIT_VAL)
    report["spray_sensitivity"] = sensitivity
    report["mixing_reference"] = {
        "v1": mixing_cooling_reference(sensitivity["dW_dv1_kgs_per_2pct"]),
        "v2": mixing_cooling_reference(sensitivity["dW_dv2_kgs_per_2pct"]),
    }
    report["persistence_increment_mae"] = persistence_increment_mae(record, SPLIT_VAL)
    report["error_floor"] = error_floor_anchors(record, SPLIT_VAL)
    report["event_study"] = {
        f"v{v + 1}": event_study_summary(valve_step_events(record, SPLIT_VAL, v))
        for v in (0, 1)
    }
    if args.checkpoint:
        arm, seed = args.arm, int(args.seed)
        spec = next(s for s in ms.t1_specs((seed,)) if s.arm == arm)
        properties = _properties(args.properties_npz)
        report["properties_probes"] = type(properties).__name__
        model = build_world_model(spec, properties).to(device)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=False)["state_dict"])
        errors = window_abs_errors(
            model, record, SPLIT_VAL, n_windows=64 if args.quick else 512, batch_size=32,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
            boundary_mode="oracle", seed=110_000 + seed, device=device,
        )
        report["residual_binning"] = binning_stats(errors)
        report["rewetting_ablation"] = rewetting_ablation(
            model, record, SPLIT_VAL, n_windows=16 if args.quick else 64,
            history_steps=ms.HISTORY_STEPS, seed=120_000 + seed, device=device,
        )
        # CF/D1 credentials (checklist 2026-08-21): constraint consistency,
        # position-binned local gain, calibration coverage.  Evidence-only.
        report["constraint_checks"] = constraint_checks(
            model, record, SPLIT_VAL, n_windows=8 if args.quick else 32,
            history_steps=ms.HISTORY_STEPS,
            rollout_steps=30 if args.quick else 120,
            seed=130_000 + seed, device=device,
        )
        report["calibration_coverage"] = calibration_coverage(
            model, record, SPLIT_VAL,
            n_windows=32 if args.quick else 256, batch_size=32,
            history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
            seed=140_000 + seed, device=device,
        )
        report["position_binned_gain"] = {
            f"v{v + 1}": position_binned_gain(
                record, SPLIT_VAL, v, model=model,
                history_steps=ms.HISTORY_STEPS,
                rollout_steps=60,
                n_windows=32 if args.quick else 256,
                seed=150_000 + seed + v, device=device,
            )
            for v in (0, 1)
        }
    # Non-default arms (v0.4 norew stack) get their own file: the frozen
    # closure_cons auditpack artifact must not be clobbered.
    arm_suffix = "" if args.arm == "closure_cons" else f"_{args.arm}"
    name = f"auditpack{('_' + args.side) if args.side else ''}{arm_suffix}{'_quick' if args.quick else ''}.json"
    _write_json(out / name, report)
    print(f"[auditpack] written: {out / name}")
    return report


def run_leakdist(args) -> dict:
    """Multi-shuffle leakage-null distribution for one trained arm (2026-08-21,
    seed1 marginal case: single-shuffle delta 5.75% vs the 5% gate).

    Diagnostic only: the frozen single-shuffle verdict stands on record; this
    distribution is the evidence base for a possible v0.4 gate amendment.
    """
    device = _device(args.device)
    record = CanonicalRecord(args.record)
    out = Path(args.out)
    seed = int(args.seed)
    spec = next(s for s in ms.t1_specs((seed,)) if s.arm == args.arm)
    model = build_world_model(spec, _properties(args.properties_npz)).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)["state_dict"])
    report = leakage_probe(
        model, record,
        n_windows=64 if args.quick else 512,
        history_steps=ms.HISTORY_STEPS,
        epochs=3 if args.quick else 20,
        n_shuffles=4 if args.quick else 16,
        seed=90_000 + seed, device=device,
    )
    report.update({"arm": args.arm, "seed": seed, "checkpoint": str(args.checkpoint)})
    name = f"leakdist_{args.arm}_seed{seed}{'_quick' if args.quick else ''}.json"
    _write_json(out / name, report)
    print(f"[leakdist] written: {out / name}")
    return report


def run_matrix(args) -> dict:
    device = _device(args.device)
    properties = _properties(args.properties_npz)
    out = Path(args.out)
    record = CanonicalRecord(args.record)
    units = args.units.split(",") if args.units else ["o1", "t1", "b1", "j1", "r1"]
    quick = bool(args.quick)
    seeds = (0,) if quick else ms.SEEDS
    if not quick and getattr(args, "seeds", None):
        seeds = tuple(int(s) for s in getattr(args, "seeds", None).split(","))

    def _filter_arm(specs):
        """Execution-side parallel-worker filter: keep only one arm's specs.
        Verdicts are computed only on the unfiltered aggregation pass."""
        if getattr(args, "arm_filter", None) is None:
            return specs
        return [s for s in specs if getattr(s, "arm", None) == getattr(args, "arm_filter", None)]

    summary: dict = {
        "quick": quick,
        "matrix_version": ms.MATRIX_VERSION,
        "required_evidence": {k: list(v) for k, v in ms.REQUIRED_EVIDENCE.items()},
        "side": args.side,
        "record": str(args.record),
        "properties": type(properties).__name__,
        "units": {},
    }
    # Quick smoke runs write a separate summary so they cannot clobber the
    # audited full-size artifact (Hermes rerun failure report 2026-08-20 §6).
    summary_name = "matrix_summary.json" if not args.side else f"matrix_summary_side{args.side}.json"
    if quick:
        summary_name = summary_name.replace(".json", "_quick.json")

    def dump_summary() -> None:
        # Incremental verdict persistence: a crash in a later unit must not
        # lose verdicts already computed (the first Linux run lost O1..J1 to
        # the R1 crash).  Rewritten after every unit.
        # 2026-08-22 audit fix: merge with the on-file summary per unit key --
        # separate invocations (e.g. `--units t1,r1` then `--units o1`) must
        # not clobber each other's verdicts.  Fresh unit blocks overwrite any
        # stale block for the SAME unit.
        path = out / summary_name
        merged = summary
        if path.exists():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = {}
            same_protocol = (
                prior.get("matrix_version") == ms.MATRIX_VERSION
                and prior.get("quick") is quick
                and prior.get("side") == args.side
            )
            if same_protocol and isinstance(prior.get("units"), dict):
                merged = {**prior, **{k: v for k, v in summary.items() if k != "units"}}
                merged["units"] = {**prior["units"], **summary["units"]}
        _write_json(path, merged)

    metrics_store: dict[str, object] = {}

    if "o1" in units:
        o1_run_specs = _filter_arm(ms.o1_specs(seeds))
        for spec in o1_run_specs:
            final, metrics = _train_and_eval(spec, record, out, device, properties, quick, _use_compile(args))
            metrics_store[final["run_id"]] = metrics
        if not quick and getattr(args, "arm_filter", None) is None:
            continuity = {}
            for spec in o1_run_specs:
                model = _load_checkpoint_model(spec, out, device, properties)
                continuity[(spec.arm, spec.seed)] = state_continuity_metrics(
                    model, record, SPLIT_VAL,
                    n_windows=256, history_steps=ms.HISTORY_STEPS,
                    gap_steps=min(ms.HORIZON, ms.HISTORY_STEPS),
                    seed=55_000 + spec.seed, device=device,
                )
            unit_verdicts = {}
            for arm in ("learned", "hybrid"):
                pairs = [
                    (metrics_store[f"o1_steady_seed{s}"], metrics_store[f"o1_{arm}_seed{s}"])
                    for s in seeds
                ]
                _h6_passes, h6 = _seed_delta_passes(pairs, horizon=6)
                h18_passes, h18 = _seed_delta_passes(pairs, horizon=18)
                continuity_details = []
                combined_passes = 0
                for i, seed in enumerate(seeds):
                    base_ci = day_block_mean_ci(
                        continuity[("steady", seed)].values,
                        continuity[("steady", seed)].day_ids,
                        seed=56_000 + seed,
                    )
                    arm_ci = day_block_mean_ci(
                        continuity[(arm, seed)].values,
                        continuity[(arm, seed)].day_ids,
                        seed=57_000 + seed,
                    )
                    continuity_ok = bool(
                        base_ci["identifiable"]
                        and arm_ci["identifiable"]
                        and arm_ci["ci_hi"] <= base_ci["point"]
                    )
                    combined_passes += int(h18[i]["pass"] and continuity_ok)
                    continuity_details.append({
                        "seed": seed,
                        "steady": base_ci,
                        "arm": arm_ci,
                        "pass": continuity_ok,
                    })
                evidence = {
                    "nll_h6": h6,
                    "nll_h18": h18,
                    "state_continuity": continuity_details,
                    "paired_nll_v07": {
                        "rule": "delta_nll_arm_minus_baseline_ci_hi_lt_0",
                        "horizon": 18,
                        "passes": h18_passes,
                        "per_seed": h18,
                    },
                    "practical_effects_h18": _practical_effects(pairs, horizon=18),
                }
                unit_verdicts[arm] = _adjudicate(
                    "o1", _verdict(combined_passes, len(seeds)), evidence,
                    quick=quick, seeds=seeds, arm_filter=getattr(args, "arm_filter", None),
                )
            summary["units"]["o1"] = unit_verdicts
        else:
            summary["units"]["o1"] = _adjudicate(
                "o1", "INCOMPLETE", {}, quick=quick, seeds=seeds,
                arm_filter=getattr(args, "arm_filter", None),
            )
        dump_summary()

    if "t1" in units:
        t1_run_specs = _filter_arm(ms.t1_specs(seeds))
        for spec in t1_run_specs:
            final, metrics = _train_and_eval(spec, record, out, device, properties, quick, _use_compile(args))
            metrics_store[final["run_id"]] = metrics
        if not quick and getattr(args, "arm_filter", None) is None:
            stability = {}
            for spec in t1_run_specs:
                model = _load_checkpoint_model(spec, out, device, properties)
                stability[(spec.arm, spec.seed)] = constant_condition_stability(
                    model, record, SPLIT_VAL,
                    n_windows=64, history_steps=ms.HISTORY_STEPS,
                    rollout_steps=60, seed=58_000 + spec.seed, device=device,
                )
            unit_verdicts = {}
            nested = [
                ("closure_cons", "physics_only"),
                ("closure_steam", "closure_cons"),
                ("latent4", "closure_cons"),
            ]
            for arm, base_arm in nested:
                pairs = [
                    (metrics_store[f"t1_{base_arm}_seed{s}"], metrics_store[f"t1_{arm}_seed{s}"])
                    for s in seeds
                ]
                _h1_passes, h1 = _seed_delta_passes(pairs, horizon=1)
                _h6_passes, h6 = _seed_delta_passes(pairs, horizon=6)
                h18_passes, h18 = _seed_delta_passes(pairs, horizon=18)
                stability_details = [
                    {
                        "seed": seed,
                        "baseline": stability[(base_arm, seed)],
                        "arm": stability[(arm, seed)],
                    }
                    for seed in seeds
                ]
                proposed = _verdict(h18_passes, len(seeds))
                if any(not item["arm"]["bounded"] for item in stability_details):
                    proposed = "REJECTED"
                evidence = {
                    "nll_h1": h1,
                    "nll_h6": h6,
                    "nll_h18": h18,
                    "constant_h60_stability": stability_details,
                    "paired_nll_v07": {
                        "rule": "delta_nll_arm_minus_baseline_ci_hi_lt_0",
                        "horizon": 18,
                        "passes": h18_passes,
                        "per_seed": h18,
                    },
                    "practical_effects_h18": _practical_effects(pairs, horizon=18),
                }
                unit_verdicts[f"{arm}_vs_{base_arm}"] = _adjudicate(
                    "t1", proposed, evidence,
                    quick=quick, seeds=seeds, arm_filter=getattr(args, "arm_filter", None),
                )
            summary["units"]["t1"] = unit_verdicts
        else:
            summary["units"]["t1"] = _adjudicate(
                "t1", "INCOMPLETE", {}, quick=quick, seeds=seeds,
                arm_filter=getattr(args, "arm_filter", None),
            )
        dump_summary()

    if "b1" in units:
        b_metrics = {}
        b_downstream = {}
        b_horizon = ms.HORIZON if quick else 36
        for spec in _filter_arm(ms.b1_specs(seeds)):
            final, _ = _train_and_eval(spec, record, out, device, properties, quick, _use_compile(args))
            model = _load_checkpoint_model(spec, out, device, properties)
            b_metrics[spec.seed] = boundary_forecast_metrics(
                model, record, SPLIT_VAL, n_windows=32 if quick else 256, batch_size=32,
                history_steps=ms.HISTORY_STEPS, horizon=b_horizon,
                seed=60_000 + spec.seed, device=device,
            )
            _save_eval_metrics(out, f"{final['run_id']}_boundary", b_metrics[spec.seed])
            if not quick:
                forecast = evaluate_windows(
                    model, record, SPLIT_VAL, n_windows=256, batch_size=32,
                    history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
                    boundary_mode="forecast", seed=61_000 + spec.seed, device=device,
                )
                oracle = evaluate_windows(
                    model, record, SPLIT_VAL, n_windows=256, batch_size=32,
                    history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
                    boundary_mode="oracle", seed=61_000 + spec.seed, device=device,
                )
                delta = forecast.nll[:, :ms.HORIZON].mean(dim=1) - oracle.nll[:, :ms.HORIZON].mean(dim=1)
                b_downstream[spec.seed] = day_block_mean_ci(
                    delta, forecast.day_ids, seed=62_000 + spec.seed,
                )
        if not quick and getattr(args, "arm_filter", None) is None:
            pairs = []
            for s in seeds:
                base = persistence_boundary_metrics(
                    record, SPLIT_VAL, n_windows=256, batch_size=32,
                    history_steps=ms.HISTORY_STEPS, horizon=b_horizon, seed=60_000 + s,
                )
                pairs.append((base, b_metrics[s]))
            _h6_passes, h6 = _seed_relative_passes(
                pairs, ms.THRESH_B1_CRPS, metric="crps", horizon=6
            )
            h18_passes, h18 = _seed_relative_passes(
                pairs, ms.THRESH_B1_CRPS, metric="crps", horizon=18
            )
            _h36_passes, h36 = _seed_relative_passes(
                pairs, ms.THRESH_B1_CRPS, metric="crps", horizon=36
            )
            evidence = {
                "boundary_h6": h6,
                "boundary_h18": h18,
                "boundary_h36": h36,
                "downstream_h18": (
                    [{"seed": seed, **b_downstream[seed]} for seed in seeds]
                    if all(b_downstream[seed]["identifiable"] for seed in seeds)
                    else None
                ),
            }
            summary["units"]["b1"] = _adjudicate(
                "b1", _verdict(h18_passes, len(seeds)), evidence,
                quick=quick, seeds=seeds, arm_filter=getattr(args, "arm_filter", None),
            )
        else:
            summary["units"]["b1"] = _adjudicate(
                "b1", "INCOMPLETE", {}, quick=quick, seeds=seeds,
                arm_filter=getattr(args, "arm_filter", None),
            )
        dump_summary()

    if "j1" in units:
        joint_metrics, staged_metrics = {}, {}
        joint_models, staged_models = {}, {}
        j1_run_specs = _filter_arm(ms.j1_specs(seeds))
        for spec in j1_run_specs:
            final, metrics = _train_and_eval(spec, record, out, device, properties, quick, _use_compile(args))
            metrics_store[final["run_id"]] = metrics
            if spec.arm == "joint":
                joint_metrics[spec.seed] = metrics
                if not quick and getattr(args, "arm_filter", None) is None:
                    joint_models[spec.seed] = _load_checkpoint_model(spec, out, device, properties)
        for seed in seeds:
            main_ckpt = out / "checkpoints" / f"j1_staged_main_seed{seed}.pt"
            if not main_ckpt.exists() and getattr(args, "arm_filter", None) is not None:
                # Parallel-worker safety: the staged boundary warm-starts from the
                # joint main checkpoint.  Under --arm-filter that checkpoint may be
                # owned by another worker; skip instead of crashing on torch.load.
                # The unfiltered aggregation pass fills the gap via resume.
                print(f"[j1] staged_boundary seed={seed} SKIPPED (main checkpoint owned by another worker)")
                continue
            bnd_spec = ms.j1_staged_boundary_spec(seed, str(main_ckpt))
            if quick:
                bnd_spec = ms.quicken(bnd_spec)
            # 2026-08-22 audit fix: the ledger spec-match resume carried no
            # code fingerprint (same hole class as the _try_resume legacy
            # path); the staged boundary head is cheap, so it always retrains.
            final = train_arm(bnd_spec, record, out, device=device, properties=properties)
            bnd_run_id = final["run_id"]
            model = build_world_model(
                ms._base("j1", "staged", seed, boundary_mode="forecast", train_boundary=True,
                         initial_state_mode="hybrid", closure_mode="conservative"),
                properties,
            ).to(device)
            model.load_state_dict(torch.load(out / "checkpoints" / f"{bnd_run_id}.pt",
                                              map_location=device, weights_only=False)["state_dict"])
            staged_models[seed] = model
            staged_metrics[seed] = evaluate_windows(
                model, record, SPLIT_VAL, n_windows=32 if quick else 256, batch_size=32,
                history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON, boundary_mode="forecast",
                seed=70_000 + seed, device=device,
            )
            _save_eval_metrics(out, f"j1_staged_seed{seed}", staged_metrics[seed])
        if not quick and getattr(args, "arm_filter", None) is None:
            for seed in seeds:
                joint_metrics[seed] = evaluate_windows(
                    joint_models[seed], record, SPLIT_VAL, n_windows=256, batch_size=32,
                    history_steps=ms.HISTORY_STEPS, horizon=ms.HORIZON,
                    boundary_mode="forecast", seed=70_000 + seed, device=device,
                )
                _save_eval_metrics(out, f"j1_joint_paired_seed{seed}", joint_metrics[seed])
            pairs = [(staged_metrics[s], joint_metrics[s]) for s in seeds]
            h18_passes, h18 = _seed_delta_passes(pairs, horizon=18)
            stability_details = []
            combined_passes = 0
            for i, seed in enumerate(seeds):
                staged_stability = constant_condition_stability(
                    staged_models[seed], record, SPLIT_VAL,
                    n_windows=128, history_steps=ms.HISTORY_STEPS, rollout_steps=36,
                    seed=72_000 + seed, device=device,
                )
                joint_stability = constant_condition_stability(
                    joint_models[seed], record, SPLIT_VAL,
                    n_windows=128, history_steps=ms.HISTORY_STEPS, rollout_steps=36,
                    seed=72_000 + seed, device=device,
                )
                stability_ok = bool(
                    joint_stability["bounded"]
                    and staged_stability["bounded"]
                    and joint_stability["p95_abs_drift_c"]
                    <= staged_stability["p95_abs_drift_c"]
                )
                combined_passes += int(h18[i]["pass"] and stability_ok)
                stability_details.append({
                    "seed": seed,
                    "staged": staged_stability,
                    "joint": joint_stability,
                    "pass": stability_ok,
                })
            evidence = {
                "h1_h6_h18_metrics": [
                    {
                        "seed": seed,
                        "staged": horizon_summary(staged_metrics[seed]),
                        "joint": horizon_summary(joint_metrics[seed]),
                    }
                    for seed in seeds
                ],
                "nll_h18": h18,
                "h36_stability": stability_details,
                "paired_nll_v07": {
                    "rule": "delta_nll_arm_minus_baseline_ci_hi_lt_0",
                    "horizon": 18,
                    "passes": h18_passes,
                    "per_seed": h18,
                },
                "practical_effects_h18": _practical_effects(pairs, horizon=18),
            }
            summary["units"]["j1"] = _adjudicate(
                "j1", _verdict(combined_passes, len(seeds)), evidence,
                quick=quick, seeds=seeds, arm_filter=getattr(args, "arm_filter", None),
            )
        else:
            summary["units"]["j1"] = _adjudicate(
                "j1", "INCOMPLETE", {}, quick=quick, seeds=seeds,
                arm_filter=getattr(args, "arm_filter", None),
            )
        dump_summary()

    if "r1" in units:
        # --r1-arm (v0.4): the frozen gate defaults to closure_cons; the norew
        # ablation stack gets its own evidence block, never clobbering 'r1'.
        r1_arm = getattr(args, "r1_arm", "closure_cons")
        r1_key = "r1" if r1_arm == "closure_cons" else f"r1_{r1_arm}"
        r1_reports = []
        for seed in seeds:
            ckpt = out / "checkpoints" / f"t1_{r1_arm}_seed{seed}.pt"
            if not ckpt.exists():
                r1_reports.append({"seed": seed, "error": f"missing t1 {r1_arm} checkpoint"})
                continue
            spec = next(s for s in ms.t1_specs((seed,)) if s.arm == r1_arm)
            model = build_world_model(spec, properties).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
            blind_ok = closure_blindness_check(model, device)["runtime_blind_ok"]
            directions = {}
            for valve_index, valve_name in ((0, "valve1"), (1, "valve2")):
                directions[valve_name] = {}
                for rollout_steps in (18, 60):
                    directions[valve_name][f"H{rollout_steps}"] = step_response_direction(
                        model, record, SPLIT_VAL, n_windows=16 if quick else 64,
                        history_steps=ms.HISTORY_STEPS, rollout_steps=rollout_steps,
                        valve_index=valve_index,
                        seed=80_000 + 1_000 * valve_index + rollout_steps + seed,
                        device=device,
                    )
            # Steady-state gain evidence (2026-08-21 audit): the 60-step gate
            # reads the transient; with learned tau_mix ~470s the 600s window
            # only reaches ~72% of steady state.  A 240-step (40 min) probe
            # approaches steady state for a fair comparison against the
            # zero-lag mixing reference.  Evidence-only, not part of the gate.
            direction_steady = step_response_direction(
                model, record, SPLIT_VAL, n_windows=16 if quick else 32,
                history_steps=ms.HISTORY_STEPS, rollout_steps=24 if quick else 240,
                seed=85_000 + seed, device=device,
            )
            if spec.closure_mode == "none":
                # 2026-08-23 physics_only R1 probe: no closure head exists, so
                # the leakage question (does the action-aware component leak
                # future information) is vacuous.  Mark it skipped; the
                # direction/blindness evidence still stands on its own.
                leak = {"skipped": True,
                        "reason": "no closure-bearing head; probe vacuous",
                        "leakage_suspected": False}
            else:
                leak = leakage_probe(
                    model, record, n_windows=64 if quick else 512,
                    history_steps=ms.HISTORY_STEPS, epochs=3 if quick else 20,
                    seed=90_000 + seed, device=device,
                )
            if spec.closure_mode == "none":
                quant = {"skipped": True,
                         "reason": "no closure; residual quantiles undefined"}
            else:
                quant = residual_quantiles(
                    model, record, SPLIT_VAL, n_windows=16 if quick else 64,
                    history_steps=ms.HISTORY_STEPS, seed=seed, device=device,
                )
            r1_reports.append({
                "seed": seed,
                "runtime_blind_ok": blind_ok,
                "directions": directions,
                # Compatibility alias for historical consumers: v2/H60.
                "direction": directions["valve2"]["H60"],
                "direction_steady": direction_steady,
                "leakage": leak,
                "residual_quantiles": quant,
            })
        verdict = "SUPPORTED"
        for rep in r1_reports:
            if "error" in rep:
                verdict = "MIXED"
                continue
            if not rep["runtime_blind_ok"] or rep["leakage"]["leakage_suspected"]:
                verdict = "REJECTED"
            for valve in rep["directions"].values():
                for direction in valve.values():
                    if not direction["ci_identifiable"]:
                        if verdict == "SUPPORTED":
                            verdict = "MIXED"
                    elif (
                        direction["mean_delta_c"] >= 0.0
                        or direction["ci_hi_c"] >= 0.0
                        or direction["frac_negative"] < 0.60
                    ):
                        verdict = "REJECTED"
        complete_reports = (
            len(r1_reports) == len(seeds)
            and all("error" not in report for report in r1_reports)
        )
        evidence = {
            "runtime_blindness": [
                {"seed": report["seed"], "pass": report["runtime_blind_ok"]}
                for report in r1_reports if "error" not in report
            ] if complete_reports else None,
            "residual_power": [
                {"seed": report["seed"], **report["residual_quantiles"]}
                for report in r1_reports if "error" not in report
            ] if complete_reports else None,
            "valve1_h18": [report["directions"]["valve1"]["H18"] for report in r1_reports]
            if complete_reports else None,
            "valve1_h60": [report["directions"]["valve1"]["H60"] for report in r1_reports]
            if complete_reports else None,
            "valve2_h18": [report["directions"]["valve2"]["H18"] for report in r1_reports]
            if complete_reports else None,
            "valve2_h60": [report["directions"]["valve2"]["H60"] for report in r1_reports]
            if complete_reports else None,
            # Task 3 and Task 4 replace the known-invalid legacy probes.
            "leakage_v07": None,
            "support_domain_v07": None,
        }
        protocol = _adjudicate(
            "r1", verdict, evidence, quick=quick, seeds=seeds,
            arm_filter=getattr(args, "arm_filter", None),
        )
        protocol.update({"arm": r1_arm, "reports": r1_reports})
        summary["units"][r1_key] = protocol
        r1_file = "r1_report.json" if r1_arm == "closure_cons" else f"r1_report_{r1_arm}.json"
        _write_json(out / r1_file, summary["units"][r1_key])
        dump_summary()

    dump_summary()
    print(f"[matrix] summary written: {out / summary_name}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=["discover", "build", "split-sides", "dsyn", "matrix", "auditpack",
                                 "leakdist"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--record", default=None)
    parser.add_argument("--side", default=None, choices=["A", "B"],
                        help="label for per-side matrix runs (recorded in the summary)")
    parser.add_argument("--out", default="artifacts/final_wm")
    parser.add_argument("--units", default=None, help="comma-separated subset: o1,t1,b1,j1,r1")
    parser.add_argument("--arm-filter", default=None,
                        help="EXECUTION-SIDE SPEED OPTION: restrict the unit's specs to one arm"
                             " (workers train disjoint arms in parallel; verdicts are NOT"
                             " computed with this flag set)")
    parser.add_argument("--seeds", default=None,
                        help="EXECUTION-SIDE SPEED OPTION: comma-separated seed subset, e.g. 0,1")
    parser.add_argument("--properties-npz", default=None, help="real IAPWS grid (else analytic fallback)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true", help="dry-run sizes; no verdicts")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the physics substep (launch-bound rollout speed lever; fp32 numerics unchanged)")
    parser.add_argument("--tf32", action="store_true",
                        help="tf32 matmul precision (tensor-core fp32; ~10-bit mantissa on GRU/MLP matmuls only, physics elementwise unaffected)")
    parser.add_argument("--checkpoint", default=None, help="auditpack: trained model checkpoint")
    parser.add_argument("--arm", default="closure_cons", help="auditpack/leakdist: T1 arm of --checkpoint")
    parser.add_argument("--r1-arm", dest="r1_arm", default="closure_cons",
                        help="r1: T1 arm whose checkpoints the gate probes (v0.4 norew stack)")
    parser.add_argument("--seed", default=0, help="auditpack: seed of --checkpoint")
    args = parser.parse_args()

    if getattr(args, "tf32", False):
        # Runner-wide speed lever; the exact flag state is recorded in every
        # ledger entry, and all arms of a verdict unit must share it (audit
        # checks uniformity).  Physics elementwise ops are not matmuls and
        # are unaffected.
        torch.set_float32_matmul_precision("high")

    if args.phase == "discover":
        if not args.data_root:
            raise FinalWMProtocolError("--data-root required for discover")
        phase_discover(args)
    elif args.phase == "build":
        if not args.data_root:
            raise FinalWMProtocolError("--data-root required for build")
        phase_build(args)
    elif args.phase == "split-sides":
        phase_split_sides(args)
    elif args.phase == "dsyn":
        run_dsyn(args)
    elif args.phase == "matrix":
        if not args.record:
            raise FinalWMProtocolError("--record required for matrix")
        run_matrix(args)
    elif args.phase == "auditpack":
        if not args.record:
            raise FinalWMProtocolError("--record required for auditpack")
        run_auditpack(args)
    elif args.phase == "leakdist":
        if not args.record or not args.checkpoint:
            raise FinalWMProtocolError("leakdist needs --record and --checkpoint")
        run_leakdist(args)


if __name__ == "__main__":
    main()
