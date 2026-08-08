#!/usr/bin/env python3
"""Evaluate one frozen Phase 3.5 checkpoint on validation or explicitly unlocked test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.phase35.data import (
    deterministic_anchor_subset,
    extract_windows,
    load_cache,
    valid_window_anchors,
)
from src.phase35.evaluation import empirical_response_summary, event_response_metrics
from src.phase35.events import (
    detect_sp_execution_events,
    detect_valve_events,
    match_quiet_controls,
    matching_diagnostics,
    matched_empirical_irf,
    quiet_control_candidates,
    events_to_jsonable,
)
from src.phase35.model import A1PhysValveWM, assert_constant_valve_identity
from src.phase35.schema import ExperimentConfig, TARGET_COLUMN, TOUT2_COLUMN, VALVE_COLUMN
from src.phase35.training import _finite_json, _json_dump, evaluate_forecast, git_sha


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[A1PhysValveWM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_mapping(checkpoint["config"])
    features = checkpoint["feature_columns"]
    model = A1PhysValveWM(config, len(features), features.index(TARGET_COLUMN)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def _model_curves(model, cache, events, event_ids, features, device, batch_size=256):
    by_id = {event.event_id: event for event in events}
    selected = [by_id[event_id] for event_id in event_ids]
    curves = []
    for start in range(0, len(selected), batch_size):
        part = selected[start:start + batch_size]
        batch = extract_windows(
            cache, [event.anchor for event in part], features, TARGET_COLUMN, VALVE_COLUMN,
            model.config.window, model.config.horizon,
        )
        output = model(
            torch.from_numpy(batch["history"]).to(device),
            torch.from_numpy(batch["future_valve"]).to(device),
            torch.from_numpy(batch["baseline_valve"]).to(device),
        )
        curves.append(output["effect"].cpu().numpy())
    return np.concatenate(curves) if curves else np.empty((0, model.config.horizon))


def _group_event_metrics(empirical, model_curves, dose, clusters, bootstrap, seed):
    result = {}
    for name, mask in {
        "open": dose > 0,
        "close": dose < 0,
        "all_oriented": np.ones(len(dose), dtype=bool),
    }.items():
        if mask.sum() < 3:
            result[name] = {"status": "insufficient_events", "n_events": int(mask.sum())}
            continue
        e, m, d = empirical[mask].copy(), model_curves[mask].copy(), dose[mask].copy()
        c = clusters[mask]
        if name == "all_oriented":
            orient = np.sign(d)[:, None]
            e, m, d = e * orient, m * orient, np.abs(d)
        result[name] = event_response_metrics(
            e, m, d, bootstrap_replicates=bootstrap, seed=seed, cluster_ids=c
        )
    return result


@torch.no_grad()
def _sp_negative_control(model, cache, sp_events, features, device):
    out = {}
    for label in ("no_execution", "executed", "ambiguous"):
        group = [event for event in sp_events if event.execution == label]
        if not group:
            out[label] = {"n_events": 0}
            continue
        batch = extract_windows(
            cache, [event.anchor for event in group], features, TARGET_COLUMN, VALVE_COLUMN,
            model.config.window, model.config.horizon,
        )
        pred = model(
            torch.from_numpy(batch["history"]).to(device),
            torch.from_numpy(batch["future_valve"]).to(device),
            torch.from_numpy(batch["baseline_valve"]).to(device),
        )["effect"].cpu().numpy()
        record = {"n_events": len(group)}
        target = cache.values[:, cache.index(TARGET_COLUMN)]
        tout2 = cache.values[:, cache.index(TOUT2_COLUMN)]
        for horizon in (6, 18, 30, 60):
            if horizon <= model.config.horizon:
                record[f"mean_abs_model_effect_h{horizon}"] = float(np.mean(np.abs(pred[:, horizon - 1])))
                target_change = [target[event.anchor + horizon] - target[event.anchor] for event in group]
                tout2_change = [tout2[event.anchor + horizon] - tout2[event.anchor] for event in group]
                record[f"mean_abs_observed_main_delta_h{horizon}"] = float(np.mean(np.abs(target_change)))
                record[f"mean_abs_observed_tout2_delta_h{horizon}"] = float(np.mean(np.abs(tout2_change)))
        out[label] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-test-access", action="store_true")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--controls-per-event", type=int, default=5)
    parser.add_argument("--caliper-quantile", type=float, default=0.5,
                        help="scaled-distance cap for matching (quantile of all candidate distances); "
                             "lower = stricter common support")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test_access:
        parser.error("test is locked; pass --allow-test-access for the preregistered one-time evaluation")

    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = checkpoint_path.parent
    ledger_path = run_dir / "access_ledger.json"
    if args.split == "test" and ledger_path.exists():
        raise FileExistsError(f"test access ledger already exists; refusing repeat access: {ledger_path}")
    if args.split == "test":
        # Write the access attempt before reading the cache or checkpoint.  A
        # crashed evaluation still consumes the lockbox access and cannot be
        # silently retried under the same run identity.
        _json_dump(ledger_path, {
            "protocol_version": "phase3.5-v1",
            "status": "started",
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "cache": str(Path(args.cache).resolve()),
            "evaluation_git_sha": git_sha(ROOT),
            "command_authorization": "--allow-test-access",
        })
    device = torch.device(args.device)
    cache = load_cache(args.cache)
    model, checkpoint = _load_model(checkpoint_path, device)
    features = checkpoint["feature_columns"]
    anchors = valid_window_anchors(
        cache, args.split, features, TARGET_COLUMN, VALVE_COLUMN, model.config.window, model.config.horizon
    )
    anchors = deterministic_anchor_subset(anchors, model.config.max_eval_anchors, args.seed)
    forecast = evaluate_forecast(model, cache, anchors, features, device)

    events = detect_valve_events(cache, args.split, model.config.window, model.config.horizon)
    if args.max_events > 0 and len(events) > args.max_events:
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(len(events), args.max_events, replace=False))
        events = [events[i] for i in chosen]
    controls = quiet_control_candidates(cache, args.split, model.config.window, model.config.horizon)
    matches = match_quiet_controls(
        cache, events, controls, args.controls_per_event,
        caliper_quantile=args.caliper_quantile,
    ) if events else {}
    balance = matching_diagnostics(cache, events, matches)
    empirical_main, doses, ids = matched_empirical_irf(
        cache, events, matches, TARGET_COLUMN, model.config.horizon
    )
    model_curves = _model_curves(model, cache, events, ids, features, device)
    event_by_id = {event.event_id: event for event in events}
    day_ns = 24 * 60 * 60 * 1_000_000_000
    clusters = np.asarray(
        [cache.timestamps_ns[event_by_id[event_id].anchor] // day_ns for event_id in ids],
        dtype=np.int64,
    )
    if len(doses) >= 3:
        main_metrics = _group_event_metrics(
            empirical_main, model_curves, doses, clusters, args.bootstrap, args.seed
        )
        empirical_tout2, tout_doses, _ = matched_empirical_irf(
            cache, events, matches, TOUT2_COLUMN, model.config.horizon
        )
        tout_metrics = empirical_response_summary(
            empirical_tout2,
            tout_doses,
            bootstrap_replicates=args.bootstrap,
            seed=args.seed,
            cluster_ids=clusters,
        ) if len(tout_doses) >= 3 else {"status": "insufficient_events", "n_events": int(len(tout_doses))}
    else:
        main_metrics = {"status": "insufficient_events", "n_events": int(len(doses))}
        tout_metrics = {"status": "insufficient_events", "n_events": 0}

    sp_events = detect_sp_execution_events(cache, args.split, model.config.window, model.config.horizon)
    negative_control = _sp_negative_control(model, cache, sp_events, features, device)
    event_payload = {
        "protocol_version": "phase3.5-v1",
        "split": args.split,
        "valve_events_detected": len(events),
        "valve_events_matched": len(doses),
        "matching_balance": balance,
        "main_temperature": main_metrics,
        "second_stage_outlet_empirical": tout_metrics,
        "sp_negative_control": negative_control,
        "warnings": [
            "matched closed-loop event responses are observational, not randomized causal effects",
            "spray-flow measurements are not used as labels, doses, or selectors",
        ],
    }
    _json_dump(run_dir / f"metrics_{args.split}.json", _finite_json(forecast))
    _json_dump(run_dir / f"event_metrics_{args.split}.json", _finite_json(event_payload))
    _json_dump(run_dir / f"event_manifest_{args.split}.json", {
        "protocol_version": "phase3.5-v1",
        "split": args.split,
        "valve_events": events_to_jsonable(events),
        "matched_event_ids": ids,
        "quiet_control_matches": matches,
        "sp_events": events_to_jsonable(sp_events),
        "cluster_unit": "UTC_calendar_day",
    })
    if args.split == "test":
        ledger = {
            "protocol_version": "phase3.5-v1",
            "status": "completed",
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_git_sha": checkpoint.get("git_sha", "unknown"),
            "evaluation_git_sha": git_sha(ROOT),
            "cache_source": cache.metadata.get("source", {}),
            "command_authorization": "--allow-test-access",
        }
        _json_dump(ledger_path, ledger)
    print(json.dumps({"split": args.split, "forecast": forecast, "events": len(doses)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
