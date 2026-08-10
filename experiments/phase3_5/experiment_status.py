#!/usr/bin/env python3
"""Read-only, fail-closed status checker for the Phase 3.5 experiment pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/phase3_5/experiment_registry.json"
LINUX_STATES = {"ready_for_linux", "linux_running", "test_authorized"}
SCRIPT_STATES = {"active", "closed", "planned", "deprecated"}


def load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("experiment registry root must be an object")
    return payload


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_registry(registry: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    root = root.resolve()
    experiments = registry.get("experiments")
    vocabulary = registry.get("status_vocabulary")
    if registry.get("schema_version") != "phase3.5-experiment-registry-v1":
        errors.append("unsupported schema_version")
    if not isinstance(experiments, dict) or not experiments:
        errors.append("experiments must be a non-empty object")
        experiments = {}
    if not isinstance(vocabulary, list) or not vocabulary:
        errors.append("status_vocabulary must be a non-empty list")
        vocabulary = []

    pipeline = registry.get("pipeline_order")
    if not isinstance(pipeline, list) or len(pipeline) != len(set(pipeline)):
        errors.append("pipeline_order must contain unique experiment IDs")
        pipeline = []
    for experiment_id in pipeline:
        if experiment_id not in experiments:
            errors.append(f"pipeline_order references missing experiment {experiment_id}")

    deprecated_tracks = sorted(
        experiment_id
        for experiment_id, experiment in experiments.items()
        if experiment.get("status") == "deprecated"
    )
    active_gate = registry.get("active_gate")
    if active_gate not in experiments:
        errors.append(f"active_gate references missing experiment {active_gate}")
    elif experiments[active_gate].get("status") == "deprecated":
        errors.append(f"active_gate {active_gate} is deprecated")

    linux_authorized_gate = registry.get("linux_authorized_gate")
    if linux_authorized_gate is not None:
        if linux_authorized_gate not in experiments:
            errors.append(
                f"linux_authorized_gate references missing experiment {linux_authorized_gate}"
            )
        elif experiments[linux_authorized_gate].get("status") not in LINUX_STATES:
            errors.append(
                f"linux_authorized_gate {linux_authorized_gate} is not in a Linux state"
            )
        if linux_authorized_gate != active_gate:
            errors.append("linux_authorized_gate must equal active_gate")
    linux_ready = sorted(
        experiment_id
        for experiment_id, experiment in experiments.items()
        if experiment.get("status") in LINUX_STATES
    )
    if len(linux_ready) > 1:
        errors.append(f"multiple Linux-authorized gates: {linux_ready}")
    if linux_ready and linux_authorized_gate != linux_ready[0]:
        errors.append(
            "linux_authorized_gate does not match the experiment in a Linux state"
        )

    for experiment_id, experiment in experiments.items():
        status = experiment.get("status")
        if status not in vocabulary:
            errors.append(f"{experiment_id} has invalid status={status!r}")
        for group_name in ("scripts", "artifacts"):
            group = experiment.get(group_name, {})
            if not isinstance(group, dict):
                errors.append(f"{experiment_id}.{group_name} must be an object")
                continue
            for role, item in group.items():
                if not isinstance(item, dict):
                    errors.append(f"{experiment_id}.{group_name}.{role} must be an object")
                    continue
                if group_name == "scripts" and item.get("status") not in SCRIPT_STATES:
                    errors.append(
                        f"{experiment_id}.scripts.{role} has invalid script status"
                    )
                relative = item.get("path")
                if not isinstance(relative, str) or not relative:
                    errors.append(f"{experiment_id}.{group_name}.{role} has no path")
                    continue
                resolved = (root / relative).resolve()
                if not _inside_root(resolved, root):
                    errors.append(
                        f"{experiment_id}.{group_name}.{role} escapes repository root"
                    )
                elif item.get("required") is True and not resolved.exists():
                    errors.append(f"missing required path: {relative}")

    return {
        "schema_version": registry.get("schema_version"),
        "valid": not errors,
        "active_gate": active_gate,
        "active_status": experiments.get(active_gate, {}).get("status"),
        "linux_authorized_gate": linux_authorized_gate,
        "pipeline_order": pipeline,
        "deprecated_tracks": deprecated_tracks,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_registry(load_registry(args.registry), ROOT)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"valid={report['valid']} active={report['active_gate']} "
            f"status={report['active_status']} "
            f"linux_authorized={report['linux_authorized_gate']}"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
    if args.check and not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
