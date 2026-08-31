"""Register jepa_b5 in the phase3.5 experiment registry and flip the active gate.

B-series (jepa_b_series) is complete on Linux (6/6 reports, push cde385e);
the B5 follow-up (action-blind slow state, user-approved) becomes the new
active Linux gate. Read-only fields of all other entries are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

REG = Path("configs/phase3_5/experiment_registry.json")


def main() -> None:
    registry = json.loads(REG.read_text(encoding="utf-8"))
    if "jepa_b5" in registry["experiments"]:
        raise SystemExit("jepa_b5 already registered; aborting")
    base = registry["experiments"]["jepa_b_series"]
    entry = {
        "title": "JEPA-B5 动作盲慢态（B2 因果修复，预注册探索批）",
        "status": "ready_for_linux",
        "owner": "execution_side_user_authorized",
        "evidence_scope": "validation_only_single_seed_state_representation_probe",
        "scripts": {
            "model_adapters": {
                "path": "src/final_wm/jepa.py",
                "status": "active",
                "required": True,
            },
            "matrix_contract": {
                "path": "experiments/final_wm/jepa_b5_spec.py",
                "status": "active",
                "required": True,
            },
            "runner": {
                "path": "experiments/final_wm/run_jepa_b.py",
                "status": "active",
                "required": True,
            },
        },
        "artifacts": {
            "design": {
                "path": "docs/plans/2026-08-31-jepa-b5-design.md",
                "required": True,
            },
            "preregistration": {
                "path": "results/final_wm/probes_20260824/PREREG_jepa_b5_20260831.md",
                "required": True,
            },
            "matrix": {
                "path": "configs/final_wm/jepa_b5_series_v1.json",
                "required": True,
            },
            "linux_runbook": {
                "path": "results/final_wm/probes_20260824/JEPA_B5_LINUX_RUN_20260831.md",
                "required": True,
            },
        },
        "protocol_state": {
            "active": True,
            "ready_for_linux": True,
            "linux_completed": False,
            "note": (
                "B5 = B2 action-blind fix: slow update drops the physical-state "
                "input (physical state is a function of logged actions). Direction "
                "gate uses original-trajectory semantics (base = real "
                "future_actions/future_boundary)."
            ),
        },
    }
    registry["experiments"]["jepa_b5"] = entry
    registry["active_gate"] = "jepa_b5"
    registry["linux_authorized_gate"] = "jepa_b5"
    REG.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("jepa_b5 registered; active_gate flipped to jepa_b5")


if __name__ == "__main__":
    main()
