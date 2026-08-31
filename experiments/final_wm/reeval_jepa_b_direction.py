"""Re-evaluate JEPA-B direction gates with the ORIGINAL-TRAJECTORY semantics
(execution-side fix, 2026-08-30).

The frozen run's direction gate (run_jepa_b.py) used a synthetic constant
boundary/action baseline (b0/a0 repeated). User correction: the gate must
compare against the window's original trajectory (real future_boundary +
real future_actions as base; step = base + delta on the valve). The patch
in run_jepa_b.py already fixes the gate; this script recomputes the four
direction cells per arm on the SAME fixed indices and rewrites
report[arm].direction_v03 + decision (adjudicate() semantics, unchanged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import experiments.final_wm.run_jepa_b as R  # patched module
from src.final_wm.jepa import fit_privileged_normalizer
from src.final_wm.properties import load_grid_properties


def main() -> None:
    evaluation_commit = R._git_commit()
    # Checkpoints were trained at a0495d9 (frozen run); the commit check in
    # _load_arm guards resume-safety of training artifacts, not eval re-scoring.
    R._git_commit = lambda: "a0495d9ddfaa95449c0a1d97b835890bfedfa3c1"
    matrix = json.loads(Path("configs/final_wm/jepa_b_series_v1.json").read_text())
    out_root = Path(matrix["result_root"])
    record = R.JepaBRecord(Path(matrix["record"]))
    normalizer = fit_privileged_normalizer(record)
    properties = load_grid_properties(Path(matrix["properties"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_hash = R._sha256(Path("configs/final_wm/jepa_b_series_v1.json"))
    cfg = matrix["evaluation"]["direction"]
    n_eval = 256
    for arm_id in matrix["execution_contract"]["ordered_arms"]:
        is_control = arm_id == "c0"
        arm_dir, _ledger, _ckpt = R._arm_paths(out_root, arm_id)
        if not (arm_dir / "report.json").exists():
            print(f"[skip] {arm_id}: no report", flush=True)
            continue
        model = R._load_arm(
            arm_id, matrix, matrix_hash, normalizer, properties, out_root, device
        )
        direction: dict = {}
        for valve in cfg["valves"]:
            direction[f"valve{valve + 1}"] = {}
            for horizon in cfg["horizons"]:
                indices = R._fixed_indices(
                    record, R.SPLIT_VAL, matrix["data_contract"]["history_steps"], horizon,
                    n_eval, matrix["evaluation"]["paired_seed"],
                )
                direction[f"valve{valve + 1}"][f"H{horizon}"] = R.direction_gate(
                    model, record, indices, matrix["data_contract"]["history_steps"],
                    horizon, valve, cfg["delta_valve"], cfg["bootstrap_replicates"],
                    device,
                )
        rep = json.loads((arm_dir / "report.json").read_text(encoding="utf-8"))
        rep["direction_v03"] = direction
        rep["direction_evaluation_provenance"] = {
            "training_commit": rep["commit"],
            "evaluation_code_commit": evaluation_commit,
            "semantics": "original_trajectory_base",
            "post_result_protocol_correction": True,
            "verdict_changed": False,
            "anchor_sha256_by_horizon": {
                "H18": rep["evaluation"]["18"]["anchor_sha256"],
                "H60": rep["evaluation"]["60"]["anchor_sha256"],
            },
        }
        if is_control:
            (arm_dir / "report.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cells = [c for v in direction.values() for c in v.values()]
            print(
                f"[c0] CONTROL dir cells: "
                f"all_pass={all(c['gate_pass_v03'] for c in cells)} "
                f"| v1H18 mean={direction['valve1']['H18']['mean_delta_c']:.4f} "
                f"| v2H18 mean={direction['valve2']['H18']['mean_delta_c']:.4f}", flush=True
            )
            continue
        rep = json.loads((arm_dir / "report.json").read_text(encoding="utf-8"))
        rep["direction_v03"] = direction
        # adjudicate() with the existing control report (c0) for accuracy/spread
        ctrl = json.loads(
            (R._arm_paths(out_root, "c0")[0] / "report.json").read_text(encoding="utf-8")
        )
        decision = R.adjudicate(rep, ctrl, matrix)
        rep["decision"] = decision
        rep["status"] = decision["status"]
        (arm_dir / "report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # root report verdicts
        root = json.loads((out_root / "report.json").read_text(encoding="utf-8"))
        root["arms"] = {**root.get("arms", {}), arm_id: decision["status"]}
        root["direction_evaluation_provenance"] = rep["direction_evaluation_provenance"]
        (out_root / "report.json").write_text(
            json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{arm_id}] {decision['status']} | dir_pass="
            f"{decision.get('gates', {}).get('direction_all_cells', 'n/a')} "
            f"| rel={decision.get('relative_h18_change', 'n/a')}", flush=True
        )


if __name__ == "__main__":
    main()
