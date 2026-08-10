#!/usr/bin/env python3
"""Fail-closed aggregation for Phase 3.5-MS2-J validation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase3_5.joint_coupling import (  # noqa: E402
    FROZEN_EXECUTION_PATHS,
    _sha256,
    expand_runs,
    load_matrix,
)
from experiments.phase3_5.summarize_multistep_mismatch import (  # noqa: E402
    _gate_metrics,
)


JOINT_ID = "j_g2_monotone_scheduled_joint"
STAGED_ID = "j_g2_monotone_scheduled_staged"
SINGLE_MODULE_IDS = (
    "j_g2_monotone_global",
    "j_g2_identity_scheduled",
)


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required MS2-J artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _assert_code_equivalent(execution_sha: str) -> None:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{execution_sha}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if exists.returncode != 0:
        raise RuntimeError(f"MS2-J execution commit unavailable: {execution_sha}")
    compared = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            execution_sha,
            "HEAD",
            "--",
            *FROZEN_EXECUTION_PATHS,
        ],
        cwd=ROOT,
        check=False,
    )
    if compared.returncode == 1:
        raise RuntimeError(
            f"MS2-J frozen execution code differs from {execution_sha}"
        )
    if compared.returncode != 0:
        raise RuntimeError("git failed while checking MS2-J code equivalence")


def build_summary(matrix_path: Path, output_root: Path) -> dict:
    matrix = load_matrix(matrix_path)
    records = []
    stage_a_nmae = {}
    trajectory_groups = {}
    current_sha = _git_sha()
    execution_shas = set()
    for run in expand_runs(matrix):
        candidate_id, seed = run["candidate_id"], int(run["seed"])
        run_dir = output_root / f"ms2j_{candidate_id}_s{seed}"
        manifest = _read_json(run_dir / "manifest.json")
        metrics = _read_json(run_dir / "metrics_validation.json")
        history = _read_json(run_dir / "history.json")
        checkpoint = run_dir / "checkpoint_best_val.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required MS2-J checkpoint missing: {checkpoint}")
        unauthorized = [
            name
            for name in (
                "metrics_test.json",
                "episode_metrics_test.json",
                "synthetic_test_access_ledger.json",
            )
            if (run_dir / name).exists()
        ]
        expected = {
            "protocol_version": matrix["protocol_version"],
            "route_id": candidate_id,
            "seed": seed,
            "training_mode": run["training_mode"],
            "test_accessed": False,
            "checkpoint_sha256": _sha256(checkpoint),
        }
        mismatches = [
            key for key, value in expected.items() if manifest.get(key) != value
        ]
        execution_sha = manifest.get("git_sha")
        if not execution_sha:
            mismatches.append("git_sha")
        else:
            execution_shas.add(execution_sha)
        if unauthorized:
            mismatches.append(f"unauthorized_test={unauthorized}")
        if not history:
            mismatches.append("empty_history")
        environment = manifest.get("environment", {})
        for key in (
            "python",
            "torch",
            "cuda_runtime",
            "cuda_available",
            "device",
            "platform",
        ):
            if key not in environment:
                mismatches.append(f"environment_{key}")
        design_hash = manifest.get("validation_trajectory_design_sha256")
        if not design_hash:
            mismatches.append("validation_trajectory_design_sha256")
        trajectory_groups.setdefault(seed, set()).add(design_hash)
        if run["training_mode"] == "staged":
            summaries = manifest.get("stage_summaries", [])
            if [item.get("stage") for item in summaries] != [
                "stage_a",
                "stage_b",
                "stage_c",
            ]:
                mismatches.append("stage_summaries")
            elif any(item.get("optimizer_updates", 0) < 1 for item in summaries):
                mismatches.append("stage_optimizer_updates")
            for stage in ("stage_a", "stage_b", "stage_c"):
                info = manifest.get("stage_checkpoints", {}).get(stage, {})
                path = run_dir / info.get("file", "missing")
                if not path.is_file() or info.get("sha256") != _sha256(path):
                    mismatches.append(f"checkpoint_{stage}")
            stage_a = _read_json(run_dir / "metrics_stage_a_validation.json")
            stage_a_nmae[seed] = float(stage_a["clean_effect_nmae"])
        if mismatches:
            raise RuntimeError(
                f"MS2-J artifact mismatch for {candidate_id}/seed={seed}: "
                f"{sorted(set(mismatches))}"
            )
        gate_failures = _gate_metrics(metrics, run["route"])
        records.append(
            {
                **run,
                "effect_mae": metrics["effect_mae"],
                "clean_effect_mae": metrics["clean_effect_mae"],
                "clean_effect_nmae": metrics["clean_effect_nmae"],
                "direction_accuracy_clean_nonzero": metrics[
                    "direction_accuracy_clean_nonzero"
                ],
                "gate_failures": gate_failures,
            }
        )
    bad_trajectory_seeds = [
        seed for seed, hashes in trajectory_groups.items() if len(hashes) != 1
    ]
    if bad_trajectory_seeds:
        raise RuntimeError(
            f"MS2-J candidates do not share validation trajectories: {bad_trajectory_seeds}"
        )
    if len(execution_shas) != 1:
        raise RuntimeError(
            f"MS2-J manifests contain multiple execution SHAs: {sorted(execution_shas)}"
        )
    execution_sha = next(iter(execution_shas))
    _assert_code_equivalent(execution_sha)

    by_id_seed = {
        (record["candidate_id"], record["seed"]): record for record in records
    }
    candidates = {}
    for candidate_id in sorted({record["candidate_id"] for record in records}):
        subset = [record for record in records if record["candidate_id"] == candidate_id]
        candidates[candidate_id] = {
            "route": subset[0]["route"],
            "training_mode": subset[0]["training_mode"],
            "clean_effect_nmae_mean": statistics.mean(
                record["clean_effect_nmae"] for record in subset
            ),
            "clean_effect_nmae_std": statistics.stdev(
                record["clean_effect_nmae"] for record in subset
            ),
            "effect_mae_mean": statistics.mean(
                record["effect_mae"] for record in subset
            ),
            "all_structural_gates_pass": all(
                not record["gate_failures"] for record in subset
            ),
        }

    module_seed_results = []
    staged_seed_results = []
    for seed in sorted(int(value) for value in matrix["seeds"]):
        joint_nmae = float(by_id_seed[(JOINT_ID, seed)]["clean_effect_nmae"])
        improvements = {}
        for baseline_id in SINGLE_MODULE_IDS:
            baseline = float(
                by_id_seed[(baseline_id, seed)]["clean_effect_nmae"]
            )
            improvements[baseline_id] = (baseline - joint_nmae) / max(
                baseline, 1e-12
            )
        module_seed_results.append(
            {
                "seed": seed,
                "joint_clean_effect_nmae": joint_nmae,
                "relative_improvements": improvements,
                "all_improvements_exceed_20pct": all(
                    value >= 0.20 for value in improvements.values()
                ),
            }
        )
        staged_nmae = float(
            by_id_seed[(STAGED_ID, seed)]["clean_effect_nmae"]
        )
        noninferiority_ratio = staged_nmae / max(joint_nmae, 1e-12)
        improvement_from_stage_a = (
            stage_a_nmae[seed] - staged_nmae
        ) / max(stage_a_nmae[seed], 1e-12)
        staged_seed_results.append(
            {
                "seed": seed,
                "joint_clean_effect_nmae": joint_nmae,
                "staged_clean_effect_nmae": staged_nmae,
                "stage_a_clean_effect_nmae": stage_a_nmae[seed],
                "staged_to_joint_ratio": noninferiority_ratio,
                "relative_improvement_from_stage_a": improvement_from_stage_a,
                "passes_noninferiority_10pct": noninferiority_ratio <= 1.10,
                "improves_stage_a_by_20pct": improvement_from_stage_a >= 0.20,
            }
        )

    structural_failures = [
        {
            "candidate_id": record["candidate_id"],
            "seed": record["seed"],
            "failures": record["gate_failures"],
        }
        for record in records
        if record["gate_failures"]
    ]
    module_pass = all(
        result["all_improvements_exceed_20pct"]
        for result in module_seed_results
    )
    staged_pass = all(
        result["passes_noninferiority_10pct"]
        and result["improves_stage_a_by_20pct"]
        for result in staged_seed_results
    )
    all_gates_pass = not structural_failures and module_pass and staged_pass
    return {
        "protocol_version": matrix["protocol_version"],
        "evidence_scope": matrix["evidence_scope"],
        "split": "validation",
        "run_count": len(records),
        "test_accessed": False,
        "execution_git_sha": execution_sha,
        "aggregation_git_sha": current_sha,
        "frozen_code_equivalence_paths": list(FROZEN_EXECUTION_PATHS),
        "all_gates_pass": all_gates_pass,
        "structural_gate_failures": structural_failures,
        "candidates": candidates,
        "joint_module_gate": {
            "candidate_id": JOINT_ID,
            "baseline_ids": list(SINGLE_MODULE_IDS),
            "seed_results": module_seed_results,
            "all_seed_improvements_exceed_20pct": module_pass,
        },
        "staged_stability_gate": {
            "staged_id": STAGED_ID,
            "joint_id": JOINT_ID,
            "seed_results": staged_seed_results,
            "all_seeds_pass": staged_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs/phase3_5/joint_coupling_matrix.json"),
    )
    parser.add_argument("--output-root", default="results/phase3_5/joint_coupling")
    parser.add_argument("--output")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    summary = build_summary(Path(args.matrix).resolve(), output_root)
    output = (
        Path(args.output).resolve()
        if args.output
        else output_root / "summary_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))
    if not summary["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
