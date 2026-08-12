"""RM3-A capacity-matched and Pareto reporting over new and audited reference runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import Phase35ProtocolError
from .rm3_reporting import verify_rm3_prediction_run
from .rm3a_contracts import RM3ARunSpec


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_rm3a(
    output_root: Path,
    reference_root: Path,
    specs: Sequence[RM3ARunSpec],
    *,
    required_artifacts: Sequence[str],
    reference_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records: dict[tuple[str, str, int], dict[str, float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for reference in reference_candidates:
        candidate = reference["candidate_id"]
        metadata[candidate] = {
            "source": "audited_rm3_reference",
            "state_elements": int(reference["state_elements"]),
        }
        for fold in ("F0", "F1"):
            for seed in (0, 1, 2):
                directory = reference_root / f"{candidate}_{fold}_s{seed}"
                payload = verify_rm3_prediction_run(directory, required_artifacts)
                metrics = payload["metrics"]["metrics"]
                records[(candidate, fold, seed)] = {
                    key: float(metrics[key])
                    for key in ("terminal_mae_c", "local_mae_c", "scope_selector_score")
                }
    for spec in specs:
        directory = output_root / spec.run_id
        payload = verify_rm3_prediction_run(directory, required_artifacts)
        manifest, metrics = payload["manifest"], payload["metrics"]["metrics"]
        if manifest.get("architecture_candidate_id") != spec.base_candidate_id:
            raise Phase35ProtocolError(f"RM3-A architecture drift: {spec.run_id}")
        if manifest.get("component_loss_weights") != spec.loss_weights:
            raise Phase35ProtocolError(f"RM3-A loss profile drift: {spec.run_id}")
        records[(spec.candidate_id, spec.fold_id, spec.seed)] = {
            key: float(metrics[key])
            for key in ("terminal_mae_c", "local_mae_c", "scope_selector_score")
        }
        metadata[spec.candidate_id] = {
            "source": "rm3a_new",
            "base_candidate_id": spec.base_candidate_id,
            "state_elements": spec.state_elements_expected,
            "loss_profile": spec.loss_profile,
        }

    def aggregate(candidate: str) -> dict[str, Any]:
        rows = [records[(candidate, fold, seed)] for fold in ("F0", "F1") for seed in (0, 1, 2)]
        return {
            "candidate_id": candidate,
            **metadata[candidate],
            **{f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in rows[0]},
        }

    def contrast(left: str, right: str) -> dict[str, Any]:
        output = {"left": left, "right": right}
        for metric in ("terminal_mae_c", "local_mae_c"):
            values = [
                records[(left, fold, seed)][metric] - records[(right, fold, seed)][metric]
                for fold in ("F0", "F1") for seed in (0, 1, 2)
            ]
            output[f"{metric}_left_minus_right"] = values
            output[f"{metric}_mean_difference"] = float(np.mean(values))
            output[f"{metric}_left_better_count"] = int(np.sum(np.asarray(values) < 0))
        return output

    candidates = sorted(metadata)
    return {
        "protocol_version": "phase3.5-ms3r-rm3a-summary-v1",
        "new_run_count": len(specs),
        "reference_run_count": len(reference_candidates) * 6,
        "candidate_summary": [aggregate(candidate) for candidate in candidates],
        "capacity_matched_contrasts": [
            contrast("A0_p3_large", "P5_hybrid_joint_latent"),
            contrast("A1_p4_large", "P5_hybrid_joint_latent"),
            contrast("A2_p5_small", "P3_gatec_paired_free"),
            contrast("A2_p5_small", "P4_gatec_a1_scheduled"),
        ],
        "pareto_table": [
            aggregate(candidate)
            for candidate in ("P5_hybrid_joint_latent", "A3_p5_local35", "A4_p5_local50")
        ],
        "single_composite_champion": None,
        "automatic_scientific_pass": None,
        "test_accessed": False,
    }
