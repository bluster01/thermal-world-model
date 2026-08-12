#!/usr/bin/env python3
"""Local-only RM3 contract dry-run and known-truth orthogonal smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.phase35.multistep.rm3_contracts import (  # noqa: E402
    rm3_identification_specs,
    rm3_prediction_specs,
    validate_rm3_matrix,
)
from src.phase35.multistep.rm3_orthogonal import (  # noqa: E402
    generate_rm3_confounded_synthetic,
    oof_nuisance_residuals,
    orthogonal_mimo_moment,
    synthetic_expanding_splits,
)


DEFAULT_MATRIX = ROOT / "configs/phase3_5/ms3r_rm3_matrix.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dry_run_payload(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_rm3_matrix(matrix)
    execution = matrix["execution_contract"]
    return {
        "protocol_version": matrix["protocol_version"],
        "scope": matrix["scope"],
        "identification_candidates": [spec.candidate_id for spec in rm3_identification_specs(matrix)],
        "prediction_candidates": [spec.candidate_id for spec in rm3_prediction_specs(matrix)],
        "prediction_candidate_count": len(rm3_prediction_specs(matrix)),
        "prediction_run_count": matrix["real_matrix_envelope"]["prediction_run_count"],
        "orthogonal_calibration_run_count": matrix["real_matrix_envelope"][
            "orthogonal_calibration_run_count"
        ],
        "total_real_run_envelope": matrix["real_matrix_envelope"]["total_run_count"],
        "real_matrix_status": matrix["real_matrix_envelope"]["status"],
        "no_composite_ranking_across_output_scopes": matrix["real_matrix_envelope"][
            "no_composite_ranking_across_output_scopes"
        ],
        "primary_response_horizons_steps": matrix["data_contract"]["primary_response_horizons_steps"],
        "raw_future_valve_auxiliary_allowed": matrix["data_contract"]["raw_future_valve_auxiliary_allowed"],
        "local_synthetic_smoke_authorized": execution["local_synthetic_smoke_authorized"],
        "local_real_training_authorized": execution["local_real_training_authorized"],
        "linux_authorized": execution["linux_authorized"],
        "test_authorized": execution["test_authorized"],
        "automatic_scientific_pass": None,
    }


def synthetic_smoke(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_rm3_matrix(matrix)
    statistics = matrix["statistics"]
    x, action, outcome, truth = generate_rm3_confounded_synthetic(seed=35120, n_rows=2400)
    residual = oof_nuisance_residuals(
        x,
        action,
        outcome,
        synthetic_expanding_splits(len(x)),
        ridge_alpha=float(statistics["ridge_alpha"]),
        epsilon=float(statistics["epsilon"]),
    )
    recovered = orthogonal_mimo_moment(
        residual.action,
        residual.outcome,
        ridge_alpha=1e-6,
        epsilon=float(statistics["epsilon"]),
        maximum_condition_number=float(statistics["maximum_input_condition_number"]),
        minimum_differential_to_common_energy=float(
            statistics["minimum_differential_to_common_energy"]
        ),
    )
    rng = np.random.default_rng(35121)
    shuffled_action = residual.action.copy()
    evaluated = np.flatnonzero(residual.evaluated)
    shuffled_action[evaluated] = shuffled_action[rng.permutation(evaluated)]
    placebo = orthogonal_mimo_moment(
        shuffled_action,
        residual.outcome,
        ridge_alpha=1e-6,
        epsilon=float(statistics["epsilon"]),
        maximum_condition_number=float(statistics["maximum_input_condition_number"]),
        minimum_differential_to_common_energy=float(
            statistics["minimum_differential_to_common_energy"]
        ),
    )
    return {
        "protocol_version": matrix["protocol_version"],
        "evaluated_row_count": int(residual.evaluated.sum()),
        "true_matrix": truth.tolist(),
        "recovered_matrix": recovered.matrix.tolist(),
        "maximum_absolute_recovery_error": float(np.max(np.abs(recovered.matrix - truth))),
        "input_condition_number": recovered.condition_number,
        "differential_to_common_energy_ratio": recovered.differential_to_common_energy_ratio,
        "independent_channels_supported": recovered.independent_channels_supported,
        "shuffled_matrix_norm": float(np.linalg.norm(placebo.matrix)),
        "recovered_matrix_norm": float(np.linalg.norm(recovered.matrix)),
        "synthetic_smoke_pass": bool(
            np.max(np.abs(recovered.matrix - truth)) < 0.03
            and np.linalg.norm(placebo.matrix) < 0.15 * np.linalg.norm(recovered.matrix)
        ),
        "real_data_claim": None,
        "automatic_scientific_pass": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    matrix = _read_json(Path(args.matrix).resolve())
    payload = dry_run_payload(matrix) if args.dry_run else synthetic_smoke(matrix)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
