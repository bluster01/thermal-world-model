"""RM3-A adapter over the audited RM3 validation-only training core."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from ..data import Phase35Cache
from ..schema import Phase35ProtocolError
from .gatec_data import paired_history_feature_names
from .rm3_prediction import RM3FairPredictionAdapter, RM3PredictionConfig
from .rm3_training import run_rm3_prediction_training
from .rm3a_contracts import RM3ARunSpec


def rm3a_state_element_count(spec: RM3ARunSpec) -> int:
    config = RM3PredictionConfig(
        candidate_id=spec.base_candidate_id,
        window=96,
        horizon=60,
        n_features=len(paired_history_feature_names()),
        d_model=spec.d_model,
        latent_dim=spec.latent_dim,
        dropout=0.1,
    )
    model = RM3FairPredictionAdapter(config, paired_history_feature_names())
    return sum(value.numel() for value in model.state_dict().values())


def run_rm3a_training(
    caches: Mapping[str, Phase35Cache],
    matrix: Mapping[str, Any],
    spec: RM3ARunSpec,
    *,
    device: str,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    actual_elements = rm3a_state_element_count(spec)
    if actual_elements != spec.state_elements_expected:
        raise Phase35ProtocolError(
            f"RM3-A state count drift: expected={spec.state_elements_expected}, actual={actual_elements}"
        )
    adapted = copy.deepcopy(dict(matrix))
    adapted["model"] = {
        "d_model": spec.d_model,
        "latent_dim": spec.latent_dim,
        "dropout": 0.1,
    }
    result = run_rm3_prediction_training(
        caches,
        adapted,
        spec,  # RM3-A spec intentionally implements the RM3 run protocol.
        device=device,
        output_dir=output_dir,
        provenance={
            **dict(provenance),
            "rm3a_candidate_id": spec.candidate_id,
            "base_architecture_candidate_id": spec.base_candidate_id,
            "loss_profile": spec.loss_profile,
            "state_element_count": actual_elements,
            "test_accessed": False,
        },
        model_candidate_id=spec.base_candidate_id,
        model_d_model=spec.d_model,
        model_latent_dim=spec.latent_dim,
        component_loss_weights=spec.loss_weights,
    )
    return result
