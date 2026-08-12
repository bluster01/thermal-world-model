from __future__ import annotations

import pytest
import torch

from src.phase35.multistep.gatec_data import paired_history_feature_names
from src.phase35.multistep.rm3_prediction import (
    PREDICTION_CANDIDATES,
    RM3FairPredictionAdapter,
    RM3PredictionConfig,
)
from src.phase35.schema import Phase35ProtocolError


FEATURES = paired_history_feature_names()


def _inputs(batch: int = 2):
    torch.manual_seed(17)
    history = torch.randn(batch, 16, len(FEATURES))
    future_sp = torch.randn(batch, 60, 2)
    logged_valve = torch.randn(batch, 60, 2)
    return history, future_sp, logged_valve


@pytest.mark.parametrize("candidate", sorted(PREDICTION_CANDIDATES))
def test_rm3_six_candidates_share_h60_paired_terminal_contract(candidate: str) -> None:
    model = RM3FairPredictionAdapter(
        RM3PredictionConfig(
            candidate_id=candidate,
            window=16,
            horizon=60,
            n_features=len(FEATURES),
            d_model=16,
            latent_dim=8,
            dropout=0.0,
        ),
        FEATURES,
    )
    history, future_sp, logged = _inputs()
    kwargs = {"logged_future_valve": logged} if candidate == "P0_m7_oracle_valve" else {}
    output = model(history, future_sp, **kwargs)
    assert output["terminal_prediction"].shape == (2, 60, 2)
    assert torch.isfinite(output["terminal_prediction"]).all()
    assert output["deployable"] is (candidate != "P0_m7_oracle_valve")
    assert output["prefix_causal_action_path"] is (not candidate.startswith("P0_") and not candidate.startswith("P1_"))


def test_rm3_logged_valve_permission_is_fail_closed() -> None:
    model = RM3FairPredictionAdapter(
        RM3PredictionConfig(
            candidate_id="P2_m9_future_sp",
            window=16,
            horizon=60,
            n_features=len(FEATURES),
            d_model=16,
            latent_dim=8,
            dropout=0.0,
        ),
        FEATURES,
    )
    history, future_sp, logged = _inputs()
    with pytest.raises(Phase35ProtocolError, match="only the declared M7 oracle"):
        model(history, future_sp, logged_future_valve=logged)


def test_rm3_deployable_m7_refuses_logged_future_valve() -> None:
    model = RM3FairPredictionAdapter(
        RM3PredictionConfig(
            candidate_id="P1_m7_predicted_valve",
            window=16,
            horizon=60,
            n_features=len(FEATURES),
            d_model=16,
            latent_dim=8,
            dropout=0.0,
        ),
        FEATURES,
    )
    history, future_sp, logged = _inputs()
    with pytest.raises(Phase35ProtocolError, match="deployable M7"):
        model(history, future_sp, logged_future_valve=logged)


def test_rm3_m9_future_sp_action_path_is_prefix_causal() -> None:
    model = RM3FairPredictionAdapter(
        RM3PredictionConfig(
            candidate_id="P2_m9_future_sp",
            window=16,
            horizon=60,
            n_features=len(FEATURES),
            d_model=16,
            latent_dim=8,
            dropout=0.0,
        ),
        FEATURES,
    ).eval()
    history, future_sp, _ = _inputs()
    changed = future_sp.clone()
    changed[:, 30:] += 10.0
    original_prediction = model(history, future_sp)["terminal_prediction"]
    changed_prediction = model(history, changed)["terminal_prediction"]
    assert torch.allclose(
        original_prediction[:, :30], changed_prediction[:, :30], atol=1e-6, rtol=0.0
    )
