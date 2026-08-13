from __future__ import annotations

import pytest
import torch

from src.phase35.multistep.gatec_data import paired_history_feature_names
from src.phase35.multistep.rm3av_contracts import RM3AV_CANDIDATE_IDS
from src.phase35.multistep.rm3av_model import (
    RM3AVModelConfig,
    build_rm3av_model,
    module_state_hashes,
)


FEATURES = paired_history_feature_names()


def _config(candidate_id: str) -> RM3AVModelConfig:
    return RM3AVModelConfig(
        candidate_id=candidate_id,
        window=4,
        horizon=60,
        n_features=len(FEATURES),
        d_model=8,
        latent_dim=4,
        dropout=0.0,
    )


def _inputs(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    history = torch.randn(batch, 4, len(FEATURES), generator=generator)
    future_sp = torch.randn(batch, 60, 2, generator=generator)
    logged_valve = torch.randn(batch, 60, 2, generator=generator)
    return history, future_sp, logged_valve


@pytest.mark.parametrize("candidate_id", RM3AV_CANDIDATE_IDS)
def test_all_rm3av_candidates_are_real_forward_models(candidate_id: str) -> None:
    model = build_rm3av_model(_config(candidate_id), FEATURES).eval()
    history, future_sp, logged_valve = _inputs()
    output = model(
        history,
        future_sp,
        logged_future_valve=(
            logged_valve if candidate_id in {"C10", "C11", "C12", "C13"} else None
        ),
    )
    assert output["terminal_prediction"].shape == (1, 60, 2)
    assert torch.isfinite(output["terminal_prediction"]).all()
    assert output["rm3av_candidate_id"] == candidate_id
    assert output["test_accessed"] is False


def test_head_interventions_change_the_executed_terminal_graph() -> None:
    history, future_sp, _ = _inputs()
    changed_sp = future_sp + 3.0

    added = build_rm3av_model(_config("C03"), FEATURES).eval()
    first = added(history, future_sp)
    second = added(history, changed_sp)
    assert torch.equal(first["terminal_bypass"], second["terminal_bypass"])
    assert torch.allclose(
        first["terminal_prediction"],
        first["terminal_physical_prediction"] + first["terminal_bypass"],
    )

    bypass_off = build_rm3av_model(_config("C04"), FEATURES).eval()(history, future_sp)
    assert torch.count_nonzero(bypass_off["terminal_bypass"]) == 0
    assert torch.equal(
        bypass_off["terminal_prediction"], bypass_off["terminal_physical_prediction"]
    )

    bypass_only = build_rm3av_model(_config("C05"), FEATURES).eval()(history, future_sp)
    terminal_indices = [
        FEATURES.index(f"{side}::末级过热器出口汽温") for side in ("A", "B")
    ]
    baseline = history[:, -1, terminal_indices]
    assert torch.allclose(
        bypass_only["terminal_prediction"],
        baseline[:, None] + bypass_only["terminal_bypass"],
    )

    response_off = build_rm3av_model(_config("C06"), FEATURES).eval()(history, future_sp)
    assert torch.count_nonzero(response_off["explicit_local_effect"]) == 0


def test_free_capacity_scan_changes_only_declared_residual_capacity() -> None:
    small = build_rm3av_model(_config("C07"), FEATURES)
    large = build_rm3av_model(_config("C08"), FEATURES)
    small_count = sum(parameter.numel() for parameter in small.base.model.residual_head.parameters())
    large_count = sum(parameter.numel() for parameter in large.base.model.residual_head.parameters())
    assert large_count > small_count


@pytest.mark.parametrize("candidate_id", ["C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24"])
def test_response_diagnostics_have_exact_constant_action_identity(candidate_id: str) -> None:
    model = build_rm3av_model(_config(candidate_id), FEATURES).eval()
    context = torch.randn(2, 8)
    baseline = torch.tensor([[30.0, 40.0], [50.0, 60.0]])
    valve = baseline[:, None].expand(-1, 60, -1).clone()
    response = model.explicit_response(context, valve, baseline)
    assert torch.count_nonzero(response["effect"]) == 0
    assert torch.isfinite(response["state"]).all()


def test_common_and_diagonal_coordinates_are_operational_not_labels() -> None:
    context = torch.zeros(1, 8)
    baseline = torch.tensor([[40.0, 40.0]])
    valve = baseline[:, None].expand(-1, 60, -1).clone()
    valve[..., 0] += 10.0

    common = build_rm3av_model(_config("C17"), FEATURES).eval()
    common_effect = common.explicit_response(context, valve, baseline)["effect"]
    assert torch.allclose(common_effect[..., 0], common_effect[..., 1])

    diagonal = build_rm3av_model(_config("C18"), FEATURES).eval()
    diagonal_effect = diagonal.explicit_response(context, valve, baseline)["effect"]
    assert torch.count_nonzero(diagonal_effect[..., 0]) > 0
    assert torch.count_nonzero(diagonal_effect[..., 1]) == 0


def test_structured_pi_candidates_use_temperature_error() -> None:
    history, future_sp, _ = _inputs()
    high_sp = future_sp + 50.0
    for candidate_id in ("C15", "C16"):
        model = build_rm3av_model(_config(candidate_id), FEATURES).eval()
        low = model(history, future_sp)["valve_prediction"]
        high = model(history, high_sp)["valve_prediction"]
        assert not torch.allclose(low, high)
        assert model.valve_decoder_family.startswith("structured_pi")


def test_module_scoped_initialization_matches_shared_encoder_across_variants() -> None:
    torch.manual_seed(1)
    p3 = build_rm3av_model(_config("C25"), FEATURES)
    torch.manual_seed(999)
    p4 = build_rm3av_model(_config("C26"), FEATURES)
    p3_encoder = {
        key.removeprefix("base.model.encoder."): value
        for key, value in p3.state_dict().items()
        if key.startswith("base.model.encoder.")
    }
    p4_encoder = {
        key.removeprefix("base.model.encoder."): value
        for key, value in p4.state_dict().items()
        if key.startswith("base.model.encoder.")
    }
    assert p3_encoder.keys() == p4_encoder.keys()
    assert all(torch.equal(p3_encoder[key], p4_encoder[key]) for key in p3_encoder)
    p3_hashes = module_state_hashes(p3)
    p4_hashes = module_state_hashes(p4)
    for module in ("encoder", "valve_policy", "tin", "free_residual", "downstream"):
        assert p3_hashes[module] == p4_hashes[module]
    assert p3_hashes["response"] == "not_applicable"
    assert p4_hashes["response"] != "not_applicable"


@pytest.mark.parametrize("candidate_id", ["C00", "C01", "C02", "C03", "C04", "C05"])
def test_diagnostic_forward_executes_all_declared_functional_interventions(candidate_id: str) -> None:
    model = build_rm3av_model(_config(candidate_id), FEATURES).eval()
    history, future_sp, logged_valve = _inputs(batch=2)
    modes = model.diagnostic_forward(
        history,
        future_sp,
        logged_future_valve=logged_valve,
        logged_future_tin=torch.randn_like(logged_valve),
        local_target=torch.randn_like(logged_valve),
    )
    assert set(modes) == {
        "normal", "bypass_off", "bypass_only", "response_off", "predicted_valve",
        "logged_valve", "logged_valve_oracle_tin", "oracle_local", "shuffled",
        "wrong_side", "lead",
    }
    assert all(value["terminal_prediction"].shape == (2, 60, 2) for value in modes.values())
    assert torch.count_nonzero(modes["response_off"]["explicit_local_effect"]) == 0
    assert torch.equal(
        modes["bypass_off"]["terminal_prediction"],
        modes["normal"]["terminal_physical_prediction"],
    )
