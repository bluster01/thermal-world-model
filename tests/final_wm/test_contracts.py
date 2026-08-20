from __future__ import annotations

import pytest
import torch

from src.final_wm.contracts import (
    ACTION_ELEMENTS,
    BOUNDARY_ELEMENTS,
    CLOSURE_BOUNDARY_CHANNELS,
    OBSERVATION_ELEMENTS,
    ORACLE_ONLY_BOUNDARY_CHANNELS,
    PHYSICAL_STATE_ELEMENTS,
    ActionSupport,
    BoundaryModelConfig,
    ClosureConfig,
    FinalWMProtocolError,
    ObservationConfig,
    ObserverConfig,
    StateLayout,
    TransitionConfig,
    WorldModelConfig,
    action_support_from_history,
    validate_world_model_config,
)


def test_registries_are_unique_and_ordered() -> None:
    assert len(set(PHYSICAL_STATE_ELEMENTS)) == 11
    assert len(set(BOUNDARY_ELEMENTS)) == 7
    assert len(set(ACTION_ELEMENTS)) == 2
    assert len(set(OBSERVATION_ELEMENTS)) == 5
    assert PHYSICAL_STATE_ELEMENTS[:3] == ("h1", "h2", "h3")
    assert "spray_flow_total" in ORACLE_ONLY_BOUNDARY_CHANNELS
    assert "spray_flow_total" not in CLOSURE_BOUNDARY_CHANNELS


def test_state_layout_slices() -> None:
    layout = StateLayout(latent_dim=4)
    assert layout.physical_dim == 11
    assert layout.dim == 15
    assert layout.h_slice == slice(0, 3)
    assert layout.tm_slice == slice(3, 6)
    assert layout.rb_index == 6
    assert layout.m_liq_slice == slice(7, 9)
    assert layout.dsw_lag_slice == slice(9, 11)
    assert layout.latent_slice == slice(11, 15)
    assert layout.physical_index("rb") == 6
    with pytest.raises(FinalWMProtocolError):
        layout.physical_index("nope")
    with pytest.raises(FinalWMProtocolError):
        StateLayout(latent_dim=-1)


def test_world_model_config_validates_modules() -> None:
    layout = validate_world_model_config(WorldModelConfig())
    assert layout.dim == 11


def test_forecast_mode_rejects_measured_spray_mode() -> None:
    config = WorldModelConfig(
        transition=TransitionConfig(spray_total_mode="boundary"),
        boundary_mode="forecast",
    )
    with pytest.raises(FinalWMProtocolError):
        validate_world_model_config(config)


def test_oracle_mode_allows_measured_spray_mode() -> None:
    config = WorldModelConfig(
        transition=TransitionConfig(spray_total_mode="boundary"),
        boundary_mode="oracle",
    )
    validate_world_model_config(config)


def test_closure_cannot_read_actions() -> None:
    config = WorldModelConfig(closure=ClosureConfig(reads_actions=True))
    with pytest.raises(FinalWMProtocolError):
        validate_world_model_config(config)


def test_double_injection_is_not_representable() -> None:
    config = WorldModelConfig(closure=ClosureConfig(injection_mode="double"))
    with pytest.raises(FinalWMProtocolError):
        validate_world_model_config(config)


def test_latent_dim_mismatch_rejected() -> None:
    config = WorldModelConfig(
        transition=TransitionConfig(latent_dim=4),
        observer=ObserverConfig(latent_dim=2),
    )
    with pytest.raises(FinalWMProtocolError):
        validate_world_model_config(config)


def test_bad_time_steps_rejected() -> None:
    config = WorldModelConfig(transition=TransitionConfig(dt_seconds=10.0, substep_seconds=3.0))
    with pytest.raises(FinalWMProtocolError):
        validate_world_model_config(config)


def test_observation_sigma_bounds() -> None:
    config = WorldModelConfig(observation=ObservationConfig(min_sigma_c=1.0, init_sigma_c=0.5))
    with pytest.raises(FinalWMProtocolError):
        validate_world_model_config(config)


def test_action_support_contains_and_margin() -> None:
    history = torch.tensor([[[0.2, 0.4], [0.5, 0.6]]])
    support = action_support_from_history(history, margin=0.05)
    assert support.lo == pytest.approx((0.15, 0.35))
    assert support.hi == pytest.approx((0.55, 0.65))
    inside = torch.tensor([[0.3, 0.5], [0.9, 0.5]])
    mask = support.contains(inside)
    assert mask.tolist() == [True, False]
    with pytest.raises(FinalWMProtocolError):
        support.contains(torch.zeros(3))
    with pytest.raises(FinalWMProtocolError):
        ActionSupport(lo=(0.6, 0.0), hi=(0.5, 1.0))


def test_boundary_config_rejects_bad_horizon() -> None:
    config = WorldModelConfig(boundary=BoundaryModelConfig(horizon=0))
    with pytest.raises(FinalWMProtocolError):
        validate_world_model_config(config)
