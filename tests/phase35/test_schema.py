import pytest

from src.phase35.schema import (
    DEFAULT_HISTORY_FEATURES,
    ExperimentConfig,
    Phase35ProtocolError,
    SP_COLUMN,
    VALVE_COLUMN,
    validate_task_action,
)


def test_plant_and_supervisory_actions_cannot_mix():
    validate_task_action("plant", VALVE_COLUMN)
    validate_task_action("supervisory", SP_COLUMN)
    with pytest.raises(Phase35ProtocolError):
        validate_task_action("plant", SP_COLUMN)


def test_nonlinear_delta_requires_absolute_baseline():
    with pytest.raises(Phase35ProtocolError, match="baseline"):
        ExperimentConfig(
            config_id="bad",
            action_mode="delta_no_baseline",
            opening_map="monotone",
        ).validate()


def test_run_config_round_trip_and_defaults():
    cfg = ExperimentConfig.from_mapping({
        "config_id": "absolute_identity",
        "action_mode": "absolute",
        "history_features": list(DEFAULT_HISTORY_FEATURES),
    })
    assert cfg.horizon == 60
    assert ExperimentConfig.from_mapping(cfg.to_dict()) == cfg
