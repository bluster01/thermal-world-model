import torch
import torch.nn as nn
import pytest

from src.phase35.multistep.contracts import OperatorConfig
from src.phase35.multistep.model import A1PhysMultiStep
from src.phase35.multistep.operators import build_response_operator
from src.phase35.multistep.synthetic import SyntheticSpec, generate_synthetic_split


class RecordingFreePredictor(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.seen_shape = None

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(context.shape)
        return context[:, :1].expand(-1, self.horizon)


def test_a1phys_ms_adds_response_to_an_action_blind_free_forecast():
    config = OperatorConfig(route="graybox", horizon=8, context_dim=3, poles=2)
    operator = build_response_operator(config)
    free = RecordingFreePredictor(horizon=8)
    model = A1PhysMultiStep(free, operator)
    context = torch.randn(4, 3)
    reference = torch.full((4, 8), 20.0)
    action = reference + 3.0
    output = model(context, action, reference)
    assert free.seen_shape == (4, 3)
    torch.testing.assert_close(output["prediction"], output["free_prediction"] + output["effect"])
    zero = model(context, reference, reference)
    torch.testing.assert_close(zero["prediction"], zero["free_prediction"], atol=0, rtol=0)


def test_synthetic_split_is_deterministic_and_contains_known_truth():
    spec = SyntheticSpec(samples=48, horizon=20, context_dim=3, seed=91, noise_std=0.0)
    first = generate_synthetic_split(spec, "train")
    second = generate_synthetic_split(spec, "train")
    torch.testing.assert_close(first.context, second.context)
    torch.testing.assert_close(first.action, second.action)
    torch.testing.assert_close(first.clean_effect, second.clean_effect)
    torch.testing.assert_close(first.target_effect, second.target_effect)
    assert first.context.shape == (48, 3)
    assert first.action.shape == first.reference.shape == first.target_effect.shape == (48, 20)
    assert set(first.profile_names) >= {"hold", "step", "pulse", "ramp", "multi_step"}
    assert first.truth["gain_c_per_pct"] < 0
    assert len(first.truth["tau_seconds"]) == 2


def test_synthetic_splits_do_not_reuse_action_paths():
    spec = SyntheticSpec(samples=24, horizon=16, context_dim=2, seed=5, noise_std=0.0)
    train = generate_synthetic_split(spec, "train")
    validation = generate_synthetic_split(spec, "validation")
    test = generate_synthetic_split(spec, "test")
    assert not torch.equal(train.action, validation.action)
    assert not torch.equal(validation.action, test.action)
    assert torch.count_nonzero(train.target_effect[train.profile_ids == 0]).item() == 0


def test_nonlinear_valve_truth_depends_on_absolute_opening_not_only_delta():
    base = SyntheticSpec(
        samples=50,
        horizon=20,
        context_dim=3,
        seed=21,
        noise_std=0.0,
        truth_regime="nonlinear_valve",
        truth_opening_map="equal_percentage_r50",
    )
    nonlinear = generate_synthetic_split(base, "train")
    linear = generate_synthetic_split(
        SyntheticSpec(**{
            **base.__dict__,
            "truth_regime": "two_pole_linear",
            "truth_opening_map": "identity",
        }),
        "train",
    )
    torch.testing.assert_close(nonlinear.action, linear.action)
    assert not torch.allclose(nonlinear.clean_effect, linear.clean_effect)


def test_context_scheduled_truth_changes_gain_and_time_constants_but_stays_stable():
    spec = SyntheticSpec(
        samples=48,
        horizon=20,
        context_dim=4,
        seed=31,
        noise_std=0.0,
        truth_regime="context_scheduled",
        context_gain_log_scale=0.35,
        context_tau_log_scale=0.30,
    )
    batch = generate_synthetic_split(spec, "validation")
    assert batch.truth["realized_gain_range"][0] < batch.truth["realized_gain_range"][1] < 0
    assert all(low > 0 and high > low for low, high in batch.truth["realized_tau_range"])
    assert torch.isfinite(batch.clean_effect).all()


def test_delayed_context_truth_is_zero_padded_and_exactly_shifts_the_same_response():
    common = dict(
        samples=48,
        horizon=20,
        context_dim=4,
        seed=41,
        noise_std=0.0,
        truth_opening_map="equal_percentage_r50",
        context_gain_log_scale=0.35,
        context_tau_log_scale=0.30,
    )
    base = generate_synthetic_split(
        SyntheticSpec(**common, truth_regime="context_scheduled"),
        "validation",
    )
    delayed = generate_synthetic_split(
        SyntheticSpec(
            **common,
            truth_regime="delayed_context_scheduled",
            input_delay_steps=2,
        ),
        "validation",
    )
    torch.testing.assert_close(delayed.action, base.action)
    torch.testing.assert_close(delayed.context, base.context)
    assert torch.count_nonzero(delayed.clean_effect[:, :2]).item() == 0
    torch.testing.assert_close(
        delayed.clean_effect[:, 2:], base.clean_effect[:, :-2], atol=1e-7, rtol=0
    )
    assert delayed.truth["input_delay_steps"] == 2
    assert delayed.truth["input_delay_seconds"] == 20.0


def test_delayed_truth_requires_a_resolvable_positive_delay():
    with pytest.raises(ValueError, match="positive input delay"):
        SyntheticSpec(
            samples=16,
            horizon=12,
            truth_regime="delayed_context_scheduled",
            context_gain_log_scale=0.3,
            context_tau_log_scale=0.3,
            input_delay_steps=0,
        ).validate()
    with pytest.raises(ValueError, match="smaller than horizon"):
        SyntheticSpec(
            samples=16,
            horizon=12,
            truth_regime="delayed_context_scheduled",
            context_gain_log_scale=0.3,
            context_tau_log_scale=0.3,
            input_delay_steps=12,
        ).validate()
