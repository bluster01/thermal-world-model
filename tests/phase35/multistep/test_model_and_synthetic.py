import torch
import torch.nn as nn

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
