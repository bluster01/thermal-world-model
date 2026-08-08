import torch

from src.phase35.model import A1PhysValveWM, MonotoneValveMap, assert_constant_valve_identity
from src.phase35.schema import ExperimentConfig


def _config(**kw):
    base = dict(
        config_id="test",
        action_mode="absolute",
        opening_map="monotone",
        window=12,
        horizon=16,
        d_model=16,
        n_heads=4,
        dropout=0.0,
        batch_size=4,
        steps_per_epoch=1,
        epochs=1,
        patience=1,
    )
    base.update(kw)
    return ExperimentConfig.from_mapping(base)


def _inputs(batch=3):
    torch.manual_seed(0)
    history = torch.randn(batch, 12, 5)
    history[:, :, 2] += 565.0
    baseline = torch.full((batch,), 20.0)
    return history, baseline


def test_monotone_valve_map_has_fixed_endpoints_and_positive_gradient():
    mapping = MonotoneValveMap("monotone")
    valve = torch.linspace(0, 100, 101, requires_grad=True)
    effective = mapping(valve)
    assert torch.isclose(effective[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(effective[-1], torch.tensor(100.0), atol=1e-5)
    assert torch.all(torch.diff(effective) >= -1e-7)
    effective.sum().backward()
    assert valve.grad is not None and torch.all(valve.grad >= -1e-7)


def test_fixed_equal_percentage_prior_is_monotone_but_not_called_flow_truth():
    mapping = MonotoneValveMap("equal_percentage_r50")
    valve = torch.linspace(0, 100, 101)
    effective = mapping(valve)
    assert torch.isclose(effective[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(effective[-1], torch.tensor(100.0), atol=1e-5)
    assert torch.all(torch.diff(effective) >= 0)
    assert effective[50] < 20.0  # convex equal-percentage prior, not identity


def test_constant_future_valve_is_exact_zero_intervention():
    model = A1PhysValveWM(_config(), n_features=5, target_index=2)
    history, baseline = _inputs()
    assert_constant_valve_identity(model, history, baseline)


def test_opening_has_nonpositive_long_run_effect():
    model = A1PhysValveWM(_config(opening_map="identity"), n_features=5, target_index=2).eval()
    history, baseline = _inputs()
    future = baseline[:, None].expand(-1, 16).clone() + 5.0
    effect = model.intervention_effect(history, future, baseline)
    assert torch.all(effect <= 1e-7)
    assert torch.all(effect[:, -1] < 0)


def test_future_action_cannot_change_earlier_response():
    model = A1PhysValveWM(_config(opening_map="identity"), n_features=5, target_index=2).eval()
    history, baseline = _inputs()
    a = baseline[:, None].expand(-1, 16).clone()
    b = a.clone()
    b[:, 10:] += 8.0
    ea = model.intervention_effect(history, a, baseline)
    eb = model.intervention_effect(history, b, baseline)
    torch.testing.assert_close(ea[:, :10], eb[:, :10], atol=1e-7, rtol=0)
    assert torch.any(torch.abs(eb[:, 10:] - ea[:, 10:]) > 1e-6)


def test_delta_with_baseline_reconstructs_absolute_identity_path():
    cfg_a = _config(config_id="a", opening_map="identity", action_mode="absolute")
    cfg_d = _config(config_id="d", opening_map="identity", action_mode="delta_with_baseline")
    ma = A1PhysValveWM(cfg_a, n_features=5, target_index=2)
    md = A1PhysValveWM(cfg_d, n_features=5, target_index=2)
    md.load_state_dict(ma.state_dict(), strict=False)
    history, baseline = _inputs()
    future = baseline[:, None] + torch.linspace(0, 5, 16)[None, :]
    da, _ = ma.action_adapter(future, baseline)
    dd, _ = md.action_adapter(future, baseline)
    torch.testing.assert_close(da, dd)
