from __future__ import annotations

import torch
import pytest

from src.phase35.multistep.gatec_contracts import GateCModelConfig, RESPONSE_ROUTES
from src.phase35.schema import Phase35ProtocolError


FEATURES = (
    "机组负荷",
    "主蒸汽压力",
    "主给水流量",
    "未校正总煤量",
    "主蒸汽流量",
    "A::二级减温器入口温度",
    "A::二级减温器出口温度",
    "A::末级过热器出口汽温",
    "A::二级减温调节阀设定",
    "A::二级减温调节门阀位",
    "B::二级减温器入口温度",
    "B::二级减温器出口温度",
    "B::末级过热器出口汽温",
    "B::二级减温调节阀设定",
    "B::二级减温调节门阀位",
)


def _config(route: str = "a1phys_three_pole") -> GateCModelConfig:
    return GateCModelConfig(
        window=12,
        horizon=8,
        n_features=len(FEATURES),
        d_model=16,
        latent_dim=8,
        local_state_dim=6,
        response_route=route,
        residual_capacity="base",
        response_scheduling="scheduled" if route != "none" else "none",
        dropout=0.0,
    )


def _inputs(batch: int = 3):
    torch.manual_seed(4)
    history = torch.randn(batch, 12, len(FEATURES))
    # Put physical-scale baselines in the indexed channels.
    history[:, :, FEATURES.index("A::二级减温调节门阀位")] = 30.0
    history[:, :, FEATURES.index("B::二级减温调节门阀位")] = 35.0
    history[:, :, FEATURES.index("A::二级减温器入口温度")] = 550.0
    history[:, :, FEATURES.index("B::二级减温器入口温度")] = 552.0
    history[:, :, FEATURES.index("A::二级减温器出口温度")] = 538.0
    history[:, :, FEATURES.index("B::二级减温器出口温度")] = 539.0
    history[:, :, FEATURES.index("A::末级过热器出口汽温")] = 540.0
    history[:, :, FEATURES.index("B::末级过热器出口汽温")] = 541.0
    future_sp = torch.randn(batch, 8, 2) + 540.0
    return history, future_sp


def test_gatec_model_shapes_boundary_isolation_and_full_terminal_mixing():
    from src.phase35.multistep.gatec_model import build_gatec_model

    history, future_sp = _inputs()
    model = build_gatec_model(_config(), FEATURES).eval()
    forecast = model(history, future_sp, boundary_mode="forecast_boundary")
    assert forecast["valve_prediction"].shape == (3, 8, 2)
    assert forecast["tin_prediction"].shape == (3, 8, 2)
    assert forecast["local_drop_prediction"].shape == (3, 8, 2)
    assert forecast["tout_prediction"].shape == (3, 8, 2)
    assert forecast["terminal_prediction"].shape == (3, 8, 2)
    assert forecast["local_effect"].shape == (3, 8, 2)
    assert forecast["terminal_effect"].shape == (3, 8, 2)
    assert forecast["local_operator_family"] == "a1phys_three_pole"
    assert torch.all(forecast["local_stable_poles"] < 1)
    assert torch.isfinite(forecast["terminal_prediction"]).all()
    with pytest.raises(Phase35ProtocolError, match="must not receive future Tin"):
        model(history, future_sp, boundary_mode="forecast_boundary", boundary_future=torch.ones(3, 8, 2))
    with pytest.raises(Phase35ProtocolError, match="requires future Tin"):
        model(history, future_sp, boundary_mode="scenario_boundary")
    oracle_tin = torch.full((3, 8, 2), 555.0)
    oracle = model(history, future_sp, boundary_mode="oracle_boundary", boundary_future=oracle_tin)
    assert torch.equal(oracle["boundary_used"], oracle_tin)
    assert model.downstream.input_projection.weight.shape[1] == 2
    assert model.downstream.output_projection.weight.shape[0] == 2


def test_train_frozen_history_normalization_preserves_physical_baselines():
    from src.phase35.multistep.gatec_model import build_gatec_model

    history, future_sp = _inputs()
    model = build_gatec_model(_config(), FEATURES).eval()
    center = history.reshape(-1, len(FEATURES)).mean(dim=0)
    scale = history.reshape(-1, len(FEATURES)).std(dim=0).clamp_min(0.1)
    model.set_history_normalization(center, scale)
    output = model(history, future_sp, boundary_mode="forecast_boundary")
    assert torch.allclose(model.history_center, center)
    assert torch.allclose(model.history_scale, scale)
    assert output["tin_prediction"].mean() > 400.0
    with pytest.raises(Phase35ProtocolError, match="normalization"):
        model.set_history_normalization(center[:-1], scale[:-1])


def test_initialization_is_persistence_anchored_and_logged_action_is_aux_only():
    from src.phase35.multistep.gatec_model import build_gatec_model

    history, future_sp = _inputs(batch=2)
    model = build_gatec_model(_config(), FEATURES).eval()
    logged = history[:, -1:, model.valve_indices].expand(-1, 8, -1).clone()
    logged[:, 3:, 0] += 5.0
    output = model(
        history,
        future_sp,
        boundary_mode="forecast_boundary",
        logged_future_valve_for_aux=logged,
    )
    baseline_valve = history[:, -1, model.valve_indices]
    baseline_tin = history[:, -1, model.tin_indices]
    baseline_local = baseline_tin - history[:, -1, model.tout_indices]
    baseline_terminal = history[:, -1, model.terminal_indices]
    assert (output["valve_prediction"] - baseline_valve[:, None]).abs().mean() < 0.1
    assert (output["tin_prediction"] - baseline_tin[:, None]).abs().mean() < 0.1
    assert (output["residual_local_prediction"] - baseline_local[:, None]).abs().mean() < 0.1
    assert (output["terminal_prediction"] - baseline_terminal[:, None]).abs().mean() < 0.1
    changed = model(
        history,
        future_sp,
        boundary_mode="forecast_boundary",
        logged_future_valve_for_aux=logged.flip(1),
    )
    assert torch.allclose(
        output["residual_local_prediction"], changed["residual_local_prediction"], atol=1e-6
    )
    assert not torch.allclose(output["logged_local_effect"], changed["logged_local_effect"])


def test_sp_decoder_is_prefix_causal_and_residual_is_future_action_invariant():
    from src.phase35.multistep.gatec_model import build_gatec_model

    history, future_sp = _inputs()
    changed = future_sp.clone()
    changed[:, 4:] += 20.0
    model = build_gatec_model(_config(), FEATURES).eval()
    normal = model(history, future_sp, boundary_mode="forecast_boundary")
    altered = model(history, changed, boundary_mode="forecast_boundary")
    assert torch.allclose(normal["valve_prediction"][:, :4], altered["valve_prediction"][:, :4], atol=1e-6)
    assert torch.allclose(normal["residual_local_prediction"], altered["residual_local_prediction"], atol=1e-6)
    assert not torch.allclose(
        normal["valve_prediction"][:, 4:],
        altered["valve_prediction"][:, 4:],
        atol=1e-7,
        rtol=0.0,
    )


def test_every_response_adapter_has_constant_action_identity_and_finite_rollout():
    from src.phase35.multistep.gatec_model import build_local_response_operator

    torch.manual_seed(8)
    context = torch.randn(4, 16)
    baseline = torch.tensor([[30.0, 35.0]]).expand(4, 2)
    constant = baseline[:, None, :].expand(4, 60, 2)
    for route in sorted(RESPONSE_ROUTES):
        operator = build_local_response_operator(
            route=route,
            context_dim=16,
            state_dim=6,
            horizon=60,
            dt_seconds=10.0,
            scheduled=route != "none",
        ).eval()
        output = operator(context, constant, baseline)
        assert output["effect"].shape == (4, 60, 2)
        assert output["state"].shape[:2] == (4, 60)
        assert float(output["effect"].abs().max()) < 1e-6
        assert torch.isfinite(output["state"]).all()
        assert output["operator_family"] == route
        assert torch.all(output["stable_poles"] >= 0)
        assert torch.all(output["stable_poles"] < 1)


def test_response_routes_are_distinct_implementations_and_prefix_causal():
    from src.phase35.multistep.gatec_model import build_local_response_operator

    torch.manual_seed(18)
    context = torch.randn(3, 16)
    baseline = torch.tensor([[28.0, 34.0]]).expand(3, 2)
    future = baseline[:, None, :].expand(3, 12, 2).clone()
    future[:, 2:, 0] += 8.0
    future[:, 5:, 1] += 5.0
    changed = future.clone()
    changed[:, 8:] += 20.0
    classes = set()
    effects = []
    for route in sorted(RESPONSE_ROUTES - {"none"}):
        operator = build_local_response_operator(
            route=route,
            context_dim=16,
            state_dim=6,
            horizon=12,
            dt_seconds=10.0,
            scheduled=True,
        ).eval()
        classes.add(type(operator).__name__)
        normal = operator(context, future, baseline)
        altered = operator(context, changed, baseline)
        assert torch.allclose(normal["effect"][:, :8], altered["effect"][:, :8], atol=1e-6)
        assert normal["effect"][:, -1, 0].mean() > 0
        effects.append(normal["effect"])
    assert classes == {
        "A1PhysThreePoleResponse",
        "StableLPVKoopmanResponse",
        "PINeuralODEResponse",
        "CausalDeepONetResponse",
    }
    assert all(
        not torch.allclose(effects[left], effects[right])
        for left in range(len(effects))
        for right in range(left + 1, len(effects))
    )


def test_all_gatec_modules_receive_gradients_under_joint_multitask_loss():
    from src.phase35.multistep.gatec_model import build_gatec_model

    history, future_sp = _inputs(batch=2)
    model = build_gatec_model(_config(), FEATURES).train()
    output = model(history, future_sp, boundary_mode="forecast_boundary")
    loss = sum(
        output[name].square().mean()
        for name in (
            "valve_prediction",
            "tin_prediction",
            "local_drop_prediction",
            "terminal_prediction",
        )
    )
    loss.backward()
    for module_name in ("encoder", "valve_policy", "tin_forecaster", "local_response", "downstream"):
        module = getattr(model, module_name)
        assert any(parameter.grad is not None for parameter in module.parameters()), module_name
