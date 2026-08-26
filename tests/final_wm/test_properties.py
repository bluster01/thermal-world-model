from __future__ import annotations

import numpy as np
import pytest
import torch

from src.final_wm.contracts import CRITICAL_PRESSURE_MPA, FinalWMProtocolError
from src.final_wm.properties import (
    AnalyticThermoProperties,
    GridThermoProperties,
    interp1d,
    interp2d,
    ste_clamp,
)


def _analytic() -> AnalyticThermoProperties:
    return AnalyticThermoProperties()


def test_analytic_enthalpy_monotone_in_temperature() -> None:
    props = _analytic()
    p = torch.tensor([10.0, 17.0, 25.0, 29.0])
    temps = torch.linspace(300, 620, 33)
    for p_val in p:
        h = props.enthalpy_of_pt(p_val.expand_as(temps), temps)
        assert bool((h[1:] > h[:-1]).all())


def test_analytic_temperature_monotone_in_enthalpy() -> None:
    props = _analytic()
    p = torch.tensor([10.0, 17.0, 25.0])
    hs = torch.linspace(2400, 3700, 27)
    for p_val in p:
        t = props.temperature_of_ph(p_val.expand_as(hs), hs)
        assert bool((t[1:] > t[:-1]).all())


def test_analytic_ph_pt_round_trip() -> None:
    props = _analytic()
    p = torch.tensor([10.0, 17.0, 25.0])
    t = torch.tensor([450.0, 550.0, 571.0])
    h = props.enthalpy_of_pt(p, t)
    t_back = props.temperature_of_ph(p, h)
    assert torch.allclose(t_back, t, atol=1e-3)


def test_analytic_tsat_monotone_and_critical_clamp() -> None:
    props = _analytic()
    p = torch.linspace(1.0, 22.0, 22)
    tsat = props.saturation_temperature(p)
    assert bool((tsat[1:] > tsat[:-1]).all())
    at_crit = props.saturation_temperature(torch.tensor([CRITICAL_PRESSURE_MPA]))
    assert at_crit.item() == pytest.approx(374.15, abs=0.2)
    above = props.saturation_temperature(torch.tensor([25.0]))
    assert above.item() == pytest.approx(374.15, abs=0.2)


def test_analytic_plausible_operating_point() -> None:
    props = _analytic()
    h = props.enthalpy_of_pt(torch.tensor(17.0), torch.tensor(550.0))
    assert 3000.0 < h.item() < 3600.0
    t = props.temperature_of_ph(torch.tensor(17.0), h)
    assert t.item() == pytest.approx(550.0, abs=1.0)


def test_analytic_separator_enthalpy_two_phase() -> None:
    props = _analytic()
    pm = torch.tensor([17.0, 25.0])
    tm_sep = torch.tensor([300.0, 500.0])  # below Tsat at 17 MPa
    h = props.separator_enthalpy(pm, tm_sep)
    tsat = props.saturation_temperature(pm[:1])
    h_floor = props.enthalpy_of_pt(pm[:1], tsat + 0.5)
    assert h[0].item() == pytest.approx(h_floor.item(), rel=1e-5)


def test_analytic_out_of_domain_stays_finite() -> None:
    props = _analytic()
    p = torch.tensor([0.1, 100.0])
    h = torch.tensor([100.0, 99999.0])
    t = props.temperature_of_ph(p, h)
    assert bool(torch.isfinite(t).all())


def test_ste_clamp_passes_gradient() -> None:
    x = torch.tensor([5.0], requires_grad=True)
    y = ste_clamp(x, 0.0, 1.0)
    assert y.item() == 1.0
    y.backward()
    assert x.grad.item() == 1.0


def test_interp_helpers_validate_and_interpolate() -> None:
    grid = torch.tensor([0.0, 1.0, 2.0])
    values = torch.tensor([0.0, 10.0, 40.0])
    out = interp1d(grid, values, torch.tensor([0.5, 1.5, 99.0]))
    assert out.tolist() == pytest.approx([5.0, 25.0, 40.0])
    with pytest.raises(FinalWMProtocolError):
        interp1d(torch.tensor([0.0, 0.0, 1.0]), values, torch.tensor([0.5]))

    table = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    out2 = interp2d(grid[:2], grid[:2], table, torch.tensor([0.5]), torch.tensor([0.5]))
    assert out2.item() == pytest.approx(1.5)


def _fake_surrogate() -> dict[str, np.ndarray]:
    """Small self-consistent grid in the legacy npz layout."""
    p_grid = np.linspace(8.0, 28.0, 5).astype(np.float32)
    h_grid = np.linspace(2400.0, 3700.0, 14).astype(np.float32)
    t_grid = np.linspace(300.0, 620.0, 17).astype(np.float32)
    # h(p, T) affine in T with pressure-dependent offset (matches analytic form).
    cp = 2.9 + 0.147 * np.clip(p_grid - 8.0, 0.0, None)
    tsat = np.clip(374.15 - (28.0 - p_grid) * 5.0, 250.0, None)
    h0 = 2600.0 - (p_grid - 8.0) * 20.0
    hpt = h0[:, None] + cp[:, None] * (t_grid[None, :] - tsat[:, None])
    tph = tsat[:, None] + (h_grid[None, :] - h0[:, None]) / cp[:, None]
    p_sub = np.linspace(0.5, 22.064, 12).astype(np.float32)
    hsatv = np.linspace(2748.0, 2358.0, 12).astype(np.float32)
    t_liq = np.linspace(0.0, 350.0, 36).astype(np.float32)
    hliq = 4.0 * t_liq
    tsat_coef = np.zeros(8, dtype=np.float32)
    tsat_coef[-1] = 300.0
    return {
        "P": p_grid, "H": h_grid, "Tg": t_grid, "Tph": tph.astype(np.float32),
        "hpT": hpt.astype(np.float32), "Psub": p_sub, "hsatV": hsatv,
        "t_liq": t_liq, "hliq_grid": hliq.astype(np.float32),
        "tsat_coef": tsat_coef, "p_crit": np.float32(CRITICAL_PRESSURE_MPA),
    }


def test_grid_properties_load_and_interpolate(tmp_path) -> None:
    arrays = _fake_surrogate()
    path = tmp_path / "iapws_surrogate.npz"
    np.savez(path, **arrays)
    from src.final_wm.properties import load_grid_properties

    props = load_grid_properties(path)
    assert props.critical_pressure == pytest.approx(CRITICAL_PRESSURE_MPA)
    b = props.bounds
    assert b.p_lo == pytest.approx(8.0) and b.p_hi == pytest.approx(28.0)

    p = torch.tensor([15.0])
    t = torch.tensor([500.0])
    h = props.enthalpy_of_pt(p, t)
    t_back = props.temperature_of_ph(p, h)
    assert t_back.item() == pytest.approx(500.0, abs=2.0)
    assert props.saturation_temperature(p).item() == pytest.approx(300.0)
    assert props.liquid_enthalpy(torch.tensor([100.0])).item() == pytest.approx(400.0)


def test_grid_properties_reject_missing_keys() -> None:
    arrays = _fake_surrogate()
    del arrays["Tph"]
    with pytest.raises(FinalWMProtocolError):
        GridThermoProperties(arrays)


def test_saturation_temperature_bit_identical_to_legacy_scalar_sync_path() -> None:
    """P1 hoist regression: pre-hoisted float coefficients must reproduce the
    legacy per-call float(tensor_scalar) evaluation exactly (zero DtoH syncs,
    same bits)."""
    arrays = _fake_surrogate()
    # non-trivial coefficients so every Horner term is exercised
    arrays["tsat_coef"] = np.array([0.5, -2.0, 1.5, 0.0, 0.25, 0.0, 0.0, 300.0],
                                   dtype=np.float32)
    props = GridThermoProperties(arrays)
    p = torch.linspace(8.0, 28.0, 257)

    def legacy(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = torch.full_like(x, float(coef[0]))
        for c in coef[1:]:
            y = y * x + float(c)
        return y

    legacy_val = legacy(props._tsat_coef, p.clamp(float(props._psub[0]), props._p_crit))
    new_val = props.saturation_temperature(p)
    torch.testing.assert_close(new_val, legacy_val, rtol=0.0, atol=0.0)
