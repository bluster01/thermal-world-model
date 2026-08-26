"""Injectable differentiable thermodynamic property providers.

The Fan2020-UDE transition needs steam/water property evaluations inside the
differentiable compute graph.  The legacy snapshot evaluated them from a
precomputed IAPWS grid (`iapws_surrogate.npz`) that is deliberately *not*
imported into this repository, so the formal package treats property
evaluation as an injectable interface:

- `GridThermoProperties` reproduces the legacy grid semantics and is the
  production path: the real IAPWS grid is supplied at execution time.
- `AnalyticThermoProperties` is a qualitative, monotone, self-consistent
  fallback for local micro-smoke and unit tests.  It is NOT an IAPWS
  replacement and must not be used for scientific numbers.

Both providers implement the `ThermoProperties` protocol.  All functions are
torch-differentiable, bounded by clamping, and finite on the declared domain.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import torch

from src.final_wm.contracts import CRITICAL_PRESSURE_MPA, FinalWMProtocolError


class PropertyBounds(NamedTuple):
    p_lo: float
    p_hi: float
    h_lo: float
    h_hi: float
    t_lo: float
    t_hi: float


class ThermoProperties(Protocol):
    """Property evaluation contract used by the transition."""

    @property
    def critical_pressure(self) -> float: ...

    @property
    def bounds(self) -> PropertyBounds: ...

    def temperature_of_ph(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """T(p, h) in degC for superheated/supercritical steam."""

    def enthalpy_of_pt(self, p: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """h(p, T) in kJ/kg for steam."""

    def saturation_temperature(self, p: torch.Tensor) -> torch.Tensor:
        """Tsat(p) in degC; clamped to the critical point above p_crit."""

    def saturated_vapor_enthalpy(self, p: torch.Tensor) -> torch.Tensor:
        """h''(p) in kJ/kg (subcritical queries only)."""

    def liquid_enthalpy(self, temperature: torch.Tensor) -> torch.Tensor:
        """h_liq(T) in kJ/kg for spray feedwater."""

    def separator_enthalpy(self, pm: torch.Tensor, tm_sep: torch.Tensor) -> torch.Tensor:
        """Separator outlet enthalpy; two-phase aware at subcritical pressure."""


# ---------------------------------------------------------------------------
# Shared interpolation / clamp helpers
# ---------------------------------------------------------------------------

def ste_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Straight-through clamp: hard value limits, gradients pass through."""
    return x + (x.clamp(lo, hi) - x).detach()


def _interp_index(grid: torch.Tensor, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.searchsorted(grid, flat.contiguous()).clamp(1, grid.numel() - 1)
    lo = grid[idx - 1]
    hi = grid[idx]
    weight = ((flat - lo) / (hi - lo)).clamp(0.0, 1.0)
    return idx, weight


def interp1d(grid: torch.Tensor, values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear 1D interpolation; x clamped to the grid range."""
    if grid.ndim != 1 or values.shape != grid.shape:
        raise FinalWMProtocolError("interp1d grid/values must be matching 1D tensors")
    if bool((grid[1:] <= grid[:-1]).any()):
        raise FinalWMProtocolError("interp1d grid must be strictly increasing")
    shape = x.shape
    flat = x.reshape(-1).clamp(float(grid[0]), float(grid[-1]))
    idx, weight = _interp_index(grid, flat)
    out = values[idx - 1] + weight * (values[idx] - values[idx - 1])
    return out.view(shape)


def interp2d(
    grid_row: torch.Tensor,
    grid_col: torch.Tensor,
    table: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """Bilinear interpolation of `table` (rows x cols); queries clamped."""
    if table.shape != (grid_row.numel(), grid_col.numel()):
        raise FinalWMProtocolError("interp2d table shape mismatch")
    shape = rows.shape
    if cols.shape != shape:
        raise FinalWMProtocolError("interp2d query shape mismatch")
    flat_r = rows.reshape(-1).clamp(float(grid_row[0]), float(grid_row[-1]))
    flat_c = cols.reshape(-1).clamp(float(grid_col[0]), float(grid_col[-1]))
    ir, wr = _interp_index(grid_row, flat_r)
    ic, wc = _interp_index(grid_col, flat_c)
    v00 = table[ir - 1, ic - 1]
    v01 = table[ir - 1, ic]
    v10 = table[ir, ic - 1]
    v11 = table[ir, ic]
    top = v00 + wc * (v01 - v00)
    bottom = v10 + wc * (v11 - v10)
    return (top + wr * (bottom - top)).view(shape)


def _polyval(coef: tuple[float, ...], x: torch.Tensor) -> torch.Tensor:
    """Horner evaluation; coef in descending powers, pre-hoisted to Python floats.

    Coefficients are fixed grid constants; hoisting them once at construction
    removes one DtoH scalar sync per coefficient per call (the dominant CPU cost
    of anchored forecasts) without changing any computed value.
    """
    y = torch.full_like(x, coef[0])
    for c in coef[1:]:
        y = y * x + c
    return y


def _two_phase_separator(
    props: ThermoProperties,
    pm: torch.Tensor,
    tm_sep: torch.Tensor,
) -> torch.Tensor:
    tsat = props.saturation_temperature(pm)
    supercrit = pm > props.critical_pressure
    t_eff = torch.where(supercrit, tm_sep, torch.maximum(tm_sep, tsat + 0.5))
    return props.enthalpy_of_pt(pm, t_eff)


# ---------------------------------------------------------------------------
# Production path: legacy IAPWS grid
# ---------------------------------------------------------------------------

class GridThermoProperties:
    """Bilinear-grid property provider matching the legacy surrogate format.

    Expected npz keys: P, H, Tg, Tph, hpT, Psub, hsatV, t_liq, hliq_grid,
    tsat_coef, p_crit -- exactly the layout produced by the recovered
    Fan2020-UDE branch's `iapws_surrogate.npz`.
    """

    REQUIRED_KEYS = ("P", "H", "Tg", "Tph", "hpT", "Psub", "hsatV", "t_liq", "hliq_grid", "tsat_coef", "p_crit")

    def __init__(self, arrays: dict[str, np.ndarray], *, device: str | torch.device = "cpu") -> None:
        missing = set(self.REQUIRED_KEYS) - set(arrays)
        if missing:
            raise FinalWMProtocolError(f"property grid is missing keys: {sorted(missing)}")

        def _t(key: str) -> torch.Tensor:
            return torch.tensor(np.asarray(arrays[key]), dtype=torch.float32, device=device)

        self._p = _t("P")
        self._h = _t("H")
        self._tg = _t("Tg")
        self._tph = _t("Tph")
        self._hpt = _t("hpT")
        self._psub = _t("Psub")
        self._hsatv = _t("hsatV")
        self._tliq = _t("t_liq")
        self._hliq = _t("hliq_grid")
        self._tsat_coef = _t("tsat_coef")
        self._tsat_coef_floats = tuple(float(v) for v in np.asarray(arrays["tsat_coef"]).astype(np.float32).tolist())
        self._psub_lo = float(np.asarray(arrays["Psub"]).astype(np.float32).ravel()[0])
        self._p_crit = float(np.asarray(arrays["p_crit"]))
        if self._tph.shape != (self._p.numel(), self._h.numel()):
            raise FinalWMProtocolError("Tph grid shape mismatch")
        if self._hpt.shape != (self._p.numel(), self._tg.numel()):
            raise FinalWMProtocolError("hpT grid shape mismatch")

    @property
    def critical_pressure(self) -> float:
        return self._p_crit

    @property
    def bounds(self) -> PropertyBounds:
        return PropertyBounds(
            p_lo=float(self._p[0]), p_hi=float(self._p[-1]),
            h_lo=float(self._h[0]), h_hi=float(self._h[-1]),
            t_lo=float(self._tg[0]), t_hi=float(self._tg[-1]),
        )

    def temperature_of_ph(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        b = self.bounds
        p = p.clamp(b.p_lo, b.p_hi)
        h = ste_clamp(h, b.h_lo, b.h_hi)
        return interp2d(self._p, self._h, self._tph, p, h)

    def enthalpy_of_pt(self, p: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        b = self.bounds
        p = p.clamp(b.p_lo, b.p_hi)
        temperature = temperature.clamp(b.t_lo, b.t_hi)
        return interp2d(self._p, self._tg, self._hpt, p, temperature)

    def saturation_temperature(self, p: torch.Tensor) -> torch.Tensor:
        return _polyval(self._tsat_coef_floats, p.clamp(self._psub_lo, self._p_crit))

    def saturated_vapor_enthalpy(self, p: torch.Tensor) -> torch.Tensor:
        return interp1d(self._psub, self._hsatv, p)

    def liquid_enthalpy(self, temperature: torch.Tensor) -> torch.Tensor:
        return interp1d(self._tliq, self._hliq, temperature)

    def separator_enthalpy(self, pm: torch.Tensor, tm_sep: torch.Tensor) -> torch.Tensor:
        return _two_phase_separator(self, pm, tm_sep)

    # executor-side fix (2026-08-18, per user instruction; Supervisor review
    # required): device movement for GPU execution.
    def to(self, *args, **kwargs) -> "GridThermoProperties":
        for name in ("_p", "_h", "_tg", "_tph", "_hpt", "_psub", "_hsatv",
                     "_tliq", "_hliq", "_tsat_coef"):
            setattr(self, name, getattr(self, name).to(*args, **kwargs))
        return self

    def _apply(self, fn) -> "GridThermoProperties":
        for name in ("_p", "_h", "_tg", "_tph", "_hpt", "_psub", "_hsatv",
                     "_tliq", "_hliq", "_tsat_coef"):
            setattr(self, name, fn(getattr(self, name)))
        return self


def load_grid_properties(path: str | Path, *, device: str | torch.device = "cpu") -> GridThermoProperties:
    """Load the production IAPWS grid supplied at execution time."""
    npz = np.load(Path(path))
    return GridThermoProperties({key: npz[key] for key in npz.files}, device=device)


# ---------------------------------------------------------------------------
# Qualitative fallback for local micro-smoke (NOT IAPWS-accurate)
# ---------------------------------------------------------------------------

# Saturation line anchors (MPa, degC), consistent with steam tables to ~1 degC.
_TSAT_P = (
    0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, CRITICAL_PRESSURE_MPA,
)
_TSAT_T = (
    151.8, 179.9, 212.4, 233.9, 250.4, 263.9, 275.6, 285.8, 295.0, 303.3,
    311.0, 318.1, 324.7, 330.8, 336.6, 342.1, 347.4, 352.3, 357.0, 361.5,
    365.8, 369.9, 373.7, 374.15,
)
# Saturated vapor enthalpy anchors (kJ/kg), qualitative.
_HSATV = (
    2748.0, 2777.0, 2798.0, 2803.0, 2801.0, 2794.0, 2785.0, 2773.0, 2759.0,
    2743.0, 2726.0, 2707.0, 2686.0, 2664.0, 2640.0, 2614.0, 2586.0, 2556.0,
    2523.0, 2487.0, 2447.0, 2404.0, 2358.0, 2100.0,
)
_H_CRIT = 2100.0
_T_CRIT = 374.15


class AnalyticThermoProperties:
    """Monotone analytic surrogate for tests and micro-smoke only.

    Model (superheated/supercritical region):
        h(p, T) = hsatv(p) + cp(p) * (T - Tsat_eff(p))
        cp(p)   = 2.9 + 0.147 * max(p - 8, 0)   [kJ/kg/K]

    with `Tsat_eff`/`hsatv` extended across the critical point by their
    critical constants.  h is affine in T, so T(p, h) inverts in closed
    form.  The approximation is qualitative (tens of kJ/kg), monotonic in
    T/h by construction, and bounded on the declared domain.  Do not use it
    for scientific numbers; inject the real IAPWS grid instead.
    """

    DOMAIN = PropertyBounds(p_lo=6.0, p_hi=30.0, h_lo=2000.0, h_hi=3800.0, t_lo=280.0, t_hi=630.0)

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self._tsat_p = torch.tensor(_TSAT_P, dtype=torch.float32, device=device)
        self._tsat_t = torch.tensor(_TSAT_T, dtype=torch.float32, device=device)
        self._hsatv = torch.tensor(_HSATV, dtype=torch.float32, device=device)

    @property
    def critical_pressure(self) -> float:
        return CRITICAL_PRESSURE_MPA

    @property
    def bounds(self) -> PropertyBounds:
        return self.DOMAIN

    def _cp(self, p: torch.Tensor) -> torch.Tensor:
        return 2.9 + 0.147 * (p - 8.0).clamp(min=0.0)

    def _tsat_eff(self, p: torch.Tensor) -> torch.Tensor:
        return torch.where(p > self.critical_pressure, torch.full_like(p, _T_CRIT), self.saturation_temperature(p))

    def _hsatv_eff(self, p: torch.Tensor) -> torch.Tensor:
        return torch.where(p > self.critical_pressure, torch.full_like(p, _H_CRIT), self.saturated_vapor_enthalpy(p))

    def temperature_of_ph(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        b = self.bounds
        p = p.clamp(b.p_lo, b.p_hi)
        h = ste_clamp(h, b.h_lo, b.h_hi)
        return self._tsat_eff(p) + (h - self._hsatv_eff(p)) / self._cp(p)

    def enthalpy_of_pt(self, p: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        b = self.bounds
        p = p.clamp(b.p_lo, b.p_hi)
        temperature = temperature.clamp(b.t_lo, b.t_hi)
        return self._hsatv_eff(p) + self._cp(p) * (temperature - self._tsat_eff(p))

    def saturation_temperature(self, p: torch.Tensor) -> torch.Tensor:
        return interp1d(self._tsat_p, self._tsat_t, p)

    def saturated_vapor_enthalpy(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(float(self._tsat_p[0]), self.critical_pressure)
        return interp1d(self._tsat_p, self._hsatv, p)

    def liquid_enthalpy(self, temperature: torch.Tensor) -> torch.Tensor:
        t = temperature.clamp(0.0, 350.0)
        return 4.0 * t + 0.001 * t * t

    def separator_enthalpy(self, pm: torch.Tensor, tm_sep: torch.Tensor) -> torch.Tensor:
        return _two_phase_separator(self, pm, tm_sep)

    # executor-side fix (2026-08-18, per user instruction; Supervisor review
    # required): device movement for GPU execution.
    def to(self, *args, **kwargs) -> "AnalyticThermoProperties":
        self._tsat_p = self._tsat_p.to(*args, **kwargs)
        self._tsat_t = self._tsat_t.to(*args, **kwargs)
        self._hsatv = self._hsatv.to(*args, **kwargs)
        return self

    def _apply(self, fn) -> "AnalyticThermoProperties":
        self._tsat_p = fn(self._tsat_p)
        self._tsat_t = fn(self._tsat_t)
        self._hsatv = fn(self._hsatv)
        return self
