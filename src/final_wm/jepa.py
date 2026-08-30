"""Pre-registered JEPA-B state-enhancement adapters.

The physical Fan2020-UDE predictor and five-temperature output equation remain
the primary model.  B1/B3 add training-only representation objectives; B2 adds
an action-blind low-frequency slow state; B4 adds a residual state with latent
state/transition consistency.  B1/B3 representation targets and B4's residual
target encoder never receive future actions; B4's deterministic physical anchor
uses the logged same-instant action required by the physical inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.final_wm.boundary import BoundarySequence
from src.final_wm.contracts import (
    BOUNDARY_ELEMENTS,
    BOUNDARY_NORM,
    FinalWMProtocolError,
    OBSERVATION_NORM,
    PHYSICAL_STATE_ELEMENTS,
    PHYSICAL_STATE_NORM,
    action_support_from_history,
)
from src.final_wm.data import SPLIT_TEST, SPLIT_TRAIN, CanonicalRecord
from src.final_wm.data_v2 import AUX_ELEMENTS, BOUNDARY_EXT_ELEMENTS, CanonicalV2Record, N_MILLS
from src.final_wm.model import FinalWorldModel, HistoryWindow, RolloutResult
from src.final_wm.properties import AnalyticThermoProperties, ThermoProperties
from src.final_wm.training import TrainSpec, build_world_model
from src.final_wm.transition import ResidualInjection


PRIVILEGED_DIM = len(BOUNDARY_EXT_ELEMENTS) + len(AUX_ELEMENTS) + N_MILLS
PHYSICS_TARGET_INDICES = (0, 1, 2, 6, 9, 10)  # h1..h3, rb, dsw_lag1..2


@dataclass(frozen=True)
class PrivilegedNormalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.shape != (PRIVILEGED_DIM,) or self.scale.shape != (PRIVILEGED_DIM,):
            raise FinalWMProtocolError("privileged normalizer width mismatch")
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.scale).all()):
            raise FinalWMProtocolError("privileged normalizer must be finite")
        if not bool((self.scale > 0).all()):
            raise FinalWMProtocolError("privileged normalizer scales must be positive")


class JepaBRecord(CanonicalV2Record):
    """Canonical v2.2 view with the A5 operating gate and 32-D privilege."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        if tuple(self.boundary_ext_elements) != tuple(BOUNDARY_EXT_ELEMENTS):
            raise FinalWMProtocolError("JEPA-B requires canonical v2.2 boundary_ext")
        arrays = np.load(path)
        source_valid = torch.from_numpy(arrays["valid"].astype(bool))
        fuel = self.boundary_ext[:, BOUNDARY_EXT_ELEMENTS.index("fuel_corrected")]
        ratio = self.boundary_ext[:, BOUNDARY_EXT_ELEMENTS.index("water_coal_ratio")]
        load = self.boundary_ext[:, BOUNDARY_EXT_ELEMENTS.index("unit_load")]
        self.operating_mask = (
            source_valid & (load > 160.0) & (ratio > 1.0) & (ratio < 8.0) & (fuel > 50.0)
        )
        self.split = self.split.clone()
        self.split[~self.operating_mask] = -1
        self._split_runs_cache = {}
        self._jepa_start_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self.privileged = torch.cat(
            [self.boundary_ext, self.aux, self.mill_on.to(torch.float32)], dim=1
        )
        if self.privileged.shape != (self.n, PRIVILEGED_DIM):
            raise FinalWMProtocolError("JEPA-B privileged registry width mismatch")
        self.unit_load = load

    def valid_window_starts(self, split_id: int, history_steps: int, horizon: int) -> torch.Tensor:
        if split_id == SPLIT_TEST:
            raise FinalWMProtocolError("test split is locked and cannot be read")
        key = (int(split_id), int(history_steps), int(horizon))
        starts = self._jepa_start_cache.get(key)
        if starts is None:
            chunks = [
                torch.arange(s + history_steps, e - horizon + 1, dtype=torch.long)
                for s, e in self.split_runs(split_id)
                if e - s >= history_steps + horizon
            ]
            if not chunks:
                raise FinalWMProtocolError("no valid JEPA-B windows for requested split/horizon")
            starts = torch.cat(chunks)
            self._jepa_start_cache[key] = starts
        return starts


def fit_privileged_normalizer(record: JepaBRecord) -> PrivilegedNormalizer:
    mask = record.split == SPLIT_TRAIN
    if int(mask.sum()) < 2:
        raise FinalWMProtocolError("privileged normalization needs at least two train samples")
    train = record.privileged[mask].to(torch.float64)
    mean = train.mean(0).to(torch.float32)
    scale = train.std(0, unbiased=False).clamp_min(1e-6).to(torch.float32)
    return PrivilegedNormalizer(mean=mean, scale=scale)


class JepaWindowBatch(NamedTuple):
    history: HistoryWindow
    future_boundary: torch.Tensor
    future_actions: torch.Tensor
    future_obs: torch.Tensor
    history_privileged: torch.Tensor
    future_privileged: torch.Tensor
    partner_history_privileged: torch.Tensor
    partner_future_privileged: torch.Tensor
    future_indices: torch.Tensor
    partner_future_indices: torch.Tensor
    day_ids: torch.Tensor
    unit_load: torch.Tensor


def _windows_at(
    record: JepaBRecord, indices: torch.Tensor, history_steps: int, horizon: int
) -> tuple[HistoryWindow, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hist_idx = indices[:, None] + torch.arange(-history_steps, 0)[None, :]
    fut_idx = indices[:, None] + torch.arange(0, horizon)[None, :]
    history = HistoryWindow(
        obs=record.obs[hist_idx], actions=record.actions[hist_idx], boundary=record.boundary[hist_idx]
    )
    return (
        history,
        record.boundary[fut_idx],
        record.actions[fut_idx],
        record.obs[fut_idx],
        record.privileged[hist_idx],
        record.privileged[fut_idx],
    )


def sample_jepa_windows(
    record: JepaBRecord,
    split_id: int,
    batch_size: int,
    history_steps: int,
    horizon: int,
    generator: torch.Generator,
    *,
    fixed_indices: torch.Tensor | None = None,
) -> JepaWindowBatch:
    """Uniform-over-valid-windows sampler with a corpus-wide derangement."""
    pool = record.valid_window_starts(split_id, history_steps, horizon)
    if len(pool) < 2:
        raise FinalWMProtocolError("fixed derangement requires at least two valid windows")
    if fixed_indices is None:
        positions = torch.randint(len(pool), (batch_size,), generator=generator)
        indices = pool[positions]
    else:
        indices = torch.as_tensor(fixed_indices, dtype=torch.long).flatten()
        if len(indices) != batch_size:
            raise FinalWMProtocolError("fixed_indices length must equal batch_size")
        pos_map = {int(value): i for i, value in enumerate(pool.tolist())}
        try:
            positions = torch.tensor([pos_map[int(value)] for value in indices], dtype=torch.long)
        except KeyError as exc:
            raise FinalWMProtocolError("fixed index is outside the valid window pool") from exc
    # A fixed half-pool cyclic shift is a corpus-wide derangement.  Partner
    # history/future remain contiguous and intact; only correspondence is false.
    offset = max(1, len(pool) // 2)
    partner_positions = (positions + offset) % len(pool)
    partner_indices = pool[partner_positions]
    if bool((partner_indices == indices).any()):
        partner_indices = pool[(positions + 1) % len(pool)]
    history, fb, fa, fo, hp, fp = _windows_at(record, indices, history_steps, horizon)
    _ph, _pfb, _pfa, _pfo, php, pfp = _windows_at(
        record, partner_indices, history_steps, horizon
    )
    return JepaWindowBatch(
        history=history,
        future_boundary=fb,
        future_actions=fa,
        future_obs=fo,
        history_privileged=hp,
        future_privileged=fp,
        partner_history_privileged=php,
        partner_future_privileged=pfp,
        future_indices=indices,
        partner_future_indices=partner_indices,
        day_ids=torch.div(record.timestamps[indices], 86400, rounding_mode="floor"),
        unit_load=record.unit_load[indices],
    )


class SlicedGaussianCFLoss(nn.Module):
    """Fixed-slice characteristic-function Gaussianity regularizer.

    This is a small, dependency-free SIGReg-style adaptation.  It matches the
    empirical characteristic function of random one-dimensional projections
    to N(0,1); it is not claimed to reproduce the official package line-wise.
    """

    def __init__(self, dim: int, num_slices: int = 16, num_knots: int = 17, seed: int = 260830) -> None:
        super().__init__()
        if dim < 1 or num_slices < 1 or num_knots < 3:
            raise FinalWMProtocolError("Gaussian-CF dimensions must be positive")
        gen = torch.Generator().manual_seed(seed)
        directions = torch.randn(dim, num_slices, generator=gen)
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
        knots = torch.linspace(-3.0, 3.0, num_knots)
        self.dim = int(dim)
        self.register_buffer("directions", directions)
        self.register_buffer("knots", knots)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.dim:
            raise FinalWMProtocolError(f"embedding width must be {self.dim}")
        projections = embedding @ self.directions  # (N, S)
        phase = projections.unsqueeze(-1) * self.knots.view(1, 1, -1)
        empirical_real = torch.cos(phase).mean(0)
        empirical_imag = torch.sin(phase).mean(0)
        target_real = torch.exp(-0.5 * self.knots.square()).view(1, -1)
        return ((empirical_real - target_real).square() + empirical_imag.square()).mean()


class B1FutureStateAux(nn.Module):
    def __init__(self, embedding_dim: int = 8) -> None:
        super().__init__()
        self.target_encoder = nn.GRU(5, 32, batch_first=True)
        self.target_projector = nn.Linear(32, embedding_dim)
        self.state_projector = nn.Linear(len(PHYSICAL_STATE_ELEMENTS), embedding_dim)
        self.gaussian = SlicedGaussianCFLoss(embedding_dim)
        self.register_buffer("obs_loc", torch.tensor([x for x, _ in OBSERVATION_NORM]))
        self.register_buffer("obs_scale", torch.tensor([x for _, x in OBSERVATION_NORM]))
        self.register_buffer("state_loc", torch.tensor([x for x, _ in PHYSICAL_STATE_NORM]))
        self.register_buffer("state_scale", torch.tensor([x for _, x in PHYSICAL_STATE_NORM]))

    def terms(self, state0: torch.Tensor, future_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        _out, hidden = self.target_encoder((future_obs - self.obs_loc) / self.obs_scale)
        target = self.target_projector(hidden[-1])
        predicted = self.state_projector(
            (state0[..., : len(PHYSICAL_STATE_ELEMENTS)] - self.state_loc) / self.state_scale
        )
        return {"prediction": F.mse_loss(predicted, target), "gaussian_cf": self.gaussian(target)}


class B2SlowState(nn.Module):
    def __init__(
        self,
        physical_dim: int,
        boundary_dim: int,
        slow_dim: int = 4,
        stride: int = 6,
        observer_hidden: int = 64,
    ) -> None:
        super().__init__()
        if stride < 1:
            raise FinalWMProtocolError("slow stride must be positive")
        self.slow_dim = int(slow_dim)
        self.stride = int(stride)
        self.init_projection = nn.Linear(observer_hidden, slow_dim)
        self.update_net = nn.Sequential(
            nn.Linear(slow_dim + physical_dim + boundary_dim, 32), nn.Tanh(), nn.Linear(32, slow_dim)
        )
        self.power_net = nn.Sequential(nn.Linear(slow_dim, 32), nn.Tanh(), nn.Linear(32, 3))
        nn.init.zeros_(self.power_net[-1].weight)
        nn.init.zeros_(self.power_net[-1].bias)
        self.gaussian = SlicedGaussianCFLoss(slow_dim)
        self.register_buffer("state_loc", torch.tensor([x for x, _ in PHYSICAL_STATE_NORM]))
        self.register_buffer("state_scale", torch.tensor([x for _, x in PHYSICAL_STATE_NORM]))
        self.register_buffer("boundary_loc", torch.tensor([x for x, _ in BOUNDARY_NORM]))
        self.register_buffer("boundary_scale", torch.tensor([x for _, x in BOUNDARY_NORM]))

    def initial(self, observer_hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.init_projection(observer_hidden))

    def update(
        self, slow: torch.Tensor, physical: torch.Tensor, boundary: torch.Tensor, *, step: int
    ) -> torch.Tensor:
        if step == 0 or step % self.stride:
            return slow
        features = torch.cat(
            [slow, (physical - self.state_loc) / self.state_scale,
             (boundary - self.boundary_loc) / self.boundary_scale], dim=-1
        )
        return slow + 0.1 * torch.tanh(self.update_net(features))

    def power(self, slow: torch.Tensor, scale_kw: float = 3.0e4) -> torch.Tensor:
        return float(scale_kw) * torch.tanh(self.power_net(slow))


class B3CrossPredictiveAux(nn.Module):
    def __init__(
        self,
        observer_hidden: int,
        normalizer: PrivilegedNormalizer,
        embedding_dim: int = 16,
        pairing: str = "corresponding",
    ) -> None:
        super().__init__()
        if pairing not in ("corresponding", "fixed_derangement"):
            raise FinalWMProtocolError("unknown B3 pairing")
        self.pairing = pairing
        self.online_projector = nn.Linear(observer_hidden, embedding_dim)
        self.privileged_encoder = nn.GRU(PRIVILEGED_DIM, 64, batch_first=True)
        self.privileged_projector = nn.Linear(64, embedding_dim)
        self.action_encoder = nn.GRU(2, 16, batch_first=True)
        self.shared_predictor = nn.Sequential(
            nn.Linear(embedding_dim + 16, 64), nn.Tanh(), nn.Linear(64, embedding_dim)
        )
        self.gaussian_o = SlicedGaussianCFLoss(embedding_dim, seed=260831)
        self.gaussian_s = SlicedGaussianCFLoss(embedding_dim, seed=260832)
        self.register_buffer("priv_mean", normalizer.mean.clone())
        self.register_buffer("priv_scale", normalizer.scale.clone())

    @staticmethod
    def _future_observer_window(batch: JepaWindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = batch.future_obs.shape[1]
        history_steps = batch.history.obs.shape[1]
        if horizon > history_steps:
            raise FinalWMProtocolError("B3 target horizon exceeds observer history")
        keep = history_steps - horizon
        obs = torch.cat([batch.history.obs[:, -keep:] if keep else batch.history.obs[:, :0],
                         batch.future_obs], dim=1)
        boundary = torch.cat([
            batch.history.boundary[:, -keep:] if keep else batch.history.boundary[:, :0],
            batch.future_boundary,
        ], dim=1)
        return obs, boundary

    def _encode_privileged(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.priv_mean) / self.priv_scale
        _out, hidden = self.privileged_encoder(normalized)
        return self.privileged_projector(hidden[-1])

    def target_embeddings(
        self, base: FinalWorldModel, batch: JepaWindowBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Target encoder is explicitly action blind.  Future actions enter only
        # the shared predictor below, matching JEPA-x's source/action/target split.
        obs, boundary = self._future_observer_window(batch)
        zeros = torch.zeros(obs.shape[0], obs.shape[1], 2, device=obs.device, dtype=obs.dtype)
        z_o = self.online_projector(base.observer.encode(obs, zeros, boundary))
        horizon = batch.future_privileged.shape[1]
        keep = batch.history_privileged.shape[1] - horizon
        hp = (batch.partner_history_privileged if self.pairing == "fixed_derangement"
              else batch.history_privileged)
        fp = (batch.partner_future_privileged if self.pairing == "fixed_derangement"
              else batch.future_privileged)
        target_priv = torch.cat([hp[:, -keep:] if keep else hp[:, :0], fp], dim=1)
        return z_o, self._encode_privileged(target_priv)

    def terms(self, base: FinalWorldModel, batch: JepaWindowBatch) -> dict[str, torch.Tensor]:
        zeros = torch.zeros_like(batch.history.actions)
        z_o = self.online_projector(
            base.observer.encode(batch.history.obs, zeros, batch.history.boundary)
        )
        hp = (batch.partner_history_privileged if self.pairing == "fixed_derangement"
              else batch.history_privileged)
        z_s = self._encode_privileged(hp)
        z_o_future, z_s_future = self.target_embeddings(base, batch)
        _out, action_hidden = self.action_encoder(batch.future_actions)
        action = action_hidden[-1]
        pred_o = self.shared_predictor(torch.cat([z_o, action], dim=1))
        pred_s = self.shared_predictor(torch.cat([z_s, action], dim=1))
        prediction = sum(
            F.mse_loss(pred, target)
            for pred in (pred_o, pred_s)
            for target in (z_o_future, z_s_future)
        )
        gaussian = self.gaussian_o(torch.cat([z_o, z_o_future], dim=0))
        gaussian = gaussian + self.gaussian_s(torch.cat([z_s, z_s_future], dim=0))
        return {"prediction": prediction, "gaussian_cf": gaussian}


class B4PhysicsResidualAux(nn.Module):
    def __init__(self, residual_dim: int = 4) -> None:
        super().__init__()
        self.residual_dim = int(residual_dim)
        self.target_encoder = nn.GRU(5, 32, batch_first=True)
        self.target_projector = nn.Linear(32, residual_dim)
        self.gaussian = SlicedGaussianCFLoss(residual_dim, seed=260834)
        state_loc = torch.tensor([PHYSICAL_STATE_NORM[i][0] for i in PHYSICS_TARGET_INDICES])
        state_scale = torch.tensor([PHYSICAL_STATE_NORM[i][1] for i in PHYSICS_TARGET_INDICES])
        self.register_buffer("state_loc", state_loc)
        self.register_buffer("state_scale", state_scale)
        self.register_buffer("obs_loc", torch.tensor([x for x, _ in OBSERVATION_NORM]))
        self.register_buffer("obs_scale", torch.tensor([x for _, x in OBSERVATION_NORM]))

    def _physical(self, state: torch.Tensor) -> torch.Tensor:
        selected = state[..., list(PHYSICS_TARGET_INDICES)]
        return (selected - self.state_loc) / self.state_scale

    def terms(
        self, base: FinalWorldModel, batch: JepaWindowBatch, result: RolloutResult
    ) -> dict[str, torch.Tensor]:
        current = base._initial_state(batch.history)
        predicted_future = result.states[:, -1]
        with torch.no_grad():
            current_anchor = base.transition.initial_steady_state(
                batch.history.boundary[:, -1], batch.history.actions[:, -1], batch.history.obs[:, -1]
            )
            future_anchor = base.transition.initial_steady_state(
                batch.future_boundary[:, -1], batch.future_actions[:, -1], batch.future_obs[:, -1]
            )
        _out, hidden = self.target_encoder((batch.future_obs - self.obs_loc) / self.obs_scale)
        target_residual = self.target_projector(hidden[-1])
        current_residual = current[..., -self.residual_dim:]
        predicted_residual = predicted_future[..., -self.residual_dim:]
        prediction = F.mse_loss(predicted_residual, target_residual)
        gaussian = self.gaussian(torch.cat(
            [current_residual, predicted_residual, target_residual], dim=0
        ))
        p0, p1 = self._physical(current), self._physical(predicted_future)
        q0, q1 = self._physical(current_anchor), self._physical(future_anchor)
        static = F.mse_loss(p0, q0) + F.mse_loss(p1, q1)
        dynamic = F.mse_loss(p1 - p0, q1 - q0)
        return {"prediction": prediction, "gaussian_cf": gaussian,
                "static": static, "dynamic": dynamic}


class JepaBModel(nn.Module):
    def __init__(
        self,
        base: FinalWorldModel,
        arm: str,
        auxiliary: nn.Module | None = None,
        slow: B2SlowState | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.arm = arm
        self.auxiliary = auxiliary
        self.slow = slow
        self.slow_mechanism_scale = 1.0

    @property
    def transition(self):
        return self.base.transition

    @property
    def observer(self):
        return self.base.observer

    @property
    def observation(self):
        return self.base.observation

    @property
    def config(self):
        return self.base.config

    @property
    def layout(self):
        return self.base.layout

    @staticmethod
    def observation_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return FinalWorldModel.observation_nll(mu, sigma, target)

    def _slow_rollout(
        self,
        history: HistoryWindow,
        state0: torch.Tensor,
        boundary: BoundarySequence,
        action_seq: torch.Tensor,
        *,
        in_support: torch.Tensor | None = None,
    ) -> RolloutResult:
        if self.slow is None:
            raise FinalWMProtocolError("slow rollout requested without B2 slow state")
        hidden = self.base.observer.encode(history.obs, history.actions, history.boundary)
        slow = self.slow.initial(hidden)
        state = state0
        states, temps = [], []
        for step in range(action_seq.shape[1]):
            boundary_t = boundary.mu[:, step]
            slow = self.slow.update(slow, state, boundary_t, step=step)
            residual = self.base.closure(state, boundary_t)
            extra = self.slow.power(slow) * float(self.slow_mechanism_scale)
            steam = residual.steam_power + extra
            metal = residual.metal_power - extra
            result = self.base.transition.step(
                state, boundary_t, action_seq[:, step],
                ResidualInjection(steam, metal, residual.latent_step),
            )
            state = result.state
            temp = self.base.transition.output_temperatures(state, boundary_t, action_seq[:, step])
            states.append(torch.cat([state, slow], dim=1))
            temps.append(temp)
        state_seq = torch.stack(states, dim=1)
        temp_seq = torch.stack(temps, dim=1)
        physical = state_seq[..., : len(PHYSICAL_STATE_ELEMENTS)]
        sigma = self.base.observation.sigma(physical.reshape(-1, physical.shape[-1])).reshape_as(temp_seq)
        return RolloutResult(state_seq, temp_seq, sigma, boundary, boundary.mode, in_support)

    def forecast(self, history: HistoryWindow, action_seq: torch.Tensor, **kwargs) -> RolloutResult:
        if self.arm != "b2":
            return self.base.forecast(history, action_seq, **kwargs)
        self.base._check_history(history)
        mode = kwargs.pop("boundary_mode", None) or self.base.config.boundary_mode
        true_future = kwargs.pop("true_future_boundary", None)
        scenario = kwargs.pop("scenario", None)
        sample_posterior = bool(kwargs.pop("sample_posterior", False))
        noise = kwargs.pop("noise", None)
        if kwargs or noise is not None:
            raise FinalWMProtocolError("B2 slow rollout received unsupported forecast options")
        boundary = self.base._boundary_sequence(
            history, action_seq.shape[1], mode, true_future, scenario
        )
        state0 = self.base._initial_state(history, sample_posterior=sample_posterior)
        return self._slow_rollout(history, state0, boundary, action_seq)

    def counterfactual(self, history: HistoryWindow, action_seq: torch.Tensor, **kwargs) -> RolloutResult:
        if self.arm != "b2":
            return self.base.counterfactual(history, action_seq, **kwargs)
        self.base._check_history(history)
        allow = bool(kwargs.pop("allow_extrapolation", False))
        initial_state = kwargs.pop("initial_state", None)
        mode = kwargs.pop("boundary_mode", None) or self.base.config.boundary_mode
        true_future = kwargs.pop("true_future_boundary", None)
        scenario = kwargs.pop("scenario", None)
        if kwargs:
            raise FinalWMProtocolError("B2 counterfactual received unsupported options")
        support = action_support_from_history(history.actions, self.base.config.support_margin)
        in_support = support.contains(action_seq)
        if not allow and not bool(in_support.all()):
            raise FinalWMProtocolError("counterfactual actions leave the history support box")
        boundary = self.base._boundary_sequence(
            history, action_seq.shape[1], mode, true_future, scenario
        )
        state0 = self.base._initial_state(history) if initial_state is None else initial_state
        if state0.shape[-1] != len(PHYSICAL_STATE_ELEMENTS):
            raise FinalWMProtocolError("B2 counterfactual initial state must be the 11-D physical state")
        return self._slow_rollout(history, state0, boundary, action_seq, in_support=in_support)

    def auxiliary_terms(
        self, batch: JepaWindowBatch, result: RolloutResult | None = None
    ) -> dict[str, torch.Tensor]:
        if self.arm == "c0":
            return {}
        if self.arm == "b1":
            state0 = self.base._initial_state(batch.history)
            return self.auxiliary.terms(state0, batch.future_obs)  # type: ignore[union-attr]
        if self.arm == "b2":
            if result is None:
                raise FinalWMProtocolError("B2 auxiliary loss requires rollout result")
            slow = result.states[..., -self.slow.slow_dim:]  # type: ignore[union-attr]
            return {"gaussian_cf": self.slow.gaussian(slow.reshape(-1, slow.shape[-1]))}  # type: ignore[union-attr]
        if self.arm in ("b3", "b3_shuffle"):
            return self.auxiliary.terms(self.base, batch)  # type: ignore[union-attr]
        if self.arm == "b4":
            if result is None:
                raise FinalWMProtocolError("B4 auxiliary loss requires rollout result")
            return self.auxiliary.terms(self.base, batch, result)  # type: ignore[union-attr]
        raise FinalWMProtocolError(f"unknown JEPA-B arm: {self.arm}")


def build_jepa_model(
    arm: str,
    *,
    history_steps: int = 96,
    properties: ThermoProperties | None = None,
    normalizer: PrivilegedNormalizer | None = None,
) -> JepaBModel:
    if arm not in ("c0", "b1", "b2", "b3", "b3_shuffle", "b4"):
        raise FinalWMProtocolError(f"unknown JEPA-B arm: {arm}")
    latent_dim = 4 if arm == "b4" else 0
    spec = TrainSpec(
        unit="jepa_b", arm=arm, seed=0, history_steps=history_steps,
        boundary_mode="oracle", initial_state_mode="hybrid",
        closure_mode="conservative_norew", latent_dim=latent_dim,
    )
    base = build_world_model(spec, properties or AnalyticThermoProperties())
    if arm == "c0":
        return JepaBModel(base, arm)
    if arm == "b1":
        return JepaBModel(base, arm, auxiliary=B1FutureStateAux(8))
    if arm == "b2":
        slow = B2SlowState(
            physical_dim=len(PHYSICAL_STATE_ELEMENTS), boundary_dim=len(BOUNDARY_ELEMENTS),
            slow_dim=4, stride=6, observer_hidden=base.config.observer.d_hidden,
        )
        return JepaBModel(base, arm, slow=slow)
    if arm in ("b3", "b3_shuffle"):
        if normalizer is None:
            raise FinalWMProtocolError("B3 requires a train-only privileged normalizer")
        pairing = "corresponding" if arm == "b3" else "fixed_derangement"
        auxiliary = B3CrossPredictiveAux(
            base.config.observer.d_hidden, normalizer, embedding_dim=16, pairing=pairing
        )
        return JepaBModel(base, arm, auxiliary=auxiliary)
    return JepaBModel(base, arm, auxiliary=B4PhysicsResidualAux(4))
