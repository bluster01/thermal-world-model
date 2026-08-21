"""R1 negative-control diagnostics: residual leakage probe.

The production closure is contractually action-blind (the
`reads_actions` flag cannot be enabled).  To *detect* whether that
constraint costs real signal, this module trains two standalone probes
to predict the frozen physics model's one-step-ahead observation
residual:

- blind probe: features = normalized state + whitelisted boundary;
- aware probe (negative control): same + normalized actions.

If the aware probe fits validation residuals significantly better than
the blind probe, the action-blind closure is leaking/absorbing valve
causality and the R1 arm must be judged REJECTED (matrix §R1 item 4).

The probes are diagnostics only: they never enter the assembled world
model and cannot influence its forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.final_wm.closure import ActionBlindClosure
from src.final_wm.contracts import FinalWMProtocolError, OBSERVATION_NORM
from src.final_wm.data import SPLIT_TRAIN, SPLIT_VAL, CanonicalRecord, sample_windows
from src.final_wm.model import FinalWorldModel


class ResidualLeakageProbe(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 5),
        )
        self.register_buffer(
            "obs_scale", torch.tensor([s for _l, s in OBSERVATION_NORM], dtype=torch.float32)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # predicts normalized obs residual


@torch.no_grad()
def _one_step_residuals(
    model: FinalWorldModel,
    record: CanonicalRecord,
    split_id: int,
    *,
    n_windows: int,
    history_steps: int,
    seed: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen physics model one-step residuals plus closure features/actions."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = sample_windows(record, split_id, n_windows, history_steps, 2, gen)
    history = batch.history.__class__(
        obs=batch.history.obs.to(device),
        actions=batch.history.actions.to(device),
        boundary=batch.history.boundary.to(device),
    )
    state = model._initial_state(history)
    boundary_t = batch.future_boundary[:, 0].to(device)
    action_t = batch.future_actions[:, 0].to(device)
    step = model.transition.step(state, boundary_t, action_t)
    pred = model.transition.output_temperatures(step.state, batch.future_boundary[:, 1].to(device), batch.future_actions[:, 1].to(device))
    residual = batch.future_obs[:, 1].to(device) - pred
    features = model.closure.features(step.state, batch.future_boundary[:, 1].to(device))
    return features.cpu(), action_t.cpu(), residual.cpu()


def leakage_probe(
    model: FinalWorldModel,
    record: CanonicalRecord,
    *,
    n_windows: int = 512,
    history_steps: int = 96,
    epochs: int = 20,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """Train blind vs action-aware probes; report val-MSE improvement."""
    if model.config.closure.injection_mode == "none":
        raise FinalWMProtocolError("leakage probe needs a closure-bearing model")
    feat_tr, act_tr, res_tr = _one_step_residuals(
        model, record, SPLIT_TRAIN, n_windows=n_windows, history_steps=history_steps, seed=seed, device=device
    )
    feat_va, act_va, res_va = _one_step_residuals(
        model, record, SPLIT_VAL, n_windows=n_windows // 4, history_steps=history_steps, seed=seed + 1, device=device
    )

    # Shuffle control (protocol fix 2026-08-21, Hermes artifact finding in
    # results/final_wm/action_signal_analysis_20260821.md §2): at the frozen
    # probe budget the blind probe is underfit, so ANY added input dimension
    # -- even information-free shuffled valves -- improves val fit by changing
    # the optimization trajectory (true 23.9% vs shuffled 23.2%).  The suspect
    # criterion is therefore the DELTA over the shuffled null, not the raw
    # aware-over-blind improvement.
    g = torch.Generator().manual_seed(seed + 2)
    perm = torch.randperm(act_tr.shape[0], generator=g)
    act_tr_shuf = act_tr[perm]
    results = {}
    for name, x_tr, x_va in (
        ("blind", feat_tr, feat_va),
        ("aware", torch.cat([feat_tr, act_tr], dim=-1), torch.cat([feat_va, act_va], dim=-1)),
        ("aware_shuffled", torch.cat([feat_tr, act_tr_shuf], dim=-1), torch.cat([feat_va, act_va], dim=-1)),
    ):
        torch.manual_seed(seed)
        probe = ResidualLeakageProbe(x_tr.shape[-1]).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        y_tr = (res_tr / probe.obs_scale.cpu()).to(device)
        y_va = (res_va / probe.obs_scale.cpu()).to(device)
        for _ in range(epochs):
            probe.train()
            opt.zero_grad()
            loss = ((probe(x_tr.to(device)) - y_tr) ** 2).mean()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val_mse = float(((probe(x_va.to(device)) - y_va) ** 2).mean())
        base_mse = float((y_va**2).mean())
        results[name] = {"val_mse_norm": val_mse, "base_mse_norm": base_mse}
    blind, aware, shuffled = results["blind"], results["aware"], results["aware_shuffled"]
    improvement = (blind["val_mse_norm"] - aware["val_mse_norm"]) / max(blind["val_mse_norm"], 1e-12)
    improvement_shuf = (blind["val_mse_norm"] - shuffled["val_mse_norm"]) / max(blind["val_mse_norm"], 1e-12)
    delta = improvement - improvement_shuf
    return {
        "blind": blind,
        "aware": aware,
        "aware_shuffled": shuffled,
        "aware_relative_improvement": improvement,
        "shuffled_relative_improvement": improvement_shuf,
        "leakage_delta": delta,
        "leakage_suspected": bool(delta > 0.05),
    }
