"""Local micro-cache forward/backward smoke for all RM3 prediction candidates."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from ..data import Phase35Cache
from ..schema import Phase35ProtocolError
from .gatec_data import extract_gatec_batch, paired_valid_anchors
from .rm3_prediction import RM3FairPredictionAdapter, RM3PredictionConfig


def run_rm3_prediction_micro_smoke(
    caches: Mapping[str, Phase35Cache],
    candidate_id: str,
    *,
    device: str = "cpu",
    window: int = 16,
    horizon: int = 60,
    anchor_count: int = 4,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase35ProtocolError("RM3 micro smoke requested unavailable CUDA")
    anchors = paired_valid_anchors(
        caches, "validation", window=window, horizon=horizon, max_age_s=180.0
    )
    if len(anchors) < anchor_count:
        raise Phase35ProtocolError("RM3 micro smoke has too few validation anchors")
    selected = anchors[:anchor_count]
    batch = extract_gatec_batch(
        caches, selected, window=window, horizon=horizon, validate_pair=False
    )
    torch.manual_seed(35030)
    model = RM3FairPredictionAdapter(
        RM3PredictionConfig(
            candidate_id=candidate_id,
            window=window,
            horizon=horizon,
            n_features=batch.history.shape[-1],
            d_model=16,
            latent_dim=8,
            dropout=0.0,
        ),
        batch.history_feature_names,
    ).to(torch_device)
    history = torch.as_tensor(batch.history, dtype=torch.float32, device=torch_device)
    future_sp = torch.as_tensor(batch.future_sp, dtype=torch.float32, device=torch_device)
    terminal = torch.as_tensor(batch.terminal_target, dtype=torch.float32, device=torch_device)
    logged_valve = torch.as_tensor(
        batch.logged_future_valve, dtype=torch.float32, device=torch_device
    )
    train_rows = paired_valid_anchors(
        caches, "train", window=window, horizon=horizon, max_age_s=180.0
    )[: min(64, len(anchors))]
    train_batch = extract_gatec_batch(
        caches, train_rows, window=window, horizon=horizon, validate_pair=False
    )
    center = torch.as_tensor(
        np.median(train_batch.history.reshape(-1, train_batch.history.shape[-1]), axis=0),
        dtype=torch.float32,
        device=torch_device,
    )
    scale = torch.as_tensor(
        np.maximum(
            np.median(
                np.abs(train_batch.history.reshape(-1, train_batch.history.shape[-1]) - center.cpu().numpy()),
                axis=0,
            ),
            1e-3,
        ),
        dtype=torch.float32,
        device=torch_device,
    )
    model.set_history_normalization(center, scale)
    kwargs = {"logged_future_valve": logged_valve} if candidate_id == "P0_m7_oracle_valve" else {}
    output = model(history, future_sp, **kwargs)
    loss = torch.nn.functional.smooth_l1_loss(output["terminal_prediction"], terminal)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not torch.isfinite(loss) or not finite_gradients:
        raise Phase35ProtocolError("RM3 micro smoke produced non-finite loss/gradients")
    return {
        "candidate_id": candidate_id,
        "anchor_count": int(anchor_count),
        "history_shape": list(history.shape),
        "terminal_shape": list(output["terminal_prediction"].shape),
        "action_access": output["action_access"],
        "deployable": bool(output["deployable"]),
        "prefix_causal_action_path": bool(output["prefix_causal_action_path"]),
        "terminal_loss": float(loss.detach().cpu()),
        "finite_gradients": bool(finite_gradients),
        "test_accessed": False,
        "automatic_scientific_pass": None,
    }
