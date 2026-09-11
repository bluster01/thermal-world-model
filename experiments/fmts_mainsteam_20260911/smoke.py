"""Synthetic smoke audit; this script never opens canonical or locked-test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.fmts_mainsteam_20260911.spec import structured_specs
from src.final_wm.model import HistoryWindow
from src.final_wm.synthetic import synthetic_history
from src.final_wm.training import build_world_model

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "smoke_audit.json"
FROZEN_COMPONENTS = (
    "src/final_wm/model.py",
    "src/final_wm/transition.py",
    "src/final_wm/closure.py",
    "src/final_wm/boundary.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    specs = {spec.arm: spec for spec in structured_specs((0,))}
    torch.manual_seed(20260911)
    gru = build_world_model(specs["fusion_gru_norew"])
    torch.manual_seed(20260911)
    token = build_world_model(specs["fusion_token_xattn_norew"])

    batch = synthetic_history(batch=2, history_steps=96, horizon=2, seed=20260911)
    history = HistoryWindow(*batch.history)
    anchor = token._steady_initial_state(history)
    mu, sigma = token.observer.posterior(
        history.obs, history.actions, history.boundary, anchor
    )
    zero_anchor = bool(torch.equal(mu, anchor))
    bounded = bool(((mu - anchor).abs() <= 0.1 * token.observer.state_scale).all())

    actions = batch.future_actions[:, :2]
    boundaries = batch.future_boundary[:, :2]
    token.eval()
    with torch.no_grad():
        first = token.forecast(
            history, actions, boundary_mode="oracle", true_future_boundary=boundaries
        ).temps_mu
        second = token.forecast(
            history, actions.clone(), boundary_mode="oracle",
            true_future_boundary=boundaries.clone(),
        ).temps_mu

    payload = {
        "canonical_data_opened": False,
        "locked_test_opened": False,
        "zero_init_anchor_identity": zero_anchor,
        "bounded_correction": bounded,
        "identical_input_rollout_identity": bool(torch.equal(first, second)),
        "finite_rollout": bool(torch.isfinite(first).all()),
        "encode_shape": list(token.observer.encode(
            history.obs, history.actions, history.boundary
        ).shape),
        "frozen_module_config_equal_to_gru": {
            "transition": gru.transition.config == token.transition.config,
            "closure": gru.closure.config == token.closure.config,
            "boundary": gru.boundary_model.config == token.boundary_model.config,
        },
        "frozen_component_source_sha256": {
            name: sha256(ROOT / name) for name in FROZEN_COMPONENTS
        },
    }
    payload["passed"] = all([
        zero_anchor,
        bounded,
        payload["identical_input_rollout_identity"],
        payload["finite_rollout"],
        *payload["frozen_module_config_equal_to_gru"].values(),
    ])
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

