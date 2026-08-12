from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.phase35.multistep.rm3_prediction import RM3FairPredictionAdapter, RM3PredictionConfig


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/phase3_5/ms3r_rm3/prediction"


def test_all_returned_rm3_checkpoints_match_manifest_and_load_strictly() -> None:
    run_dirs = sorted(path for path in RESULTS.iterdir() if path.is_dir())
    assert len(run_dirs) == 36
    candidate_ids = set()
    for run_dir in run_dirs:
        checkpoint = torch.load(
            run_dir / "checkpoint_best_validation.pt", map_location="cpu", weights_only=False
        )
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads(
            (run_dir / "metrics_validation.json").read_text(encoding="utf-8")
        )
        assert checkpoint["protocol_version"] == manifest["protocol_version"]
        checkpoint_spec = json.loads(json.dumps(checkpoint["run_spec"], sort_keys=True))
        assert checkpoint_spec == manifest["run_spec"]
        assert checkpoint["run_spec"]["run_id"] == run_dir.name
        assert checkpoint["best_update"] == metrics["best_update"]
        assert checkpoint["best_selector_score"] == metrics["best_selector_score"]
        config = RM3PredictionConfig(**checkpoint["model_config"])
        model = RM3FairPredictionAdapter(config, checkpoint["feature_names"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert np.array_equal(
            model.model.history_center.detach().numpy(), checkpoint["history_center"]
        )
        assert np.array_equal(
            model.model.history_scale.detach().numpy(), checkpoint["history_scale"]
        )
        candidate_ids.add(config.candidate_id)
    assert candidate_ids == {
        "P0_m7_oracle_valve",
        "P1_m7_predicted_valve",
        "P2_m9_future_sp",
        "P3_gatec_paired_free",
        "P4_gatec_a1_scheduled",
        "P5_hybrid_joint_latent",
    }
