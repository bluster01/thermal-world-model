import json

import numpy as np
import torch

from src.phase35.data import Phase35Cache
from src.phase35.schema import ExperimentConfig, REQUIRED_COLUMNS, TARGET_COLUMN, VALVE_COLUMN
from src.phase35.training import train_one


def _cache(n=180):
    rng = np.random.default_rng(2)
    cols = tuple(REQUIRED_COLUMNS)
    values = rng.normal(size=(n, len(cols))).astype(np.float32)
    valve = 20 + np.sin(np.arange(n) / 12) * 3
    temp = 565 + np.sin(np.arange(n) / 20) - 0.03 * valve
    values[:, cols.index(VALVE_COLUMN)] = valve
    values[:, cols.index(TARGET_COLUMN)] = temp
    ages = np.zeros_like(values)
    return Phase35Cache(
        timestamps_ns=np.arange(n, dtype=np.int64) * 10_000_000_000,
        values=values,
        ages_s=ages,
        columns=cols,
        metadata={"side": "A", "step_seconds": 10},
    )


def test_training_selects_on_validation_and_never_marks_test_access(tmp_path):
    cfg = ExperimentConfig.from_mapping({
        "config_id": "smoke",
        "action_mode": "absolute",
        "window": 8,
        "horizon": 6,
        "d_model": 8,
        "n_heads": 2,
        "dropout": 0.0,
        "batch_size": 4,
        "steps_per_epoch": 1,
        "epochs": 2,
        "patience": 2,
        "max_eval_anchors": 16,
    })
    result = train_one(_cache(), cfg, "A", 0, tmp_path / "run", device=torch.device("cpu"))
    assert result.checkpoint.is_file()
    with (tmp_path / "run" / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["checkpoint_selector"] == "validation_integrated_mae"
    assert manifest["test_accessed"] is False
    assert (tmp_path / "run" / "metrics_validation.json").is_file()
