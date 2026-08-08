import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.phase35.data import Phase35Cache, save_cache
from src.phase35.schema import LOAD_COLUMN, REQUIRED_COLUMNS, SP_COLUMN, TARGET_COLUMN, VALVE_COLUMN


ROOT = Path(__file__).resolve().parents[2]


def test_train_and_validation_evaluate_cli_smoke(tmp_path):
    n = 4000
    columns = tuple(REQUIRED_COLUMNS)
    values = np.zeros((n, len(columns)), dtype=np.float32)
    ages = np.zeros_like(values)
    values[:, columns.index(LOAD_COLUMN)] = 500.0
    values[:, columns.index(TARGET_COLUMN)] = 565.0 + np.sin(np.arange(n) / 200.0)
    values[:, columns.index(SP_COLUMN)] = 568.0
    values[:, columns.index(VALVE_COLUMN)] = 20.0
    for onset, step in ((2500, 3.0), (2700, -3.0), (2900, 3.0)):
        values[onset:, columns.index(VALVE_COLUMN)] += step
    cache_path = tmp_path / "cache_A.npz"
    save_cache(
        Phase35Cache(
            np.arange(n, dtype=np.int64) * 10_000_000_000,
            values,
            ages,
            columns,
            {"protocol_version": "smoke", "side": "A", "step_seconds": 10},
        ),
        cache_path,
    )
    matrix = {
        "protocol_version": "phase3.5-smoke",
        "sides": ["A", "B"],
        "seeds": [0],
        "defaults": {
            "window": 12,
            "horizon": 6,
            "d_model": 8,
            "n_heads": 2,
            "dropout": 0.0,
            "loss": "huber",
            "batch_size": 16,
            "steps_per_epoch": 1,
            "epochs": 1,
            "patience": 1,
            "max_train_anchors": 64,
            "max_eval_anchors": 64,
        },
        "experiments": [
            {
                "config_id": "absolute_identity",
                "action_mode": "absolute",
                "opening_map": "identity",
                "rate_branch": False,
                "free_head": True,
            }
        ],
        "gates": {},
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    run_root = tmp_path / "runs"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/phase3_5/train.py"),
            "--matrix", str(matrix_path),
            "--config-id", "absolute_identity",
            "--cache", str(cache_path),
            "--side", "A",
            "--seed", "0",
            "--output-root", str(run_root),
            "--device", "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = run_root / "A_absolute_identity_s0"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/phase3_5/evaluate.py"),
            "--checkpoint", str(run_dir / "checkpoint_best_val.pt"),
            "--cache", str(cache_path),
            "--split", "validation",
            "--device", "cpu",
            "--controls-per-event", "2",
            "--bootstrap", "20",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (run_dir / "metrics_validation.json").is_file()
    assert (run_dir / "event_metrics_validation.json").is_file()
    assert (run_dir / "event_manifest_validation.json").is_file()
    assert not (run_dir / "access_ledger.json").exists()

    test_command = [
        sys.executable,
        str(ROOT / "experiments/phase3_5/evaluate.py"),
        "--checkpoint", str(run_dir / "checkpoint_best_val.pt"),
        "--cache", str(cache_path),
        "--split", "test",
        "--device", "cpu",
        "--allow-test-access",
        "--bootstrap", "20",
    ]
    subprocess.run(test_command, cwd=ROOT, check=True, capture_output=True, text=True)
    ledger = json.loads((run_dir / "access_ledger.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "completed"
    repeat = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "refusing repeat access" in repeat.stderr
