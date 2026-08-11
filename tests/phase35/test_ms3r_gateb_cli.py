from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.phase35.data import save_cache
from tests.phase35.test_ms3r_gateb import CONFIG_PATH, _cache_pair, _config


def test_gateb_runner_bundle_finalization_and_cache_free_replay(tmp_path):
    from experiments.phase3_5.audit_ms3r_gateb_point_closure import replay
    from experiments.phase3_5.ms3r_gateb_point_closure import finalize, run

    config = _config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    cache_paths = {"A": tmp_path / "A.npz", "B": tmp_path / "B.npz"}
    for side, cache in _cache_pair(11).items():
        save_cache(cache, cache_paths[side])
    output = tmp_path / "results"
    summary = run(
        config_path=config_path, cache_paths=cache_paths, output=output, require_clean=False
    )
    for name in ("stdout.log", "stderr.log", "resource_usage.txt"):
        (output / name).write_text("synthetic test placeholder\n", encoding="utf-8")
    ledger = finalize(config_path=config_path, output=output)
    assert set(ledger) == set(config["execution_contract"]["required_artifacts"]) - {"artifact_ledger.json"}
    assert summary["manifest"]["test_accessed"] is False
    with np.load(output / "replay_arrays_validation.npz", allow_pickle=False) as arrays:
        report = replay(config, arrays)
    assert report["maximum_daily_matrix_error"] < 1e-12
    assert report["maximum_paired_contrast_error"] < 1e-12
    assert report["cache_accessed"] is False
