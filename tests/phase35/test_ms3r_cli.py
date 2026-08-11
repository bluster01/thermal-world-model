from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.phase35.data import save_cache
from tests.phase35.test_ms3r import _cache, _config


ROOT = Path(__file__).resolve().parents[2]


def test_gate1_runner_writes_validation_only_machine_bundle(tmp_path):
    from experiments.phase3_5.ms3r_gate1_point_identifiability import run

    config = _config()
    parent = ROOT / config["parent_matrix"]["path"]
    import hashlib

    config["parent_matrix"]["sha256"] = hashlib.sha256(parent.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    cache_paths = {"A": tmp_path / "A.npz", "B": tmp_path / "B.npz"}
    for side, seed in (("A", 10), ("B", 20)):
        save_cache(_cache(side, seed), cache_paths[side])
    output = tmp_path / "results"
    summary = run(
        config_path=config_path,
        cache_paths=cache_paths,
        output=output,
        require_clean=False,
    )
    expected = {
        "run_manifest.json",
        "branch_semantics.json",
        "point_quality_validation.json",
        "path_diagnostics_validation.json",
        "rank_diagnostics_validation.json",
        "analysis_arrays_validation.npz",
        "summary_validation.json",
        "artifact_ledger.json",
    }
    assert expected == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "validation"
    assert manifest["test_accessed"] is False
    assert manifest["training_executed"] is False
    assert manifest["automatic_scientific_pass"] is None
    assert summary["automatic_scientific_pass"] is None
    with np.load(output / "analysis_arrays_validation.npz") as arrays:
        assert set(arrays.files) == {
            "anchors",
            "timestamps_ns",
            "innovation_A",
            "innovation_B",
            "fold_id_A",
            "fold_id_B",
        }
