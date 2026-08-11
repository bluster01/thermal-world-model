from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.phase35.data import save_cache
from tests.phase35.test_ms3r import _cache, _config


ROOT = Path(__file__).resolve().parents[2]


def test_gate1_local_replay_closes_artifact_contract(tmp_path):
    from experiments.phase3_5.audit_ms3r_gate1_point_identifiability import run_audit
    from experiments.phase3_5.ms3r_gate1_point_identifiability import run

    config = _config()
    parent = ROOT / config["parent_matrix"]["path"]
    config["parent_matrix"]["sha256"] = hashlib.sha256(parent.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    cache_paths = {"A": tmp_path / "A.npz", "B": tmp_path / "B.npz"}
    for side, seed in (("A", 10), ("B", 20)):
        save_cache(_cache(side, seed), cache_paths[side])
    results = tmp_path / "results"
    run(
        config_path=config_path,
        cache_paths=cache_paths,
        output=results,
        require_clean=False,
    )
    audit = run_audit(results, cache_paths)
    assert audit["passes"] is True
    assert audit["scientific_decision"] is None
    assert all(audit["contract_checks"].values())
