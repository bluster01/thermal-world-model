import json

from src.phase35.matrix import expand_matrix, load_matrix
from src.phase35.reporting import summarize


def test_frozen_matrix_expands_to_42_runs():
    matrix = load_matrix("configs/phase3_5/experiment_matrix.json")
    runs = expand_matrix(matrix)
    assert len(runs) == 42
    assert len({run.run_id for run in runs}) == 42


def test_missing_event_results_remain_inconclusive(tmp_path):
    run = tmp_path / "runs" / "A_absolute_identity_s0"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "A_absolute_identity_s0",
        "side": "A",
        "seed": 0,
        "config": {"config_id": "absolute_identity"},
    }), encoding="utf-8")
    (run / "metrics_validation.json").write_text(json.dumps({"integrated_mae": 0.5}), encoding="utf-8")
    result = summarize(tmp_path / "runs", "configs/phase3_5/experiment_matrix.json", "validation")
    assert result["n_runs"] == 1
    assert result["gates"]["A"]["E3_empirical_response"]["status"] == "INCONCLUSIVE"
    assert "absolute_identity" in result["markdown"]
