import json

from src.phase35.matrix import expand_matrix, load_matrix
from src.phase35.reporting import evaluate_gates, summarize


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


def test_invalid_empirical_reference_blocks_model_gate_and_small_sp_groups_are_inconclusive():
    def metric(value):
        return {"mean": value, "n": 3, "sd": 0.0}

    metrics = {
        "event.valve_events_matched": metric(93),
        "event.main_temperature.all_oriented.n_clusters": metric(11),
        "event.main_temperature.open.n_events": metric(93),
        "event.main_temperature.close.n_events": metric(0),
        "event.main_temperature.all_oriented.empirical_ci_high_h60": metric(1.0),
        "event.main_temperature.open.empirical_direction_rate": metric(0.32),
        "event.main_temperature.close.empirical_direction_rate": metric(0.32),
        "event.matching_balance.max_abs_smd": metric(0.30),
        "event.matching_balance.main_temperature_pretrend_difference_c": metric(0.04),
        "event.main_temperature.open.model_direction_rate": metric(1.0),
        "event.main_temperature.close.model_direction_rate": metric(1.0),
        "event.main_temperature.open.irf_wmae": metric(0.4),
        "event.main_temperature.close.irf_wmae": metric(0.4),
        "event.sp_negative_control.no_execution.mean_abs_model_effect_h6": metric(0.0),
        "event.sp_negative_control.executed.mean_abs_model_effect_h6": metric(0.1),
        "event.sp_negative_control.no_execution.n_events": metric(4),
        "event.sp_negative_control.executed.n_events": metric(134),
    }
    aggregate = {"A": {"absolute_identity": {"metrics": metrics}}}
    gates = evaluate_gates(aggregate, load_matrix("configs/phase3_5/experiment_matrix.json"))["A"]
    assert gates["E3_empirical_response"]["status"] == "INCONCLUSIVE"
    assert gates["E4_model_response"]["status"] == "BLOCKED"
    assert gates["E5_sp_negative_control"]["status"] == "INCONCLUSIVE"
