import json

from src.phase35.matrix import expand_matrix, load_matrix


def _matrix_file(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "protocol_version": "test",
                "sides": ["A", "B"],
                "seeds": [0, 1, 2],
                "defaults": {
                    "window": 8,
                    "horizon": 4,
                    "d_model": 8,
                    "n_heads": 2,
                    "action_mode": "none",
                },
                "experiments": [{"config_id": "free"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_expand_matrix_uses_development_seeds_by_default(tmp_path):
    matrix = load_matrix(_matrix_file(tmp_path))
    assert [run.seed for run in expand_matrix(matrix)] == [0, 1, 2, 0, 1, 2]


def test_expand_matrix_can_use_preregistered_final_seeds(tmp_path):
    matrix = load_matrix(_matrix_file(tmp_path))
    runs = expand_matrix(matrix, seeds=[3, 4])
    assert [run.seed for run in runs] == [3, 4, 3, 4]
    assert {run.side for run in runs} == {"A", "B"}
