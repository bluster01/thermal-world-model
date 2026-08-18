import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


q33 = load_module(ROOT / "33_qnav_residual_feedback_probe.py", "q33_under_test")


class DoubleResidual:
    def __call__(self, features):
        return features[:, :3] * 2.0


class Q32RTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = q33.load_config(
            ROOT / "configs" / "qnav_residual_feedback_probe.json"
        )

    def test_record_replay_and_feature_freeze_contracts(self):
        base = DoubleResidual()
        reference = torch.arange(10, dtype=torch.float32)[None, :]
        recorder = q33.RecordResidual(base)
        expected = recorder(reference)

        replay = q33.ReplayResidual(recorder.outputs)
        torch.testing.assert_close(replay(reference + 100.0), expected)
        replay.assert_consumed()

        freeze = q33.FreezeFeaturesResidual(base, recorder.features, slice(0, 3))
        torch.testing.assert_close(freeze(reference + 100.0), expected)
        freeze.assert_consumed()

        scaled = q33.ScaledResidual(base, 0.5)
        torch.testing.assert_close(scaled(reference), expected * 0.5)

    def test_quantile_points_are_deterministic_and_in_bounds(self):
        exo = np.zeros((20, 9), dtype=np.float32)
        exo[:, 2] = q33.q32.P_CRIT + 1.0
        exo[:, 6] = 0.5
        first = q33.select_quantile_points(exo, 2, 18, 4)
        second = q33.select_quantile_points(exo, 2, 18, 4)
        self.assertEqual(first, second)
        self.assertEqual(first, [2, 7, 12, 17])

    def test_development_loader_stops_before_reserved_rows(self):
        columns = list(dict.fromkeys(q33.q32.E0_COLS + q33.q32.OUTPUTS))
        frame = pd.DataFrame(
            {column: np.arange(10, dtype=np.float32) for column in columns}
        )
        config = {
            **self.config,
            "data": {
                **self.config["data"],
                "window_start": 2,
                "development_rows": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.csv"
            frame.to_csv(path, index=False)
            exo, targets, _ = q33.load_development_data(path, config)
        self.assertEqual(len(exo), 4)
        self.assertEqual(len(targets), 4)
        self.assertAlmostEqual(float(exo[0, 0]), 2.0 / 3.6, places=6)

    def test_real_checkpoint_probe_smoke(self):
        model = q33.q32.load_evap_model(ROOT / "out" / "model_e0_evap_seed0.pt")
        residual = q33.load_residual(
            ROOT
            / "out"
            / "qnav_first_principles"
            / "h_now_F0_s0"
            / "residual_best_validation.pt"
        )
        row = np.array(
            [360.0, 250.0, 24.0, 450.0, 300.0, 0.4, 0.5, 22.0, 3.0],
            dtype=np.float32,
        )
        observed = np.array([500.0, 480.0, 530.0, 510.0, 570.0], dtype=np.float32)
        config = {
            **self.config,
            "probe": {
                **self.config["probe"],
                "steps": 2,
            },
        }
        result = q33.probe_point(model, residual, row, observed, 4.0, config)
        self.assertEqual(set(result), {"valve_only", "coupled"})
        for path in result.values():
            self.assertTrue(set(config["probe"]["modes"]).issubset(path))
            self.assertTrue(np.isfinite(path["live"]["steady_main_c"]))

    def test_execute_smoke_closes_the_full_call_path(self):
        rows = 50
        values = {
            "主蒸汽流量": np.full(rows, 1296.0),
            "未校正总煤量": np.full(rows, 250.0),
            "分离器出口压力": np.full(rows, 24.0),
            "分离器出口温度": np.full(rows, 450.0),
            "省煤器出口给水温度": np.full(rows, 300.0),
            "一级减温调节门阀位": np.full(rows, 40.0),
            "二级减温调节门阀位": np.linspace(20.0, 80.0, rows),
            "末级过热器出口压力": np.full(rows, 22.0),
            "减温水总流量": np.linspace(2.0, 4.0, rows),
        }
        for index, column in enumerate(q33.q32.OUTPUTS):
            values[column] = np.full(rows, 500.0 + 15.0 * index)
        frame = pd.DataFrame(values)
        config = {
            **self.config,
            "data": {
                **self.config["data"],
                "window_start": 0,
                "development_rows": rows,
            },
            "folds": {
                "F0": {"training": [0, 25], "evaluation": [25, 35]},
                "F1": {"training": [0, 25], "evaluation": [35, 45]},
            },
            "probe": {
                **self.config["probe"],
                "points_per_fold": 1,
                "steps": 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "smoke.csv"
            frame.to_csv(csv_path, index=False)
            args = SimpleNamespace(csv=str(csv_path), output=str(directory / "results"))
            with patch.object(
                q33.q32,
                "estimate_w_coupling",
                return_value={"wet": None, "dry": 4.0},
            ):
                q33.execute(
                    args,
                    config,
                    ROOT / "configs" / "qnav_residual_feedback_probe.json",
                )
            summary = directory / "results" / "summary_development.json"
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
