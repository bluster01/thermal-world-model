import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


q34 = load_module(ROOT / "34_qnav_shared_disturbance_loop.py", "q34_under_test")


class Q32STests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "configs" / "qnav_shared_disturbance_loop.json"
        cls.config = q34.load_config(cls.config_path)

    def test_parent_verification_uses_canonical_git_bytes(self):
        q34.verify_parent(self.config)

    def test_loader_includes_power_and_stops_at_development_rows(self):
        columns = list(
            dict.fromkeys(q34.q32.E0_COLS + q34.q32.OUTPUTS + [q34.q32.POWER_COLUMN])
        )
        frame = pd.DataFrame(
            {column: np.arange(10, dtype=np.float32) + 1 for column in columns}
        )
        config = {
            **self.config,
            "data": {**self.config["data"], "window_start": 2, "development_rows": 4},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.csv"
            frame.to_csv(path, index=False)
            exo, targets, power, _ = q34.load_development_data(path, config)
        self.assertEqual(len(exo), 4)
        self.assertEqual(len(targets), 4)
        self.assertEqual(len(power), 4)
        self.assertAlmostEqual(float(power[0]), 3.0)

    def test_real_checkpoint_three_mode_smoke(self):
        model = q34.q32.load_evap_model(ROOT / "out" / "model_e0_evap_seed0.pt")
        residual = q34.q33.load_residual(
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
                "warm_steps": 2,
                "control_steps": 2,
            },
        }
        results = {
            mode: q34.run_deviation_loop(
                model, residual, row, observed, 4.0, 500.0, config, mode
            )
            for mode in config["probe"]["modes"]
        }
        self.assertEqual(set(results), {"physical", "live", "shared"})
        self.assertTrue(np.isfinite(results["shared"]["tracking_error_c"]))
        self.assertEqual(
            results["shared"]["residual_calls_shared"],
            2 * q34.q32.t02.N_SUB,
        )

    def test_execute_smoke_closes_wet_and_dry_call_paths(self):
        rows = 60
        pressure = np.tile([21.0, 24.0], rows // 2)
        values = {
            "主蒸汽流量": np.full(rows, 1296.0),
            "未校正总煤量": np.full(rows, 250.0),
            "分离器出口压力": pressure,
            "分离器出口温度": np.full(rows, 450.0),
            "省煤器出口给水温度": np.full(rows, 300.0),
            "一级减温调节门阀位": np.full(rows, 40.0),
            "二级减温调节门阀位": np.linspace(20.0, 80.0, rows),
            "末级过热器出口压力": np.full(rows, 20.0),
            "减温水总流量": np.linspace(2.0, 4.0, rows),
            "机组负荷": np.full(rows, 500.0),
        }
        for index, column in enumerate(q34.q32.OUTPUTS):
            values[column] = np.full(rows, 500.0 + 15.0 * index)
        frame = pd.DataFrame(values)
        config = {
            **self.config,
            "data": {**self.config["data"], "window_start": 0, "development_rows": rows},
            "folds": {
                "F0": {"training": [0, 20], "evaluation": [20, 40]},
                "F1": {"training": [0, 20], "evaluation": [40, 60]},
            },
            "probe": {
                **self.config["probe"],
                "points_per_state_per_fold": 1,
                "warm_steps": 1,
                "control_steps": 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "smoke.csv"
            frame.to_csv(csv_path, index=False)
            args = SimpleNamespace(csv=str(csv_path), output=str(directory / "results"))
            with patch.object(
                q34.q32,
                "estimate_w_coupling",
                return_value={"wet": 2.0, "dry": 4.0},
            ):
                q34.execute(args, config, self.config_path)
            summary = json_load(directory / "results" / "summary_development.json")
        self.assertEqual(set(summary["folds"]["F0"]["states"]), {"wet", "dry"})
        self.assertIsNone(summary["scientific_verdict"])


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
