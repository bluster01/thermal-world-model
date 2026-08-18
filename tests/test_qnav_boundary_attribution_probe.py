import importlib.util
import json
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


q35 = load_module(ROOT / "35_qnav_boundary_attribution_probe.py", "q35_under_test")


class Q32TTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "configs" / "qnav_boundary_attribution_probe.json"
        cls.config = q35.load_config(cls.config_path)

    def test_parent_contract_closes_exactly_sixteen_points(self):
        parent = q35.load_parent_results(self.config)
        count = sum(
            len(state["points"])
            for fold in parent["folds"].values()
            for state in fold["states"].values()
        )
        self.assertEqual(count, 16)

    def test_deadband_lpf_and_antiwindup_are_explicit(self):
        command, integral, filtered, held, in_deadband = q35.controller_command(
            measurement=-10.0,
            target=0.5,
            filtered=0.0,
            integral=0.0,
            valve0=0.1,
            power_mw=500.0,
            controller="aw_only",
            config=self.config,
        )
        self.assertTrue(held)
        self.assertEqual(integral, 0.0)
        self.assertEqual(command, 0.0)

        command, integral, filtered, held, in_deadband = q35.controller_command(
            measurement=0.49,
            target=0.5,
            filtered=0.0,
            integral=0.0,
            valve0=0.4,
            power_mw=500.0,
            controller="deadband_aw",
            config=self.config,
        )
        self.assertTrue(in_deadband)
        self.assertAlmostEqual(command, 0.4)

        _, _, filtered, _, _ = q35.controller_command(
            measurement=1.0,
            target=0.5,
            filtered=0.0,
            integral=0.0,
            valve0=0.4,
            power_mw=500.0,
            controller="lpf_aw",
            config=self.config,
        )
        self.assertAlmostEqual(filtered, 0.25)

    def test_real_checkpoint_object_controller_and_initialization_smoke(self):
        model = q35.q32.load_evap_model(ROOT / "out" / "model_e0_evap_seed0.pt")
        residual = q35.q33.load_residual(
            ROOT
            / "out"
            / "qnav_first_principles"
            / "h_now_F0_s0"
            / "residual_best_validation.pt"
        )
        exo = np.array(
            [
                [360.0, 250.0, 24.0, 450.0, 300.0, 0.4, 0.5, 22.0, 3.0],
                [360.0, 250.0, 24.0, 450.0, 300.0, 0.4, 0.5, 22.0, 3.0],
                [360.0, 250.0, 24.0, 450.0, 300.0, 0.4, 0.5, 22.0, 3.0],
            ],
            dtype=np.float32,
        )
        targets = np.array(
            [[500.0, 480.0, 530.0, 510.0, 570.0]] * 3, dtype=np.float32
        )
        config = json.loads(json.dumps(self.config))
        config["object_panel"]["steps"] = 2
        config["controller_panel"]["warm_steps"] = 1
        config["controller_panel"]["control_steps"] = 2
        config["initialization_panel"]["history_steps"] = 1
        config["initialization_panel"]["constant_warm_horizons"] = [1, 2]

        object_result = q35.object_response(
            model, residual, exo[1], targets[1], 4.0, config
        )
        self.assertEqual(set(object_result["+0.020"]), {"physical", "live", "shared"})

        initial = q35.q34.warm_state(
            model, residual, exo[1], targets[1], 1, True
        )
        controller = q35.run_controller(
            model,
            residual,
            exo[1],
            4.0,
            500.0,
            initial,
            config,
            "shared",
            "deadband_lpf_aw",
        )
        self.assertEqual(
            controller["residual_calls_shared"],
            2 * q35.q32.t02.N_SUB,
        )
        init = q35.initialization_diagnostics(
            model, residual, exo, targets, 1, config, "live"
        )
        self.assertTrue(np.isfinite(init["one_step_error_c"]))
        self.assertEqual(set(init["constant_warm_offsets_c"]), {"1", "2"})

    def test_execute_smoke_closes_all_three_panels(self):
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
        for index, column in enumerate(q35.q32.OUTPUTS):
            values[column] = np.full(rows, 500.0 + 15.0 * index)
        frame = pd.DataFrame(values)
        config = json.loads(json.dumps(self.config))
        config["data"].update({"window_start": 0, "development_rows": rows})
        config["folds"] = {
            "F0": {"training": [0, 20], "evaluation": [20, 40]},
            "F1": {"training": [0, 20], "evaluation": [40, 60]},
        }
        config["object_panel"]["steps"] = 2
        config["controller_panel"].update({"warm_steps": 1, "control_steps": 2})
        config["initialization_panel"].update(
            {"history_steps": 1, "constant_warm_horizons": [1, 2]}
        )
        parent = {
            "folds": {
                "F0": {
                    "states": {
                        "wet": {"points": [{"row": 20}], "aggregate": {}},
                        "dry": {"points": [{"row": 21}], "aggregate": {}},
                    }
                },
                "F1": {
                    "states": {
                        "wet": {"points": [{"row": 40}], "aggregate": {}},
                        "dry": {"points": [{"row": 41}], "aggregate": {}},
                    }
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            csv_path = directory / "smoke.csv"
            frame.to_csv(csv_path, index=False)
            _, _, _, data_hash = q35.q34.load_development_data(csv_path, config)
            config["parent"]["development_arrays_sha256"] = data_hash
            args = SimpleNamespace(csv=str(csv_path), output=str(directory / "results"))
            with patch.object(q35, "load_parent_results", return_value=parent), patch.object(
                q35.q32,
                "estimate_w_coupling",
                return_value={"wet": 2.0, "dry": 4.0},
            ):
                q35.execute(args, config, self.config_path)
            summary = json.loads(
                (directory / "results" / "summary_development.json").read_text()
            )
        self.assertEqual(summary["status"], "results_returned")
        self.assertIsNone(summary["scientific_verdict"])
        point = summary["folds"]["F0"]["states"]["wet"]["points"][0]
        self.assertEqual(set(point), {
            "row", "state", "pressure_mpa", "valve2_fraction", "spray_w",
            "power_mw", "object", "controllers", "initialization"
        })


if __name__ == "__main__":
    unittest.main()
