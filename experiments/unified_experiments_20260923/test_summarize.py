"""Synthetic result-JSON tests. No industrial data, model loading, or training."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.unified_experiments_20260923.summarize import summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.protocol = {"experiment_id": "synthetic_matrix", "tasks": ["scr", "main_steam"],
                         "arms": ["gru_direct", "mechanism", "mechanism_no_slow"],
                         "seeds": [11, 22, 33], "fit_count": 18,
                         "primary_profile": "h60m_f10m", "long_profile": "h60m_f30m",
                         "training_horizon": 60, "long_horizon": 180}
        levels = {"gru_direct": [3., 4., 5.], "mechanism": [1., 2., 4.], "mechanism_no_slow": [2., 2., 6.]}
        for task in self.protocol["tasks"]:
            scale = 100. if task == "scr" else 1.
            for arm in self.protocol["arms"]:
                for position, seed in enumerate(self.protocol["seeds"]):
                    path = self.out / "runs" / f"{task}__{arm}__s{seed}"
                    path.mkdir(parents=True)
                    self.write(path / "status.json", {"status": "BUDGET_EXHAUSTED" if seed == 33 else "EARLY_STOPPED",
                                                      "epoch": 80 if seed == 33 else 27,
                                                      "identity": {"task": task, "arm": arm, "seed": seed,
                                                                   "model_parameters": 10}})
                    (path / "ledger.jsonl").write_text(
                        json.dumps({"epoch": 1, "improved": True}) + "\n" +
                        json.dumps({"epoch": 4, "improved": True}) + "\n" +
                        json.dumps({"epoch": 5, "improved": False}) + "\n", encoding="utf-8")
                    value = levels[arm][position]
                    targets = {name: {"count": 100, "MAE": value * scale,
                                      "RMSE": (value + 1) * scale, "persistence_MAE": 9 * scale,
                                      "increment_count": 50 if name == "first" else 0,
                                      "increment_MAE": value * scale / 2 if name == "first" else None,
                                      "endpoints": {"6": {"count": 5, "MAE": value * scale, "RMSE": (value + 1) * scale},
                                                    "60": {"count": 0, "MAE": None, "RMSE": None}}}
                               for name in ("first", "second")}
                    for split in ("validation", "historical_test"):
                        for profile, horizon in (("h60m_f10m", 60), ("h60m_f30m", 180)):
                            self.write(path / f"evaluation_{split}_{profile}.json",
                                       {"origins": 10, "horizon_steps": horizon,
                                        "target_macro_normalized_mse": value, "targets": targets})

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    def evaluation_path(self):
        return self.out / "runs" / "scr__mechanism__s11" / "evaluation_validation_h60m_f10m.json"

    def test_full_matrix_sample_std_paired_difference_and_units(self):
        result = summarize(self.out, self.protocol)
        self.assertEqual(result["fits_completed"], 18)
        self.assertEqual(result["epoch_cap_count"], 6)
        self.assertEqual(len(result["sections"]), 8)
        self.assertEqual(len(result["source_files"]), 72)
        scr = next(s for s in result["sections"] if (s["task"], s["split"], s["profile"]) == ("scr", "validation", "h60m_f10m"))
        thermal = next(s for s in result["sections"] if (s["task"], s["split"], s["profile"]) == ("main_steam", "validation", "h60m_f10m"))
        statistic = scr["arms"]["mechanism"]["target_macro_normalized_mse"]
        self.assertAlmostEqual(statistic["mean"], 7 / 3)
        self.assertAlmostEqual(statistic["sample_std"], math.sqrt(7 / 3))
        self.assertEqual(statistic["per_seed"], {"11": 1., "22": 2., "33": 4.})
        paired = scr["paired_mechanism_minus_no_slow"]["target_macro_normalized_mse"]
        self.assertEqual(paired["per_seed"], {"11": -1., "22": 0., "33": -2.})
        self.assertEqual(paired["mean"], -1.)
        self.assertEqual(paired["sample_std"], 1.)
        self.assertAlmostEqual(scr["arms"]["mechanism"]["targets"]["first"]["MAE"]["mean"], 700 / 3)
        self.assertAlmostEqual(thermal["arms"]["mechanism"]["targets"]["first"]["MAE"]["mean"], 7 / 3)
        self.assertNotIn("overall_physical_MAE", result)
        self.assertEqual(result["fits"][0]["best_epoch"], 4)
        self.assertEqual(json.loads((self.out / "summary.json").read_text(encoding="utf-8")), result)
        markdown = (self.out / "SUMMARY_ZH.md").read_text(encoding="utf-8")
        self.assertIn("BUDGET_EXHAUSTED", markdown)
        self.assertIn("原记录单位", markdown)
        self.assertIn("°C", markdown)

    def test_undefined_increment_and_endpoint_stay_undefined(self):
        result = summarize(self.out, self.protocol)
        group = result["sections"][0]
        target = group["arms"]["mechanism"]["targets"]["second"]
        self.assertEqual(target["increment_MAE"]["n"], 0)
        self.assertIsNone(target["increment_MAE"]["mean"])
        self.assertIsNone(target["increment_MAE"]["sample_std"])
        self.assertFalse(target["increment_MAE"]["complete"])
        self.assertIsNone(target["endpoints"]["60"]["MAE"]["mean"])
        self.assertIsNone(group["paired_mechanism_minus_no_slow"]["targets"]["second"]["increment_MAE"]["mean"])

    def test_missing_evaluation_refuses_partial_summary(self):
        self.evaluation_path().unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(self.out, self.protocol)
        self.assertFalse((self.out / "summary.json").exists())
        self.assertFalse((self.out / "SUMMARY_ZH.md").exists())

    def test_mask_support_mismatch_is_rejected(self):
        file = self.evaluation_path()
        report = json.loads(file.read_text())
        report["targets"]["first"]["count"] -= 1
        self.write(file, report)
        with self.assertRaisesRegex(ValueError, "mask counts mismatch"):
            summarize(self.out, self.protocol)
        self.assertFalse((self.out / "summary.json").exists())

    def test_persistence_mismatch_is_rejected(self):
        file = self.evaluation_path()
        report = json.loads(file.read_text())
        report["targets"]["first"]["persistence_MAE"] += 1
        self.write(file, report)
        with self.assertRaisesRegex(ValueError, "Persistence baseline mismatch"):
            summarize(self.out, self.protocol)

    def test_nonfinite_and_duplicate_seeds_are_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["seeds"] = [11, 11, 33]
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            summarize(self.out, protocol)
        file = self.evaluation_path()
        report = json.loads(file.read_text())
        report["target_macro_normalized_mse"] = float("nan")
        self.write(file, report)
        with self.assertRaisesRegex(ValueError, "finite metric"):
            summarize(self.out, self.protocol)

    def test_running_fit_is_not_treated_as_complete(self):
        status_path = self.out / "runs" / "main_steam__mechanism_no_slow__s33" / "status.json"
        status = json.loads(status_path.read_text())
        status["status"] = "RUNNING"
        self.write(status_path, status)
        with self.assertRaisesRegex(ValueError, "Incomplete fit"):
            summarize(self.out, self.protocol)

    def test_horizon_must_match_protocol_profile(self):
        file = self.evaluation_path()
        report = json.loads(file.read_text())
        report["horizon_steps"] = 180
        self.write(file, report)
        with self.assertRaisesRegex(ValueError, "horizon disagrees"):
            summarize(self.out, self.protocol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
