"""Synthetic runner regression checks; no industrial files or fits are used."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.unified_experiments_20260923 import run


class TinyEvaluationDataset(Dataset):
    input_names = ["context", "b", "a"]
    target_names = ["a", "b"]
    target_indices = np.array([0, 1])
    scale = np.array([2., 10.])
    profile = {"forecast_steps": 3}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "origin_time_ns": np.int64(100),
            "history_normalized": np.array([[0., 1., 2.], [0., 999., 2.]], dtype=np.float32),
            "history_valid": np.array([[True, True, True], [True, False, True]]),
            "history_observed": np.array([[True, True, True], [True, False, True]]),
            "history_age_s": np.zeros((2, 3), dtype=np.float32),
            "targets_normalized": np.array([[1., 2.], [2., 999.], [4., 3.]], dtype=np.float32),
            "target_mask": np.array([[True, True], [True, False], [True, True]]),
        }


class FixedPrediction(nn.Module):
    def forward(self, batch, horizon):
        if horizon != 3:
            raise AssertionError("Unexpected synthetic horizon")
        return batch["history_normalized"].new_tensor([[[1., 0.], [0., 12345.], [6., 4.]]])


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_macro_mse_weights_targets_and_ignores_missing_values(self):
        prediction = torch.tensor([[[1., 4.], [3., 1e8]]], requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[True, True], [True, False]]])
        loss = run.macro_mse(prediction, target, mask)
        self.assertAlmostEqual(loss.item(), 10.5)  # ((1+9)/2 + 16/1)/2
        loss.backward()
        self.assertEqual(prediction.grad[0, 1, 1].item(), 0.)
        torch.testing.assert_close(prediction.grad[0, :, 0], torch.tensor([.5, 1.5]))

    def test_macro_mse_absent_target_and_all_missing_batch(self):
        prediction = torch.tensor([[[2., 100.]]], requires_grad=True)
        mask = torch.tensor([[[True, False]]])
        self.assertEqual(run.macro_mse(prediction, torch.zeros_like(prediction), mask).item(), 4.)
        loss = run.macro_mse(prediction, torch.zeros_like(prediction), torch.zeros_like(mask))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_increment_loss_requires_both_endpoint_labels(self):
        prediction = torch.tensor([[[0.], [1000.], [4.], [7.]]], requires_grad=True)
        target = torch.tensor([[[0.], [-999.], [2.], [3.]]])
        mask = torch.tensor([[[True], [False], [True], [True]]])
        batch = {"targets_normalized": target, "target_mask": mask}
        spec = {"increment_lags": [1, 2], "increment_weight": .2}
        # Level = (0 + 4 + 16)/3. lag1 only t2->t3 contributes 4.
        # lag2 only t0->t2 contributes 4. Masked t1 is never a delta endpoint.
        self.assertAlmostEqual(run.objective(prediction, batch, spec).item(), 20 / 3 + .8, places=5)
        run.objective(prediction, batch, spec).backward()
        self.assertEqual(prediction.grad[0, 1, 0].item(), 0.)

    def test_anchor_respects_target_order_and_last_valid_measurement(self):
        batch = {
            "history_normalized": torch.tensor([[[7., 3., 5.], [8., 99., 6.], [9., 100., 7.]]]),
            "history_valid": torch.tensor([[[False, True, True], [False, False, True], [False, False, True]]]),
        }
        torch.testing.assert_close(run.anchor(batch, [2, 1, 0]), torch.tensor([[7., 3., 0.]]))

    def test_evaluation_raw_units_masks_persistence_and_endpoint_counts(self):
        protocol = {"validation": {"batch_size": 2, "endpoints": [1, 2, 3, 6]}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metrics"
            report = run.evaluate(FixedPrediction(), TinyEvaluationDataset(), protocol,
                                  torch.device("cpu"), workers=0, output=output)
            self.assertAlmostEqual(report["target_macro_normalized_mse"], 31 / 12)
            a, b = report["targets"]["a"], report["targets"]["b"]
            self.assertEqual(a["count"], 3)
            self.assertEqual(b["count"], 2)
            self.assertAlmostEqual(a["MAE"], 8 / 3)
            self.assertAlmostEqual(b["MAE"], 15.)
            self.assertAlmostEqual(a["persistence_MAE"], 2.)
            self.assertAlmostEqual(b["persistence_MAE"], 15.)
            self.assertAlmostEqual(a["increment_MAE"], 6.)
            self.assertIsNone(b["increment_MAE"])
            self.assertEqual(b["increment_count"], 0)
            self.assertEqual(b["endpoints"]["2"]["count"], 0)
            self.assertIsNone(b["endpoints"]["2"]["MAE"])
            self.assertNotIn("6", a["endpoints"])
            self.assertTrue(output.with_suffix(".npz").exists())
            self.assertEqual(json.loads(output.with_suffix(".json").read_text()), report)

    def test_checkpoint_preserves_state_and_rng_without_training(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        identity = {"task": "synthetic", "seed": 723}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            with mock.patch.object(torch.cuda, "get_rng_state_all", return_value=[]):
                run.save_checkpoint(path, model, optimizer, scheduler, 9, .125, 3, identity)
            expected_python, expected_numpy, expected_torch = random.random(), np.random.rand(), torch.rand(3)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual((saved["epoch"], saved["best"], saved["wait"]), (9, .125, 3))
            self.assertEqual(saved["identity"], identity)
            self.assertEqual(saved["optimizer"]["param_groups"][0]["lr"], .001)
            self.assertEqual(saved["scheduler"], scheduler.state_dict())
            random.setstate(saved["rng"]["python"])
            np.random.set_state(saved["rng"]["numpy"])
            torch.set_rng_state(saved["rng"]["torch"])
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.rand(), expected_numpy)
            torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)

    def test_historical_test_gate_checks_entire_fit_matrix_first(self):
        protocol = {"seeds": [11, 22, 33], "tasks": ["a", "b"], "arms": ["m1", "m2"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in protocol["seeds"]:
                for task in protocol["tasks"]:
                    for arm in protocol["arms"]:
                        path = root / "runs" / f"{task}__{arm}__s{seed}"
                        path.mkdir(parents=True)
                        state = "RUNNING" if (seed, task, arm) == (33, "b", "m2") else "EARLY_STOPPED"
                        run.write_json(path / "status.json", {"status": state})
            with mock.patch.object(run, "IndustrialDataset") as dataset:
                with self.assertRaisesRegex(RuntimeError, "complete frozen fit matrix"):
                    run.final_evaluation(Path("unused"), root, protocol, torch.device("cpu"), 0)
                dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
