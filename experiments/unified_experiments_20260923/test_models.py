"""Synthetic software checks; no industrial data are read or fitted."""
import unittest

import torch

from experiments.unified_experiments_20260923.models import (
    ARMS, build_model, implicit_star_step,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(723)
        self.inputs = ["a_in", "a_out", "b_in", "b_out", "load", "reagent_a"]
        self.targets = self.inputs[:4]
        self.batch = {
            "history_normalized": torch.randn(2, 360, 6) * 0.1,
            "history_valid": torch.ones(2, 360, 6, dtype=torch.bool),
            "history_observed": torch.ones(2, 360, 6, dtype=torch.bool),
            "history_age_s": torch.zeros(2, 360, 6),
        }

    def model(self, arm, task="scr"):
        return build_model(arm, self.inputs, self.targets, task, hidden=16,
                           target_center=[100, 20, 110, 22], target_scale=[10, 4, 12, 5])

    def test_all_arms_shapes_finite_and_gradients(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm)
                y = model(self.batch, 60)
                self.assertEqual(y.shape, (2, 60, 4))
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_no_targets_or_future_fields_consumed(self):
        poisoned = dict(self.batch, targets=torch.full((2, 180, 4), float("nan")),
                        targets_normalized=torch.full((2, 180, 4), 1e30),
                        target_mask=torch.zeros(2, 180, 4),
                        recorded_future_actions=torch.full((2, 180, 6), float("nan")),
                        recorded_future_boundaries=torch.full((2, 180, 6), float("inf")))
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm).eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch, 60), model(poisoned, 60), rtol=0, atol=0)

    def test_horizon_prefix_consistency(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm).eval()
                with torch.no_grad():
                    short, long = model(self.batch, 60), model(self.batch, 180)
                torch.testing.assert_close(short, long[:, :60], rtol=1e-5, atol=1e-6)

    def test_anchor_uses_last_valid_and_missing_target_uses_zero(self):
        self.batch["history_valid"][:, -20:, 0] = False
        self.batch["history_normalized"][:, -21, 0] = 3.5
        self.batch["history_normalized"][:, -20:, 0] = 1e8
        self.batch["history_valid"][:, :, 1] = False
        for arm in ARMS:
            _, anchor = self.model(arm).history(self.batch)
            torch.testing.assert_close(anchor[:, 0], torch.full((2,), 3.5))
            torch.testing.assert_close(anchor[:, 1], torch.zeros(2))

    def test_invalid_values_and_infinite_ages_are_safe(self):
        self.batch["history_valid"][:, :40] = False
        self.batch["history_normalized"][:, :40] = float("nan")
        self.batch["history_age_s"][:, :40] = float("inf")
        for arm in ARMS:
            with self.subTest(arm=arm):
                self.assertTrue(torch.isfinite(self.model(arm)(self.batch, 60)).all())

    def test_scr_concentration_positive_and_no_implicit_reagent_routing(self):
        for arm in ("mechanism", "mechanism_no_slow"):
            model = self.model(arm)
            raw = model(self.batch, 180) * model.target_scale + model.target_center
            self.assertGreaterEqual(raw.min().item(), 0)
            self.assertEqual(model.description()["reagent_group_to_side_assignment"], "none")
            self.assertEqual(model.scr_incoming.sum().item(), 2)

    def test_slow_ablation_preserves_ports_and_observer(self):
        full, ablated = self.model("mechanism"), self.model("mechanism_no_slow")
        self.assertEqual(full.slow_count, 3)
        self.assertEqual(ablated.slow_count, 0)
        self.assertEqual(full.ports.weight.shape, ablated.ports.weight.shape)
        self.assertEqual(full.observer.gru.weight_ih_l0.shape, ablated.observer.gru.weight_ih_l0.shape)

    def test_thermal_main_and_reheat_never_impose_tag_order_chain(self):
        for task in ("main_steam", "reheat"):
            for arm in ("mechanism", "mechanism_no_slow"):
                model = self.model(arm, task)
                prediction = model(self.batch, 60)
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertEqual(model.description()["process_topology"], "independent_target_local_storage_stars")
                self.assertEqual(model.scr_incoming.sum().item(), 0)

    def test_exchange_storage_balance_and_dissipation(self):
        dtype = torch.float64
        fast = torch.tensor([[3.0, 2.0]], dtype=dtype)
        slow = torch.tensor([[[1.0, 2.0, 0.5], [3.0, 0.5, 1.0]]], dtype=dtype)
        capacity = torch.tensor([[0.5, 2., 1.], [1., 0.3, 2.]], dtype=dtype)
        k = torch.tensor([[0.03, 0.01, 0.02], [0.01, 0.02, 0.03]], dtype=dtype)
        zero = torch.zeros_like(fast)
        fn, sn = implicit_star_step(fast, slow, capacity, k, zero, zero, 10.)
        initial_inventory = fast + (capacity * slow).sum(-1)
        final_inventory = fn + (capacity * sn).sum(-1)
        torch.testing.assert_close(initial_inventory, final_inventory, rtol=0, atol=1e-14)
        before = .5 * (fast.square() + (capacity * slow.square()).sum(-1))
        after = .5 * (fn.square() + (capacity * sn.square()).sum(-1))
        self.assertTrue((after <= before).all())

    def test_declared_port_balance_and_positive_update(self):
        dtype = torch.float64
        fast = torch.rand(2, 4, dtype=dtype)
        slow = torch.rand(2, 4, 3, dtype=dtype)
        capacity, k = torch.ones(4, 3, dtype=dtype), torch.rand(2, 4, 3, dtype=dtype)
        source, sink = torch.rand(2, 4, dtype=dtype), torch.rand(2, 4, dtype=dtype)
        fn, sn = implicit_star_step(fast, slow, capacity, k, source, sink, 10.)
        inventory_delta = fn + (capacity * sn).sum(-1) - fast - (capacity * slow).sum(-1)
        torch.testing.assert_close(inventory_delta, 10 * (source - sink * fn), rtol=0, atol=1e-14)
        self.assertTrue((fn >= 0).all() and (sn >= 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
