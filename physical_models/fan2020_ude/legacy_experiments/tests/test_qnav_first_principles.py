import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


q32 = load_module(ROOT / "32_qnav_first_principles.py", "q32_under_test")


class Q32ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix_path = ROOT / "configs" / "qnav_first_principles_matrix.json"
        cls.matrix = q32.load_matrix(cls.matrix_path)

    def test_frozen_matrix_expands_to_ten_units(self):
        units = q32.expand_units(self.matrix)
        self.assertEqual(len(units), 10)
        self.assertEqual({unit["fold"] for unit in units}, {"F0", "F1"})
        self.assertTrue(all(unit["seed"] == 0 for unit in units))

    def test_energy_injection_switches_are_explicit(self):
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        zero = torch.zeros_like(z)
        expected = {
            "none": (zero, zero),
            "double": (z, z),
            "h_only": (zero, z),
            "conservative": (-z, z),
        }
        for mode, pair in expected.items():
            actual = q32.residual_fluxes(mode, z)
            torch.testing.assert_close(actual[0], pair[0])
            torch.testing.assert_close(actual[1], pair[1])

    def test_no_w_residual_is_invariant_to_measured_spray_flow(self):
        batch = 3
        ts = torch.full((3, batch), 500.0)
        metal = torch.full((3, batch), 700.0)
        scalar = torch.full((batch,), 1.0)
        w0 = torch.zeros(batch)
        w1 = torch.full((batch,), 9.0)
        no_w_0 = q32.residual_features(
            ts, metal, scalar, scalar, scalar, scalar, scalar, scalar, w0, False
        )
        no_w_1 = q32.residual_features(
            ts, metal, scalar, scalar, scalar, scalar, scalar, scalar, w1, False
        )
        with_w_0 = q32.residual_features(
            ts, metal, scalar, scalar, scalar, scalar, scalar, scalar, w0, True
        )
        with_w_1 = q32.residual_features(
            ts, metal, scalar, scalar, scalar, scalar, scalar, scalar, w1, True
        )
        self.assertEqual(no_w_0.shape[1], 10)
        self.assertEqual(with_w_0.shape[1], 11)
        torch.testing.assert_close(no_w_0, no_w_1)
        self.assertFalse(torch.equal(with_w_0, with_w_1))

    def test_current_double_w_path_matches_original_qnav_integrator(self):
        original = load_module(ROOT / "27_fix_evap_residual.py", "q27_reference")
        model = q32.load_evap_model(ROOT / "out" / "model_e0_evap_seed0.pt")
        residual = q32.r09.ResMLP(11, q32.r09.Q_SCALE).to(q32.DEVICE)
        residual.load_state_dict(
            torch.load(
                ROOT / "out" / "model_res_qnav_seed0.pt",
                map_location=q32.DEVICE,
                weights_only=True,
            )
        )
        residual.eval()
        row = torch.tensor(
            [[360.0, 250.0, 20.0, 450.0, 300.0, 0.4, 0.5, 18.0, 3.0]],
            device=q32.DEVICE,
        )
        observed = torch.tensor(
            [[500.0, 480.0, 530.0, 510.0, 570.0]], device=q32.DEVICE
        )
        exo = row[:, None, :].repeat(1, 2, 1)
        state_a = q32.r26.init_states_evap(model, row, observed)
        state_b = tuple(value.clone() for value in state_a)
        with torch.no_grad():
            reference = original.integrate_evap_res(
                model, residual, exo, *state_a, 2
            )
            candidate = q32.integrate(
                model, residual, exo, *state_b, 2, "double", True
            )
        for actual, expected in zip(candidate, reference):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_dry_run_is_closed_and_emits_no_verdict(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "32_qnav_first_principles.py"), "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["unit_count"], 10)
        self.assertIsNone(payload["scientific_verdict"])


if __name__ == "__main__":
    unittest.main()
