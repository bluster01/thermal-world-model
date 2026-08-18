import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("direct_wm_v2", ROOT / "36_direct_wm.py")
WM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WM)


def test_folds_are_disjoint_and_match_q32():
    assert WM.FOLDS["F0"] == {"train": (0, 20000), "val": (20000, 25000), "eval": (25000, 30000)}
    assert WM.FOLDS["F1"] == {"train": (0, 30000), "val": (30000, 35000), "eval": (35000, 40000)}
    for fold in WM.FOLDS.values():
        assert fold["train"][1] <= fold["val"][0] <= fold["val"][1] <= fold["eval"][0]


def test_beta_nll_preserves_batch_and_horizon_axes():
    loss = WM.BetaNLLLoss(beta=-0.3)
    assert loss(torch.zeros(3, WM.H), torch.zeros(3, WM.H), torch.ones(3, WM.H)).shape == (3, WM.H)


def test_fixed_indices_are_seed_independent_and_cover_block():
    a = WM.fixed_indices(WM.FOLDS["F0"]["eval"])
    b = WM.fixed_indices(WM.FOLDS["F0"]["eval"])
    assert a.tolist() == b.tolist() and len(a) == WM.EVAL_N
    assert a[0] == WM.FOLDS["F0"]["eval"][0]
    assert a[-1] < WM.FOLDS["F0"]["eval"][1] - WM.W - WM.H
    assert WM.indices_sha256(a) == WM.indices_sha256(b)


def test_action_intervention_is_persistent_and_reports_clipping():
    action = torch.full((2, WM.H, 2), 99.0)
    shifted, actual = WM.apply_persistent_shift(action, 1, 2.0)
    assert torch.all(shifted[:, :, 0] == 99.0)
    assert torch.all(shifted[:, :, 1] == 100.0)
    assert actual == 1.0


def test_frozen_primary_is_h18_and_valve_only():
    assert WM.ANCHORS[-1] == 17
    assert WM.SENSITIVITY_DELTAS == (-2.0, 2.0)
