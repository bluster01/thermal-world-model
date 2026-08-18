"""End-to-end smoke of the matrix runner (quick mode, no verdicts).

Quick mode exists for local smoke and Linux dry-run only; the frozen
verdicts are computed exclusively in full mode.
"""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np

from experiments.final_wm import matrix_spec as ms
from experiments.final_wm.run_matrix import run_dsyn, run_matrix
from src.final_wm.synthetic import synthetic_canonical_arrays


def _args(tmp_path, **kw) -> Namespace:
    base = dict(
        data_root=None, mapping=None, record=None, out=str(tmp_path / "out"),
        units=None, properties_npz=None, device="cpu", quick=True,
    )
    base.update(kw)
    return Namespace(**base)


def test_dsyn_quick_gate_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    args = _args(tmp_path)
    verdict = run_dsyn(args)
    assert (tmp_path / "out" / "dsyn_verdict.json").exists()
    assert verdict["quick"] is True
    for entry in verdict["per_seed"]:
        assert np.isfinite(entry["student_val_nll"])


def test_matrix_quick_o1_and_b1_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    arrays = synthetic_canonical_arrays(total_steps=1500, seed=5)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **arrays)
    args = _args(tmp_path, record=str(record_path), units="o1,b1")
    summary = run_matrix(args)
    out = tmp_path / "out"
    assert (out / "matrix_summary.json").exists()
    ledger_lines = (out / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # 3 O1 arms + 1 B1 arm, quick mode = 1 seed
    run_ids = {json.loads(l)["run_id"] for l in ledger_lines}
    assert "o1_steady_seed0" in run_ids
    assert "o1_learned_seed0" in run_ids
    assert "o1_hybrid_seed0" in run_ids
    assert "b1_gru_seed0" in run_ids
    assert summary["units"] == {} or summary["quick"] is True  # no verdicts in quick mode
