"""End-to-end smoke of the matrix runner (quick mode, no verdicts).

Quick mode exists for local smoke and Linux dry-run only; the frozen
verdicts are computed exclusively in full mode.
"""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import torch

from experiments.final_wm import matrix_spec as ms
from experiments.final_wm.run_matrix import (
    _adjudicate,
    closure_blindness_check,
    run_dsyn,
    run_matrix,
)
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.training import build_world_model


def _args(tmp_path, **kw) -> Namespace:
    base = dict(
        data_root=None, mapping=None, record=None, side=None, out=str(tmp_path / "out"),
        units=None, properties_npz=None, device="cpu", quick=True,
    )
    base.update(kw)
    return Namespace(**base)


def _complete_evidence(unit: str) -> dict[str, bool]:
    return {name: True for name in ms.REQUIRED_EVIDENCE[unit]}


def test_verdict_is_fail_closed_when_protocol_evidence_is_missing() -> None:
    evidence = _complete_evidence("o1")
    evidence.pop("state_continuity")
    result = _adjudicate(
        "o1", "SUPPORTED", evidence,
        quick=False, seeds=ms.SEEDS, arm_filter=None,
    )
    assert result["verdict"] == "INCOMPLETE"
    assert result["missing_evidence"] == ["state_continuity"]
    assert result["required_evidence"] == list(ms.REQUIRED_EVIDENCE["o1"])


def test_verdict_requires_full_unfiltered_execution() -> None:
    evidence = _complete_evidence("b1")
    full = _adjudicate(
        "b1", "SUPPORTED", evidence,
        quick=False, seeds=ms.SEEDS, arm_filter=None,
    )
    quick = _adjudicate(
        "b1", "SUPPORTED", evidence,
        quick=True, seeds=(0,), arm_filter=None,
    )
    partial = _adjudicate(
        "b1", "SUPPORTED", evidence,
        quick=False, seeds=(0, 1), arm_filter=None,
    )
    filtered = _adjudicate(
        "b1", "SUPPORTED", evidence,
        quick=False, seeds=ms.SEEDS, arm_filter="gru",
    )
    assert full["verdict"] == "SUPPORTED"
    assert quick["verdict"] == "SMOKE"
    assert partial["verdict"] == "INCOMPLETE"
    assert filtered["verdict"] == "INCOMPLETE"


def test_matrix_version_and_required_evidence_are_v07() -> None:
    assert ms.MATRIX_VERSION == "0.7"
    assert set(ms.REQUIRED_EVIDENCE) == {"o1", "t1", "b1", "j1", "r1"}
    assert "state_continuity" in ms.REQUIRED_EVIDENCE["o1"]
    assert "boundary_h36" in ms.REQUIRED_EVIDENCE["b1"]
    assert "constant_h60_stability" in ms.REQUIRED_EVIDENCE["t1"]
    assert "h36_stability" in ms.REQUIRED_EVIDENCE["j1"]
    assert {"valve1_h18", "valve1_h60", "valve2_h18", "valve2_h60"} <= set(
        ms.REQUIRED_EVIDENCE["r1"]
    )


def test_dsyn_quick_gate_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    args = _args(tmp_path)
    verdict = run_dsyn(args)
    # Quick tier writes a *_quick.json sibling (rerun failure report §6).
    assert (tmp_path / "out" / "dsyn_verdict_quick.json").exists()
    assert not (tmp_path / "out" / "dsyn_verdict.json").exists()
    assert verdict["quick"] is True
    for entry in verdict["per_seed"]:
        assert np.isfinite(entry["student_val_nll"])


def _ledger_final_count(out) -> int:
    ledger = out / "ledger.jsonl"
    return sum(1 for line in ledger.read_text(encoding="utf-8").splitlines()
               if json.loads(line).get("final"))


def test_matrix_rerun_resumes_without_retraining(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **synthetic_canonical_arrays(total_steps=1200, seed=3))
    args = _args(tmp_path, record=str(record_path), units="o1")
    run_matrix(args)
    n_final = _ledger_final_count(tmp_path / "out")
    assert n_final == 3  # steady/learned/hybrid, seed 0 in quick mode
    summary = run_matrix(args)  # second run: resume everything, no retraining
    assert _ledger_final_count(tmp_path / "out") == n_final
    assert summary["matrix_version"] == ms.MATRIX_VERSION
    assert (tmp_path / "out" / "matrix_summary_quick.json").exists()
    assert not (tmp_path / "out" / "matrix_summary.json").exists()


def test_matrix_rerun_retrains_when_spec_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **synthetic_canonical_arrays(total_steps=1200, seed=3))
    args = _args(tmp_path, record=str(record_path), units="o1")
    run_matrix(args)
    n_final = _ledger_final_count(tmp_path / "out")
    monkeypatch.setattr(ms, "HORIZON", 12)  # spec change -> fingerprint mismatch -> retrain
    run_matrix(args)
    assert _ledger_final_count(tmp_path / "out") == n_final + 3


def test_matrix_quick_t1_and_r1_run(tmp_path, monkeypatch) -> None:
    """End-to-end coverage of the R1 unit path (the unit that crashed the
    first Linux run): trains the four T1 arms quick, then runs the R1 probes
    against the closure_cons checkpoints."""
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **synthetic_canonical_arrays(total_steps=1500, seed=9))
    args = _args(tmp_path, record=str(record_path), units="t1,r1")
    summary = run_matrix(args)
    out = tmp_path / "out"
    assert (out / "r1_report.json").exists()
    reports = summary["units"]["r1"]["reports"]
    assert len(reports) == 1 and "error" not in reports[0]
    assert reports[0]["runtime_blind_ok"] is True
    assert "leakage" in reports[0] and "direction" in reports[0]


def test_closure_blindness_check_passes(tmp_path) -> None:
    spec = ms._base("t1", "closure_cons", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative")
    model = build_world_model(spec, properties=None)
    report = closure_blindness_check(model, torch.device("cpu"))
    assert report["runtime_blind_ok"] is True


def test_legacy_metrics_blob_never_resumes(tmp_path, monkeypatch) -> None:
    """2026-08-22 audit regression: the legacy flat metrics format carries no
    code fingerprint; resuming from it re-emitted pre-repair verdicts after
    the batch-1 observer change (spec fields unchanged).  Legacy blobs must
    always retrain."""
    from experiments.final_wm.run_matrix import _try_resume

    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **synthetic_canonical_arrays(total_steps=1200, seed=3))
    args = _args(tmp_path, record=str(record_path), units="o1")
    run_matrix(args)
    n_final = _ledger_final_count(tmp_path / "out")
    # Downgrade one metrics blob to the legacy flat format (drop the
    # fingerprint/final wrapper); the identical rerun must RETRAIN that arm.
    mpath = tmp_path / "out" / "metrics" / "o1_steady_seed0.pt"
    blob = torch.load(mpath, map_location="cpu", weights_only=False)
    torch.save(blob["metrics"], mpath)
    run_matrix(args)
    assert _ledger_final_count(tmp_path / "out") == n_final + 1


def test_summary_merges_across_invocations(tmp_path, monkeypatch) -> None:
    """2026-08-22 audit regression: separate invocations (`--units t1,r1`
    then `--units o1`) must not clobber each other's summary blocks."""
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **synthetic_canonical_arrays(total_steps=1200, seed=3))
    run_matrix(_args(tmp_path, record=str(record_path), units="o1"))
    summary_path = tmp_path / "out" / "matrix_summary_quick.json"
    first = json.loads(summary_path.read_text(encoding="utf-8"))
    # Forge a foreign unit block, then rerun: the block must survive.
    first.setdefault("units", {})["t1"] = {"marker": "from-another-invocation"}
    summary_path.write_text(json.dumps(first), encoding="utf-8")
    run_matrix(_args(tmp_path, record=str(record_path), units="o1"))
    merged = json.loads(summary_path.read_text(encoding="utf-8"))
    assert merged["units"]["t1"] == {"marker": "from-another-invocation"}


def test_r1_arm_targets_norew_stack_without_clobbering(tmp_path, monkeypatch) -> None:
    """Amendment v0.4 regression: --r1-arm closure_cons_norew probes the norew
    checkpoints and writes r1_closure_cons_norew / r1_report_<arm>.json; the
    frozen 'r1' block and r1_report.json are untouched."""
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **synthetic_canonical_arrays(total_steps=1200, seed=3))
    out = tmp_path / "out"
    args = _args(tmp_path, record=str(record_path), units="r1", seeds="0",
                 r1_arm="closure_cons_norew")
    summary = run_matrix(args)
    block = summary["units"]["r1_closure_cons_norew"]
    assert block["arm"] == "closure_cons_norew"
    assert block["verdict"] == "SMOKE"  # quick tier never emits a directional verdict
    assert block["status"] == "SMOKE"
    assert "norew" in block["reports"][0]["error"]
    assert "r1" not in summary["units"]
    assert (out / "r1_report_closure_cons_norew.json").exists()
    assert not (out / "r1_report.json").exists()


def test_matrix_quick_o1_and_b1_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ms, "HISTORY_STEPS", 16)
    arrays = synthetic_canonical_arrays(total_steps=1500, seed=5)
    record_path = tmp_path / "record.npz"
    np.savez_compressed(record_path, **arrays)
    args = _args(tmp_path, record=str(record_path), units="o1,b1")
    summary = run_matrix(args)
    out = tmp_path / "out"
    assert (out / "matrix_summary_quick.json").exists()
    ledger_lines = (out / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # 3 O1 arms + 1 B1 arm, quick mode = 1 seed
    run_ids = {json.loads(l)["run_id"] for l in ledger_lines}
    assert "o1_steady_seed0" in run_ids
    assert "o1_learned_seed0" in run_ids
    assert "o1_hybrid_seed0" in run_ids
    assert "b1_gru_seed0" in run_ids
    assert summary["units"] == {} or summary["quick"] is True  # no verdicts in quick mode
