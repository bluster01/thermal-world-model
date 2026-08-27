"""Guard against the 2026-08-26 silent-properties defect.

train_arm(properties=None) falls back to AnalyticThermoProperties while probe
evaluation builds the model with the IAPWS grid -> train/eval physics mismatch,
which silently produced H18 MAE 4-8 (vs a 0.723 baseline) across five arms
before the ledger's `properties` field exposed it.

Import this in every probe that trains, and call assert_grid(props) before
train_arm, plus verify_ledger_properties(out_dir) after.
"""
from __future__ import annotations

import json
from pathlib import Path

EXPECTED = "GridThermoProperties"


def assert_grid(props) -> None:
    name = type(props).__name__
    if name != EXPECTED:
        raise RuntimeError(
            f"probe would train with {name}, expected {EXPECTED}. "
            "Load with load_grid_properties(artifacts/final_wm/iapws_surrogate.npz) "
            "and pass properties=props to train_arm.")
    print(f"[guard] training properties = {name} OK", flush=True)


def verify_ledger_properties(out_dir: str | Path) -> str:
    """Read back what the ledger recorded; raise if the arm trained analytic."""
    f = Path(out_dir) / "ledger.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    finals = [r for r in rows if r.get("final")]
    if not finals:
        raise RuntimeError(f"no final ledger entry in {f}")
    got = finals[-1].get("properties")
    if got != EXPECTED:
        raise RuntimeError(
            f"ledger says the arm trained with {got!r}, expected {EXPECTED!r} -- "
            "the run is VOID, do not report its numbers.")
    print(f"[guard] ledger properties = {got} OK", flush=True)
    return got
