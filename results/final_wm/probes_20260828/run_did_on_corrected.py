"""Re-run the plant DiD event study on the CORRECTED (v2.1) record.

WHY
---
`probes_20260828/plant_did_event_study.py` ran on `canonical_sideA.npz` (the OLD
record).  In that record the "valve1" column is the WRONG side (一级B) -- the
known wiring defect.  So its v1 result (n=0 up events, n=6 down, DiD H60 =
+0.113 degC, placebo p=0.485, i.e. a clean null) may be a DIRECT CONSEQUENCE of
the defect rather than a property of the plant.

Re-running the identical, unmodified analysis on `canonical_sideA_v1fixed.npz`
therefore gives a MODEL-FREE test of the wiring correction: if the corrected v1
(一级A) shows a real causal effect where the old column showed none, the wiring
fix is validated by plant-side statistics alone, with no training involved.

The analysis code is NOT modified -- this runner reads the original source and
rebinds only RECORD and OUT, so the protocol (min_step 0.04, contamination
horizon 60, k=5 matched controls, 200 placebo draws, one-sided score test)
stays byte-identical.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parent / "plant_did_event_study.py"
NEW_RECORD = 'ROOT / "results/final_wm/probes_20260824/v1fix_probe/canonical_sideA_v1fixed.npz"'
NEW_OUT = 'Path(__file__).resolve().parent / "corrected_record"'

src = SRC.read_text()
old_rec = 'RECORD = ROOT / "artifacts/final_wm/canonical_sideA.npz"'
old_out = 'OUT = Path(__file__).resolve().parent'
assert old_rec in src, "record line not found -- upstream script changed"
assert old_out in src, "out line not found -- upstream script changed"

src = src.replace(old_rec, f"RECORD = {NEW_RECORD}")
src = src.replace(old_out, f"OUT = {NEW_OUT}")

out_dir = SRC.parent / "corrected_record"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[runner] record -> {NEW_RECORD}")
print(f"[runner] out    -> {out_dir}")
print("[runner] analysis code unmodified apart from those two constants\n", flush=True)

g = {"__file__": str(SRC), "__name__": "__main__"}
exec(compile(src, str(SRC), "exec"), g)
