from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/phase3_5/ms3r_gatec_real_subset.py"


def test_gatec_real_subset_dry_run_is_local_and_test_closed() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["fraction_denominator"] == 100
    assert payload["allowed_splits"] == ["train", "validation"]
    assert payload["test_allowed"] is False
    assert payload["local_real_smoke_authorized"] is True
    assert payload["linux_authorized"] is False
    assert payload["real_full_matrix_authorized"] is False
    assert len(payload["routes"]) == 4
    assert payload["automatic_scientific_pass"] is None
