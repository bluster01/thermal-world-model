from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_independent_ms3d_bootstrap_is_deterministic():
    from experiments.phase3_5.audit_ms3d_asymmetry_diagnosis import (
        independent_bootstrap_median_ci,
    )

    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    first = independent_bootstrap_median_ci(values, samples=2000, seed=11)
    second = independent_bootstrap_median_ci(values, samples=2000, seed=11)
    assert first == second
    assert first[0] <= np.median(values) <= first[1]


def test_repository_ms3d_artifacts_recompute_exactly():
    from experiments.phase3_5.audit_ms3d_asymmetry_diagnosis import run_audit

    result = run_audit(ROOT / "results/phase3_5/ms3d_asymmetry_diagnosis")
    assert result["passes"] is True
    assert result["max_numeric_recomputation_error"] <= 1e-12
    assert result["test_accessed"] is False
