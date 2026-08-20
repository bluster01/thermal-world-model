"""Audit-pack analyses: event study, references, floors, binning, ablation."""

from __future__ import annotations

import numpy as np
import torch

from src.final_wm.analysis import (
    binning_stats,
    error_floor_anchors,
    event_study_summary,
    mixing_cooling_reference,
    persistence_increment_mae,
    rewetting_ablation,
    spray_sensitivity,
    valve_step_events,
)
from src.final_wm.contracts import KAPPA_TPH_TO_KGS
from src.final_wm.data import CanonicalRecord
from src.final_wm.synthetic import synthetic_canonical_arrays
from src.final_wm.training import build_world_model
from experiments.final_wm import matrix_spec as ms


def _record(tmp_path, arrays) -> CanonicalRecord:
    path = tmp_path / "record.npz"
    np.savez_compressed(path, **arrays)
    return CanonicalRecord(path)


def _crafted_record(tmp_path, n: int = 400) -> CanonicalRecord:
    """v2 steps +0.1 at t=200; final temp ramps down 0.01 C/step afterwards.
    All rows validation so the event study can read them."""
    arrays = synthetic_canonical_arrays(total_steps=n, seed=7)
    arrays["actions"][:, 0] = 0.3
    arrays["actions"][:200, 1] = 0.4
    arrays["actions"][200:, 1] = 0.5
    obs = arrays["obs"]
    obs[:, 4] = 550.0
    obs[200:, 4] = 550.0 - 0.01 * np.arange(n - 200)
    arrays["split"] = np.ones(n, dtype=np.int8)  # all val
    return _record(tmp_path, arrays)


def test_event_study_recovers_known_step(tmp_path) -> None:
    record = _crafted_record(tmp_path)
    events = valve_step_events(record, 1, valve_index=1, min_step=0.04, horizon=60)
    summary = event_study_summary(events)
    assert summary["up"]["n"] == 1
    assert summary["down"]["n"] == 0
    # event at t=200, baseline obs[199]=550; H60 -> obs[259] = 550 - 0.59
    h60 = summary["up"]["horizons"].index(60)
    assert abs(summary["up"]["mean_delta"][h60] - (-0.59)) < 1e-3  # float32 record
    assert summary["up"]["frac_correct"][h60] == 1.0


def test_event_study_excludes_contaminated_steps(tmp_path) -> None:
    arrays = synthetic_canonical_arrays(total_steps=400, seed=7)
    arrays["actions"][:, 0] = 0.3
    arrays["actions"][:, 1] = 0.4
    arrays["actions"][100:, 1] = 0.55   # event 1: follow-up within horizon -> excluded
    arrays["actions"][120:, 1] = 0.70   # event 2: preceded within horizon -> excluded
    arrays["actions"][390:, 1] = 0.85   # too close to the run end -> not counted
    arrays["split"] = np.ones(400, dtype=np.int8)
    record = _record(tmp_path, arrays)
    events = valve_step_events(record, 1, valve_index=1, min_step=0.04, horizon=60)
    assert events["up"]["n"] == 0
    assert events["n_excluded_followup"] == 1
    assert events["n_excluded_preceded"] == 1


def test_persistence_increment_mae_constant_signal(tmp_path) -> None:
    arrays = synthetic_canonical_arrays(total_steps=300, seed=7)
    arrays["obs"][:] = 500.0
    arrays["split"] = np.ones(300, dtype=np.int8)
    record = _record(tmp_path, arrays)
    mae = persistence_increment_mae(record, 1)
    assert all(v == 0.0 for v in mae.values())


def test_spray_sensitivity_recovers_slopes(tmp_path) -> None:
    arrays = synthetic_canonical_arrays(total_steps=500, seed=7)
    v1, v2 = arrays["actions"][:, 0], arrays["actions"][:, 1]
    arrays["boundary"][:, 6] = 2.0 + 8.0 * v1 + 30.0 * v2  # W, t/h
    arrays["split"] = np.ones(500, dtype=np.int8)
    record = _record(tmp_path, arrays)
    fit = spray_sensitivity(record, 1)
    # Tolerance 1e-5 (was 1e-6): aarch64/BLAS float differences give relative
    # error ~4e-8, i.e. 1.14e-6 absolute at slope 30 -- see the Hermes rerun
    # failure report 2026-08-20 §5 (red item, user ruling A: relax here).
    assert abs(fit["dW_dv1_tph_per_full"] - 8.0) < 1e-5
    assert abs(fit["dW_dv2_tph_per_full"] - 30.0) < 1e-5
    assert abs(fit["dW_dv2_kgs_per_2pct"] - 30.0 * 0.02 * KAPPA_TPH_TO_KGS) < 1e-5
    assert fit["r2"] > 0.999
    assert fit["closed_loop_warning"] is True


def test_mixing_reference_matches_evidence_anchors() -> None:
    # 1 kg/s extra spray: 1.36 C at D=560, 3.8 C at D=200 (evidence chain sec.2)
    ref = mixing_cooling_reference(1.0, d_lo=200.0, d_hi=560.0)
    assert abs(ref["delta_t_at_d_hi"] - 1.36) < 0.02
    assert abs(ref["delta_t_at_d_lo"] - 3.80) < 0.05


def test_error_floor_anchors_separate_fast_and_regime(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 4000
    arrays = synthetic_canonical_arrays(total_steps=n, seed=7)
    flow = np.linspace(200, 560, n)
    arrays["boundary"][:, 0] = flow
    slow_regime = 0.05 * (flow - 380.0)
    fast_noise = rng.normal(0.0, 0.5, n)
    arrays["obs"][:, 4] = 550.0 + slow_regime + fast_noise
    arrays["split"] = np.ones(n, dtype=np.int8)
    record = _record(tmp_path, arrays)
    floors = error_floor_anchors(record, 1, n_bins=5, median_window=61)
    assert 0.3 < floors["fast_sigma"]["final_outlet_temp"] < 0.8
    assert floors["within_bin_sigma"]["final_outlet_temp"] < 1.6


def _window_errors(tmp_path) -> "object":
    from src.final_wm.analysis import WindowErrors
    load = torch.linspace(200, 560, 200)
    abs_err = torch.rand(200, 18, 5, generator=torch.Generator().manual_seed(0))
    return WindowErrors(abs_err=abs_err, load=load, day_ids=torch.zeros(200, dtype=torch.int64))


def test_binning_stats_flags_regime_dependence(tmp_path) -> None:
    from src.final_wm.analysis import WindowErrors
    base = _window_errors(tmp_path)
    stats = binning_stats(base, n_bins=5, horizons=(1,))
    ratio_flat = stats["H1"]["final_outlet_temp"]["between_ratio"]
    assert ratio_flat < 0.05  # iid errors: no regime signal
    abs_err = base.abs_err.clone()
    abs_err[:, :, 4] += ((base.load - 380.0) / 100.0).unsqueeze(1)  # load-dependent bias
    stats2 = binning_stats(WindowErrors(abs_err=abs_err, load=base.load, day_ids=base.day_ids),
                           n_bins=5, horizons=(1,))
    assert stats2["H1"]["final_outlet_temp"]["between_ratio"] > 0.5


def test_auditpack_phase_record_only(tmp_path) -> None:
    from argparse import Namespace

    from experiments.final_wm.run_matrix import run_auditpack

    path = tmp_path / "record.npz"
    np.savez_compressed(path, **synthetic_canonical_arrays(total_steps=3000, seed=13))
    args = Namespace(record=str(path), out=str(tmp_path / "out"), side="A", checkpoint=None,
                     properties_npz=None, device="cpu", quick=True, arm="closure_cons", seed=0)
    report = run_auditpack(args)
    # Quick tier must not clobber the audited artifact (rerun failure report
    # 2026-08-20 §6): quick runs write the *_quick.json sibling instead.
    assert (tmp_path / "out" / "auditpack_A_quick.json").exists()
    assert not (tmp_path / "out" / "auditpack_A.json").exists()
    for key in ("spray_sensitivity", "mixing_reference", "persistence_increment_mae",
                "error_floor", "event_study"):
        assert key in report
    assert set(report["event_study"]) == {"v1", "v2"}
    assert "residual_binning" not in report  # model-based probes need --checkpoint


def test_rewetting_ablation_restores_parameters(tmp_path) -> None:
    arrays = synthetic_canonical_arrays(total_steps=600, seed=11)
    record = _record(tmp_path, arrays)
    spec = ms._base("t1", "closure_cons", 0, boundary_mode="oracle",
                    initial_state_mode="hybrid", closure_mode="conservative")
    model = build_world_model(spec, properties=None)
    before = [model.transition.raw[n].data.clone() for n in ("aW1", "aW2")]
    report = rewetting_ablation(model, record, 1, n_windows=4, history_steps=16, seed=0)
    after = [model.transition.raw[n].data for n in ("aW1", "aW2")]
    for b, a in zip(before, after):
        assert torch.equal(b, a)  # restoration
    for key in ("intact", "rewet_zeroed"):
        assert np.isfinite(report[key]["mean_delta_c"])
        assert 0.0 <= report[key]["frac_negative"] <= 1.0
