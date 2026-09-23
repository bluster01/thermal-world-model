"""Mocked execution/recovery checks; no industrial reads or optimization steps."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.unified_experiments_20260923 import prepare_linux, run


class RecoveryDataset:
    input_names = ["x"]
    target_names = ["x"]

    def __len__(self):
        return 4


@pytest.mark.parametrize("saved_epoch,wait,wanted_status", [(4, 3, "EARLY_STOPPED"), (9, 0, "BUDGET_EXHAUSTED")])
def test_complete_terminal_epoch_resumes_without_more_training(tmp_path, monkeypatch, saved_epoch, wait, wanted_status):
    protocol = {"primary_profile": "toy", "validation": {"max_selection_origins": 2},
                "training": {"learning_rate": .001, "weight_decay": .0001,
                             "plateau_factor": .5, "plateau_patience": 1,
                             "minimum_learning_rate": .00001, "min_epochs": 2,
                             "max_epochs": 10, "early_stop_patience": 3}}
    common = {"release": "synthetic"}
    model = nn.Linear(1, 1)
    identity = {"task": "toy", "arm": "mock", "seed": 7, "common": common,
                "input_names": ["x"], "target_names": ["x"], "model_parameters": 2,
                "model_class": "Linear", "training_origins": 4, "selection_origins": 2}
    path = tmp_path / "runs/toy__mock__s7"
    path.mkdir(parents=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=1, min_lr=.00001)
    best_name = "best_epoch_002.pt"
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    run.save_checkpoint(path / best_name, model, optimizer, scheduler, 1, .125, 0, identity, best_name)
    best_hash = run.sha(path / best_name)
    with torch.no_grad():
        model.weight.add_(10)
    run.save_checkpoint(path / "last.pt", model, optimizer, scheduler, saved_epoch, .125, wait, identity, best_name)
    run.write_json(path / "status.json", {"status": "FAILED", "identity": identity})
    monkeypatch.setattr(run, "IndustrialDataset", lambda *a: RecoveryDataset())
    monkeypatch.setattr(run, "construct", lambda *a: nn.Linear(1, 1))
    monkeypatch.setattr(run, "seed_all", lambda _: None)
    forbidden = mock.Mock(side_effect=AssertionError("Terminal recovery must not execute another batch"))
    monkeypatch.setattr(run, "loader", forbidden)
    monkeypatch.setattr(run, "evaluate", forbidden)
    run.fit_one("toy", "mock", 7, tmp_path / "unused", tmp_path, protocol,
                torch.device("cpu"), 0, common, resume=True)
    forbidden.assert_not_called()
    status = run.read_json(path / "status.json")
    assert status["status"] == wanted_status
    assert status["best_checkpoint"] == best_name
    assert status["checkpoint_sha256"] == best_hash == run.sha(path / "best.pt")
    assert torch.load(path / "best.pt", weights_only=False)["epoch"] == 1
    assert status["best_epoch"] == 2


class ManyOrigins(Dataset):
    input_names = target_names = ["toy"]
    target_indices = np.array([0])
    scale = np.array([2.])
    profile = {"forecast_steps": 3}

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {"origin_time_ns": np.int64(index * 100),
                "history_normalized": np.zeros((2, 1), dtype=np.float32),
                "history_valid": np.ones((2, 1), dtype=bool),
                "history_observed": np.ones((2, 1), dtype=bool),
                "history_age_s": np.zeros((2, 1), dtype=np.float32),
                "targets_normalized": np.full((3, 1), index + 1, dtype=np.float32),
                "target_mask": np.ones((3, 1), dtype=bool)}


class ZeroPrediction(nn.Module):
    def forward(self, batch, horizon):
        return batch["targets_normalized"].new_zeros((len(batch["target_mask"]), horizon, 1))


def test_evaluation_saves_every_origin_before_completion_marker(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    protocol = {"validation": {"batch_size": 2, "endpoints": [1, 3]}}
    original_write = run.write_json

    def checked_write(path, report):
        assert output.with_suffix(".npz").is_file()
        assert report["artifact_sha256"] == run.sha(output.with_suffix(".npz"))
        original_write(path, report)

    monkeypatch.setattr(run, "write_json", checked_write)
    report = run.evaluate(ZeroPrediction(), ManyOrigins(), protocol, torch.device("cpu"), 0, output)
    assert report["complete"] is True and report["origins"] == 7
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["origin_time_ns"], np.arange(7) * 100)
        np.testing.assert_array_equal(archive["label_count"], np.full((7, 1), 3))
        np.testing.assert_array_equal(archive["normalized_sse"][:, 0], 3 * np.arange(1, 8) ** 2)
        np.testing.assert_array_equal(archive["physical_absolute_error"][:, 0], 6 * np.arange(1, 8))
        assert archive["normalized_sse"].sum() / archive["label_count"].sum() == report["target_macro_normalized_mse"]
        previews = [name for name in archive.files if name.startswith("preview_origin_")]
        assert len(previews) == 3 and sum(len(archive[k]) for k in previews) == 6


def test_failed_array_write_never_commits_evaluation_json(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    monkeypatch.setattr(np, "savez_compressed", mock.Mock(side_effect=OSError("synthetic disk failure")))
    with pytest.raises(OSError, match="synthetic disk failure"):
        run.evaluate(ZeroPrediction(), ManyOrigins(), {"validation": {"batch_size": 2, "endpoints": [1]}},
                     torch.device("cpu"), 0, output)
    assert not output.with_suffix(".json").exists()
    assert not output.with_suffix(".npz").exists()


def setup_mock_main(tmp_path, monkeypatch, *, locked=False):
    spec, data, out = (tmp_path / name for name in ("spec", "data", "output"))
    for path in (spec, data, out):
        path.mkdir()
    protocol = {"experiment_id": "synthetic", "fit_count": 0, "tasks": [], "arms": [], "seeds": [],
                "execution": {"cpu_threads": 1}}
    run.write_json(spec / "protocol.json", protocol)
    run.write_json(spec / "expected_data.json", {})
    run.write_json(out / "preflight.json", {"status": "PASS"})
    events = []

    def flock(*args):
        events.append("lock")
        if locked:
            raise BlockingIOError("synthetic lock owner")

    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(run, "HERE", spec)
    monkeypatch.setattr(run, "linux_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic device")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(run, "verify_release", lambda: "synthetic-release")
    monkeypatch.setattr(run.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(run.subprocess, "check_output", lambda *a, **k: "synthetic-commit")
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(run, "preflight", mock.Mock(side_effect=AssertionError("Reuse existing preflight only")))

    def verify(*args):
        events.append("verify")
        return {"status": "PASS"}

    verifier = mock.Mock(side_effect=verify)
    monkeypatch.setattr(prepare_linux, "verify_pack", verifier)
    monkeypatch.setattr(sys, "argv", ["run", "preflight", "--data", str(data), "--out", str(out), "--resume"])
    return out, events, verifier


def test_every_resume_freshly_verifies_data_after_owning_lock(tmp_path, monkeypatch):
    out, events, verifier = setup_mock_main(tmp_path, monkeypatch)
    run.main()
    run.main()
    assert events == ["lock", "verify", "lock", "verify"]
    assert verifier.call_count == 2
    assert run.read_json(out / "public_status.json")["status"] == "PREFLIGHT_PASSED"


def test_second_process_cannot_verify_or_run_under_owned_lock(tmp_path, monkeypatch):
    _, events, verifier = setup_mock_main(tmp_path, monkeypatch, locked=True)
    with pytest.raises(RuntimeError, match="Another process already owns"):
        run.main()
    assert events == ["lock"]
    verifier.assert_not_called()


def test_resume_rejects_changed_release_before_reading_data(tmp_path, monkeypatch):
    out, events, verifier = setup_mock_main(tmp_path, monkeypatch)
    run.write_json(out / "run_identity.json", {"common": {"release_sha256": "different"}})
    with pytest.raises(RuntimeError, match="different released experiment"):
        run.main()
    verifier.assert_not_called()
