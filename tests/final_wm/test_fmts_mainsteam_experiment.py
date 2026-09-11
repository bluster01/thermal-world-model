from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.fmts_mainsteam_20260911.manifests import (
    sample_start_indices,
    save_manifest_bundle,
    windows_from_starts,
)
from experiments.fmts_mainsteam_20260911.spec import serialized_spec, structured_specs
from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.data import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, CanonicalRecord
from src.final_wm.synthetic import synthetic_canonical_arrays


def _record(tmp_path) -> CanonicalRecord:
    path = tmp_path / "record.npz"
    np.savez_compressed(path, **synthetic_canonical_arrays(total_steps=1800, seed=11))
    return CanonicalRecord(path)


def test_registered_specs_are_complete_and_stable() -> None:
    specs = structured_specs()
    assert len(specs) == 9
    assert {(spec.arm, spec.seed) for spec in specs} == {
        (arm, seed)
        for arm in ("greybox_steady_none", "fusion_gru_norew", "fusion_token_xattn_norew")
        for seed in (0, 1, 2)
    }
    payload = serialized_spec()
    assert payload["common_evaluation"]["locked_test_enabled"] is False
    assert payload["token_advance_rule"]["fallback"] == "fusion_gru_norew"


def test_manifest_round_trip_is_deterministic_and_matched(tmp_path) -> None:
    record = _record(tmp_path)
    first = sample_start_indices(record, SPLIT_VAL, 20, 16, 4, seed=30_000)
    second = sample_start_indices(record, SPLIT_VAL, 20, 16, 4, seed=30_000)
    torch.testing.assert_close(first, second)
    batch = windows_from_starts(record, first, SPLIT_VAL, 16, 4)
    assert batch.history.obs.shape == (20, 16, 5)
    assert batch.future_obs.shape == (20, 4, 5)
    metadata = save_manifest_bundle(
        record,
        tmp_path / "manifest",
        train_split=SPLIT_TRAIN,
        validation_split=SPLIT_VAL,
        history_steps=16,
        horizon=4,
        train_windows=30,
        validation_windows=20,
        response_windows=10,
    )
    assert metadata["locked_test_included"] is False
    assert metadata["arrays"]["response_starts"]["count"] == 10


def test_manifest_code_fails_closed_on_test_or_cross_split(tmp_path) -> None:
    record = _record(tmp_path)
    with pytest.raises(FinalWMProtocolError, match="locked test"):
        sample_start_indices(record, SPLIT_TEST, 2, 16, 4, seed=1)
    with pytest.raises(FinalWMProtocolError, match="locked test"):
        windows_from_starts(record, torch.tensor([10]), SPLIT_TEST, 16, 4)
    train_end = int(torch.nonzero(record.split == SPLIT_TRAIN)[-1])
    with pytest.raises(FinalWMProtocolError, match="crosses"):
        windows_from_starts(record, torch.tensor([train_end]), SPLIT_TRAIN, 16, 4)

