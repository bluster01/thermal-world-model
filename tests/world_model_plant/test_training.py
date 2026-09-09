"""Small runner checks: support masks, selection coverage and source transfer."""
from types import SimpleNamespace
import hashlib
import json

import pytest
import torch

from experiments.world_model_plant.train import (TrainingConfig, supported_training_loss,
                                               validation_selection, verify_release)
from experiments.world_model_plant.model import PROPERTY_SHA256


def test_unsupported_window_and_missing_labels_cannot_contribute_gradients():
    prediction = torch.tensor([[[2., 3.]], [[7., 8.]]], requires_grad=True)
    batch = SimpleNamespace(keys=[0, 1], observed_right_mask=torch.tensor([[[True, False]], [[True, True]]]),
                            observations_right=torch.tensor([[[1., float('nan')]], [[0., 0.]]]))
    result = SimpleNamespace(domain_diagnostics={'unsupported_rows': [False, True]},
                             free_predictions=prediction, prior_predictions=prediction)
    model = SimpleNamespace(backend=SimpleNamespace(observation_scale=torch.ones(2)))
    loss, counts = supported_training_loss(result, batch, model, TrainingConfig())
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[4., 0.]], [[0., 0.]]]))
    assert counts == dict(supported_windows=1, unsupported_windows=1, supported_observed_scalars=1)
    result.domain_diagnostics['unsupported_rows'] = [True, True]
    assert supported_training_loss(result, batch, model, TrainingConfig())[0] is None


def test_validation_requires_full_coverage_but_keeps_conditional_score():
    records = [dict(state='supported', free_normalized_mse=2.),
               dict(state='unsupported', free_normalized_mse=100.)]
    result = validation_selection(records, 2)
    assert result['selection_value'] is None and result['conditional_supported_free_normalized_mse'] == 2.
    records[1] = dict(state='supported', free_normalized_mse=4.)
    assert validation_selection(records, 2)['selection_value'] == 3.


def test_source_freeze_accepts_git_newlines_and_rejects_code_change(tmp_path):
    files = {}
    for folder in ('src/final_wm', 'src/world_model_vnext', 'src/world_model_streaming',
                   'src/world_model_actuated', 'src/world_model_evidence', 'experiments/world_model_plant'):
        path = tmp_path / folder / '__init__.py'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'x = 1\r\n')
        files[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(b'x = 1\n').hexdigest()
    config = tmp_path / 'task.json'
    config.write_bytes(b'{}\n')
    files['task.json'] = hashlib.sha256(config.read_bytes()).hexdigest()
    release = tmp_path / 'release.json'
    release.write_text(json.dumps(dict(kind='plant_oracle_training_release_v1',
        scientific_training_authorized=True, files=files, properties_sha256=PROPERTY_SHA256)))
    verify_release(tmp_path, release, config)
    config.write_bytes(b'{"changed": true}\n')
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_release(tmp_path, release, config)
