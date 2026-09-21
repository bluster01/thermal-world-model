import copy
import json
from pathlib import Path

import pytest

from .data import sha256
from .physical import continuation_manifest, copy_parent_fits
from .run import HERE


def parent_and_config():
    parent = HERE.parents[1]/'results/action_predictor_bench_20260919/physical33_seed11'
    config = json.loads((parent/'config.json').read_text())
    config.update(max_epochs=90, source_sha256={p.name: sha256(p) for p in HERE.glob('*.py')})
    return parent, config


def test_returned_fits_extend_without_reusing_old_evaluations(tmp_path):
    parent, config = parent_and_config()
    manifest = continuation_manifest(parent, config)
    assert set(manifest['arms']) == {'P0', 'P1', 'P2'}
    assert all(value['epoch'] == 60 for value in manifest['arms'].values())
    copy_parent_fits(parent, tmp_path, manifest)
    for arm in manifest['arms']:
        folder = tmp_path/'fits/seed11'/arm
        assert not (folder/'fit.json').exists()
        assert sha256(folder/'last.pt') == sha256(parent/'fits/seed11'/arm/'last.pt')
    assert not (tmp_path/'seed11').exists()
    # Already plateaued arms are carried intact; they must not get extra updates.
    carried = copy.deepcopy(manifest)
    carried['arms']['P0']['reached_validation_plateau'] = True
    copy_parent_fits(parent, tmp_path/'carried', carried)
    assert (tmp_path/'carried/fits/seed11/P0/fit.json').exists()


@pytest.mark.parametrize('key,value', [('data_sha256', 'changed'), ('latent_weight', .1), ('batch_size', 64)])
def test_continuation_rejects_changed_scientific_settings(key, value):
    parent, config = parent_and_config()
    config[key] = value
    with pytest.raises(ValueError, match='experiment settings'):
        continuation_manifest(parent, config)


def test_continuation_rejects_changed_model_or_no_extra_budget():
    parent, config = parent_and_config()
    config['source_sha256']['physical_models.py'] = 'changed'
    with pytest.raises(ValueError, match='source mismatch'):
        continuation_manifest(parent, config)
    config['max_epochs'] = 60
    with pytest.raises(ValueError, match='larger total'):
        continuation_manifest(parent, config)
