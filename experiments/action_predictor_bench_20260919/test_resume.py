"""Small physical fits must survive interruption without changing the experiment."""
import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from .full_baselines import fit
from .physical_models import Physical


@pytest.fixture
def tiny_data():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    mean = np.array([450, 440, 490, 480, 540, .3, .3, 700, 400, 20, 420, 280, 18], dtype=np.float32)
    scale = np.array([10]*5+[.1, .1, 100, 100, 2, 10, 10, 2], dtype=np.float32)
    bank = np.zeros((6, 192, 24), dtype=np.float32)
    bank[:, :, :13] = mean
    time = np.arange(192, dtype=np.float32)
    for i in range(len(bank)):
        bank[i, :, :5] += ((i+1)*.1+np.sin(time/25+i)*.5)[:, None]
        bank[i, :, 5] += .02*np.sin(time/40+i)
        bank[i, :, 6] += .03*np.cos(time/45+i)
        bank[i, :64, 13:] = (i+1)*.05
    yield dict(mean=mean, scale=scale, train=bank[:4], selector=bank[4:])
    torch.set_num_threads(previous_threads)


def _run(folder, arm, data, *, resume=False, interrupt=False, max_epochs=3,
         min_epochs=3, stop_patience=9):
    folder.mkdir(parents=True, exist_ok=True)
    # A different freshly constructed model makes a missing checkpoint load visible.
    torch.manual_seed(923 if resume else 11)
    model = Physical(data['mean'], data['scale'], np.zeros(11), np.ones(11), arm)
    args = Namespace(device='cpu', batch_size=2, learning_rate=.001, min_lr=.0005,
                     lr_patience=1, stop_patience=stop_patience,
                     min_epochs=min_epochs, max_epochs=max_epochs,
                     min_delta=1e6, smoke=False)
    calls = 0

    def objective(current_model, batch):
        nonlocal calls
        calls += 1
        if interrupt and calls == 5:
            raise RuntimeError('simulated interruption after two committed epochs')
        return current_model.training_objective(batch)

    return fit(arm, 11, data, args, folder, model=model,
               training_objective=objective, supervision_horizon=128, resume=resume)


def _load(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def _rows(folder):
    return [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def _interrupt(folder, arm, data):
    with pytest.raises(RuntimeError, match='simulated interruption'):
        _run(folder, arm, data, interrupt=True)
    assert not (folder/'fit.json').exists()
    checkpoint = _load(folder/'last.pt')
    assert checkpoint['epoch'] == 2 and checkpoint['updates'] == 4
    assert checkpoint['stale'] == 1
    assert checkpoint['optimizer']['param_groups'][0]['lr'] == .0005
    return checkpoint


@pytest.mark.parametrize('arm', ['P0', 'P1', 'P2'])
def test_physical_resume_matches_uninterrupted_training(tmp_path, tiny_data, arm):
    uninterrupted, resumed = tmp_path/'uninterrupted', tmp_path/'resumed'
    reference = _run(uninterrupted, arm, tiny_data)
    partial = _interrupt(resumed, arm, tiny_data)
    initial_log = _rows(resumed)
    result = _run(resumed, arm, tiny_data, resume=True)
    actual, expected = _load(resumed/'last.pt'), _load(uninterrupted/'last.pt')
    for key in ('model', 'optimizer', 'epoch', 'updates', 'milestones', 'stale', 'best', 'best_epochs'):
        _assert_equal(actual[key], expected[key])
    for choice in ('short', 'balanced'):
        _assert_equal(_load(resumed/f'best_{choice}.pt'), _load(uninterrupted/f'best_{choice}.pt'))
    for key in ('epochs_run', 'updates', 'best_epochs', 'best_selector', 'final_selector', 'final_lr'):
        _assert_equal(result[key], reference[key])
    rows = _rows(resumed)
    assert [r['epoch'] for r in rows] == [1, 2, 3]
    assert [r['updates'] for r in rows] == [2, 4, 6]
    assert rows[:2] == initial_log
    for actual_row, expected_row in zip(rows, _rows(uninterrupted)):
        _assert_equal({k: v for k, v in actual_row.items() if k != 'seconds'},
                      {k: v for k, v in expected_row.items() if k != 'seconds'})
    assert result['train_seconds'] >= rows[-1]['seconds'] >= rows[-2]['seconds']
    if arm == 'P1':
        teacher_keys = [k for k in actual['model'] if k.startswith('teacher.')]
        assert teacher_keys
        assert any(not torch.equal(actual['model'][k], partial['model'][k]) for k in teacher_keys)


def test_legacy_checkpoint_resumes_and_recovers_elapsed_time(tmp_path, tiny_data):
    uninterrupted, resumed = tmp_path/'uninterrupted', tmp_path/'legacy'
    _run(uninterrupted, 'P1', tiny_data)
    checkpoint = _interrupt(resumed, 'P1', tiny_data)
    legacy_keys = ('model', 'optimizer', 'epoch', 'updates', 'milestones', 'stale', 'best', 'best_epochs')
    torch.save({key: checkpoint[key] for key in legacy_keys}, resumed/'last.pt')
    rows = _rows(resumed)
    for row in rows:
        row['seconds'] = 3600.*row['epoch']
    (resumed/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = _run(resumed, 'P1', tiny_data, resume=True)
    actual, expected = _load(resumed/'last.pt'), _load(uninterrupted/'last.pt')
    for key in legacy_keys:
        _assert_equal(actual[key], expected[key])
    assert _rows(resumed)[:2] == rows
    assert result['train_seconds'] >= _rows(resumed)[-1]['seconds'] >= 7200.


@pytest.mark.parametrize('log_state', ['missing_committed_row', 'extra_uncommitted_row'])
def test_resume_reconciles_log_with_committed_checkpoint(tmp_path, tiny_data, log_state):
    _interrupt(tmp_path, 'P0', tiny_data)
    committed_rows = _rows(tmp_path)
    if log_state == 'missing_committed_row':
        rows = committed_rows[:1]
    else:
        rows = committed_rows+[{**committed_rows[-1], 'epoch': 3, 'updates': 6, 'train_loss': -1.}]
    (tmp_path/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    _run(tmp_path, 'P0', tiny_data, resume=True)
    rows = _rows(tmp_path)
    assert rows[:2] == committed_rows
    assert [row['epoch'] for row in rows] == [1, 2, 3]
    assert [row['updates'] for row in rows] == [2, 4, 6]
    assert rows[-1]['train_loss'] >= 0


@pytest.mark.parametrize('stopping_rule', ['epoch_budget', 'validation_plateau'])
def test_resume_recovers_missing_completion_record_without_another_update(tmp_path, tiny_data, monkeypatch, stopping_rule):
    options = dict(max_epochs=2, min_epochs=1) if stopping_rule == 'epoch_budget' else dict(min_epochs=1, stop_patience=1)
    reference = _run(tmp_path, 'P1', tiny_data, **options)
    assert reference['epochs_run'] == 2
    assert reference['stop_reason'] == ('budget_limit' if stopping_rule == 'epoch_budget' else 'validation_plateau')
    checkpoint, rows = _load(tmp_path/'last.pt'), _rows(tmp_path)
    (tmp_path/'fit.json').unlink()

    def no_training(*args, **kwargs):
        raise AssertionError('a committed terminal checkpoint must not train again')

    monkeypatch.setattr(Physical, 'training_objective', no_training)
    result = _run(tmp_path, 'P1', tiny_data, resume=True, **options)
    assert (tmp_path/'fit.json').exists()
    for key in ('epochs_run', 'updates', 'stop_reason', 'reached_validation_plateau', 'best_epochs', 'final_selector'):
        assert result[key] == reference[key]
    _assert_equal(_load(tmp_path/'last.pt'), checkpoint)
    assert _rows(tmp_path) == rows


def test_resume_rejects_best_checkpoint_ahead_of_last_without_overwriting(tmp_path, tiny_data):
    checkpoint = _interrupt(tmp_path, 'P0', tiny_data)
    best = _load(tmp_path/'best_short.pt')
    best['epoch'] = checkpoint['epoch']+1
    torch.save(best, tmp_path/'best_short.pt')
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='Inconsistent best_'):
        _run(tmp_path, 'P0', tiny_data, resume=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == originals
