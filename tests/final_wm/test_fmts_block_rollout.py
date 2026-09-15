"""Engineering-only fixtures: no optimizer, training, or plant-data evaluation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fmts_block_rollout_20260915.run import (
    eligibility, rollout_batch, metrics, aggregate, private_output_path,
)
from experiments.fmts_mainsteam_20260911.models import RichBlackbox, RichFusion
from experiments.fmts_mainsteam_20260911.spec import structured_specs
from src.final_wm.model import HistoryWindow


@pytest.fixture
def record():
    torch.set_num_threads(1)
    n = 400
    r = SimpleNamespace(n=n)
    r.obs = torch.tensor([420., 410., 500., 490., 540.]).repeat(n, 1)
    r.obs[:, 0] += torch.arange(n) / 100
    r.actions = torch.tensor([.2, .3]).repeat(n, 1)
    r.boundary = torch.tensor([450., 180., 24., 400., 310., 22., 30.]).repeat(n, 1)
    r.boundary_ext = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 9)
    r.timestamps = torch.arange(n, dtype=torch.int64) * 10
    r.split = torch.ones(n, dtype=torch.long)
    r.base_valid = torch.ones(n, dtype=torch.bool)
    r.ext_valid = torch.ones(n, dtype=torch.bool)
    return r


class Spy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, h, ext, actions, boundary):
        assert not torch.is_grad_enabled()
        self.calls.append((h.obs.clone(), ext.clone(), actions.clone(), boundary.clone()))
        return h.obs[:, -1:, 4] + torch.arange(1, 19, device=h.obs.device)


def test_complete_block_feedback_and_truncated_tail(record):
    model = Spy()
    pred = rollout_batch(model, record, torch.tensor([100]), 'cpu')
    assert pred.shape == (1, 120)
    assert len(model.calls) == 7
    torch.testing.assert_close(pred[0], 540. + torch.arange(1, 121))
    h2, ext2, a2, b2 = model.calls[1]
    torch.testing.assert_close(h2[0, -18:, 4], 540. + torch.arange(1, 19))
    torch.testing.assert_close(h2[0, :, :4], record.obs[22:118, :4])
    torch.testing.assert_close(ext2[0], record.boundary_ext[22:118])
    torch.testing.assert_close(a2[0], record.actions[118:136])
    torch.testing.assert_close(b2[0], record.boundary[118:136])
    torch.testing.assert_close(model.calls[-1][0][0, :, 4], 540. + torch.arange(13, 109))
    assert torch.equal(record.obs[:, 4], torch.full((400,), 540.))


def test_future_main_temperature_never_fed_back(record):
    starts = torch.tensor([100])
    original = rollout_batch(Spy(), record, starts, 'cpu')
    record.obs[100:, 4] = -123456.
    torch.testing.assert_close(rollout_batch(Spy(), record, starts, 'cpu'), original)


@pytest.mark.parametrize('kind', ['blackbox', 'fusion', 'greybox'])
def test_original_forward_first_block_and_unchanged_weights(record, kind):
    if kind == 'blackbox':
        model = RichBlackbox(torch.zeros(23), torch.ones(23))
    else:
        if kind == 'fusion':
            spec = next(s for s in structured_specs((0,)) if s.arm == 'fusion_gru_norew')
        else:
            from experiments.fmts_greybox_norew_20260913.spec import specs
            spec = specs()[0]
        model = RichFusion(spec, torch.zeros(23), torch.ones(23))
    model.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    h = HistoryWindow(record.obs[4:100][None], record.actions[4:100][None], record.boundary[4:100][None])
    with torch.no_grad():
        direct = model(h, record.boundary_ext[4:100][None], record.actions[100:118][None], record.boundary[100:118][None])
    direct = direct if isinstance(direct, torch.Tensor) else direct.temps_mu[:, :, 4]
    pred = rollout_batch(model, record, torch.tensor([100]), 'cpu')
    torch.testing.assert_close(pred[:, :18], direct, rtol=0, atol=0)
    assert torch.isfinite(pred).all()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_common_eligibility_preserves_duplicates_and_order(record):
    result = eligibility(record, np.array([200, 100, 200]))
    assert result['eligible'].tolist() == [True, True, True]
    assert result['starts'].tolist() == [200, 100, 200]


@pytest.mark.parametrize('field, index, value, reason', [
    ('split', 225, 2, 'non_validation'),
    ('timestamps', 110, -1, 'time_gap'),
    ('base_valid', 225, False, 'invalid_base'),
    ('ext_valid', 207, False, 'invalid_history_extension'),
])
def test_quality_gate_exact_consumed_range(record, field, index, value, reason):
    getattr(record, field)[index] = value
    result = eligibility(record, np.array([100]))
    assert not result['eligible'][0]
    assert result[reason][0]


def test_unconsumed_extension_and_outside_future_not_screened(record):
    record.ext_valid[208:] = False
    record.split[226:] = 2
    assert eligibility(record, np.array([100]))['eligible'][0]
    assert eligibility(record, np.array([95, 300]))['out_of_bounds'].all()


def test_day_balancing_and_seed_sd():
    target = np.zeros((3, 120))
    pred = np.repeat([[1.], [3.], [9.]], 120, axis=1)
    m = metrics(pred, target, np.array([0, 0, 1]))
    assert m['H120'] == 5.5
    rows = [dict(arm=a, seed=s, metrics=metrics(pred + s, target, np.array([0, 0, 1])))
            for a in ['blackbox_itransformer', 'fusion_gru_norew', 'greybox_steady_none_norew'] for s in [0, 1, 2]]
    summary = aggregate(rows)
    assert summary['fusion_gru_norew']['mean'][119] == 6.5
    assert summary['fusion_gru_norew']['seed_sd'][119] == 1.
    with pytest.raises(AssertionError):
        aggregate(rows[:-1])


def test_private_directory_must_be_outside_repository(tmp_path):
    from experiments.fmts_block_rollout_20260915.run import ROOT
    with pytest.raises(ValueError, match='outside'):
        private_output_path(ROOT / 'results' / 'unsafe')
    assert private_output_path(tmp_path / 'traces') == (tmp_path / 'traces').resolve()


@pytest.fixture
def exported_fixture(record, tmp_path, monkeypatch):
    """Exercise packaging/replay with explicit mocked provenance, never a plant run."""
    from experiments.fmts_block_rollout_20260915 import run as R
    starts = np.full(256, 100, dtype=np.int64)
    parent, gnr = tmp_path / 'parent', tmp_path / 'gnr'
    mapping = tmp_path / 'mapping.json'
    mapping.write_text('{}')
    protocol = dict(R.P, fixed_case_starts=[100, 100])
    for prefix, base, arms in [('parent', parent, R.P['arms'][:2]), ('gnr', gnr, R.P['arms'][2:])]:
        base.mkdir()
        R.write(base / 'identity.json', {'test_fixture': True})
        np.savez_compressed(base / 'indices.npz', validation=starts)
        reports = []
        for arm in arms:
            for seed in range(3):
                folder = base / f'{arm}_seed{seed}'
                folder.mkdir()
                (folder / 'best.pt').write_bytes(b'engineering test stub - not a model')
                np.savez_compressed(folder / 'prediction.npz', starts=starts, days=np.zeros(256, dtype=np.int64),
                                    prediction=np.tile(540. + np.arange(1, 19, dtype=np.float32), (256, 1)),
                                    target=np.full((256, 18), 540., dtype=np.float32),
                                    persistence=np.full((256, 18), 540., dtype=np.float32))
                report = dict(arm=arm, seed=seed, checkpoint_sha256=R.sha256(folder / 'best.pt'),
                              prediction_sha256=R.sha256(folder / 'prediction.npz'))
                R.write(folder / 'report.json', report)
                reports.append(report)
        R.write(base / 'summary.json', dict(runs=reports))
        protocol[prefix + '_identity'] = R.sha256(base / 'identity.json')
        protocol[prefix + '_summary'] = R.sha256(base / 'summary.json')
    protocol['indices'] = R.sha256(parent / 'indices.npz')
    monkeypatch.setattr(R, 'P', protocol)
    monkeypatch.setattr(R, 'validate_sources', lambda *args: None)
    monkeypatch.setattr(R, 'RichRecord', lambda *args: record)
    monkeypatch.setattr(R, 'load_grid_properties', lambda *args: None)
    monkeypatch.setattr(R, 'load_model', lambda *args: Spy())
    record.candidates = lambda split: torch.tensor([100])
    def batch(idx, split, device):
        assert split == 1
        past, future = idx[:, None] + torch.arange(-96, 0), idx[:, None] + torch.arange(18)
        return (HistoryWindow(record.obs[past], record.actions[past], record.boundary[past]),
                record.boundary_ext[past], record.actions[future], record.boundary[future],
                record.obs[future], record.timestamps[idx] // 86400)
    record.batch = batch
    args = SimpleNamespace(private_out=tmp_path / 'private', parent=parent, gnr=gnr, mapping=mapping,
                           record='mocked-record', properties='mocked-properties', device='cpu')
    R.execute(args)
    return args


def test_saved_artifacts_are_not_real_replay(exported_fixture):
    from experiments.fmts_block_rollout_20260915.audit import audit
    a = exported_fixture
    result = audit(a.private_out, a.parent, a.gnr, mapping=a.mapping)
    assert result['artifact_checks_passed'] and result['cells_checked'] == 9
    assert not result['complete'] and result['cells_replayed'] == 0


def test_mocked_full_replay_gate(exported_fixture):
    from experiments.fmts_block_rollout_20260915.audit import audit
    a = exported_fixture
    result = audit(a.private_out, a.parent, a.gnr, a.record, a.properties, a.mapping)
    assert result['complete'] and result['cells_replayed'] == 9  # Fixture pipeline, not scientific evidence.


@pytest.mark.parametrize('tamper', ['file', 'missing_cell', 'metric', 'replay_failure'])
def test_audit_rejects_tampering_and_failed_replay(exported_fixture, tamper, monkeypatch):
    from experiments.fmts_block_rollout_20260915 import run as R
    from experiments.fmts_block_rollout_20260915.audit import audit
    a = exported_fixture
    if tamper == 'file':
        (a.private_out / 'observed.npz').write_bytes(b'corrupt')
    elif tamper == 'missing_cell':
        manifest = R.read(a.private_out / 'manifest.json')
        del manifest['fusion_gru_norew_seed2.npz']
        R.write(a.private_out / 'manifest.json', manifest)
    elif tamper == 'metric':
        path = a.private_out / 'fusion_gru_norew_seed0.json'
        report = R.read(path)
        report['metrics']['H120'] = -1
        R.write(path, report)
        manifest = R.read(a.private_out / 'manifest.json')
        manifest[path.name] = R.sha256(path)
        R.write(a.private_out / 'manifest.json', manifest)
    else:
        monkeypatch.setattr(R, 'collect', lambda *args: np.zeros((256, 120)))
    with pytest.raises(AssertionError):
        audit(a.private_out, a.parent, a.gnr, a.record if tamper == 'replay_failure' else None,
              a.properties, a.mapping)


def test_never_overwrite_existing_export(exported_fixture):
    from experiments.fmts_block_rollout_20260915.run import execute
    with pytest.raises(FileExistsError):
        execute(exported_fixture)
