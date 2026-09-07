"""Authorized inference-only replay; never invokes a training or leakage probe."""
from __future__ import annotations
import hashlib
import argparse
import json
import platform
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.final_wm.data import CanonicalRecord, SPLIT_VAL
from src.final_wm.evaluation import evaluate_windows, step_response_direction, paired_difference_ci
from src.final_wm.properties import load_grid_properties
from src.final_wm.training import TrainSpec, build_world_model

PACKAGE = ROOT / 'artifacts/final_wm/v07_full_reissue_v1'
OUT = ROOT / 'results/final_wm/norew_local_replay_20260907'
ARMS = ('physics_only', 'closure_cons', 'closure_cons_norew')
# Cross-platform float32 replay criterion, frozen before inference. Not a
# scientific acceptance threshold: failure stops rather than loosens tolerance.
ATOL, RTOL = 0.002, 0.0001

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read_npy_prefix(archive, name, rows=None):
    with archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        assert version in ((1, 0), (2, 0)), version
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        assert not fortran and not dtype.hasobject
        count_rows = shape[0] if rows is None else rows
        assert 0 <= count_rows <= shape[0]
        count = int(np.prod((count_rows, *shape[1:])))
        raw = stream.read(count * dtype.itemsize)
        assert len(raw) == count * dtype.itemsize
        return np.frombuffer(raw, dtype=dtype).copy().reshape(count_rows, *shape[1:])

def isolated_record(path):
    # split is partition metadata only. Never decode any test-valued numeric
    # row. ZIP may buffer compressed bytes; those are not interpreted as values.
    with zipfile.ZipFile(path) as archive:
        split = read_npy_prefix(archive, 'split')
        test = np.flatnonzero(split == 2)
        assert len(test) and np.array_equal(test, np.arange(test[0], len(split)))
        end = int(test[0])
        assert set(np.unique(split[:end])) == {0, 1}
        record = CanonicalRecord.__new__(CanonicalRecord)
        record.path = path.resolve()
        record.n = end
        record.split = torch.from_numpy(split[:end].astype(np.int64))
        record._split_runs_cache = {}
        for name in ('boundary', 'actions', 'obs', 'timestamps'):
            values = read_npy_prefix(archive, name, end)
            dtype = np.int64 if name == 'timestamps' else np.float32
            setattr(record, name, torch.from_numpy(values.astype(dtype)))
    return record, {'original_rows': len(split), 'decoded_numeric_rows': end,
                    'first_test_index': end, 'test_numeric_rows_decoded': 0,
                    'original_indices_preserved': True}

def sampled_indices(record, n_windows, history_steps, horizon, seed):
    """Mirror the unchanged sampler RNG draws, for auditable row identities."""
    span = history_steps + horizon
    runs = [(s,e) for s,e in record.split_runs(SPLIT_VAL) if e-s >= span]
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for _ in range(n_windows):
        s,e = runs[int(torch.randint(len(runs),(1,),generator=generator))]
        indices.append(int(torch.randint(s,e-span+1,(1,),generator=generator))+history_steps)
    return indices

def main():
    global PACKAGE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True, help='Original package containing inputs/ and sideA/')
    parser.add_argument('--out', type=Path, required=True, help='New, nonexistent output directory')
    args = parser.parse_args()
    PACKAGE, OUT = args.package.resolve(), args.out.resolve()
    if sys.flags.optimize:
        raise RuntimeError('Do not use python -O: protocol assertions must remain enabled')
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen replay requires CUDA; no silent fallback')
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision('highest')
    protocol = {'authorization': 'User explicitly authorized checkpoint prediction/response replay, no retraining',
        'arms': ARMS, 'seeds': [0,1,2], 'prediction': {'windows':256,'batch':32,'horizon':18,'history':96,'seed':'50000 + seed'},
        'response': {'windows':64,'horizons':[18,60],'valves':[0,1],'delta_v':0.05,'summary':'last ten steps, final temperature; equal-day bootstrap',
                     'seed':'80000 + 1000*valve + horizon + seed','n_boot':1000,'allow_extrapolation_for_flagged_diagnostics':True},
        'replay_tolerance': {'atol':ATOL,'rtol':RTOL,'applies_to':'every saved MAE/NLL/CRPS element; day ids exact'},
        'paired_mae_ci':'each seed separately, equal UTC-day weighting, 1000 day bootstrap samples; no pooled seed significance',
        'training':False,'leakage_probe':False,'full_R1_verdict':False,'device':'cuda',
        'environment':{'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),
                       'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'matmul_precision':torch.get_float32_matmul_precision(),
                       'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32},
        'script_sha256':digest(Path(__file__))}
    write(OUT/'protocol.json',protocol)
    manifest = json.loads((PACKAGE/'sideA/manifest.json').read_text())
    for item in manifest['inputs'].values():
        assert digest(PACKAGE/'sideA'/item['package_relative_path']) == item['sha256']
    record, isolation = isolated_record(PACKAGE/'inputs/canonical_sideA_v2.npz')
    write(OUT/'isolation.json',isolation)
    properties = load_grid_properties(PACKAGE/'inputs/iapws_surrogate.npz',device='cuda')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('highest')
    reports, metrics_by_run = [], {}
    for seed in range(3):
        for arm in ARMS:
            started = time.monotonic()
            run = f't1_{arm}_seed{seed}'
            print('START '+run,flush=True)
            ckpt_path = PACKAGE/'sideA/checkpoints'/f'{run}.pt'
            metrics_path = PACKAGE/'sideA/metrics'/f'{run}.pt'
            for p in (ckpt_path, metrics_path):
                assert digest(p) == manifest['files'][p.relative_to(PACKAGE/'sideA').as_posix()]
            payload = torch.load(ckpt_path,map_location='cpu',weights_only=True)
            spec = TrainSpec(**payload['spec'])
            assert (spec.arm,spec.seed,spec.initial_state_mode,spec.boundary_mode)==(arm,seed,'hybrid','oracle')
            assert (spec.epochs,spec.patience,spec.horizon,spec.history_steps,spec.latent_dim)==(120,20,18,96,0)
            assert spec.closure_mode == {'physics_only':'none','closure_cons':'conservative','closure_cons_norew':'conservative_norew'}[arm]
            model = build_world_model(spec,properties).to('cuda').eval()
            model.load_state_dict(payload['state_dict'],strict=True)
            before = {k:v.detach().clone() for k,v in model.named_parameters()}
            metrics = evaluate_windows(model,record,SPLIT_VAL,n_windows=256,batch_size=32,history_steps=96,horizon=18,boundary_mode='oracle',seed=50000+seed,device='cuda')
            saved = torch.load(metrics_path,map_location='cpu',weights_only=True)['metrics']
            checks = {}
            assert torch.equal(metrics.day_ids,saved['day_ids'])
            for key in ('mae','nll','crps'):
                new, old = getattr(metrics,key),saved[key]
                checks[key]={'max_abs_difference':float((new-old).abs().max()),'bit_exact':torch.equal(new,old),'allclose':torch.allclose(new,old,atol=ATOL,rtol=RTOL)}
            anchors = sampled_indices(record,256,96,18,50000+seed)
            assert torch.equal(torch.div(record.timestamps[torch.tensor(anchors)],86400,rounding_mode='floor'), metrics.day_ids)
            report = {'run':run,'seed':seed,'arm':arm,'prediction_mae':float(metrics.mae.mean()),'prediction_indices':anchors,'replay_checks':checks,'responses':{}}
            torch.save(metrics._asdict(),OUT/f'{run}_prediction.pt')
            write(OUT/f'{run}.json',report)
            assert all(c['allclose'] for c in checks.values()), checks
            metrics_by_run[(arm,seed)] = metrics
            print('PREDICTION_OK '+run+' '+str(report['prediction_mae']),flush=True)
            for valve in (0,1):
                for horizon in (18,60):
                    key=f'valve{valve+1}_H{horizon}'
                    response=step_response_direction(model,record,SPLIT_VAL,n_windows=64,history_steps=96,rollout_steps=horizon,valve_index=valve,seed=80000+1000*valve+horizon+seed,device='cuda',allow_extrapolation=True)
                    response['first_future_indices']=sampled_indices(record,64,96,1,80000+1000*valve+horizon+seed)
                    report['responses'][key]=response
                    write(OUT/f'{run}.json',report)
                    print(run+' '+key+' delta='+str(response['mean_delta_c'])+' unsupported='+str(response['n_unsupported']),flush=True)
            assert all(torch.equal(before[k],v) for k,v in model.named_parameters()), 'parameters changed'
            report['parameters_unchanged']=True
            report['elapsed_seconds']=time.monotonic()-started
            write(OUT/f'{run}.json',report)
            reports.append(report)
            del model, before
    paired=[]
    for seed in range(3):
        for base,arm in (('closure_cons','closure_cons_norew'),('physics_only','closure_cons')):
            result=paired_difference_ci(metrics_by_run[(base,seed)],metrics_by_run[(arm,seed)],horizon=18,metric='mae',seed=seed)
            paired.append({'base':base,'arm':arm,'seed':seed,**result._asdict()})
    write(OUT/'summary.json',{'runs':reports,'paired_mae_equal_day_ci':paired,'full_R1_verdict':'NOT_ASSESSED_NO_LEAKAGE_TRAINING','test_values_accessed':False,'training_run':False})
    print('COMPLETE',flush=True)

if __name__ == '__main__':
    main()
