"""One new arm x 3 seeds; reuse frozen v0.2 training/evaluation without edits."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import numpy as np
import torch
from experiments.fmts_mainsteam_20260911.data import RichRecord
from experiments.fmts_mainsteam_20260911.manifests import sha256
from experiments.fmts_mainsteam_20260911.run import train_one, summarize, write_json, day_mean, block_interval
from src.final_wm.contracts import FinalWMProtocolError
from src.final_wm.properties import load_grid_properties
from .spec import ARM, PARENT_ARM, specs, serialized_spec

ROOT=Path(__file__).resolve().parents[2]


def validate_parent(parent,record,mapping,properties,smoke=False):
    parent=Path(parent)
    identity=json.loads((parent/'identity.json').read_text())
    summary=json.loads((parent/'summary.json').read_text())
    if identity['smoke'] != smoke or summary['locked_test_evaluated']:
        raise FinalWMProtocolError('parent smoke/test status mismatch')
    if identity['protocol']['protocol_version'] != '0.2':
        raise FinalWMProtocolError('requires frozen v0.2 parent')
    for field,path in [('record',record),('mapping',mapping),('indices',parent/'indices.npz')]:
        if sha256(path)!=identity[field]:
            raise FinalWMProtocolError(f'parent {field} hash mismatch')
    if not smoke and (not properties or sha256(properties)!=identity['properties']):
        raise FinalWMProtocolError('identical parent IAPWS grid required')
    for source,digest in identity['sources'].items():
        local=ROOT/(source if not source.startswith('/') else
                    'experiments/fmts_mainsteam_20260911/'+Path(source).name)
        if sha256(local)!=digest:
            raise FinalWMProtocolError(f'frozen parent code changed: {source}')
    expected={(arm,s) for arm in [PARENT_ARM,'fusion_gru_norew','blackbox_itransformer','fusion_token_xattn_norew']
              for s in ((0,) if smoke else (0,1,2))}
    actual={(r['arm'],r['seed']) for r in summary['runs'] if r['status']=='complete'}
    if actual!=expected:
        raise FinalWMProtocolError('parent results incomplete')
    for report in summary['runs']:
        folder=parent/f"{report['arm']}_seed{report['seed']}"
        for name in ['prediction','response']:
            if sha256(folder/f'{name}.npz')!=report[f'{name}_sha256']:
                raise FinalWMProtocolError('parent result artifact hash mismatch')
    return identity


def compare(parent,out,reports):
    """Pair retained original results with the new arm; no checkpoint selection."""
    rows=[]
    for report in reports:
        if report['status']!='complete':
            continue
        seed=report['seed']
        with np.load(out/f'{ARM}_seed{seed}'/'prediction.npz') as f:
            new={k:f[k] for k in f.files}
        with np.load(out/f'{ARM}_seed{seed}'/'response.npz') as f:
            response={k:f[k] for k in f.files}
        for reference in [PARENT_ARM,'fusion_gru_norew','blackbox_itransformer']:
            with np.load(parent/f'{reference}_seed{seed}'/'prediction.npz') as old:
                for key in ['starts','days','target','persistence']:
                    if not np.array_equal(new[key],old[key]):
                        raise FinalWMProtocolError(f'unpaired prediction/{key}')
                delta=(np.abs(new['prediction'].astype(float)-new['target']).mean(1)
                       -np.abs(old['prediction'].astype(float)-old['target']).mean(1))
                prediction=block_interval(delta,new['days'])
            with np.load(parent/f'{reference}_seed{seed}'/'response.npz') as old:
                for key in ['starts','days','valve1_support','valve2_support','valve1_dose','valve2_dose']:
                    if not np.array_equal(response[key],old[key]):
                        raise FinalWMProtocolError(f'unpaired response/{key}')
                valves={}
                for v in [1,2]:
                    new_delta=response[f'valve{v}_prediction'].astype(float)-response['base']
                    old_delta=old[f'valve{v}_prediction'].astype(float)-old['base']
                    valves[str(v)]={'new_curve':day_mean(new_delta,response['days']).tolist(),
                                    'reference_curve':day_mean(old_delta,old['days']).tolist(),
                                    'new_minus_reference':block_interval(new_delta-old_delta,response['days'])}
            rows.append({'candidate':ARM,'reference':reference,'seed':seed,
                         'paired_cumulative_H18_mae':prediction,'response':valves})
    write_json(out/'comparison.json',{'status':'DESCRIPTIVE_ONLY_NO_PLANT_TRUTH','pairs':rows,
                                    'locked_test_evaluated':False,'parent_results_retained':True})


def execute(parent,record_path,mapping,properties,out,device='cpu',smoke=False):
    parent,out=Path(parent),Path(out)
    frozen=Path(__file__).with_name('frozen_spec.json')
    if json.loads(frozen.read_text())!=serialized_spec():
        raise FinalWMProtocolError('executable spec differs from frozen supplement registration')
    if out.exists():
        raise FinalWMProtocolError('output exists; no overwrite or implicit resume')
    parent_id=validate_parent(parent,record_path,mapping,properties,smoke)
    record=RichRecord(record_path,mapping)
    with np.load(parent/'indices.npz') as f:
        indices={k:torch.from_numpy(f[k].copy()) for k in f.files}
    for key,split in [('train',0),('validation',1),('response',1)]:
        if not torch.isin(indices[key],record.candidates(split)).all():
            raise FinalWMProtocolError(f'ineligible parent {key} indices')
    mean,std=record.normalization()
    for name,value in [('mean',mean),('std',std)]:
        if not torch.allclose(value,torch.tensor(parent_id[f'normalization_{name}']),rtol=1e-6,atol=1e-6):
            raise FinalWMProtocolError('parent normalization mismatch')
    out.mkdir(parents=True)
    (out/'.gitattributes').write_text('* -text\n',encoding='ascii')
    shutil.copyfile(parent/'indices.npz',out/'indices.npz')
    identity=dict(parent_id)
    identity['parent_protocol']=identity.pop('protocol')
    identity.update(protocol=serialized_spec(),frozen_spec_sha256=sha256(frozen),
                    registration_sha256=sha256(ROOT/'docs/fmts2026/PREREG_GREYBOX_NOREW_20260913.md'),
                    parent_identity_sha256=sha256(parent/'identity.json'),
                    parent_summary_sha256=sha256(parent/'summary.json'),
                    git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
                    supplement_sources={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p)
                                        for p in sorted(Path(__file__).parent.glob('*.py'))},
                    python=platform.python_version(),torch=torch.__version__,device=device,
                    cuda_version=torch.version.cuda,matmul_precision=torch.get_float32_matmul_precision(),
                    cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    write_json(out/'identity.json',identity)
    reports=[]
    for spec in specs((0,) if smoke else (0,1,2)):
        print(f'Starting {ARM} seed{spec.seed}',flush=True)
        try:
            props=load_grid_properties(properties) if properties else None
            report=train_one(ARM,spec.seed,spec,record,indices,mean,std,props,device,out,smoke)
            # Only finalize metadata in this NEW checkpoint: no weight changes.
            checkpoint=out/f'{ARM}_seed{spec.seed}'/'best.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            payload['parent_protocol']=payload.pop('protocol')
            payload['protocol']=serialized_spec()
            torch.save(payload,checkpoint)
            report['checkpoint_sha256']=sha256(checkpoint)
            write_json(checkpoint.parent/'report.json',report)
        except Exception as exc:
            report={'arm':ARM,'seed':spec.seed,'status':'failed','error':str(exc)}
            write_json(out/f'{ARM}_seed{spec.seed}_failure.json',report)
        reports.append(report)
        write_json(out/'progress.json',reports)
    summarize(out,reports,smoke)
    summary=json.loads((out/'summary.json').read_text())
    summary.update(protocol_id='FMTS-GNR1',selected_hybrid='UNCHANGED_PARENT_fusion_gru_norew',
                   supplement_only=True,scientific_verdict='PENDING_AUDIT_NOT_PLANT_FIDELITY')
    write_json(out/'summary.json',summary)
    compare(parent,out,reports)
    if any(r['status']!='complete' for r in reports):
        raise FinalWMProtocolError('supplement has failed runs; preserve failure artifacts')
    return reports


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--parent',default='results/fmts_mainsteam_20260911/linux_full_v02')
    ap.add_argument('--record',required=True)
    ap.add_argument('--mapping',default='configs/final_wm/channel_mapping_v2.json')
    ap.add_argument('--properties')
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    execute(args.parent,args.record,args.mapping,args.properties,args.out,args.device,args.smoke)


if __name__=='__main__':
    main()
