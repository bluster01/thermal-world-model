"""Full-origin acceptance of the returned pack, optionally against original mmap A/B."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bilateral import load_data
from .bilateral_data import OUT_DEFAULT,REF_MAP,NAMES
from .bilateral_models import SIDE_RAW
from .data import sha256
from .run import HERE,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=OUT_DEFAULT)
    p.add_argument('--mmap',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data,meta=load_data(args.data)
    checks={}
    with np.load(HERE/'data/screen_A_33pct_h128.npz') as ref,np.load(HERE/'data/hist_bypass_A_33pct_v1.npz') as aux:
        names=aux['channel_names'].tolist()
        for split in ('train','selector','evaluation'):
            assert np.array_equal(data[f'{split}_starts'],ref[f'{split}_starts'])
            assert np.array_equal(data[f'{split}_time'],ref[f'{split}_time'])
            assert np.array_equal(data[f'{split}_time'],aux[f'{split}_time'])
            original=ref[split]
            actual=data[split][:,:,list(REF_MAP.values())]
            difference=float(np.abs(actual-original).max())
            if difference!=0: raise ValueError(f'{split}: A/reference mismatch {difference}')
            for name in NAMES[21:]:
                assert np.array_equal(data[split][:,:64,NAMES.index(name)],aux[f'hist12_{split}'][:,:,names.index(name)])
            # Previously duplicated two B sensors now have one canonical slot.
            assert np.array_equal(data[split][:,:64,5],aux[f'hist12_{split}'][:,:,names.index('b_sh1_in')])
            assert np.array_equal(data[split][:,:64,9],aux[f'hist12_{split}'][:,:,names.index('b_sh_out')])
            checks[split]=dict(windows=len(data[split]),A_all_rows_max_abs_difference=difference,auxiliary_bit_identical=True)
        for old,new in REF_MAP.items():
            assert data['mean'][new]==ref['mean'][old] and data['scale'][new]==ref['scale'][old]
    if args.mmap:
        times=np.load(args.mmap/'timestamps.npy',mmap_mode='r')
        for side,name in enumerate(('A','B')):
            source=np.load(args.mmap/f'values_{name}.npy',mmap_mode='r')
            for split in ('train','selector','evaluation'):
                maximum=0.
                for start in range(0,len(data[split]),512):
                    bank=data[split][start:start+512];ids=data[f'{split}_starts'][start:start+512,None]+np.arange(bank.shape[1])
                    assert np.array_equal(times[ids[:,63]],data[f'{split}_time'][start:start+512])
                    maximum=max(maximum,float(np.abs(bank[:,:,SIDE_RAW[side]]-source[ids]).max()))
                if maximum!=0: raise ValueError(f'{split}/{name}: original mmap mismatch {maximum}')
                checks[split][f'mmap_{name}_all_rows_max_abs_difference']=maximum
    save_json(args.output,dict(status='accepted',pack_sha256=meta['pack_sha256'],source_sha256=sha256(__file__),
        checks=checks,mmap_used=str(args.mmap) if args.mmap else None,
        normalization='A/core and auxiliary frozen; added channels train-history only',
        strict_counts={k:int(data[f'{k}_strict01'].sum()) for k in ('train','selector','evaluation')},
        old_event_analysis='v1 not accepted; use reviewed v2',training_launched=False))
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
