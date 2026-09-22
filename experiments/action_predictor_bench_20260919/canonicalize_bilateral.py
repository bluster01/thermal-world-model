"""Replace CSV-converted core inputs with the exact frozen A/B mmap values.

This is an acceptance repair, not a new sample or a training step. The original
returned pack is retained in Git at 9ab31ba; its digest is saved in the manifest.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .bilateral_data import OUT_DEFAULT, REF_MAP, SPLITS, sha256
from .bilateral_models import SIDE_RAW


def canonicalize(path, mmap):
    path, mmap = Path(path), Path(mmap)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    previous = sha256(path)
    if previous != meta['pack_sha256']:
        raise ValueError('Source pack digest mismatch')
    times = np.load(mmap/'timestamps.npy', mmap_mode='r')
    difference = {}
    for side, name in enumerate(('A', 'B')):
        source = np.load(mmap/f'values_{name}.npy', mmap_mode='r')
        for split in SPLITS:
            h = arrays[f'hist30_{split}']
            H = arrays[f'future_temp_{split}'].shape[1]
            maximum = np.zeros(13)
            for start in range(0, len(h), 256):
                stop = min(start+256, len(h))
                ids = arrays[f'{split}_starts'][start:stop, None]+np.arange(64+H)
                assert np.array_equal(times[ids[:,63]], arrays[f'{split}_time'][start:stop])
                assert np.all(np.diff(times[ids], axis=1)==10)
                raw = source[ids]
                maximum = np.maximum(maximum, np.abs(h[start:stop,:,SIDE_RAW[side]]-raw[:,:64]).max((0,1)))
                for j, channel in enumerate(SIDE_RAW[side]):
                    h[start:stop,:,channel] = raw[:,:64,j]
                arrays[f'future_temp_{split}'][start:stop,:,side*5:side*5+5] = raw[:,64:,:5]
                for j, channel in enumerate(SIDE_RAW[side][5:7]):
                    arrays[f'future_act_{split}'][start:stop,:,channel-10] = raw[:,63:,5+j]
                for j, channel in enumerate(SIDE_RAW[side][7:]):
                    arrays[f'future_bnd_{split}'][start:stop,:,channel-14] = raw[:,63:,7+j]
            difference[f'{split}_{name}_history_max_difference_per_core_channel'] = maximum.tolist()
    # Preserve all old normalization. Refit only the eight added channels on the
    # identical unique TRAIN history rows, using the original float32 convention.
    epochs = arrays['train_time'][:,None]+(np.arange(64)-63)*10
    _, first = np.unique(epochs.ravel(), return_index=True)
    unique = arrays['hist30_train'].reshape(-1,30)[first]
    mean, scale = unique.mean(0), unique.std(0)
    for channel in (5,6,7,8,9,11,12,20):
        arrays['mean'][channel], arrays['scale'][channel] = mean[channel], scale[channel]
    for split in SPLITS:
        u = np.concatenate((arrays[f'hist30_{split}'][:,:,10:14],arrays[f'future_act_{split}'][:,1:]),1)
        assert np.array_equal(arrays[f'strict01_{split}'],((u>=0)&(u<=1)).all((1,2)))
        arrays[f'near_closed_{split}'] = (np.abs(u)<.02).any((1,2))
        arrays[f'u2a_negative_{split}'] = (u[:,:,2]<0).any(1)
    temporary = path.with_name(path.stem+'_canonical_tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    meta['returned_csv_verification'] = meta.pop('verification', {})
    meta['canonicalization'] = dict(previous_pack_sha256=previous, original_return_commit='9ab31ba',
        reason='CSV float64 conversion vs frozen mmap float32 conversion; temperatures already identical',
        source_files={name:sha256(mmap/name) for name in ('timestamps.npy','values_A.npy','values_B.npy')},
        code_sha256=sha256(__file__), differences=difference,
        preserved='origins, timestamps, nine auxiliary history channels, old core/aux normalization')
    meta['pack_sha256'] = sha256(path)
    meta['produced_by'] += '; canonicalize_bilateral.py acceptance repair 2026-09-23'
    meta['history_alignment'] += '; core values replaced by exact frozen mmap values'
    path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta['canonicalization'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=OUT_DEFAULT)
    p.add_argument('--mmap',type=Path,required=True)
    a=p.parse_args(); canonicalize(a.data,a.mmap)
