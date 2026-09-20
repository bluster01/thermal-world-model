"""Read returned arrays and produce descriptive tables; no fitting or selection."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    repo = Path(__file__).resolve().parents[3]
    root = repo / 'results/action_predictor_bench_20260919/full33_seed11'
    source = repo / 'experiments/action_predictor_bench_20260919'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    config = read(root / 'config.json')
    state = read(root / 'state.json')
    assert state['status'] == 'complete' and not state['failures']
    assert not state['not_confirmed_plateau'] and not config['smoke']
    for name, digest in config['source_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    pack = source / 'data/screen_A_33pct_h128.npz'
    assert hashlib.sha256(pack.read_bytes()).hexdigest() == config['data_sha256']
    bank = np.load(root / 'evaluation_inputs.npz')['bank']
    truth = bank[:, 64:, :5].astype(float)
    assert truth.shape == (256, 512, 5)
    fits = [read(p) for p in sorted((root / 'fits/seed11').glob('*/fit.json'))]
    assert {f['model'] for f in fits} == set(config['models'])
    folders = sorted((root / 'seed11').iterdir())
    assert len(folders) == 21
    verified, errors = 0, []
    for folder in folders:
        result = read(folder / 'result.json')
        assert result['status'] == 'complete'
        with np.load(folder / 'forecasts.npz') as arrays:
            for mode in ('block', 'native'):
                key = mode + '_recorded'
                if key not in arrays:
                    continue
                prediction = arrays[key].astype(float)
                assert prediction.shape == truth.shape and np.isfinite(prediction).all()
                error = np.abs(prediction - truth)
                saved = result['forecast'][key]
                difference = abs(float(error[:, :, 4].mean()) - saved['main_mae_C'])
                assert difference < 1e-6, (folder.name, mode, difference)
                verified += 1
                errors.append(dict(model=folder.name, mode=mode,
                                   H32_all=float(error[:, :32].mean()),
                                   H512_all=float(error.mean()), difference=difference))
    lines = ['# Full33 补充读数', '',
             '从回传数组重算；仅描述性分析，不重新选模。精度单位 °C。', '',
             '## 全五温度平均误差（short checkpoint）', '',
             '| 模型 | 协议 | H32 | H512 |', '|---|---|---:|---:|']
    for row in errors:
        if row['model'].endswith('_short') or row['model'] == 'persistence':
            lines.append(f"| {row['model']} | {row['mode']} | {row['H32_all']:.4f} | {row['H512_all']:.4f} |")
    lines += ['', '## 响应按形状和动作发生位置分组（short checkpoint）', '',
              '每行是同组方案等权平均，不把不同有效窗口数当成相同的统计样本量。',
              '方向不符比例仅针对定义了固定符号指标的阶跃/脉冲/斜坡/双脉冲；连续正负摆动不套用该指标。',
              '响应绝对幅度是敏感度，不是动作响应真值误差。窗口前=-24，早=0，中=64，晚=112，跨块=31/32/33。', '',
              '| 模型 | 协议 | 分组 | 方案数 | 平均响应绝对幅度 | 方向不符均值 | 动作发生前最大变化（全通道） |',
              '|---|---|---|---:|---:|---:|---:|']
    for folder in folders:
        if not (folder.name.endswith('_short') or folder.name == 'persistence'):
            continue
        probes = read(folder / 'response_metrics.json')
        for mode in sorted({p['mode'] for p in probes}):
            valid = [p for p in probes if p['mode'] == mode and p['status'] == 'ok']
            groups = [(s, [p for p in valid if p['shape'] == s]) for s in sorted({p['shape'] for p in valid})]
            groups += [(f'onset={t}', [p for p in valid if p['relative_onset'] == t]) for t in (-24, 0, 31, 32, 33, 64, 112)]
            for label, rows in groups:
                if not rows:
                    continue
                signed = [p['opposite_sign_fraction'] for p in rows if 'opposite_sign_fraction' in p]
                wrong = f'{np.mean(signed):.2%}' if signed else '—'
                amplitude = np.mean([p['mean_abs_main_C'] for p in rows])
                pre = max(p['pre_onset_max_abs_C'] for p in rows)
                lines.append(f'| {folder.name} | {mode} | {label} | {len(rows)} | {amplitude:.5f} | {wrong} | {pre:.5f} |')
    (root / 'SUPPLEMENT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    check = dict(return_commit='c696bf9', rows=21, fits=len(fits),
                 recorded_forecast_modes_recomputed=verified, all_finite=True,
                 max_main_H512_recompute_difference=max(r['difference'] for r in errors),
                 source_hashes_match=True, data_hash_matches=True,
                 total_fit_seconds=sum(f['train_seconds'] for f in fits),
                 scope='Returned-array arithmetic and source/data identity; no model replay, no training, one seed')
    (root / 'return_review.json').write_text(json.dumps(check, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    main()
