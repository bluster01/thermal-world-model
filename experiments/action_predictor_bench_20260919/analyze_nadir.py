"""Summarize saved valve-response extrema without fitting or relabelling them as truth."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/action_predictor_bench_20260919'
MODELS = ('B_block_recorded', 'B_native_recorded', 'D_block_recorded',
          'D_native_recorded', 'P0_native_recorded')
CHANNELS = ('T1', 'T2', 'T3', 'T4', 'main')
DT = 10.0
EPS = 1e-4


def distribution(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0, 'mean': None, 'std': None, 'min': None,
                'q25': None, 'median': None, 'q75': None, 'max': None}
    q = np.quantile(x, (0, .25, .5, .75, 1))
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
                **dict(zip(('min', 'q25', 'median', 'q75', 'max'), map(float, q))))


def trajectory_metrics(raw, sign):
    """Rows are windows; signed<0 aligns opening-cooling and closing-heating."""
    signed = raw * sign
    idx = signed.argmin(axis=1)
    nadir = np.take_along_axis(signed, idx[:, None], axis=1)[:, 0]
    terminal = signed[:, -1]
    tail = signed[:, -6:].mean(axis=1)
    desired = np.maximum(-nadir, 0)
    eligible = nadir < -EPS
    recovery = np.full(len(raw), np.nan)
    recovery[eligible] = (terminal[eligible] - nadir[eligible]) / (-nadir[eligible])
    tail_recovery = np.full(len(raw), np.nan)
    tail_recovery[eligible] = (tail[eligible] - nadir[eligible]) / (-nadir[eligible])
    # Every saved output represents the end of a 10-second interval. Use a
    # declared right-endpoint rectangular sum, not an implicit integral scheme.
    good_area = np.maximum(-signed, 0).sum(axis=1) * DT
    wrong_area = np.maximum(signed, 0).sum(axis=1) * DT
    return dict(signed_nadir_C=nadir, signed_nadir_time_s=(idx+1)*DT,
                raw_delta_at_signed_nadir_C=nadir*sign,
                desired_direction_peak_C=desired,
                raw_min_C=raw.min(axis=1), raw_max_C=raw.max(axis=1),
                raw_terminal_C=raw[:, -1], raw_tail60_mean_C=raw[:, -6:].mean(axis=1),
                signed_terminal_C=terminal, signed_tail60_mean_C=tail,
                desired_direction_area_Cs=good_area, reverse_direction_area_Cs=wrong_area,
                signed_area_Cs=signed.sum(axis=1)*DT,
                recovery_to_terminal_fraction=recovery,
                recovery_to_tail60_fraction=tail_recovery,
                signed_nadir_at_last_sample=(idx == raw.shape[1]-1).astype(float),
                any_desired_direction=(desired > EPS).astype(float))


def summarize(raw, sign, positions):
    per = trajectory_metrics(raw, sign)
    mean_raw = raw.mean(axis=0)
    mean_signed = mean_raw * sign
    i = int(mean_signed.argmin())
    nadir = float(mean_signed[i])
    aggregate = {key: distribution(value) for key, value in per.items()}
    mean_curve = dict(signed_nadir_C=nadir, raw_delta_at_signed_nadir_C=nadir*sign,
                      signed_nadir_time_s=(i+1)*DT, desired_direction_peak_C=max(0., -nadir),
                      raw_terminal_C=float(mean_raw[-1]), raw_tail60_mean_C=float(mean_raw[-6:].mean()),
                      signed_terminal_C=float(mean_signed[-1]), signed_tail60_mean_C=float(mean_signed[-6:].mean()),
                      recovery_to_terminal_fraction=(float(mean_signed[-1])-nadir)/(-nadir) if nadir < -EPS else None,
                      raw_curve_C=mean_raw.tolist())
    records = []
    for j, position in enumerate(positions):
        row = {'evaluation_position': int(position)}
        for key, value in per.items():
            row[key] = float(value[j]) if np.isfinite(value[j]) else None
        records.append(row)
    return dict(windows=len(raw), distribution=aggregate, mean_curve=mean_curve, per_window=records)


def paired_comparisons(saved, case_idx, group, sign):
    records = []
    group_mask = saved['case_labels'][case_idx] == group
    d = saved['D_block_recorded'][case_idx, :, :, 4]
    for base in ('B_block_recorded', 'B_native_recorded'):
        b = saved[base][case_idx, :, :, 4]
        valid = group_mask & np.isfinite(b).all(axis=1) & np.isfinite(d).all(axis=1)
        bm, dm = trajectory_metrics(b[valid], sign), trajectory_metrics(d[valid], sign)
        row = dict(baseline=base, candidate='D_block_recorded', group=group,
                   case=str(saved['case_ids'][case_idx]), n=int(valid.sum()),
                   evaluation_positions=saved['positions'][valid].tolist())
        for key in ('desired_direction_peak_C', 'desired_direction_area_Cs', 'reverse_direction_area_Cs'):
            bmean, dmean = float(bm[key].mean()), float(dm[key].mean())
            row[key] = dict(baseline_mean=bmean, candidate_mean=dmean,
                            candidate_over_baseline=dmean/bmean if bmean > 1e-12 else None,
                            paired_difference=distribution(dm[key]-bm[key]))
        records.append(row)
    return records


def write_summary(payload, output):
    main = [row for row in payload['rows'] if row['channel'] == 'main' and row['model'] != 'D_native_recorded']
    lines = ['# 保存曲线的谷底、末端与恢复：不训练', '',
             '源数据为既有权重在真实未来边界下，阀门记录轨迹整体加减3个百分点的预测差值。',
             '采样间隔10秒，观察10–1280秒（21.33分钟）；不是实测干预曲线，也不能把末端当稳态。', '',
             '原差值 ΔT = 改动作预测 − 原动作预测。统计时 s = sign(阀门改变量) × ΔT；s<0 对应开大冷却或关小升温。',
             '图保留原 ΔT 符号，正负动作分别画；JSON 同时保存原符号和对齐后符号。', '',
             '“每窗谷值均值”先对每个起点找最低值再平均；“均值曲线谷值”先平均曲线再找最低值。两者不可混用。',
             '表内均为对齐后有符号温差；正谷值表示该均值曲线从未进入所参考的响应方向。', '',
             '|模型|负荷|动作|n|每窗谷值均值°C|每窗谷值Q25/Q50/Q75°C|均值曲线谷值°C|其时刻s|末端原ΔT°C|末60s原ΔT°C|每窗正确/反向面积均值°C·s|',
             '|---|---|---|---:|---:|---|---:|---:|---:|---:|---|']
    for r in main:
        s, m = r['distribution'], r['mean_curve']
        q = s['signed_nadir_C']
        lines.append(f"|{r['model'].replace('_recorded','')}|{r['group']}|v{r['valve']} {r['dose_pp']:+.0f}pp|{r['windows']}|{q['mean']:.4f}|{q['q25']:.4f}/{q['median']:.4f}/{q['q75']:.4f}|{m['signed_nadir_C']:.4f}|{m['signed_nadir_time_s']:.0f}|{m['raw_terminal_C']:.4f}|{m['raw_tail60_mean_C']:.4f}|{s['desired_direction_area_Cs']['mean']:.2f}/{s['reverse_direction_area_Cs']['mean']:.2f}|")
    lines += ['', '## 配对的幅度变化', '',
              '以下只对同一组有效起点计算，峰值为 max(0, -min(s))，不是绝对峰值；反方向峰不会被计为有效响应。', '',
              '|负荷|动作|相对基线|D/基线每窗正确方向峰值均值|D/基线正确方向面积均值|',
              '|---|---|---|---:|---:|']
    for r in payload['paired_comparisons_main']:
        peak = r['desired_direction_peak_C']['candidate_over_baseline']
        area = r['desired_direction_area_Cs']['candidate_over_baseline']
        lines.append(f"|{r['group']}|{r['case']}|{r['baseline'].replace('_recorded','')}|{peak:.3f}|{area:.3f}|")
    lines += ['', '## 可以与不可以得出的结论', '',
              '- 阀2中，D 的每窗冷却/升温峰值确实远弱于 B 连续；这是可以直接复核的模型差异，不能解释成 D 幅度已经正确。',
              '- 阀1中，B 连续每窗谷值均值与均值曲线谷值相差很大；不同起点的谷底时刻、方向和回弹差异很大。只选各窗最深点会放大“已经恢复幅度”的观感。',
              '- B 分块阀2的均值谷底集中在340–350秒，接近320秒重编码边界；连续递推谷底更晚。当前结果需检查边界伪影，不能直接将较深谷底认定为真实物理峰值。',
              '- JSON 的恢复比例是 (末端s−谷底s)/(-谷底s)，仅对谷底<-0.0001°C计算；0为未恢复，1为回到零，>1为越过零。处于观察末端的谷底是右截断，不能说没有回弹。',
              '- P0 经常在最后一个样本达到谷底；较大单调响应同样不能自动当作物理幅度真值。',
              '- 当前仅每类至多8个固定起点、单训练种子；描述窗口间分布，不提供虚假的独立样本置信区间。', '',
              '## 下一轮如何辨别幅度', '',
              '1. 固定权重先做同起点 H512 响应，并保留 H128 截断读数；配对比较连续状态、每32步重编码，另加16/64步边界移动。如果谷底跟着计算边界移动，它不足以支持物理延迟解释。',
              '2. 阶跃同时做±1/±3/±5pp，脉冲做30/60/120秒，并区分阀门回到原计划前后的恢复。阶跃仍在时的回弹与撤销脉冲后的恢复不是同一个指标。',
              '3. 同时预测两侧局部喷前/喷后温度与终端温度，检查阀门→局部温差→下游链路的时序、剂量比例和路径；不只盯主温谷底。',
              '4. 双侧负对照允许共享扰动，但对经核实不可达的短时路径、动作前差值、时间打乱动作做对照。A−B差分可削弱部分共同扰动，不能保证移除侧别独立扰动。',
              '5. 从训练/选择区提取阀门变化而煤量/流量相对平稳、另一阀少动的事件，估计匹配条件下的响应区间；与模型峰值/面积/时刻对照。事件是否满足可比条件需要记录，不能将记录相关性直接称作干预真值。', '']
    (output/'nadir_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(saved, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'axes.unicode_minus':False})
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14), sharex=True)
    colors = {'B_block_recorded':'#0072B2', 'B_native_recorded':'#56B4E9',
              'D_block_recorded':'#009E73', 'P0_native_recorded':'#D55E00'}
    labels = {'B_block_recorded':'B 分块', 'B_native_recorded':'B 连续',
              'D_block_recorded':'D 保护', 'P0_native_recorded':'P0'}
    t = (np.arange(128)+1)*DT/60
    for row, (group, dose) in enumerate((('rise', .03), ('rise', -.03), ('fall', .03), ('fall', -.03))):
        for col, valve in enumerate((1, 2)):
            ax = axes[row, col]
            case = f'step_t0_v{valve}_{dose:+.2f}'
            i = saved['case_ids'].tolist().index(case)
            mask = saved['case_labels'][i] == group
            for key, color in colors.items():
                raw = saved[key][i, mask, :, 4]
                raw = raw[np.isfinite(raw).all(axis=1)]
                mean = raw.mean(axis=0)
                ax.plot(t, mean, color=color, lw=1.8, label=f'{labels[key]} n={len(raw)}')
                extremum = (mean*np.sign(dose)).argmin()
                ax.scatter(t[extremum], mean[extremum], s=30, color=color, marker='o', zorder=5)
                if key == 'B_native_recorded':
                    q1, q3 = np.quantile(raw, (.25, .75), axis=0)
                    ax.fill_between(t, q1, q3, color=color, alpha=.13)
            for boundary in (32, 64, 96):
                ax.axvline(boundary*DT/60, color='#aaaaaa', alpha=.35, lw=.8, ls='--')
            ax.axhline(0, color='#555555', lw=.7)
            direction = '开大应冷却（负方向）' if dose > 0 else '关小应升温（正方向）'
            ax.set_title(f'{"升" if group=="rise" else "降"}负荷 · 阀{valve} {dose*100:+.0f}pp · {direction}', fontsize=11)
            ax.set_ylabel('原始主温差 ΔT (°C)')
            ax.grid(alpha=.12)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            if row == 3:
                ax.set_xlabel('动作后时间（分钟）')
    fig.suptitle('谷底是不是响应幅度？先分清原始符号、窗口差异与递推边界', fontsize=15)
    fig.text(.5, .018, '圆点：均值曲线的期望方向极值；蓝色阴影：B连续窗口间四分位区间（非置信区间）。\n灰虚线：32步分块边界；最大观察21.33分钟。全部为模型预测差值，不是实测干预曲线。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .97))
    fig.savefig(output/'nadir.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=RESULTS/'ssm_dynamic_20260922/dynamic_responses.npz')
    parser.add_argument('--output', type=Path, default=RESULTS/'response_spec_20260922')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, comparisons = [], []
    with np.load(args.input) as saved:
        assert all(saved[key].shape == (28, 32, 128, 5) for key in MODELS)
        for i, case in enumerate(saved['case_ids']):
            if not str(case).startswith('step_t0_'):
                continue
            valve, dose = int(str(case).split('_')[2][1:]), float(str(case).split('_')[3])
            for group in ('rise', 'fall'):
                mask = saved['case_labels'][i] == group
                for model in MODELS:
                    full = saved[model][i]
                    valid = mask & np.isfinite(full).all(axis=(1, 2))
                    for channel, label in enumerate(CHANNELS):
                        summary = summarize(full[valid, :, channel], np.sign(dose), saved['positions'][valid])
                        rows.append(dict(model=model, group=group, case=str(case), valve=valve,
                                         dose_pp=dose*100, channel=label, excluded=int((mask & ~valid).sum()),
                                         **summary))
                comparisons.extend(paired_comparisons(saved, i, group, np.sign(dose)))
        payload = dict(schema_version=1, source=str(args.input.resolve()),
                       source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                       no_training=True, dt_s=DT, outputs=128, observation_s=[10, 1280],
                       definitions=dict(raw='candidate minus recorded valve prediction, degrees C',
                                        signed='sign(dose) times raw; desired-direction reference is signed < 0',
                                        desired_peak='max(0, -min_t(signed)); not absolute peak',
                                        desired_area='sum_t(max(-signed,0))*10, right endpoint degrees C seconds',
                                        reverse_area='sum_t(max(signed,0))*10, right endpoint degrees C seconds',
                                        recovery='(terminal_signed - signed_nadir)/(-signed_nadir), only nadir < -1e-4 C',
                                        distributions='Across saved valid origins; std is population descriptive std; not CI',
                                        plateau='Not assessed; terminal/last60s is not steady-state gain'),
                       rows=rows, paired_comparisons_main=comparisons,
                       d_block_native_max_difference=float(np.nanmax(np.abs(saved['D_block_recorded']-saved['D_native_recorded']))))
        (args.output/'nadir_metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        write_summary(payload, args.output)
        plot(saved, args.output)
    print(f'Wrote {len(rows)} channel summaries, {len(comparisons)} paired comparisons to {args.output}')


if __name__ == '__main__':
    main()
