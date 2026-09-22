"""Fixed-checkpoint valve responses during observed load changes; no fitting."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .data import sha256
from .dynamic_factual import classify_load, select_probe_windows
from .evaluation import forecast
from .focused_models import Focused
from .physical_models import Physical
from .run import HERE, save_json

ROOT = HERE.parents[1]/'results/action_predictor_bench_20260919'


def cases(horizon=256):
    result = []
    for shape, onsets in [('step', (0, 64, 128)), ('ramp', (0, 64, 128)), ('pulse', (64,))]:
        for onset in onsets:
            elapsed = np.arange(horizon+1)-onset
            profile = (elapsed >= 0).astype(float)
            if shape == 'ramp': profile *= np.clip(elapsed/16, 0, 1)
            if shape == 'pulse': profile *= elapsed < 12
            for valve in (0, 1):
                for dose in (-.03, .03):
                    delta = np.zeros((horizon+1, 2))
                    delta[:, valve] = profile*dose
                    result.append(dict(id=f'{shape}_t{onset}_v{valve+1}_{dose:+.2f}',
                        shape=shape, onset=onset, valve=valve, dose=dose, delta=delta))
    return result


def response_stats(delta, case, valid, selected):
    good = valid & selected
    result = dict(windows=int(selected.sum()), eligible=int(good.sum()), excluded=int((selected & ~valid).sum()))
    if not good.any(): return result
    x = delta[good]
    if not np.isfinite(x).all(): raise ValueError('Nonfinite response')
    onset, valve = case['onset'], case['valve']
    post = x[:, onset:onset+128]
    main = post[:, :, 4]
    upstream = [0] if valve == 0 else [0, 1, 2]
    reachable = [1, 2, 3, 4] if valve == 0 else [3, 4]
    # The diagnostic reference is cooling for increased valve opening. It is
    # not a label from a plant intervention or a proof of global monotonicity.
    signed = post*np.sign(case['dose'])
    return dict(**result, main_wrong_fraction=float((signed[:, :, 4] > 1e-4).mean()),
        reachable_wrong_fraction=float((signed[:, :, reachable] > 1e-4).mean()),
        terminal_wrong_window_fraction=float((signed[:, -1, 4] > 1e-4).mean()),
        terminal_main_C=float(main[:, -1].mean()), peak_abs_main_C=float(np.abs(main).max(1).mean()),
        near_zero_window_fraction=float((np.abs(main).max(1) < 1e-4).mean()),
        upstream_max_C=float(np.abs(x[:, :, upstream]).max()),
        pre_onset_max_C=float(np.abs(x[:, :onset]).max()) if onset else 0.,
        main_at_10_60_180_640_1280s_C=main[:, [0, 5, 17, 63, 127]].mean(0).tolist())


def load_model(name):
    source = ROOT/('physical33_seed11_to90' if name == 'P0' else 'focused33_seed11')/'seed11'/f'{name}_short/best.pt'
    state = torch.load(source, weights_only=True, map_location='cpu')['model']
    cls = Physical if name == 'P0' else Focused
    model = cls(state['mean'], state['scale'], state['aux_mean'], state['aux_scale'], name)
    model.load_state_dict(state)
    return model.double().eval(), source


@torch.no_grad()
def evaluate(output, windows_per_group=8):
    context = np.load(output/'load_context.npz')
    saved = np.load(ROOT/'focused33_seed11/evaluation_inputs.npz')
    if not np.array_equal(saved['times'], context['times']): raise ValueError('Load/evaluation time mismatch')
    labels = classify_load(context['load_MW'])
    grouped = select_probe_windows(labels, n=windows_per_group)
    positions = np.asarray(sorted({i for ids in grouped.values() for i in ids}), dtype=int)
    groups = labels[positions]
    bank = torch.as_tensor(saved['bank'][positions], dtype=torch.float64)
    history, actions, boundary = bank[:, :64], bank[:, 63:320, 5:7], bank[:, 63:320, 7:13]
    held_actions, held_boundary = actions[:, :1].expand_as(actions), boundary[:, :1].expand_as(boundary)
    plans, rows, curves, factorial, checkpoints = cases(), [], {}, {}, {}
    case_labels = np.stack([classify_load(context['load_MW'][positions,c['onset']:]) for c in plans])
    for name in ('A', 'B', 'D', 'P0'):
        model, source = load_model(name)
        checkpoints[name] = dict(path=str(source), sha256=sha256(source))
        for mode in (('native',) if name == 'P0' else ('block', 'native')):
            # Four paired forecasts reveal sensitivity to the recorded valve
            # plan, the joint external-boundary trajectory, and their interaction.
            base = forecast(model, history, actions, boundary, mode=mode)
            fixed_u = forecast(model, history, held_actions, boundary, mode=mode)
            fixed_d = forecast(model, history, actions, held_boundary, mode=mode)
            fixed_both = forecast(model, history, held_actions, held_boundary, mode=mode)
            for label, value in [('recorded', base), ('held_valves', fixed_u),
                                  ('held_boundary', fixed_d), ('held_both', fixed_both)]:
                factorial[f'{name}_{mode}_{label}'] = value.numpy()
            # All models see the same changing boundary. B also gets a held-
            # boundary ablation to compare with the earlier static probes.
            for protocol in (('recorded', 'held') if name == 'B' else ('recorded',)):
                d, reference = (boundary, base) if protocol == 'recorded' else (held_boundary, fixed_d)
                responses = []
                for index, case in enumerate(plans):
                    end = case['onset']+128
                    factual_actions = actions[:, :end+1]
                    candidate = factual_actions + torch.as_tensor(case['delta'][:end+1])[None]
                    valid = ((candidate >= 0) & (candidate <= 1)).all((1, 2))
                    safe = torch.where(valid[:, None, None], candidate, factual_actions)
                    delta = (forecast(model, history, safe, d[:, :end+1], mode=mode)-reference[:, :end]).numpy()
                    for group in ('all', *grouped):
                        selected = np.ones(len(groups), bool) if group == 'all' else case_labels[index] == group
                        row = {key:value for key,value in case.items() if key != 'delta'}
                        row.update(model=name, mode=mode, boundary=protocol, group=group,
                                   **response_stats(delta, case, valid.numpy(), selected))
                        rows.append(row)
                    post = delta[:, case['onset']:case['onset']+128].copy()
                    post[~valid.numpy()] = np.nan
                    responses.append(post)
                curves[f'{name}_{mode}_{protocol}'] = np.stack(responses)
                print(f'Completed {name}/{mode}/{protocol}: {len(plans)} plans x {len(positions)} origins', flush=True)
    np.savez_compressed(output/'dynamic_responses.npz', **curves, positions=positions, labels=groups,case_labels=case_labels,
                        case_ids=np.asarray([case['id'] for case in plans]))
    np.savez_compressed(output/'dynamic_factorial.npz', **factorial, positions=positions, labels=groups)
    save_json(output/'dynamic_response_metrics.json', rows)
    save_json(output/'dynamic_response_config.json', dict(checkpoints=checkpoints, windows=grouped,
        load_sha256=sha256(output/'load_context.npz'), evaluation_sha256=sha256(ROOT/'focused33_seed11/evaluation_inputs.npz'),
        source_sha256={p.name:sha256(p) for p in (Path(__file__), HERE/'dynamic_factual.py', HERE/'evaluation.py')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol='Add +/-3pp step/ramp/pulse to RECORDED future valve plans; keep the same historical state and boundary in each pair',
        observation='128 steps after action, dt10s; first affected output is 10s after onset',
        response_grouping='Reclassify actual MW change over each action-onset to onset+128 interval; original 32 origins selected by origin-time groups',
        future_load_usage='Only post-hoc regime labels. MW is NOT an input in future plans.',
        boundary_ablation='Joint observed coal/steamflow/pressure/temperature trajectory, not a calibrated MW intervention',
        no_training=True, cases=[{k:v for k,v in c.items() if k != 'delta'} for c in plans]))
    summarize(rows, factorial, groups, output)
    return curves, groups, plans


def summarize(rows, factorial, groups, output):
    lines = ['# 升降负荷期间的阀门响应（固定权重）', '',
        '真实未来工况回放下，在原阀门轨迹上加减3个百分点。误差方向参考为开大降温；它不是实测干预误差。', '',
        '| 模型/递推 | 分组 | 有效情景数 | 主温反号比例 | 可达温度反号比例 | 主温峰值均值°C |',
        '|---|---|---:|---:|---:|---:|']
    for name, mode in [('A','block'),('A','native'),('B','block'),('B','native'),('D','block'),('D','native'),('P0','native')]:
        for group in ('rise','fall','vary','steady'):
            subset = [r for r in rows if (r['model'],r['mode'],r['boundary'],r['group']) == (name,mode,'recorded',group) and r['eligible']]
            if subset:
                values = [np.mean([r[key] for r in subset]) for key in ('main_wrong_fraction','reachable_wrong_fraction','peak_abs_main_C')]
                lines.append(f'| {name}/{mode} | {group} | {len(subset)} | {values[0]:.1%} | {values[1]:.1%} | {values[2]:.3f} |')
    lines += ['', '上述比例先按单情景的有效窗口×时间（或×可达温度）统计，再对情景等权平均；不同情景越界排除数量见JSON。',
              'P0在block/native采用相同连续状态，只算一遍。未来MW没有注入模型；每个情景按动作起效后21.3分钟真实负荷变化重新分组，晚期动作不会沿用起点负荷方向标签。']
    (output/'dynamic_response_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    effects = []
    for key, prediction in factorial.items():
        if not key.endswith('_recorded'): continue
        prefix = key[:-len('_recorded')]
        pu, pd, p00 = [factorial[prefix+'_'+suffix] for suffix in ('held_valves','held_boundary','held_both')]
        for group in np.unique(groups):
            selected = groups == group
            effects.append(dict(model_mode=prefix, group=str(group), windows=int(selected.sum()),
                observed_valve_effect_mean_abs_main_C=float(np.abs(prediction[selected,:128,4]-pu[selected,:128,4]).mean()),
                joint_boundary_effect_with_held_valves_mean_abs_main_C=float(np.abs(pu[selected,:128,4]-p00[selected,:128,4]).mean()),
                interaction_mean_abs_main_C=float(np.abs(prediction[selected,:128,4]-pu[selected,:128,4]-pd[selected,:128,4]+p00[selected,:128,4]).mean())))
    save_json(output/'dynamic_factorial_metrics.json',effects)


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    data = np.load(output/'dynamic_responses.npz')
    ids = data['case_ids'].tolist()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    interactive = make_subplots(rows=2,cols=2,subplot_titles=['升负荷 · 阀1','升负荷 · 阀2','降负荷 · 阀1','降负荷 · 阀2'])
    palette = [('#0072B2','B_block_recorded','SSM B 分块'),('#56B4E9','B_native_recorded','SSM B 连续'),
               ('#009E73','D_block_recorded','D 响应保护'),('#D55E00','P0_native_recorded','P0 物理状态')]
    t = (np.arange(128)+1)*10/60
    for row, group in enumerate(('rise','fall')):
        for col, valve in enumerate((1,2)):
            index = ids.index(f'step_t0_v{valve}_+0.03')
            labels = data['case_labels'][index]
            axis = axes[row,col]
            for color,key,label in palette:
                values = data[key][index,labels==group,:,4]
                valid = np.isfinite(values).all(1)
                values = values[valid]
                if not len(values): continue
                mean = values.mean(0)
                axis.plot(t,mean,color=color,label=f'{label} (n={len(values)})',lw=1.8)
                if key=='B_block_recorded':
                    axis.fill_between(t,np.quantile(values,.25,axis=0),np.quantile(values,.75,axis=0),color=color,alpha=.12)
                interactive.add_trace(go.Scatter(x=t,y=mean,name=f'{label} / {group} v{valve}',line=dict(color=color),
                    hovertemplate='%{x:.2f} min<br>ΔT=%{y:.3f}°C<extra>%{fullData.name}</extra>'),row=row+1,col=col+1)
            axis.axhline(0,color='#777777',lw=.8)
            axis.set(title=f'{"升" if group=="rise" else "降"}负荷 · 阀{valve} +3个百分点',ylabel='主温响应 ΔT (°C)',xlabel='动作后时间 (分钟)')
            axis.legend(fontsize=8,frameon=False)
            axis.grid(alpha=.15)
            interactive.update_xaxes(title_text='动作后分钟',row=row+1,col=col+1)
            interactive.update_yaxes(title_text='主温响应 ΔT (°C)',zeroline=True,row=row+1,col=col+1)
    fig.suptitle('相同升降负荷工况下，单独开大阀门：预测差值应如何变化？',fontsize=14)
    fig.text(.5,.015,'记录中的未来工况回放；阀门轨迹整体 +3pp。曲线为有效起点均值，B 分块阴影为起点间四分位区间，不是置信区间。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(output/'dynamic_response.png',dpi=180)
    fig.savefig(output/'dynamic_response.svg')
    plt.close(fig)
    interactive.update_layout(height=780,title='阀门 +3pp：升降负荷背景下的模型响应（预测差值，非实测干预）',template='plotly_white')
    interactive.write_html(output/'dynamic_response.html',include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'ssm_dynamic_20260922')
    parser.add_argument('--windows-per-group',type=int,default=8)
    parser.add_argument('--plot-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not args.plot_only: evaluate(args.output,args.windows_per_group)
    plots(args.output)


if __name__=='__main__': main()
