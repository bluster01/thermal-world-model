"""Fixed-weight H512 response and block-size sensitivity on saved rise/fall origins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .analyze_nadir import RESULTS, CHANNELS, summarize
from .dynamic_response import load_model
from .evaluation import forecast, metrics


PROTOCOLS = ('B_native', 'B_block16', 'B_block32', 'B_block64', 'D_native')
CASES = [(v, dose) for v in (1, 2) for dose in (-.03, .03)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(model, history, actions, boundary, protocol):
    if protocol.endswith('native'):
        return forecast(model, history, actions, boundary, mode='native')
    if model.arm != 'B':
        raise ValueError('Movable blocks implemented only for B')
    # Exact Focused.forecast_plan semantics, with only its hard-coded 32 changed:
    # original auxiliary context is fixed; only generated 13-channel history
    # is re-encoded; right-endpoint action/boundary rows accompany predictions.
    block = int(protocol.split('block')[1])
    current, context = history[:, :, :13], model.context(history)
    parts, horizon = [], actions.shape[1]-1
    for start in range(0, horizon, block):
        stop = min(start+block, horizon)
        p = model.background(current, actions[:, start:stop], boundary[:, start:stop], initial_context=context)
        parts.append(p)
        rows = torch.cat((p, actions[:, start+1:stop+1], boundary[:, start+1:stop+1]), -1)
        current = torch.cat((current, rows), 1)[:, -64:]
    return torch.cat(parts, 1)


def case_id(valve, dose):
    return f'step_t0_v{valve}_{dose:+.2f}'


def valid_action(actions):
    return ((actions >= 0) & (actions <= 1)).all((1, 2))


def report(payload, output):
    lines = ['# 固定权重 H512 与重编码间隔检查', '',
             '无训练。B 为同一权重，分别连续推演或每16/32/64步重编码。D连续为参考。',
             '动作：原记录阀门轨迹整体±3pp；未来边界按记录回放；10秒采样，最长512步=85.33分钟。',
             'rise/fall 仍指原起点前128步的实际负荷标签，不代表整个85分钟保持升/降负荷。',
             '每个动作以全H512合法起点为固定人群，H128与H512使用同一人群；另保存原H128合法人群的对照，不混用样本。', '',
             '## 开大3pp的主温差（同一H512合法人群）', '',
             '谷值是均值曲线的最低值，负值表示冷却；逐窗谷值完整分布另见JSON。末端不是稳态。', '',
             '|阀|分组|协议|n|H128谷值°C/时刻s|H512谷值°C/时刻s|H512末端°C|H512每窗冷却面积均值°C·s|',
             '|---|---|---|---:|---|---|---:|---:|']
    lookup = {(r['protocol'], r['valve'], r['dose_pp'], r['group'], r['channel'], r['cohort'], r['horizon']): r for r in payload['response_rows']}
    for valve in (1, 2):
        for group in ('rise', 'fall'):
            for protocol in PROTOCOLS:
                a = lookup[(protocol,valve,3.,group,'main','H512_eligible',128)]
                b = lookup[(protocol,valve,3.,group,'main','H512_eligible',512)]
                am, bm = a['mean_curve'], b['mean_curve']
                lines.append(f"|{valve}|{group}|{protocol}|{b['windows']}|{am['signed_nadir_C']:.4f}/{am['signed_nadir_time_s']:.0f}|{bm['signed_nadir_C']:.4f}/{bm['signed_nadir_time_s']:.0f}|{bm['raw_terminal_C']:.4f}|{b['distribution']['desired_direction_area_Cs']['mean']:.1f}|")
    lines += ['', '## 同一16起点的事实预测MAE', '',
              '阀门、边界均用原记录；各协议完全相同的16个起点。不能把下列小样本替代256窗总体排名。', '',
              '|协议|H32主温°C|H128主温°C|H512主温°C|H512全5温度°C|',
              '|---|---:|---:|---:|---:|']
    for protocol in PROTOCOLS:
        m = payload['factual_metrics'][protocol]['all_16']
        lines.append(f"|{protocol}|{m['H32']['main_mae_C']:.4f}|{m['H128']['main_mae_C']:.4f}|{m['H512']['main_mae_C']:.4f}|{m['H512']['all_mae_C']:.4f}|")
    lines += ['', '## 配对核对与解释边界', '',
              f"- block32 手写复现与现有 forecast 的H512最大绝对差：{payload['checks']['block32_vs_existing_forecast_H512_max_abs_C']:.3g}°C。",
              '- 旧H128响应数组重放、同协议H512前缀与独立H128的差值全部在JSON checks中记录；超过1e-8°C会使脚本失败。',
              '- 改变分块间隔会改变推演算法和反馈历史分布；谷底/恢复变化可以确认边界敏感性，不能仅凭这一项判定哪条曲线是真实物理响应。',
              '- H512全程不越界的共同人群按动作分别固定；+3pp均16窗，-3pp有效窗更少。不同动作的差异不可自动解释为非线性。',
              '- 本结果是模型相对原计划的温差，缺少真实反事实曲线。谷底更深、反向更少、事实MAE更低是不同维度，不相互替代。', '']
    (output/'horizon_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(arrays, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    colors = ('#0072B2','#E69F00','#56B4E9','#CC79A7','#009E73')
    t = (np.arange(512)+1)/6
    fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True)
    for row, group in enumerate(('rise','fall')):
        for col, valve in enumerate((1,2)):
            ax=axes[row,col]
            ci=CASES.index((valve,.03))
            good=(arrays['labels']==group)&arrays['valid_H512'][ci]
            for protocol,color in zip(PROTOCOLS,colors):
                x=arrays[f'response_{protocol}'][ci,good,:,4].mean(axis=0)
                ax.plot(t,x,color=color,label=protocol,lw=1.7)
                k=x.argmin()
                ax.scatter(t[k],x[k],color=color,s=22,zorder=4)
            ax.axvline(128/6,color='#777777',ls='--',lw=.9,label='原H128观察终点')
            ax.axhline(0,color='#777777',lw=.7)
            ax.set(title=f'原H128{"升" if group=="rise" else "降"}负荷组 · 阀{valve} +3pp · n={good.sum()}',ylabel='主温预测差 ΔT (°C)',xlabel='动作后分钟')
            ax.legend(fontsize=8,ncol=2,frameon=False)
            ax.grid(alpha=.15)
    fig.suptitle('同一权重与同一起点：延长时域、改变重编码间隔后的阀门响应',fontsize=14)
    fig.text(.5,.018,'只改变推演协议，无训练；记录边界、记录阀门轨迹 +3pp。圆点为均值曲线谷值，不是实测物理峰值。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(output/'horizon_response.png',dpi=180)
    plt.close(fig)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RESULTS/'response_spec_20260922')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    source=RESULTS/'ssm_dynamic_20260922/dynamic_responses.npz'
    evaluation=RESULTS/'focused33_seed11/evaluation_inputs.npz'
    old=np.load(source)
    selected=np.isin(old['labels'],['rise','fall'])
    positions,labels=old['positions'][selected],old['labels'][selected]
    bank=torch.as_tensor(np.load(evaluation)['bank'][positions],dtype=torch.float64)
    history,actions,boundary,truth=bank[:,:64],bank[:,63:576,5:7],bank[:,63:576,7:13],bank[:,64:576,:5]
    candidates=[]
    for valve,dose in CASES:
        c=actions.clone(); c[:,:,valve-1]+=dose; candidates.append(c)
    valid512=torch.stack([valid_action(c) for c in candidates])
    valid128=torch.stack([valid_action(c[:,:129]) for c in candidates])
    arrays=dict(positions=positions,labels=labels,case_ids=np.asarray([case_id(*c) for c in CASES]),
                valid_H512=valid512.numpy(),valid_H128=valid128.numpy(),truth=truth.numpy())
    rows,factual,checks,checkpoints=[],{},{},{}
    b,bpath=load_model('B'); d,dpath=load_model('D')
    checkpoints={'B':dict(path=str(bpath),sha256=sha(bpath)),'D':dict(path=str(dpath),sha256=sha(dpath))}
    for protocol in PROTOCOLS:
        model=d if protocol.startswith('D') else b
        base=predict(model,history,actions,boundary,protocol)
        base128=predict(model,history,actions[:,:129],boundary[:,:129],protocol)
        checks[f'{protocol}_factual_prefix_max_abs_C']=float((base[:,:128]-base128).abs().max())
        if protocol=='B_block32':
            checks['block32_vs_existing_forecast_H512_max_abs_C']=float((base-forecast(model,history,actions,boundary,mode='block')).abs().max())
        factual[protocol]={'all_16':metrics(base.numpy(),truth.numpy(),history[:,-1].numpy())}
        for group in ('rise','fall'):
            mask=labels==group
            factual[protocol][group]=metrics(base.numpy()[mask],truth.numpy()[mask],history[:,-1].numpy()[mask])
        arrays[f'factual_{protocol}']=base.numpy()
        long_curves,short_curves=[],[]
        for ci,((valve,dose),candidate) in enumerate(zip(CASES,candidates)):
            safe512=torch.where(valid512[ci,:,None,None],candidate,actions)
            safe128=torch.where(valid128[ci,:,None,None],candidate[:,:129],actions[:,:129])
            delta=(predict(model,history,safe512,boundary,protocol)-base).numpy()
            delta128=(predict(model,history,safe128,boundary[:,:129],protocol)-base128).numpy()
            common=valid512[ci].numpy()
            checks[f'{protocol}_{case_id(valve,dose)}_prefix_max_abs_C']=float(np.max(np.abs(delta[common,:128]-delta128[common])))
            delta[~common]=np.nan
            delta128[~valid128[ci].numpy()]=np.nan
            if protocol in ('B_native','B_block32','D_native'):
                oldkey={'B_native':'B_native_recorded','B_block32':'B_block_recorded','D_native':'D_native_recorded'}[protocol]
                oi=old['case_ids'].tolist().index(case_id(valve,dose))
                previous=old[oldkey][oi,selected]
                if not np.array_equal(np.isfinite(previous),np.isfinite(delta128)):
                    raise ValueError('Original H128 eligibility mismatch')
                checks[f'{protocol}_{case_id(valve,dose)}_old_H128_max_abs_C']=float(np.nanmax(np.abs(previous-delta128)))
            long_curves.append(delta); short_curves.append(delta128)
            for group in ('rise','fall'):
                for cohort,horizon,values,eligible in [('H512_eligible',128,delta[:,:128],valid512[ci].numpy()),
                                                       ('H512_eligible',512,delta,valid512[ci].numpy()),
                                                       ('original_H128_eligible',128,delta128,valid128[ci].numpy())]:
                    good=(labels==group)&eligible
                    for channel,label in enumerate(CHANNELS):
                        summary=summarize(values[good,:,channel],np.sign(dose),positions[good])
                        rows.append(dict(protocol=protocol,valve=valve,dose_pp=dose*100,group=group,
                                         channel=label,cohort=cohort,horizon=horizon,**summary))
        arrays[f'response_{protocol}']=np.stack(long_curves)
        arrays[f'original_H128_response_{protocol}']=np.stack(short_curves)
        print(f'Completed {protocol}: 4 actions x 16 origins x H512 + independent H128',flush=True)
    if max(checks.values())>1e-8:
        raise ValueError(f'Prefix/original replay check failed: {checks}')
    payload=dict(no_training=True,dtype='float64',device='cpu',torch_threads=1,dt_s=10,horizon=512,
                 sources={str(source):sha(source),str(evaluation):sha(evaluation),str(Path(__file__)):sha(__file__)},
                 checkpoints=checkpoints,positions=positions.tolist(),labels=labels.tolist(),
                 label_scope='Original first H128 load regime, not the whole H512 future',
                 cohort_definition='Per-action H512-legal common origins across all protocols and H128/H512; original-H128 legal cohort separately reproduced',
                 limitations='Model differences only; changing block length changes algorithm, not a causal proof of response fidelity',
                 response_rows=rows,factual_metrics=factual,checks=checks)
    np.savez_compressed(args.output/'horizon_curves.npz',**arrays)
    (args.output/'horizon_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report(payload,args.output)
    plot(arrays,args.output)
    print(f'Wrote {len(rows)} response rows; maximum replay/prefix error {max(checks.values()):.3g} C',flush=True)


if __name__=='__main__':
    main()
