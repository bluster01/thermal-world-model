"""Decompose the saved protected response into carrier and readout-gain changes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .analyze_nadir import RESULTS, CHANNELS, distribution, summarize
from .dynamic_response import load_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dists(values):
    return distribution(np.asarray(values).reshape(-1))


def plot(curves, output, channels=CHANNELS, filename='path_response.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig, axes=plt.subplots(len(channels),2,figsize=(13,3*len(channels)+.5),sharex=True)
    t=(np.arange(512)+1)/6
    for row,channel in enumerate(channels):
        channel_index=CHANNELS.index(channel)
        for col,valve in enumerate((1,2)):
            ax=axes[row,col]
            for name,color in [('B_native','#0072B2'),('D_native','#009E73')]:
                x=curves[name][col,:,:,channel_index]
                mean=x.mean(axis=0)
                ax.plot(t,mean,color=color,lw=1.6,label=name)
                ax.fill_between(t,*np.quantile(x,(.25,.75),axis=0),color=color,alpha=.10)
                k=mean.argmin()
                ax.scatter(t[k],mean[k],color=color,s=24)
            ax.axhline(0,color='#777777',lw=.7)
            ax.axvline(128/6,color='#777777',ls='--',lw=.7)
            label={'T4':'T4：二级喷水后温度','main':'T5：末级出口主温'}.get(channel,channel)
            ax.set_title(f'阀{valve} +3pp → {label}',fontsize=11)
            ax.set_ylabel('预测差 ΔT (°C)')
            ax.grid(alpha=.15)
            if row==0: ax.legend(frameon=False,fontsize=9)
            if row==len(channels)-1: ax.set_xlabel('动作后分钟')
    fig.suptitle('沿温度测点定位：局部喷水响应与末级出口响应',fontsize=15)
    fig.text(.5,.015,'同16起点（原H128升/降负荷各8），未来记录边界；无训练。曲线均值，阴影为窗口间四分位区间，非置信区间。\n灰虚线为H128观察终点。两模型均为连续推演；原始正负温差未翻转。模型差值不是实测因果响应。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065 if len(channels)==2 else .045,1,.94 if len(channels)==2 else .97))
    fig.savefig(output/filename,dpi=160)
    plt.close(fig)


def report(payload,output):
    lines=['# D 的载体与温度读出：模型内部路径定位', '',
           '同一D权重，无训练，16个既有起点，记录轨迹的单阀+3pp，记录未来边界，H512/10秒。',
           '六条路径为 v1→T2/T3/T4/main、v2→T4/main。候选动作可能改变全部路径的载体或增益，全部保存。', '',
           '模型中每条路径响应 R = −K·c。配对变化精确分解为：', '',
           '`ΔR = −K_base·(c_candidate−c_base) − (K_candidate−K_base)·c_candidate`', '',
           '前项是固定原增益时载体改变的贡献，后项是增益调制的贡献。它是模型内部代数分解，不是物理因果归因。',
           '`K_C_per_pp = K_C_per_normalized_action × 0.01 / scale_u[该路径的阀]`；这里只换算读出系数，不把它叫总导数。',
           '载体比值 `Δcarrier / (0.03 / scale_u[本次扰动的阀])`。不同于路径所属阀时，该比值描述交叉调制。', '',
           '## 最后60秒，全部16起点', '',
           '|扰动|路径|Δ载体/输入Δ均值|基准K °C/pp|候选K °C/pp|载体贡献°C|增益贡献°C|总路径差°C|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in payload['path_rows']:
        if r['group']!='all': continue
        m=r['tail60_window_mean_distributions']
        lines.append(f"|v{r['perturbed_valve']}+3pp|{r['path']}|{m['delta_carrier_over_input_delta']['mean']:.4f}|{m['base_gain_C_per_pp']['mean']:.5f}|{m['candidate_gain_C_per_pp']['mean']:.5f}|{m['carrier_contribution_C']['mean']:.5f}|{m['gain_contribution_C']['mean']:.5f}|{m['total_path_delta_C']['mean']:.5f}|")
    lines+=['','## 两模型各温度测点的均值曲线极值（全部16起点）','','|扰动|模型|通道|最低ΔT°C|最高ΔT°C|末60秒ΔT°C|10秒ΔT°C|','|---|---|---|---:|---:|---:|---:|']
    for r in payload['channel_rows']:
        if r['group']!='all': continue
        m=r['mean_curve']
        curve=m['raw_curve_C']
        lines.append(f"|v{r['perturbed_valve']}+3pp|{r['model']}|{r['channel']}|{min(curve):.4f}|{max(curve):.4f}|{m['raw_tail60_mean_C']:.4f}|{curve[0]:.4f}|")
    lines+=['','## 校验与解释范围','',
            f"- 路径响应精确分解最大误差：{payload['checks']['path_decomposition_max_abs_C']:.3g}°C。",
            f"- 六路径汇总与core实际五温度响应差最大误差：{payload['checks']['path_sum_vs_core_response_max_abs_C']:.3g}°C。",
            f"- 与上一轮H512的完整D候选−基准预测差最大误差：{payload['checks']['path_sum_vs_saved_D_total_max_abs_C']:.3g}°C。",
            '- gain是每条路径独立学习的幅度；单位DC载体抵达下游并不强制主温与喷后温度有同样幅度。',
            '- 载体比值接近1只能说明该模型内部动作信号已传到该路径，不能据此说物理传热已经识别。',
            '- 较弱主温读出可以是历史闭环数据下的条件拟合，也可能是不正确的衰减；需要双侧及局部温差的观测依据区分，不能直接放大到SSM峰值。',
            '- rise/fall仍是原H128标签，不代表85分钟始终保持该负荷方向。','']
    (output/'path_summary.md').write_text('\n'.join(lines),encoding='utf-8')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RESULTS/'response_spec_20260922')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    horizon_file=args.output/'horizon_curves.npz'
    saved=np.load(horizon_file)
    positions,labels=saved['positions'],saved['labels']
    evaluation=RESULTS/'focused33_seed11/evaluation_inputs.npz'
    bank=torch.as_tensor(np.load(evaluation)['bank'][positions],dtype=torch.float64)
    history,actions,boundary=bank[:,:64,:13],bank[:,63:575,5:7],bank[:,63:575,7:13]
    model,checkpoint=load_model('D');core=model.response.core
    pv=core.path_valve.cpu().numpy();ps=core.path_stage.cpu().numpy()
    scale_u=core.scale[5:7].cpu().numpy()
    path_names=[f'v{v+1}->{CHANNELS[s]}' for v,s in zip(pv,ps)]
    base=core(history,actions,boundary)
    cb=base['transport_state'][:,:,core.path_valve,core.path_stage].numpy()
    kb=base['path_gain_C_per_normalized_action'].numpy()
    base_gain_pp=kb*(.01/scale_u[pv])
    arrays=dict(positions=positions,labels=labels,path_valve=pv+1,path_stage=ps,path_names=np.asarray(path_names),
                base_carrier=cb,base_gain_C_per_normalized_action=kb,base_gain_C_per_pp=base_gain_pp,
                scale_u=scale_u)
    rows,channel_rows,curves=[],[],{'B_native':[],'D_native':[]}
    checks=dict(path_decomposition_max_abs_C=0.,path_sum_vs_core_response_max_abs_C=0.,path_sum_vs_saved_D_total_max_abs_C=0.)
    for valve in (1,2):
        candidate=actions.clone();candidate[:,:,valve-1]+=.03
        if not ((candidate>=0)&(candidate<=1)).all(): raise ValueError('Expected all16 +3pp valid')
        cc_out=core(history,candidate,boundary)
        cc=cc_out['transport_state'][:,:,core.path_valve,core.path_stage].numpy()
        kc=cc_out['path_gain_C_per_normalized_action'].numpy()
        carrier_term=-kb*(cc-cb)
        gain_term=-(kc-kb)*cc
        total=-kc*cc+kb*cb
        actual=(cc_out['action_response_C']-base['action_response_C']).numpy()
        reconstructed=total@core.path_output.numpy()
        ci=saved['case_ids'].tolist().index(f'step_t0_v{valve}_+0.03')
        previous=saved['response_D_native'][ci]
        checks['path_decomposition_max_abs_C']=max(checks['path_decomposition_max_abs_C'],float(np.abs(total-carrier_term-gain_term).max()))
        checks['path_sum_vs_core_response_max_abs_C']=max(checks['path_sum_vs_core_response_max_abs_C'],float(np.abs(reconstructed-actual).max()))
        checks['path_sum_vs_saved_D_total_max_abs_C']=max(checks['path_sum_vs_saved_D_total_max_abs_C'],float(np.abs(reconstructed-previous).max()))
        parts=dict(candidate_carrier=cc,delta_carrier=cc-cb,
                   delta_carrier_over_input_delta=(cc-cb)/(.03/scale_u[valve-1]),
                   candidate_gain_C_per_normalized_action=kc,candidate_gain_C_per_pp=kc*(.01/scale_u[pv]),
                   carrier_contribution_C=carrier_term,gain_contribution_C=gain_term,total_path_delta_C=total)
        for key,value in parts.items(): arrays[f'v{valve}_{key}']=value
        for group in ('all','rise','fall'):
            good=np.ones(len(positions),bool) if group=='all' else labels==group
            for path in range(6):
                values={**parts,'base_gain_C_per_normalized_action':kb,'base_gain_C_per_pp':base_gain_pp,'base_carrier':cb}
                tail={key:distribution(value[good,-6:,path].mean(axis=1)) for key,value in values.items()}
                sample={str(t):dict(delta_carrier_over_input_delta=float(parts['delta_carrier_over_input_delta'][good,t-1,path].mean()),
                                   base_gain_C_per_pp=float(base_gain_pp[good,t-1,path].mean()),candidate_gain_C_per_pp=float(parts['candidate_gain_C_per_pp'][good,t-1,path].mean()),
                                   total_path_delta_C=float(total[good,t-1,path].mean())) for t in (1,32,128,256,512)}
                rows.append(dict(perturbed_valve=valve,dose_pp=3,group=group,windows=int(good.sum()),path=path_names[path],
                                 path_valve=int(pv[path]+1),path_stage=CHANNELS[ps[path]],
                                 tail60_window_mean_distributions=tail,at_steps=sample,
                                 mean_curve_min_C=float(total[good,:,path].mean(axis=0).min()),
                                 mean_curve_max_C=float(total[good,:,path].mean(axis=0).max()),
                                 signed_positive_fraction=float((total[good,:,path]>1e-4).mean())))
            for name in curves:
                values=saved[f'response_{name}'][ci]
                for channel,label in enumerate(CHANNELS):
                    channel_rows.append(dict(model=name,perturbed_valve=valve,group=group,channel=label,
                                             **summarize(values[good,:,channel],1.,positions[good])))
        for name in curves: curves[name].append(saved[f'response_{name}'][ci])
    if max(checks.values())>1e-8: raise ValueError(f'Exact decomposition mismatch {checks}')
    curves={key:np.stack(value) for key,value in curves.items()}
    payload=dict(no_training=True,dtype='float64',threads=1,H=512,dt_s=10,dose_pp=3,
                 definitions=dict(decomposition='DeltaR = -Kbase*DeltaCarrier - DeltaK*CarrierCandidate',
                                  gain_unit='K_C_per_pp = K_C_per_normalized_action*0.01/scale_u[path_valve]; coefficient, not total derivative',
                                  carrier_ratio='DeltaCarrier/(0.03/scale_u[perturbed_valve]); other-valve ratios express cross modulation',
                                  averaging='Tail60: first average last6 per window, then distribution across windows',
                                  limitation='Algebraic explanation inside model, not physical causal identification'),
                 sources={str(checkpoint):sha(checkpoint),str(evaluation):sha(evaluation),str(horizon_file):sha(horizon_file),str(Path(__file__)):sha(__file__)},
                 positions=positions.tolist(),labels=labels.tolist(),scale_u=scale_u.tolist(),
                 path_rows=rows,channel_rows=channel_rows,checks=checks)
    np.savez_compressed(args.output/'path_transport.npz',**arrays)
    (args.output/'path_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report(payload,args.output);plot(curves,args.output)
    plot(curves,args.output,channels=('T4','main'),filename='path_local_terminal.png')
    print(json.dumps(checks,indent=2));print(f'Wrote {len(rows)} path summaries and {len(channel_rows)} channel summaries')


if __name__=='__main__': main()
