"""Review all three bilateral seeds from saved arrays; no fitting or reselection."""
import json
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .bilateral import HERE, OUT_DEFAULT, PARENT, load_data
from .bilateral_models import make_model
from .bilateral_evaluation import load_groups
from .data import sha256
from .review_bilateral import ROOT, DEPLOY, COLORS, LOCAL, MAIN, read, factual, analyze_responses
from .run import save_json

ARMS = ['S0_A','S0_B','S1','S2','S3','R3','P3']
NAMES = DEPLOY[2:]
ZH = ['A侧基础','B侧基础','双侧历史','四阀自由','双侧自由','双侧输运','双侧保护']
OUT = ROOT/'bilateral_three_seed_review_20260923'


def run_for(seed):
    return ROOT/('bilateral33_seed11' if seed == 11 else 'bilateral33_seed23_37')


def validate():
    data,_ = load_data(OUT_DEFAULT)
    _, masks, _, _ = load_groups(data)
    fits=[]; replay=[]; errors=[]; checked=0
    torch.set_num_threads(1)
    for seed in (11,23,37):
        run=run_for(seed); config=read(run/'config.json');state=read(run/'state.json')
        assert state['status']=='complete' and not state['failures'] and not state['not_confirmed_plateau']
        assert config['data_sha256']==sha256(OUT_DEFAULT)
        assert all(sha256(HERE/name)==digest for name,digest in config['source_sha256'].items())
        assert config['training_windows']==20371 and not config['smoke']
        for arm in ARMS:
            folder=run/f'fits/seed{seed}'/arm; fit=read(folder/'fit.json')
            logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
            assert fit['reached_validation_plateau'] and fit['final_lr']==.0001
            assert len(logs)==fit['epochs_run'] and fit['updates']==160*len(logs)
            assert logs[-1]['stale_at_current_lr']>=6
            last=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
            assert last['epoch']==fit['epochs_run'] and last['best_epochs']==fit['best_epochs']
            for choice, epoch in fit['best_epochs'].items():
                p=folder/f'best_{choice}.pt'; ck=torch.load(p,weights_only=True,map_location='cpu')
                result=read(run/f'seed{seed}'/f'{arm}_{choice}'/'result.json')
                assert ck['epoch']==epoch==result['checkpoint_epoch'] and sha256(p)==result['checkpoint_sha256']
            assert fit['seed']==seed
            fits.append(fit)
        for folder in sorted((run/f'seed{seed}').iterdir()):
            if not (folder/'result.json').exists(): continue
            result=read(folder/'result.json')
            with np.load(folder/'predictions.npz') as z:
                assert np.array_equal(z['truth'],data['evaluation'][:,64:,:10])
                assert np.array_equal(z['times'],data['evaluation_time'])
                outputs=z['output_indices'].tolist()
                for mode,sides in result['factual'].items():
                    assert np.isfinite(z[mode]).all()
                    for side,item in sides.items():
                        if side not in ('A','B') or 'H32' not in item:continue
                        channel=4 if side=='A' else 9
                        error=np.abs(z[mode][:,:,outputs.index(channel)].astype(float)-z['truth'][:,:,channel])
                        for group, mask in {'all':np.ones(256,bool),**masks}.items():
                            target=item if group=='all' else item['groups'][group]
                            assert target['windows']==int(mask.sum())
                            for h in (32,128,512):
                                errors.append(abs(float(error[mask,:h].mean())-target[f'H{h}']['main_mae_C']))
                        errors.append(abs(float(error[:,128:].mean())-item['tail_129_512']['main_mae_C']))
                checked+=1
        # Small direct forward replay for each deployed checkpoint, both modes.
        ids=np.array([0,127,255]); bank=torch.tensor(data['evaluation'][ids])
        h,u,d=bank[:,:64],bank[:,63:,10:14],bank[:,63:,14:21]
        for arm,name in zip(ARMS,NAMES):
            model=make_model(arm,data['mean'],data['scale'],seed,PARENT).eval()
            choice=name[len(arm)+1:]
            ck=torch.load(run/f'fits/seed{seed}'/arm/f'best_{choice}.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(ck['model'])
            with np.load(run/f'seed{seed}'/name/'predictions.npz') as z,torch.no_grad():
                for mode in ('native','block_joint_refresh' if len(model.output_indices)==10 else 'block_context_fixed'):
                    difference=np.abs(model.forecast(h,u,d,mode).numpy()-z[mode][ids])
                    replay.append(dict(seed=seed,model=arm,mode=mode,max_abs_C=float(difference.max())))
    assert checked==63 and max(errors)<1e-10
    assert max(r['max_abs_C'] for r in replay)<.002
    return dict(return_commit='5696e5f',local_torch=str(torch.__version__),returned_torch=config['torch_version'],
                fits=fits,prediction_rows_checked=checked,
                recomputed_metric_count=len(errors),maximum_mae_difference_C=max(errors),
                replay=replay,data_sha256=sha256(OUT_DEFAULT),blind_test=False)


def summarize(rows, keys, metrics):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,items in groups.items():
        assert sorted(r['seed'] for r in items)==[11,23,37]
        r=dict(zip(keys,key));r['seeds']=[11,23,37]
        for metric in metrics:
            a=np.array([next(x[metric] for x in items if x['seed']==s) for s in (11,23,37)])
            r[metric]=dict(mean=float(a.mean()),std=float(a.std(ddof=1)),values=a.tolist())
        result.append(r)
    return result


def plots(curves,stats):
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next(n for n in ('Microsoft YaHei','Noto Sans CJK SC','SimHei') if n in available)
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(17,8.5),sharex=True)
    handles={}
    for name,label,color in zip(NAMES,ZH,COLORS[2:]):
        for v in range(4):
            key=(11,name,v,'native')
            if key not in curves:continue
            values=[]
            for seed in (11,23,37):
                raw,outputs=curves[(seed,name,v,'native')]
                assert len(raw)==32
                values.append(np.stack([(raw[:,:,outputs.index(LOCAL[v])]-raw[:,:,outputs.index(LOCAL[v]-1)]).mean(0),
                                         raw[:,:,outputs.index(MAIN[v])].mean(0)]))
            values=np.stack(values);t=(np.arange(512)+1)/6
            for row in range(2):
                mean=values[:,row].mean(0);sd=values[:,row].std(0,ddof=1)
                ax=axes[row,v];line,=ax.plot(t,mean,color=color,lw=1.5,label=label)
                ax.fill_between(t,mean-sd,mean+sd,color=color,alpha=.10,linewidth=0);handles[label]=line
    for v,title in enumerate(['一级A阀 → A侧','一级B阀 → B侧','二级A阀 → B侧（交叉）','二级B阀 → A侧（交叉）']):
        axes[0,v].set_title(title+'\n局部：喷水后减去喷水前')
        axes[1,v].set_title('末级出口汽温')
        for ax in axes[:,v]:
            ax.axhline(0,c='gray',lw=.6);ax.grid(alpha=.12);ax.set_xlabel('动作后时间（分钟）')
    axes[0,0].set_ylabel('局部温差的预测变化（℃）');axes[1,0].set_ylabel('末级汽温的预测变化（℃）')
    fig.suptitle('四阀局部与末级响应：三次独立训练',fontsize=19)
    fig.text(.5,.925,'开大3个百分点｜相同32个起点｜其他阀位和边界固定｜连续推演',ha='center')
    fig.legend(handles.values(),handles.keys(),loc='lower center',bbox_to_anchor=(.5,.05),ncol=4,frameon=False)
    fig.text(.5,.02,'实线：三种子均值；阴影：种子间±1个样本标准差。每个种子先平均32个起点；不是实测干预或置信区间。',ha='center')
    fig.subplots_adjust(left=.065,right=.99,top=.83,bottom=.23,wspace=.27,hspace=.4)
    for ext in ('png','pdf'):fig.savefig(OUT/f'response_paths_zh.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(side,mode) in zip(axes.flat,[('A','native'),('B','native'),('A','rollout'),('B','rollout')]):
        selected=[r for r in stats if r['side']==side and r['group']=='rise' and
                  r['mode']==('native' if mode=='native' else ('block_joint_refresh' if '_AB_' in r['model'] else 'block_context_fixed'))]
        for j,h in enumerate(('H128','H512')):
            ax.errorbar(np.arange(len(selected))+.08*(j-.5),[r[h]['mean'] for r in selected],
                        yerr=[r[h]['std'] for r in selected],fmt='o',capsize=3,label={'H128':'前21.3分钟','H512':'前85.3分钟'}[h])
        ax.set_xticks(range(len(selected)),[ZH[NAMES.index(r['model'])] for r in selected],rotation=20)
        ax.set_title(side+'侧｜'+('连续推演' if mode=='native' else '分块滚动'))
        ax.set_ylabel('升负荷末级汽温 MAE（℃）');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('升负荷预测精度：均值与种子间标准差（越低越好）',fontsize=17)
    fig.text(.5,.015,'相同75个评价窗口；升负荷按前21.3分钟划分，不代表85分钟持续升负荷。输运模型的两种推演方式实际相同。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ('png','pdf'):fig.savefig(OUT/f'rising_accuracy_zh.{ext}',dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);accept=validate();save_json(OUT/'acceptance.json',accept)
    factual_rows=[];response_rows=[];curves={};routing=[]
    for seed in (11,23,37):
        for row in factual(run_for(seed),seed,NAMES):
            for group,item in {'all':{f'H{h}':row[f'H{h}'] for h in (32,128,512)},**row['groups']}.items():
                factual_rows.append(dict(seed=seed,model=row['model'],mode=row['mode'],side=row['side'],group=group,
                                         **{f'H{h}':item[f'H{h}'] for h in (32,128,512)}))
        rows,c,_,_=analyze_responses(run_for(seed),seed,NAMES)
        response_rows.extend(dict(seed=seed,**r) for r in rows)
        for (name,v,mode),(raw,outputs) in c.items():
            curves[(seed,name,v,mode)]=(raw,outputs)
            if len(outputs)==10 and mode=='native':
                routing.append(dict(seed=seed,model=name,valve=v,tail_by_channel_C=raw[:,-6:].mean((0,1)).tolist()))
    stats=summarize(factual_rows,['model','mode','side','group'],['H32','H128','H512'])
    save_json(OUT/'factual_summary.json',stats)
    save_json(OUT/'response_summary.json',response_rows)
    save_json(OUT/'routing_summary.json',routing)
    plots(curves,stats)
    lines=['# 双侧模型三种子回传复核','',
           '固定使用单侧 balanced、双侧 AB_balanced 检查点；没有根据升负荷结果重选。数字为种子11/23/37的均值±样本标准差，单位℃。',
           '','21次主模型训练全部达到预设验证平台；63份预测数组（包含旧固定模型及校准模型）复算，最大MAE差为 '+str(accept['maximum_mae_difference_C'])+'。',
           '42项跨PyTorch版本局部重放，最大差 '+str(max(r['max_abs_C'] for r in accept['replay']))+'℃。',
           '','三个种子共用同一评价数据，不是新的盲测；种子方差不代表数据抽样不确定性。G2旧保护增益校准及D固定父模型仅seed11，见上一轮报告，不混入三种子统计。']
    for group in ('all','rise','fall','vary','steady','support_outside'):
        for mode in ('native','rollout'):
            lines += ['',f'## {group} / {mode}','','| 模型 | A侧 H32 | A侧 H128 | A侧 H512 | B侧 H32 | B侧 H128 | B侧 H512 |','|---|---|---|---|---|---|---|']
            for name,label in zip(NAMES,ZH):
                mm='native' if mode=='native' else 'block_joint_refresh' if '_AB_' in name else 'block_context_fixed'
                values=[]
                for side in ('A','B'):
                    r=next((r for r in stats if r['model']==name and r['side']==side and r['group']==group and r['mode']==mm),None)
                    values.extend([f"{r[h]['mean']:.4f} ± {r[h]['std']:.4f}" if r else '—' for h in ('H32','H128','H512')])
                lines.append('| '+label+'（'+name+'） | '+' | '.join(values)+' |')
    lines += ['','## 四阀响应：开大3个百分点，记录计划，连续推演','',
              '末60秒平均值为开大后的预测减去原计划预测；负值为相对降温。表中每个数字依次对应seed11/23/37，正确数使用同一合法起点集合。升负荷每种子8个起点，不是75个精度窗口。']
    for group in ('all','rise','fall'):
        lines += ['',f'### {group}','','| 模型 | 阀门 | 末段温差（℃） | 降温起点数 / 合法起点数 |','|---|---|---|---|']
        for name,label in zip(NAMES,ZH):
            for v in ('u1A','u1B','u2A','u2B'):
                rr=[r for r in response_rows if r['model']==name and r['valve']==v and r['group']==group and r['shape']=='step' and r['onset']==0 and r['dose_pp']==3 and r['protocol']=='recorded' and r['mode']=='native']
                if not rr:continue
                assert len(rr)==3
                lines.append('| '+label+' | '+v+' | '+' / '.join(f"{r['tail60_C']:+.4f}" for r in rr)+' | '+' / '.join(f"{round(r['correct_tail_fraction']*r['n'])}/{r['n']}" for r in rr)+' |')
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(fits=len(accept['fits']),predictions=accept['prediction_rows_checked'],output=str(OUT)),ensure_ascii=False))


if __name__=='__main__':main()
