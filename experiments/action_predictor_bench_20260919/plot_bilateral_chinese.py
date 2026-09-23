"""Chinese rendering of the unchanged returned four-valve response curves."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .review_bilateral import ROOT,DEPLOY,COLORS,LOCAL,MAIN,read


def main():
    available={f.name for f in font_manager.fontManager.ttflist}
    candidates=['Microsoft YaHei','Noto Sans CJK SC','SimHei','WenQuanYi Micro Hei']
    font=next((name for name in candidates if name in available),None)
    if font is None: raise RuntimeError('A Chinese font is required to render this figure')
    plt.rcParams.update({'font.family':font,'font.size':11,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    names=['旧保护模型（D）','两增益校准（G2）','A侧基础模型（S0_A）','B侧基础模型（S0_B）',
           '双侧历史模型（S1）','四阀自由模型（S2）','双侧自由模型（S3）',
           '双侧输运模型（R3）','双侧保护模型（P3）']
    titles=['一级A阀 → A侧','一级B阀 → B侧','二级A阀 → B侧（交叉接线）','二级B阀 → A侧（交叉接线）']
    fig,axes=plt.subplots(2,4,figsize=(16,8.5),sharex=True)
    handles={}
    for name,label,color in zip(DEPLOY,names,COLORS):
        folder=ROOT/'bilateral33_seed11/seed11'/name
        catalog=read(folder/'response_catalog.json')
        with np.load(folder/'responses.npz') as z:
            for valve in range(4):
                cases=[c for c in catalog if c['shape']=='step' and c['onset']==0 and c['dose']==.03
                    and c['vector']==[int(i==valve) for i in range(4)] and c['mode']=='native'
                    and c['boundary_action_protocol']=='held_both' and c['all_perturbed_inputs_supplied']
                    and MAIN[valve] in c['output_indices']]
                if not cases: continue
                c=cases[0];a=z[c['key']][np.array(c['valid'])];out=c['output_indices']
                assert len(a)==32 and np.isfinite(a).all()
                time=(np.arange(a.shape[1])+1)/6
                local=a[:,:,out.index(LOCAL[valve])]-a[:,:,out.index(LOCAL[valve]-1)]
                for ax,y in zip(axes[:,valve],(local.mean(0),a[:,:,out.index(MAIN[valve])].mean(0))):
                    line,=ax.plot(time,y,color=color,lw=1.8,label=label);handles[label]=line
    for valve in range(4):
        axes[0,valve].set_title(titles[valve]+'\n局部：喷水后减去喷水前',fontsize=12)
        axes[1,valve].set_title('末级：'+('A侧' if valve in (0,3) else 'B侧')+'出口汽温',fontsize=12)
        for ax in axes[:,valve]:
            ax.axhline(0,color='#999999',lw=.7);ax.grid(alpha=.12)
            ax.set_xlabel('阀门动作后的时间（分钟）');ax.tick_params(labelbottom=True)
    axes[0,0].set_ylabel('局部温差的预测变化（℃）')
    axes[1,0].set_ylabel('末级汽温的预测变化（℃）')
    fig.suptitle('四阀局部与末级响应',fontsize=20,y=.98)
    fig.text(.5,.925,'各阀开大3个百分点｜相同32个起点｜其他阀位与边界条件固定｜连续推演',ha='center',fontsize=12)
    fig.legend([handles[n] for n in names],names,loc='lower center',bbox_to_anchor=(.5,.046),ncol=3,frameon=False,fontsize=11)
    fig.text(.5,.018,'曲线表示“开大阀门后的预测 − 原计划预测”的平均差值。下排负值表示相对降温；这些曲线不是实测干预结果。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.065,right=.987,top=.83,bottom=.245,hspace=.53,wspace=.27)
    out=ROOT/'bilateral_review_20260923';out.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(out/f'response_paths_zh.{suffix}',dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
