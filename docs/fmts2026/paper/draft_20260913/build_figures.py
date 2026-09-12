"""Real-data FMTS figures. Python; SVG text remains editable; no bitmap tracing.

Data contract in ARGUMENT_AND_FIGURE_CONTRACT.md. All registered windows retained.
Lines: equal-day then equal-seed means. Bands: three-seed range, NOT a CI.
"""
from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE/'figures'
OUT.mkdir(exist_ok=True)
BB, GREY, GRU, TOKEN = ('blackbox_itransformer','greybox_steady_none',
                        'fusion_gru_norew','fusion_token_xattn_norew')
COLORS = {BB:'#2874A6',GREY:'#D83F3A',GRU:'#008477',TOKEN:'#8055A8'}
LABELS = {BB:('Black box','纯黑箱'),GREY:('Grey box','纯灰箱'),
          GRU:('Hybrid GRU','GRU 融合'),TOKEN:('Hybrid token','Token 融合')}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,'xtick.labelsize':6.5,
    'ytick.labelsize':6.5,'legend.fontsize':7,'axes.linewidth':.6,
    'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':360})


def finish(fig, name):
    fig.savefig(OUT/f'{name}.pdf',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',facecolor='white')
    fig.savefig(OUT/f'{name}.png',facecolor='white',dpi=360)
    tree=ET.parse(OUT/f'{name}.svg')
    assert len(tree.findall('.//{http://www.w3.org/2000/svg}text'))>10
    assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
    plt.close(fig)


def language(zh):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','DejaVu Sans'] if zh else ['DejaVu Sans']


def architecture(zh=False):
    language(zh)
    fig,ax=plt.subplots(figsize=(5.5,2.85))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,100),ylim=(0,57));ax.axis('off')
    ink, neural, physics, action='#233645','#008477','#516576','#C77227'
    def txt(x,y,en,cn=None,size=6.7,c=ink,weight='normal',ha='center'):
        ax.text(x,y,cn if zh and cn else en,ha=ha,va='center',fontsize=size,color=c,weight=weight)
    def box(x,y,w,h,en,cn=None,c=physics,fill='#F0F3F5',size=6.4):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1,rounding_size=.7',
                                  fc=fill,ec=c,lw=.65))
        txt(x+w/2,y+h/2,en,cn,size,c)
    def arr(a,b,c=ink,style='-',rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=.75,
                                    color=c,linestyle=style,connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))
    txt(2,54.5,'a  From steam-temperature forecasting to valve evaluation',
        'a  从主汽温预测到阀门动作评估',7.6,weight='bold',ha='left')
    # Schematic matches three heated stages and two spray junctions, not a P&ID.
    for x,w,en,cn in [(3,16,'Heating 1','一级加热'),(35,16,'Heating 2','二级加热'),(68,16,'Heating 3','末级加热')]:
        box(x,44.7,w,4.8,en,cn,size=6.5)
    for a,b in [(19,35),(51,68),(84,97)]: arr((a,47.1),(b,47.1),physics)
    for x,name,cn in [(27,'Valve 1','一级减温阀'),(60,'Valve 2','二级减温阀')]:
        ax.add_patch(Polygon([(x-1,51),(x+1,49.8),(x+1,51),(x-1,49.8)],ec=action,fc='white',lw=.7))
        arr((x,49.8),(x,47.3),action)
        txt(x,52.2,name,cn,6.4,action)
    txt(91,44.8,'Main steam','主蒸汽',6.3)
    txt(3,41.8,'600 MW once-through boiler  |  physical process schematic',
        '600 MW 直流炉  |  物理流程示意',6.1,physics,ha='left')
    ax.plot([2,98],[39.4,39.4],color='#CDD5DA',lw=.6)
    txt(2,37.4,'b  Selected hybrid: learned initialization, physical action pathway',
        'b  选定融合模型：学习初态，物理方程承载动作',7.4,weight='bold',ha='left')
    box(2,20,19,12,'Past 96 steps\n5 temperatures\n2 valves + 7 base\n+ 9 extensions',
        '过去 96 步\n5 温度\n2 阀位 + 7 基础边界\n+ 9 扩展变量',size=6.2)
    box(26,29,17,4.5,'Physical anchor','物理锚点',size=6.5)
    box(26,20,17,6.4,'GRU observer\nbounded correction','GRU 观察器\n有界修正',neural,'#EDF8F5',6.4)
    arr((21,29.9),(26,31),physics)
    arr((21,24),(26,23.2),neural)
    ax.add_patch(Circle((48,27.3),1.2,ec=physics,fc='white',lw=.7));txt(48,27.3,'+',size=8)
    arr((43,31),(47.5,28.5),physics); arr((43,23.2),(47.5,26.1),neural)
    txt(46.8,22.2,'mask','掩码',5.9,neural)
    arr((49.2,27.3),(55,27.3)); txt(52,29,'x(t)',size=6.2)
    box(55,22,23,10,'Physical transition\nspray → mixing\nsteam–metal balances',
        '物理状态转移\n喷水 → 混合\n蒸汽—金属能量平衡',size=6.25)
    box(55,33,23,2.7,'Valves u + boundaries b','阀位 u + 边界 b',action,'#FFF5E9',6.1)
    arr((66.5,33),(66.5,32),action)
    arr((78,27.3),(83,27.3))
    box(83,22,15,10,'State x(t+k)\nphysical map g\n5 temperatures',
        '状态 x(t+k)\n物理观测映射 g\n5 个温度',size=6.1)
    txt(74,19,'step × 18','转移 × 18',6.2,physics)
    box(26,7.9,28,8,'Residual MLP: r(x, b)\nno direct valve / spray input',
        '残差 MLP：r(x, b)\n不直接读取阀位 / 喷水流量',neural,'#EDF8F5',6.2)
    arr((59,22),(51,16),neural)
    arr((54,11.9),(70,22),neural)
    txt(64.5,12.8,'+r steam\n−r metal','蒸汽 +r\n金属 −r',6.2,neural)
    txt(3,15.8,'Learned','学习模块',6.2,neural,ha='left')
    txt(3,12.3,'Physics','物理模块',6.2,physics,ha='left')
    arr((90.5,22),(90.5,14),physics)
    txt(90.5,10.5,'Report:\nmain-steam T','报告：\n主汽温 T',6.4)
    txt(2,3.8,'Probe: same history + held boundaries; change only one valve by +0.05.',
        '探针：相同历史、固定边界，仅将一个阀位增加 0.05。',6.3,ha='left')
    finish(fig,'fig1_architecture_'+('zh' if zh else 'en'))


def load_rows(name):
    with (HERE/'source_data'/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def curves(rows,arm,field,valve=None):
    chosen=[r for r in rows if r['arm']==arm and (valve is None or int(r['valve'])==valve)]
    return np.array([[float(r[field]) for r in sorted(
        [r for r in chosen if int(r['seed'])==s],key=lambda r:int(r['horizon']))] for s in range(3)])


def evidence(zh=False,ablation=False):
    language(zh)
    pred,resp=load_rows('prediction.csv'),load_rows('response.csv')
    arms=[GRU,TOKEN] if ablation else [BB,GREY,GRU]
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.04))
    fig.subplots_adjust(left=.095,right=.985,bottom=.23,top=.71,wspace=.48)
    x=np.arange(1,19)*10
    handles=[Line2D([0],[0],color=COLORS[a],lw=1.8,label=LABELS[a][int(zh)]) for a in arms]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.0),ncol=len(arms),
               frameon=False,handlelength=1.5,columnspacing=1.4)
    for idx,ax in enumerate(axes):
        for arm in arms:
            y=curves(pred,arm,'cumulative_mae') if idx==0 else curves(resp,arm,'response',idx)
            ax.fill_between(x,y.min(0),y.max(0),color=COLORS[arm],alpha=.13,lw=0)
            ax.plot(x,y.mean(0),color=COLORS[arm],lw=1.65,
                    marker='o' if arm==GRU else None,markevery=[5,11,17],markersize=2.5)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(10,180);ax.set_xticks([10,60,120,180]);ax.tick_params(length=2.5,pad=2)
        ax.grid(axis='y',color='#E5E9EC',lw=.45)
        ax.set_xlabel('Horizon (s)' if not zh else '预测时长（秒）',labelpad=2)
        title=[('a  Prediction','a  预测精度'),('b  Valve 1','b  一级减温阀'),('c  Valve 2','c  二级减温阀')][idx]
        ax.set_title(title[int(zh)],loc='left',weight='bold',pad=6)
        if idx==0:
            ax.set_ylabel('Cumulative MAE (°C)' if not zh else '累计 MAE（°C）',labelpad=2)
            ax.set_ylim(0, .77 if ablation else 1.05)
            if not ablation:
                ax.text(.04,.80,'+0.0725°C\nvs black box' if not zh else '较黑箱\n+0.0725°C',
                        transform=ax.transAxes,fontsize=6.4,color=COLORS[GRU])
        else:
            ax.axhline(0,color='#7D8991',ls=':',lw=.7,zorder=0)
            ax.set_ylim(-.22,.055);ax.set_yticks([-.2,-.1,0])
            ax.set_xlabel('After valve step (s)' if not zh else '阀位阶跃后（秒）',labelpad=2)
            if idx==1: ax.set_ylabel('Δ main-steam T (°C)' if not zh else '主汽温变化（°C）',labelpad=2)
    fig.text(.54,.028,'Lines: 3-seed means  |  bands: seed min–max; not confidence intervals'
             if not zh else '实线：三种子均值  |  阴影：种子最小—最大值，非置信区间',ha='center',fontsize=6.1,color='#56636C')
    finish(fig,('figS1_observer_ablation_' if ablation else 'fig2_prediction_response_')+('zh' if zh else 'en'))


if __name__=='__main__':
    audit=json.loads((HERE/'audit.json').read_text())
    assert audit['status']=='SAVED_ARRAY_AUDIT_PASSED'
    for zh in [False,True]:
        architecture(zh)
        evidence(zh)
        evidence(zh,True)
    print('Built six figures, each as editable SVG + PDF + 360 dpi PNG.')
