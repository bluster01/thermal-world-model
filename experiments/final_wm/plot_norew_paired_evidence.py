"""Render audited aggregate evidence, not reconstructed response trajectories."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/final_wm/norew_paired_audit_20260908'
OUT=ROOT/'docs/fmts2026/figures_20260908/norew_paired'
ARMS=['physics_only','closure_cons','closure_cons_norew']
LABELS=['Physics-only','Conservative closure','Conservative closure\nwithout rewetting']
COLORS=['#737373','#0072B2','#D55E00']
MARKERS=['o','s','^']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (DATA/'prediction.csv').open(encoding='utf-8-sig') as f: predictions=list(csv.DictReader(f))
    with (DATA/'response.csv').open(encoding='utf-8-sig') as f: responses=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(11.4,5.9))
    gs=fig.add_gridspec(2,3,width_ratios=[1.12,1,1],left=.19,right=.98,top=.82,bottom=.27,wspace=.24,hspace=.4)
    ax=fig.add_subplot(gs[:,0])
    for i,arm in enumerate(ARMS):
        vals=np.array([float(p['mae_h18_c']) for p in predictions if p['arm']==arm])
        ax.plot(vals,[i-.15,i,i+.15],linestyle='none',marker='o',color=COLORS[i],markersize=4)
        ax.plot(vals.mean(),i,marker='|',markersize=23,markeredgewidth=2,color=COLORS[i])
        ax.text(vals.mean(),i+.31,f'{vals.mean():.3f}',ha='center',color=COLORS[i],fontsize=9)
    ax.set(yticks=range(3),yticklabels=LABELS,xlim=(1.48,2.04),ylim=(2.65,-.55),xlabel='Five-temperature MAE (°C)')
    ax.set_title('a   Prediction · H18',loc='left',fontweight='bold',pad=8)
    ax.grid(axis='x',color='#eeeeee',linewidth=.7)
    fig.text(.19,.12,'Closure → no rewetting\n+0.283°C (+17.7%)',color=COLORS[2],fontsize=9)
    letters=['b','c','d','e']
    for n,(valve,horizon) in enumerate([(1,18),(2,18),(1,60),(2,60)]):
        axis=fig.add_subplot(gs[n//2,n%2+1])
        axis.axvline(0,color='#999999',linewidth=.8,linestyle='--')
        for i,arm in enumerate(ARMS):
            rows=[r for r in responses if r['arm']==arm and int(r['valve'])==valve and int(r['horizon'])==horizon]
            for r in rows:
                seed=int(r['seed']); y=i+(seed-1)*.2
                mean,lo,hi=[float(r[k]) for k in ['mean_delta_c','ci_lo_c','ci_hi_c']]
                supported=r['all_steps_supported']=='True'
                axis.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt=MARKERS[seed],color=COLORS[i],mfc=COLORS[i] if supported else 'white',mec=COLORS[i],markersize=4,capsize=2,elinewidth=.9)
        axis.set(xlim=(-.4,.1),ylim=(2.5,-.5),yticks=range(3),yticklabels=['','',''],xticks=[-.4,-.2,0,.1])
        axis.set_title(f'{letters[n]}   Valve {valve} · H{horizon}',loc='left',fontweight='bold')
        axis.grid(axis='y',color='#eeeeee',linewidth=.7)
        if n>=2: axis.set_xlabel('Terminal temperature response (°C)')
    fig.suptitle('Prediction accuracy and action-response direction',x=.19,ha='left',y=.975,fontsize=14,fontweight='bold')
    fig.text(.19,.91,'Side A · fixed validation windows · unchanged checkpoints',fontsize=9,color='#555555')
    handles=[Line2D([0],[0],marker=m,linestyle='none',color='#444444',mfc='white',label=f'Seed {s}') for s,m in enumerate(MARKERS)]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.48,.085),ncol=3,frameon=False)
    fig.text(.19,.055,'Response: +0.05 valve position; last 10 steps; equal-day 95% bootstrap CI. Open markers: unsupported steps present.',fontsize=8)
    fig.text(.19,.025,'11/12 response cells per arm include unsupported steps. Direction diagnostics do not establish plant response fidelity.',fontsize=8,color='#555555')
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'fig2_paired_direction.{ext}',dpi=300)
    svg=OUT/'fig2_paired_direction.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    # Shape/line encoding remains legible without color.
    for artist in fig.findobj():
        if isinstance(artist,Line2D):
            artist.set_color('#444444')
            artist.set_markeredgecolor('#444444')
            if artist.get_markerfacecolor() not in ('white','none','None'):
                artist.set_markerfacecolor('#444444')
    fig.savefig(OUT/'fig2_paired_direction_grayscale.png',dpi=150)
    plt.close(fig)
    print(OUT)

if __name__=='__main__':main()
