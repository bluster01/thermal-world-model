"""Train-only, timestamp-deduplicated 60s isolated actions and aligned controls.

Replaces the execution-side v1 analysis (10s detection, post-action features,
misaligned control anchors). Old results are preserved, never used as calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .bilateral_data import OUT_DEFAULT
from .data import sha256
from .run import HERE,save_json

VALVES=('u1A','u1B','u2A','u2B')
LOCAL=((1,0),(6,5),(8,7),(3,2))
MAIN=(4,9,9,4)
SIDE=(0,5,5,0)
OUT=HERE.parents[1]/'results/action_predictor_bench_20260919/bilateral_events_reviewed_20260923'


def unique_train_rows(pack):
    h=pack['hist30_train']
    epochs=pack['train_time'][:,None]+(np.arange(64)-63)*10
    times,first=np.unique(epochs.ravel(),return_index=True)
    return times,h.reshape(-1,30)[first]


def eligible_origins(times):
    # 64 pre rows (including anchor) +128 future rows; no bridging data gaps.
    ids=np.arange(63,len(times)-128)
    return ids[(times[ids+128]-times[ids-63])==1910]


def find_events(times,values,dose=.02,guard=.002,separation=1920):
    ids=eligible_origins(times)
    u=values[:,10:14]
    change=u[ids+6]-u[ids]
    excursion=np.stack([np.abs(u[ids+k]-u[ids]) for k in range(1,7)]).max(0)
    events=[];counts={}
    for valve in range(4):
        good=(np.abs(change[:,valve])>=dose)&(np.delete(excursion,valve,axis=1)<=guard).all(1)
        candidates=ids[good]; counts[VALVES[valve]]=len(candidates)
        last=-10**18
        for i in candidates:
            if times[i]-last<separation: continue
            events.append(dict(valve=valve,index=int(i),epoch=int(times[i]),dose=float(u[i+6,valve]-u[i,valve])))
            last=int(times[i])
    # A quiet CONTROL means all four valves change <=guard during the same60s.
    quiet=ids[(excursion<=guard).all(1)]
    return events,quiet,counts


def features(values,ids,side):
    block=values[ids[:,None]+np.arange(-31,1)[None]]
    t=block[:,:,side:side+5]
    return np.concatenate((block[:,-1,10:14],t.mean(1),t[:,-1]-t[:,0],block[:,:,[24,15,14]].mean(1)),1)


def match_controls(times,values,events,quiet,k=5):
    # A sparse fixed-time pool is sufficient; prohibit event/control window overlap.
    selected=[];last=-10**18
    for i in quiet:
        if times[i]-last>=320: selected.append(i);last=times[i]
    quiet=np.asarray(selected,dtype=int)
    if len(quiet)<k: return [],dict(pool_size=len(quiet),unmatched=len(events))
    pools={}
    for side in (0,5):
        x=features(values,quiet,side).astype(np.float64)
        sd=np.maximum(x.std(0),1e-6)
        pools[side]=(x,sd)
    matched=[]
    for event in events:
        i=event['index'];x,sd=pools[SIDE[event['valve']]]
        feat=features(values,np.array([i]),SIDE[event['valve']])[0].astype(np.float64)
        difference=(x-feat)/sd
        distance=np.sqrt((difference**2).sum(1))
        distance[np.abs(times[quiet]-times[i])<1920]=np.inf
        selected=[]
        for rank in np.argsort(distance):
            if not np.isfinite(distance[rank]): break
            control=int(quiet[rank])
            if any(abs(times[control]-times[prev])<1920 for prev in selected): continue
            selected.append(control)
            if len(selected)==k: break
        if len(selected)<k: continue
        cfeatures=features(values,np.asarray(selected),SIDE[event['valve']])
        matched.append(dict(**event,controls=selected,control_epochs=times[selected].tolist(),
            match_distance=float(distance[np.where(quiet==selected[0])[0][0]]),
            pre_balance_standardized=((cfeatures.mean(0)-feat)/sd).tolist()))
    return matched,dict(pool_size=len(quiet),unmatched=len(events)-len(matched))


def responses(times,values,events):
    arrays={};summary={}
    for valve,name in enumerate(VALVES):
        rows=[r for r in events if r['valve']==valve]
        summary[name]=dict(events=len(rows))
        if not rows: continue
        ids=np.asarray([r['index'] for r in rows]);control=np.asarray([r['controls'] for r in rows])
        offsets=np.arange(129)
        y=values[ids[:,None]+offsets,:10];yc=values[control[:,:,None]+offsets,:10]
        dy=y-y[:,:1];dc=yc-yc[:,:,:1]
        paired=dy-dc.mean(1)
        pre=values[ids[:,None]+np.arange(-63,1),:]
        pre_control=values[control[:,:,None]+np.arange(-63,1),:10]
        pre_difference=(pre[:,:,:10]-pre[:,-1:,:10])-(pre_control-pre_control[:,:,-1:]).mean(1)
        plans=values[ids[:,None]+offsets,10:21]
        arrays.update({f'{name}_history':pre,f'{name}_future_act':plans[:,:,:4],
            f'{name}_future_bnd':plans[:,:,4:],f'{name}_truth':y[:,1:],
            f'{name}_event_delta':dy,f'{name}_control_delta':dc,f'{name}_matched_difference':paired,
            f'{name}_event_epoch':times[ids],f'{name}_control_epoch':times[control],
            f'{name}_dose':np.asarray([r['dose'] for r in rows]),
            f'{name}_control_actions':values[control[:,:,None]+offsets,10:14],
            f'{name}_matched_pretrend':pre_difference,
            f'{name}_future_MW_coal_flow':values[ids[:,None]+offsets][:,:,[24,15,14]],
            f'{name}_control_MW_coal_flow':values[control[:,:,None]+offsets][:,:,:,[24,15,14]],
            f'{name}_pre_balance':np.asarray([r['pre_balance_standardized'] for r in rows])})
        for label,mask in (('opening',arrays[f'{name}_dose']>0),('closing',arrays[f'{name}_dose']<0)):
            if not mask.any(): summary[name][label]=dict(events=0);continue
            local=paired[mask,:,LOCAL[valve][0]]-paired[mask,:,LOCAL[valve][1]]
            downstream=paired[mask,:,MAIN[valve]]
            summary[name][label]=dict(events=int(mask.sum()),local_0_180_mean_C=float(local[:,1:19].mean()),
                downstream_180_1280_mean_C=float(downstream[:,19:].mean()),
                median_dose_pp=float(np.median(arrays[f'{name}_dose'][mask])*100))
        summary[name]['pre_balance_abs_mean_by_feature']=np.abs(arrays[f'{name}_pre_balance'].mean(0)).tolist()
        summary[name]['pretrend_max_abs_mean_per_channel_C']=np.abs(pre_difference.mean(0)).max(0).tolist()
    return arrays,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',type=Path,default=OUT_DEFAULT);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--dose',type=float,default=.02);p.add_argument('--guard',type=float,default=.002)
    p.add_argument('--controls',type=int,default=5)
    args=p.parse_args()
    if args.dose<=0 or args.guard<0 or args.controls<1: p.error('Invalid event rules')
    with np.load(args.pack) as pack: times,values=unique_train_rows(pack)
    events,quiet,counts=find_events(times,values,args.dose,args.guard)
    matched,match_info=match_controls(times,values,events,quiet,args.controls)
    arrays,groups=responses(times,values,matched)
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'event_responses.npz',**arrays)
    summary=dict(version=2,source_sha256=sha256(args.pack),code_sha256=sha256(__file__),
        protocol='60s net target movement, max absolute excursion of other3 over same60s; timestamp unique; 1920s same-valve event separation',
        rule=dict(dose=args.dose,guard=args.guard,window_steps=6,separation_seconds=1920,control_count=args.controls),
        train_unique_rows=len(times),eligible_origins=len(eligible_origins(times)),candidates=counts,
        separated_events=len(events),matched_events=len(matched),matching=match_info,groups=groups,
        caveat='Train observational comparison, not randomized effect; quiet controls guaranteed only first60s; later valve traces saved; reuse across events possible',
        supersedes='bilateral_events_20260922: wrong10s detection, post-action feature contamination, misaligned control anchor, duplicated epochs; do not calibrate from v1')
    save_json(args.out/'event_summary.json',summary);save_json(args.out/'event_manifest.json',matched)
    lines=['# Reviewed natural actions: train only','','60s detection, 64-row history, 128-step future; all controls anchored at their matching endpoint.',
        'Opening and closing are separate. Matched differences remain observational, not physical-response ground truth.','',
        '| Valve | Candidates | Matched independent-spaced events | Opening | Closing |','|---|---:|---:|---:|---:|']
    for name in VALVES:
        g=groups[name];lines.append(f'| {name} | {counts[name]} | {g["events"]} | {g.get("opening",{}).get("events",0)} | {g.get("closing",{}).get("events",0)} |')
    lines+=['',f'Control pool: {match_info["pool_size"]}; unmatched: {match_info["unmatched"]}.',
            'All prefeatures use anchor-310s through anchor only. Event/control 1920s windows do not overlap.',
            'Previous v1 event report is superseded; its counts and response curves are not accepted for calibration.']
    (args.out/'event_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
