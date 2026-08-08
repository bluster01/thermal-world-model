#!/usr/bin/env python3
"""B 线: 1s 时序优先识别 — 强 compliance 子集的增益估计。

方法: SP 阶跃事件中, 30s 内阀位快速响应 (|dv30|>1% 且 dv·dsp<0) 的子集,
利用时间优先结构 (阀位 1-5s 响应, 温度 600s 响应): 30s 内的阀位变化不可能
由未来温度引起。对子集做 pretrend 检查 + dT600/dv 增益估计。

不做的: 不用温度响应筛选事件 (避免 collider); 不称 IV。
"""
import json
import numpy as np

EV = '/home/bluster/projectA/thermal-world-model/results/phase35_sp1s_events.json'

def main():
    ev = json.load(open(EV))['events']
    n_all = len(ev)
    dsp = np.array([e['dsp'] for e in ev])
    dv30 = np.array([e['valve_dv_30s'] for e in ev])
    dv600 = np.array([e['valve_dv_600s'] for e in ev])
    dT600 = np.array([e['dT_post_600'] for e in ev])

    # 强 compliance 子集: 30s 内 |dv|>1% 且方向与 SP 相反
    comp = (np.abs(dv30) > 1.0) & (dv30 * dsp < 0)
    print(f'all events: {n_all} | strong-compliance: {comp.sum()} ({comp.sum()/n_all:.1%})')

    for name, mask in [('all', np.ones(n_all, bool)), ('strong-compliance', comp)]:
        n = mask.sum()
        if n < 5:
            print(f'{name}: n={n} too small'); continue
        dv30_m, dv600_m, dT_m, dsp_m = dv30[mask], dv600[mask], dT600[mask], dsp[mask]
        # 增益 (600s 响应 / 30s 阀位动作)
        g30 = dT_m / dv30_m           # °C/%  (dv30<0, dT>0 → 负)
        g600 = dT_m / dv600_m
        # pretrend: 事件前温度范围 vs 响应
        # (无趋势场, 用 dT 与 |dv| 的相关性 + 方向率)
        opp = np.mean(dT_m * dv30_m < 0)
        print(f'\n{name} (n={n}):')
        print(f'  dT600/dv30: median={np.median(g30)*1000:8.1f} m°C/%  mean={np.mean(g30)*1000:8.1f}  [Q25={np.quantile(g30,0.25)*1000:.1f} Q75={np.quantile(g30,0.75)*1000:.1f}]')
        print(f'  dT600/dv600: median={np.median(g600)*1000:8.1f} m°C/%')
        print(f'  方向一致率 (dT·dv<0): {opp:.1%}   dsp 中位={np.median(dsp_m):.1f}°C  |dv30|中位={np.median(np.abs(dv30_m)):.2f}%')
        print(f'  dT600 中位={np.median(dT_m):+.2f}°C  (Q25={np.quantile(dT_m,0.25):+.2f} Q75={np.quantile(dT_m,0.75):+.2f})')
        # 分层: 按事件前温度范围 (粗稳态代理)
        tr = np.array([e['temp_range_600'] for e in ev])[mask]
        for lo, hi, lbl in [(0, 2.0, 'temp-range<2°C'), (2.0, 5.0, '2-5°C'), (5.0, 99, '>5°C')]:
            m2 = (tr >= lo) & (tr < hi)
            if m2.sum() >= 3:
                g = dT_m[m2] / dv30_m[m2]
                print(f'    {lbl}: n={m2.sum()} gain_med={np.median(g)*1000:.1f} m°C/%  方向率={np.mean(dT_m[m2]*dv30_m[m2]<0):.1%}')

    # 输出子集清单供后续 IRF
    out = {'n_all': n_all, 'n_comp': int(comp.sum()),
           'comp_indices': [i for i in range(n_all) if comp[i]]}
    with open('/home/bluster/projectA/thermal-world-model/results/phase35_sp1s_comp_subset.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('\nsaved: results/phase35_sp1s_comp_subset.json')

if __name__ == '__main__':
    main()
