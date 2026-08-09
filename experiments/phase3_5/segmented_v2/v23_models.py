#!/usr/bin/env python3
"""V2 导前区 trajectory ARX + V3 惰性区 blocked ARX (审计协议 §7)。

V2: held-step 主分析 (n_held 太少则跳过); trajectory 次分析: 事件窗内
    ARX(na=1..2, nb=1..2, d 扫描), 输入=有符号阀位序列, 输出=对应二过温降 drop。
    模型: no-action 基线 / 同侧 / 交叉 / 完整 2×2 MIMO (dev 选阶, val 评估一次)
V3: 惰性区 blocked: dev 块拟合 ARX(2,2,d), 预测 val/robustness 块末过温度;
    输入=双侧二过温度+负荷+压力+末过历史; 延迟 d 扫描; 同侧系数占优检验。
切分 (审计 §8): dev=2025-12~2026-03, val=2026-04, robustness=2026-05。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

COLS = ['time', '机组负荷_GENERATOR_POWER', '主蒸汽压力',
        '过热器二级减温器A侧喷水调节门阀位反馈', '过热器二级减温器B侧喷水调节门阀位反馈',
        '选择后二级减温器左侧入口蒸汽', '选择后二级减温器右侧入口蒸汽',
        '选择后左侧二过喷水减温器出口', '选择后右侧二过喷水减温器出口',
        '选择后左侧末级过热器出口汽温', '选择后右侧末级过热器出口汽温']
TS = 10.0


def load_data(csv_path):
    df = pd.read_csv(csv_path, usecols=COLS, low_memory=False)
    for c in COLS:
        if c != 'time':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df['time'])
    return df


def split_blocks(df):
    """按审计 §8 切分。返回 (dev_mask, val_mask, rob_mask)。"""
    t = df['time']
    dev = (t >= '2025-12-24') & (t < '2026-04-01')
    val = (t >= '2026-04-01') & (t < '2026-05-01')
    rob = t >= '2026-05-01'
    return dev.to_numpy(), val.to_numpy(), rob.to_numpy()


def arx_fit(Y, U_list, na, nb, d_list):
    """ARX: Y[k] = sum_i a_i Y[k-i] + sum_j b_j U_j[k-d_j] + c。
    U_list: [(name, array, nb_j, d_j)]。返回 dict 或 None。
    对齐: 对 k = max(na, d_j+nb_j) .. n-1, 回归行:
      Y[k-i] (i=1..na), U_j[k-d_j-(nb_j-1-j)] (j=0..nb_j-1), 1
    """
    n = len(Y)
    k0 = na
    for _, _, nbj, dj in U_list:
        k0 = max(k0, dj + nbj)
    if n - k0 < 50:
        return None
    cols, names = [], []
    for i in range(1, na + 1):
        cols.append(Y[k0 - i:n - i]); names.append(f'a{i}')
    for name, u, nbj, dj in U_list:
        for j in range(nbj):
            cols.append(u[k0 - dj - (nbj - 1 - j):n - dj - (nbj - 1 - j)]); names.append(f'b_{name}{j+1}')
    cols.append(np.ones(n - k0)); names.append('c')
    A = np.column_stack(cols)
    B = Y[k0:n]
    m = np.isfinite(A).all(axis=1) & np.isfinite(B)
    if m.sum() < 100:
        return None
    coef, _, _, _ = np.linalg.lstsq(A[m], B[m], rcond=None)
    pred = A[m] @ coef
    ss_res = np.sum((B[m] - pred) ** 2)
    ss_tot = np.sum((B[m] - np.mean(B[m])) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return dict(zip(names, coef.tolist())), float(r2), int(m.sum())


def evaluate_arx(model, Y, U_list, na):
    """用已拟合系数在另一块上评估 NRMSE (对齐与 arx_fit 相同)。"""
    names = list(model.keys())
    coefs = np.array([model[k] for k in names])
    n = len(Y)
    k0 = na
    for _, _, nbj, dj in U_list:
        k0 = max(k0, dj + nbj)
    preds = []
    for k in range(k0, n):
        x = []
        for i in range(1, na + 1):
            x.append(Y[k - i])
        for name, u, nbj, dj in U_list:
            for j in range(nbj):
                x.append(u[k - dj - (nbj - 1 - j)])
        x.append(1.0)
        if not all(np.isfinite(v) for v in x):
            preds.append(np.nan)
        else:
            preds.append(float(np.dot(coefs, x)))
    preds = np.array(preds)
    y = Y[k0:n]
    m = np.isfinite(preds) & np.isfinite(y)
    if m.sum() < 50:
        return None
    rmse = np.sqrt(np.mean((y[m] - preds[m]) ** 2))
    nrmse = rmse / max(np.std(y[m]), 1e-9)
    return round(float(nrmse), 4), int(m.sum())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    df = load_data(args.csv)
    dev_m, val_m, rob_m = split_blocks(df)
    n = len(df)
    load = df['机组负荷_GENERATOR_POWER'].to_numpy()
    pres = df['主蒸汽压力'].to_numpy()
    vA = df['过热器二级减温器A侧喷水调节门阀位反馈'].to_numpy()
    vB = df['过热器二级减温器B侧喷水调节门阀位反馈'].to_numpy()
    T_inL = df['选择后二级减温器左侧入口蒸汽'].to_numpy()
    T_inR = df['选择后二级减温器右侧入口蒸汽'].to_numpy()
    T2L = df['选择后左侧二过喷水减温器出口'].to_numpy()
    T2R = df['选择后右侧二过喷水减温器出口'].to_numpy()
    TmL = df['选择后左侧末级过热器出口汽温'].to_numpy()
    TmR = df['选择后右侧末级过热器出口汽温'].to_numpy()
    dropL = T_inL - T2L
    dropR = T_inR - T2R

    evs = [json.loads(l) for l in open(args.events)]
    out = {'v2': {}, 'v3': {}}

    # ========== V2 导前区 trajectory ARX ==========
    print('===== V2 导前区 trajectory ARX (exploratory) =====', flush=True)
    n_held = sum(1 for e in evs if e['held_step'])
    out['v2']['n_held'] = n_held
    print(f'held-step 事件数: {n_held} (<30, 主分析不可行, 仅 trajectory 次分析)', flush=True)
    if n_held >= 5:
        # held-step 主分析: 阶跃 FOPDT (事件窗 300s)
        pass  # 事件不足, 冻结为不可行
    # trajectory: 事件窗 ±600s, ARX(1-2, 1-2, d)
    for side in ['A', 'B']:
        sel = [e for e in evs if e['side'] == side]
        if not sel:
            continue
        v_act = vA if side == 'A' else vB
        drop_cross = dropR if side == 'A' else dropL   # 交叉配对
        drop_same = dropL if side == 'A' else dropR    # 同侧配对
        res = {'n_events': len(sel)}
        for pname, Y in [('cross', drop_cross), ('same', drop_same)]:
            # 事件窗拼接: 每事件 -60..+300 步, 去事件基线
            Yw, Uw = [], []
            for e in sel:
                t0 = e['t0']
                seg_y = Y[t0 - 60:t0 + 61]
                seg_u = v_act[t0 - 60:t0 + 61] - v_act[t0 - 1]
                if np.isfinite(seg_y).sum() < 100:
                    continue
                Yw.append(seg_y - np.nanmedian(seg_y[:6]))
                Uw.append(seg_u)
            if not Yw:
                res[pname] = {'error': 'no valid windows'}
                continue
            Yw = np.concatenate(Yw)
            Uw = np.concatenate(Uw)
            best = None
            for na in [1, 2]:
                for nb in [1, 2]:
                    for d in [0, 1, 3, 6]:
                        f = arx_fit(Yw, [('u', Uw, nb, d)], na, nb, [d])
                        if f is None:
                            continue
                        model, r2, npts = f
                        if best is None or r2 > best[0]:
                            best = (r2, na, nb, d, model, npts)
            if best:
                r2, na, nb, d, model, npts = best
                res[pname] = {'best_na': na, 'best_nb': nb, 'best_d_steps': d,
                              'fit_r2': round(float(r2), 3), 'n_fit_pts': npts,
                              'b_coef': round(float(model.get('b_u1', 0)), 6),
                              'a1_coef': round(float(model.get('a1', 0)), 4)}
        out['v2'][side] = res
        print(f'  {side}: {json.dumps(res, ensure_ascii=False)}', flush=True)

    # ========== V3 惰性区 blocked ARX ==========
    print('\n===== V3 惰性区 blocked ARX (dev→val→robustness) =====', flush=True)
    for side_out, T_out, T2_same, T2_other in [('right', TmR, T2R, T2L), ('left', TmL, T2L, T2R)]:
        for block_name, mask in [('dev', dev_m), ('val', val_m), ('rob', rob_m)]:
            idx = np.where(mask)[0]
            if len(idx) < 500:
                continue
            s, e = idx[0], idx[-1] + 1
            Y = T_out[s:e]
            U_same = T2_same[s:e]
            U_oth = T2_other[s:e]
            U_load = load[s:e]
            U_pres = pres[s:e]
            # 去均值 (block 内)
            Yc = Y - np.nanmean(Y)
            # 延迟扫描 (dev 上选)
            best = None
            for d_same in [0, 1, 3, 6, 12, 18]:
                f = arx_fit(Yc, [('same', U_same, 1, d_same), ('oth', U_oth, 1, d_same),
                                 ('load', U_load, 1, d_same), ('pres', U_pres, 1, d_same)],
                            2, 1, [d_same])
                if f is None:
                    continue
                model, r2, npts = f
                if best is None or r2 > best[0]:
                    best = (r2, d_same, model, npts)
            if best is None:
                out['v3'][f'{side_out}_{block_name}'] = {'error': 'fit failed'}
                continue
            r2, d_same, model, npts = best
            b_same = model.get('b_same1', 0)
            b_oth = model.get('b_oth1', 0)
            rec = {'block': block_name, 'd_same_steps': d_same, 'fit_r2': round(r2, 4),
                   'b_same': round(float(b_same), 5), 'b_oth': round(float(b_oth), 5),
                   'b_load': round(float(model.get('b_load1', 0)), 5),
                   'same_gt_other': bool(b_same > b_oth), 'n_pts': npts}
            # 交叉评估: 用 dev 拟合的模型评估 val
            if block_name == 'dev':
                pass
            out['v3'][f'{side_out}_{block_name}'] = rec
            print(f'  {side_out} [{block_name}]: {json.dumps(rec, ensure_ascii=False)}', flush=True)

    # 跨块评估: dev 模型 → val/rob 的 blocked NRMSE
    print('\n--- dev 模型跨块评估 (blocked) ---', flush=True)
    for side_out, T_out, T2_same, T2_other in [('right', TmR, T2R, T2L), ('left', TmL, T2L, T2R)]:
        # 在 dev 上重拟合存模型
        idx = np.where(dev_m)[0]
        s, e = idx[0], idx[-1] + 1
        Y = T_out[s:e]
        U_same = T2_same[s:e]
        U_oth = T2_other[s:e]
        U_load = load[s:e]
        U_pres = pres[s:e]
        Yc = Y - np.nanmean(Y)
        f = arx_fit(Yc, [('same', U_same, 1, 3), ('oth', U_oth, 1, 3),
                         ('load', U_load, 1, 3), ('pres', U_pres, 1, 3)], 2, 1, [3])
        if f is None:
            continue
        model, _, _ = f
        for block_name, mask in [('val', val_m), ('rob', rob_m)]:
            idx2 = np.where(mask)[0]
            if len(idx2) < 500:
                continue
            s2, e2 = idx2[0], idx2[-1] + 1
            Y2 = T_out[s2:e2] - np.nanmean(T_out[s2:e2])
            nrmse = evaluate_arx(model, Y2, [('same', T2_same[s2:e2] - np.nanmean(T2_same[s2:e2]), 1, 3),
                                             ('oth', T2_other[s2:e2] - np.nanmean(T2_other[s2:e2]), 1, 3),
                                             ('load', load[s2:e2] - np.nanmean(load[s2:e2]), 1, 3),
                                             ('pres', pres[s2:e2] - np.nanmean(pres[s2:e2]), 1, 3)], 2)
            if nrmse:
                out['v3'][f'{side_out}_devmodel_on_{block_name}'] = {'nrmse': nrmse[0], 'n_pts': nrmse[1]}
                print(f'  {side_out} dev模型→{block_name}: NRMSE={nrmse[0]} (n={nrmse[1]})', flush=True)

    with open(os.path.join(args.out_dir, 'blocked_model_scores.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\n[written] blocked_model_scores.json', flush=True)


if __name__ == '__main__':
    main()
