#!/usr/bin/env python3
"""07_strat_ablation.py: 方向②分层消融 —— e0(修复前后) vs 基线 v0/v2/v2o, 湿/干分层。

协议（与 05_strat / e0_windowed_eval / baseline_rollout 同一代码路径）:
- 层定义: wet = pm <= P_CRIT(22.064 MPa), dry = pm > P_CRIT
- rollout 分层: 1800 步递归轨迹按每步 pm 分类（e0: 落盘 npz; 基线: rollout_{v}_seed*.npz）
- windowed 分层: 每窗重置 60 步, stride 10, 窗口起点 pm 分类
  - e0: 与 e0_windowed_eval 同积分路径（平衡初始化 + integrate 60 步, 外生真值）
  - 基线: 与 baseline_rollout 同反馈协议（真外生 AR, 输出回填, phys 列重算, dspray_6 用真值）
- 漂移曲线: 窗口内误差 vs 步数(0..59), 分层按窗均值
产物:
  out/strat_ablation_summary.json  分层指标全表 + 决策检查
  out/strat_drift_curves.npz       漂移曲线数组
  out/figs/fig5_strat_bars.png     分层柱状对比 (rollout/windowed × 指标)
  out/figs/fig6_strat_drift.png    漂移曲线 (main/sh1_in × wet/dry × 模型)
  out/figs/fig7_strat_rollout.png  e0-post vs v2 长轨迹 + 湿态底纹 + 误差带
自校验: 重算的 e0-post 分层数字须与 strat_post_fixA.json 一致(Δ<0.1);
        基线整体 rollout rmse 须与 results_{v}_seed*.json 一致。
"""
import argparse
import importlib.util
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t02", os.path.join(os.getcwd(), "02_train.py"))
t02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t02)
import numpy as np
import pandas as pd
import torch

PAIRS_PHYS = [(1, 0), (3, 2), (1, 2), (3, 4), (0, 2)]  # 与 step1_summary 物理对一致
START = t02.TRAIN_N + t02.VAL_N
P_CRIT = t02.P_CRIT
GATES = {"P1_rollout_rmse_main_C": 2.48, "P2_rollout_band_viol_frac": 0.005}

E0_COLS = ["主蒸汽流量", "未校正总煤量", "分离器出口压力", "分离器出口温度",
           "省煤器出口给水温度", "一级减温调节门阀位", "二级减温调节门阀位",
           "末级过热器出口压力", "减温水总流量"]


def load_e0_df():
    return pd.read_csv(t02.CSV, usecols=E0_COLS + t02.OUTPUTS + [
        "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
        dtype=np.float32).iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)


def load_baseline_df():
    return pd.read_csv(t02.CSV, usecols=t02.EXO + t02.EXO_EXTRA + t02.OUTPUTS + [
        "一级减温调节门阀位", "二级减温调节门阀位", "分离器出口压力", "末级过热器出口压力"],
        dtype=np.float32).iloc[t02.WIN_START: t02.WIN_START + t02.WIN].ffill().bfill().reset_index(drop=True)


# ---------------- rollout 分层（npz → 每步 pm 分类） ----------------
def strat_rollout(preds, truths, pm, mu_o, sd_o):
    out = {}
    for mode, mask in (("wet", pm <= P_CRIT), ("dry", pm > P_CRIT)):
        if mask.sum() == 0:
            out[mode] = None
            continue
        p, t = preds[mask], truths[mask]
        viol = np.zeros(mask.sum(), dtype=bool)
        for lo, hi in PAIRS_PHYS:
            viol |= (p[:, lo] >= p[:, hi])
        z = np.abs((p - mu_o) / sd_o)
        out[mode] = {
            "n": int(mask.sum()),
            "pm_mean": round(float(pm[mask].mean()), 2),
            "rmse_main": round(float(np.sqrt(np.mean((p[:, 4] - t[:, 4]) ** 2))), 3),
            "rmse_all": round(float(np.sqrt(np.mean((p - t) ** 2))), 3),
            "bias_5": [round(float((p[:, j] - t[:, j]).mean()), 2) for j in range(5)],
            "band_viol_frac": round(float(np.mean((p[:, 4] > t02.T_BAND[1]) | (p[:, 4] < t02.T_BAND[0]))), 4),
            "viol_phys_frac": round(float(viol.mean()), 4),
            "drift_main_mean_z": round(float(z[:, 4].mean()), 3),
        }
    return out


# ---------------- e0 windowed 分层（与 05_strat.strat_windowed 同代码路径, 返回数组） ----------------
def k_eq(model, pm):
    if hasattr(model, "k_of"):
        return model.k_of(pm)
    return model.tri("k")[:, None].expand(3, pm.shape[0])


def e0_windowed_arrays(df, model):
    Xte, Yte, Ite, Ite_T = t02.e0_build_windows(df, START, len(df) - 1, 10)
    model.eval()
    errs_main, errs_sh1, preds_main, pm0_list = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(Xte), 256):
            xb = torch.from_numpy(Xte[b: b + 256]).to(t02.DEVICE)
            yb = torch.from_numpy(Yte[b: b + 256]).to(t02.DEVICE)
            ib = Ite[b: b + 256]
            pm = torch.from_numpy(ib[:, 2]).to(t02.DEVICE)
            p_out = torch.from_numpy(ib[:, 7]).to(t02.DEVICE)
            p0 = pm + (p_out - pm) / 3.0
            p1 = pm + 2.0 * (p_out - pm) / 3.0
            obs = torch.from_numpy(Ite_T[b: b + 256]).to(t02.DEVICE)
            h0 = t02.h_of_pT(p0, obs[:, 0])
            h1 = t02.h_of_pT(p1, obs[:, 2])
            h2 = t02.h_of_pT(p_out, obs[:, 4])
            h = torch.stack([h0, h1, h2])
            ts = t02.T_of_ph(torch.stack([p0, p1, p_out]), h)
            rB = torch.from_numpy(ib[:, 1]).to(t02.DEVICE).clone()
            Tm = (ts + k_eq(model, pm) * rB[None, :] / 3600.0 / model.tri("UA")[:, None]
                  + model.tri("dTm")[:, None])
            pred = model.integrate(xb, h, Tm, rB, xb.shape[1])
            err = pred - yb
            errs_main.append(err[:, :, 4].cpu().numpy())
            errs_sh1.append(err[:, :, 0].cpu().numpy())
            preds_main.append(pred[:, :, 4].cpu().numpy())
            pm0_list.append(ib[:, 2])
    return (np.concatenate(errs_main), np.concatenate(errs_sh1),
            np.concatenate(preds_main), np.concatenate(pm0_list))


# ---------------- 基线 windowed 分层（baseline_rollout 同反馈协议, 窗口并行 AR） ----------------
def baseline_windowed_arrays(df_b, model, info, phys):
    exo_cols = info["exo_cols"]
    phy_cols = info["phy_cols"]
    cols = exo_cols + t02.OUTPUTS + (phy_cols if phys else [])
    mu = pd.Series(info["mu"])
    sd = pd.Series(info["sd"])
    feats = t02.physics_features(df_b) if phys else None
    full = pd.concat([df_b[exo_cols], df_b[t02.OUTPUTS]] + ([feats] if feats is not None else []), axis=1)
    Z = ((full[cols] - mu[cols]) / sd[cols]).to_numpy(np.float32)
    out_start = len(cols) - len(phy_cols) - 5 if phys else len(cols) - 5
    mu_o = np.array([info["mu"][c] for c in t02.OUTPUTS], np.float32)
    sd_o = np.array([info["sd"][c] for c in t02.OUTPUTS], np.float32)
    i0 = np.arange(START, len(df_b) - 1 - t02.SEQ, 10)
    hist = np.stack([Z[s - t02.SEQ: s] for s in i0])  # (Nw, SEQ, F)
    errs_main = np.zeros((len(i0), t02.SEQ), np.float32)
    errs_sh1 = np.zeros((len(i0), t02.SEQ), np.float32)
    preds_main = np.zeros((len(i0), t02.SEQ), np.float32)
    truth_all = df_b[t02.OUTPUTS].to_numpy()
    cmd = df_b["二级减温喷水调节门指令"].to_numpy()
    w_exo = df_b["减温水总流量"].to_numpy()
    feco = df_b["省煤器出口给水温度"].to_numpy()
    sep = df_b["分离器出口温度"].to_numpy()
    steam = df_b["主蒸汽流量"].to_numpy()
    coal = df_b["未校正总煤量"].to_numpy()
    pm_series = df_b["分离器出口压力"].to_numpy()
    model.eval()
    with torch.no_grad():
        for t in range(t02.SEQ):
            idx = i0 + t
            yz = model(torch.from_numpy(hist).to(t02.DEVICE))
            pred = t02.model_out(yz, info).cpu().numpy()  # (Nw,5)
            truth = truth_all[idx]
            errs_main[:, t] = pred[:, 4] - truth[:, 4]
            errs_sh1[:, t] = pred[:, 0] - truth[:, 0]
            preds_main[:, t] = pred[:, 4]
            new_z = Z[idx].copy()
            new_z[:, out_start: out_start + 5] = (pred - mu_o) / sd_o
            if phys:
                sh1i, sh1o, sh2i, sh2o, main = [pred[:, j] for j in range(5)]
                feats_arr = np.stack([
                    w_exo[idx] * (sh1i - feco[idx]),
                    w_exo[idx] * (sh2i - feco[idx]),
                    steam[idx] * (sh1i - sep[idx]),
                    steam[idx] * (sh2i - sh1o),
                    steam[idx] * (main - sh2o),
                    coal[idx] / (steam[idx] + 1.0),
                    cmd[idx] - cmd[idx - 6],
                ], axis=1)  # (Nw,7) 顺序同 physics_features
                for k, c in enumerate(phy_cols):
                    new_z[:, out_start + 5 + k] = (feats_arr[:, k] - mu[c]) / sd[c]
            hist = np.concatenate([hist[:, 1:], new_z[:, None]], axis=1)
    return errs_main, errs_sh1, preds_main, pm_series[i0]


# ---------------- 窗口内分层聚合 ----------------
def layer_agg(errs_main, errs_sh1, preds_main, pm0):
    out = {}
    for mode, mask in (("wet", pm0 <= P_CRIT), ("dry", pm0 > P_CRIT)):
        if mask.sum() == 0:
            out[mode] = None
            continue
        m = errs_main[mask]
        s1 = errs_sh1[mask]
        p = preds_main[mask]
        first = m[:, 0]
        out[mode] = {
            "n_win": int(mask.sum()),
            "first_rmse_main": round(float(np.sqrt(np.mean(first ** 2))), 3),
            "win60_rmse_main": round(float(np.sqrt(np.mean(m ** 2))), 3),
            "first_bias_sh1in": round(float(s1[:, 0].mean()), 2),
            "first_bias_main": round(float(first.mean()), 2),
            "win60_band_viol_frac": round(float(np.mean((p > t02.T_BAND[1]) | (p < t02.T_BAND[0]))), 4),
        }
    return out


def crossing_frac(df):
    E = df[E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    E = E.to_numpy(np.float32)
    i0 = np.arange(START, len(df) - 1 - t02.SEQ, 10)
    cross = np.array([
        bool(((E[s: s + t02.SEQ, 2] > P_CRIT).any()) and ((E[s: s + t02.SEQ, 2] <= P_CRIT).any()))
        for s in i0])
    return round(float(cross.mean()), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    df = load_e0_df()
    df_b = load_baseline_df()
    mu_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].mean().to_numpy(np.float32)
    sd_o = df[t02.OUTPUTS].iloc[:t02.TRAIN_N].std().replace(0, 1.0).to_numpy(np.float32)
    pm_all = df["分离器出口压力"].to_numpy(np.float32)
    pm_roll = pm_all[START: START + t02.ROLL_STEPS]
    wet_frac = {
        "train": round(float((pm_all[:t02.TRAIN_N] <= P_CRIT).mean()), 4),
        "val": round(float((pm_all[t02.TRAIN_N:START] <= P_CRIT).mean()), 4),
        "test": round(float((pm_all[START:] <= P_CRIT).mean()), 4),
    }
    cross_frac_val = crossing_frac(df)
    print(f"[info] wet_frac={wet_frac} crossing_win_frac={cross_frac_val}", flush=True)

    summ = {"layer_def": f"wet = pm <= {P_CRIT} MPa", "wet_frac": wet_frac,
            "crossing_win_frac": cross_frac_val, "gates": GATES,
            "rollout": {}, "windowed": {}}
    drift = {}

    # ============ rollout 分层 ============
    for v in ("e0_post", "v0", "v2", "v2o"):
        summ["rollout"][v] = {}
        for s in seeds:
            npz_tag = "e0" if v == "e0_post" else v
            d = np.load(os.path.join(t02.OUT, f"rollout_{npz_tag}_seed{s}.npz"))
            r = strat_rollout(d["preds"], d["truths"], pm_roll, mu_o, sd_o)
            summ["rollout"][v][str(s)] = r
            # 自校验: 整体 rmse 与 results json 一致
            rmse_all_steps = float(np.sqrt(np.mean((d["preds"][:, 4] - d["truths"][:, 4]) ** 2)))
            ref = json.load(open(os.path.join(t02.OUT, f"results_{npz_tag}_seed{s}.json")))["rmse_main"]
            if abs(rmse_all_steps - ref) > 0.01:
                raise SystemExit(f"[FAIL] rollout 整体 rmse 与 results 不一致: {v} s{s} {rmse_all_steps} vs {ref}")
        # 跨 seed 聚合
        rows = [summ["rollout"][v][str(s)] for s in seeds]
        for mode in ("wet", "dry"):
            sub = [r[mode] for r in rows if r[mode] is not None]
            if sub:
                agg = {k: round(float(np.mean([r[k] for r in sub])), 4) for k in
                       ("rmse_main", "rmse_all", "band_viol_frac", "viol_phys_frac", "drift_main_mean_z")}
                agg["n"] = int(sub[0]["n"])
                agg["bias_5"] = [round(float(np.mean([r["bias_5"][j] for r in sub])), 2) for j in range(5)]
                summ["rollout"][v][f"agg_{mode}"] = agg
    print("[ok] rollout 分层完成 (e0_post/v0/v2/v2o × 3 seeds, 自校验与 results json 一致)", flush=True)

    # ============ e0-post windowed ============
    e0_arrs = {}
    for s in seeds:
        model = t02.E0Model().to(t02.DEVICE)
        model.load_state_dict(torch.load(os.path.join(t02.OUT, f"model_e0_seed{s}.pt"),
                                         map_location=t02.DEVICE, weights_only=True))
        e0_arrs[s] = e0_windowed_arrays(df, model)
        r = layer_agg(*e0_arrs[s])
        summ["windowed"].setdefault("e0_post", {})[str(s)] = r
        print(f"[e0_post s{s}] wet={r['wet']}", flush=True)
    # 与 strat_post_fixA.json 自校验
    sp = json.load(open(os.path.join(t02.OUT, "strat_post_fixA.json")))
    for mode in ("wet", "dry"):
        for k_ref, k_mine in (("win60_rmse_main", "win60_rmse_main"), ("first_rmse_main", "first_rmse_main")):
            ref_val = sp["windowed"][mode][k_ref]
            mine_vals = [summ["windowed"]["e0_post"][str(s)][mode][k_mine] for s in seeds]
            mine = float(np.mean(mine_vals))
            if abs(mine - ref_val) > 0.1:
                raise SystemExit(f"[FAIL] e0_post windowed {mode}.{k_ref}: 重算 {mine} vs strat_post {ref_val}")
    print("[ok] e0_post windowed 与 strat_post_fixA.json 一致", flush=True)

    # ============ 基线 windowed (v2/v2o) ============
    for v in ("v2", "v2o"):
        phys = True
        info = t02.baseline_build_data(df_b, v, phys)[7]
        for s in seeds:
            model = t02.Net(info["F"]).to(t02.DEVICE)
            model.load_state_dict(torch.load(os.path.join(t02.OUT, f"model_{v}_seed{s}.pt"),
                                             map_location=t02.DEVICE, weights_only=True))
            arrs = baseline_windowed_arrays(df_b, model, info, phys)
            r = layer_agg(*arrs)
            summ["windowed"].setdefault(v, {})[str(s)] = r
            drift.setdefault(v, []).append((arrs[0], arrs[1], arrs[3]))
            print(f"[{v} s{s}] wet={r['wet']}", flush=True)
    for s in seeds:
        drift.setdefault("e0_post", []).append((e0_arrs[s][0], e0_arrs[s][1], e0_arrs[s][3]))

    # ============ e0-pre: 从 strat_pre_fixA.json 读聚合 ============
    spre = json.load(open(os.path.join(t02.OUT, "strat_pre_fixA.json")))
    summ["rollout"]["e0_pre"] = {f"agg_{mode}": spre["rollout"][mode] for mode in ("wet", "dry")}
    summ["windowed"]["e0_pre"] = {f"agg_{mode}": spre["windowed"][mode] for mode in ("wet", "dry")}

    # ============ 跨 seed 聚合 windowed ============
    for v, rows in summ["windowed"].items():
        if v == "e0_pre":
            continue
        seed_vals = {str(s): rows[str(s)] for s in seeds if str(s) in rows}
        for mode in ("wet", "dry"):
            sub = [r[mode] for r in seed_vals.values() if r[mode] is not None]
            if sub:
                agg = {k: round(float(np.mean([r[k] for r in sub])), 3) for k in
                       ("first_rmse_main", "win60_rmse_main", "first_bias_sh1in",
                        "first_bias_main", "win60_band_viol_frac")}
                agg["n_win"] = int(sub[0]["n_win"])
                summ["windowed"][v][f"agg_{mode}"] = agg

    # ============ 漂移曲线 (pool seeds, 分层均值) ============
    drift_out = {}
    for v, arrs_list in drift.items():
        em = np.concatenate([a[0] for a in arrs_list], axis=0)   # (Nw,60)
        es = np.concatenate([a[1] for a in arrs_list], axis=0)
        pm0 = np.concatenate([a[2] for a in arrs_list])
        drift_out[v] = {}
        for mode, mask in (("wet", pm0 <= P_CRIT), ("dry", pm0 > P_CRIT)):
            drift_out[v][mode] = {
                "main_mean": em[mask].mean(0).round(3).tolist(),
                "main_std": em[mask].std(0).round(3).tolist(),
                "sh1in_mean": es[mask].mean(0).round(3).tolist(),
                "sh1in_std": es[mask].std(0).round(3).tolist(),
                "n_win": int(mask.sum()),
            }
    np.savez(os.path.join(t02.OUT, "strat_drift_curves.npz"),
             drift=json.dumps(drift_out).encode("utf-8"))
    print("[ok] 漂移曲线已存", flush=True)

    # ============ 决策检查 ============
    ck = {}
    e0_r = summ["rollout"]["e0_post"]
    v2_r = summ["rollout"]["v2"]
    e0_w = summ["windowed"]["e0_post"]
    v2_w = summ["windowed"]["v2"]
    ck["Q1_rollout_dry_fails_e0_post"] = {
        "wet": e0_r["agg_wet"]["rmse_main"], "dry": e0_r["agg_dry"]["rmse_main"],
        "gate": GATES["P1_rollout_rmse_main_C"],
        "dry_better_than_wet": e0_r["agg_dry"]["rmse_main"] < e0_r["agg_wet"]["rmse_main"],
    }
    ck["Q2_gap_vs_v2_rollout"] = {
        "wet": round(e0_r["agg_wet"]["rmse_main"] - v2_r["agg_wet"]["rmse_main"], 2),
        "dry": round(e0_r["agg_dry"]["rmse_main"] - v2_r["agg_dry"]["rmse_main"], 2),
    }
    ck["Q3_gap_vs_v2_windowed60"] = {
        "wet": round(e0_w["agg_wet"]["win60_rmse_main"] - v2_w["agg_wet"]["win60_rmse_main"], 2),
        "dry": round(e0_w["agg_dry"]["win60_rmse_main"] - v2_w["agg_dry"]["win60_rmse_main"], 2),
    }
    ck["Q4_band_vs_v2_rollout"] = {
        "wet": {"e0": e0_r["agg_wet"]["band_viol_frac"], "v2": v2_r["agg_wet"]["band_viol_frac"]},
        "dry": {"e0": e0_r["agg_dry"]["band_viol_frac"], "v2": v2_r["agg_dry"]["band_viol_frac"]},
    }
    summ["decision_checks"] = ck

    with open(os.path.join(t02.OUT, "strat_ablation_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("[json] out/strat_ablation_summary.json", flush=True)

    # ============ 图 ============
    make_figs(df, summ, drift_out)

    # ============ 打印决策表 ============
    print("\n=== 分层消融总表 (3 seeds 聚合) ===", flush=True)
    hdr = f"{'指标':<28}{'e0-pre':>12}{'e0-post':>12}{'v0':>10}{'v2':>10}{'v2o':>10}"
    print(hdr, flush=True)
    for mode in ("wet", "dry"):
        for key, lab in (("rmse_main", "rollout rmse_main"), ("band_viol_frac", "rollout band"),
                         ("viol_phys_frac", "rollout phys viol")):
            row = f"[{mode}] {lab:<20}"
            for v in ("e0_pre", "e0_post", "v0", "v2", "v2o"):
                d = summ["rollout"][v].get(f"agg_{mode}")
                row += f"{d[key]:>12.3f}" if d else f"{'—':>12}"
            print(row, flush=True)
    for mode in ("wet", "dry"):
        for key, lab in (("win60_rmse_main", "win60 rmse_main"), ("first_rmse_main", "first rmse_main"),
                         ("win60_band_viol_frac", "win60 band")):
            row = f"[{mode}] {lab:<20}"
            for v in ("e0_pre", "e0_post", "v2", "v2o"):
                d = summ["windowed"][v].get(f"agg_{mode}")
                row += f"{d[key]:>12.3f}" if d and key in d else f"{'—':>12}"
            print(row, flush=True)
    print("\n=== 决策检查 ===", flush=True)
    print(json.dumps(ck, ensure_ascii=False, indent=2), flush=True)


def make_figs(df, summ, drift_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = df[E0_COLS].copy()
    E["主蒸汽流量"] = E["主蒸汽流量"] / 3.6
    E["一级减温调节门阀位"] = E["一级减温调节门阀位"].clip(lower=0) / 100.0
    E["二级减温调节门阀位"] = E["二级减温调节门阀位"].clip(lower=0) / 100.0
    E = E.to_numpy(np.float32)
    pm_roll = E[START: START + t02.ROLL_STEPS, 2]
    t_axis = np.arange(t02.ROLL_STEPS) / 6.0

    C_WET, C_DRY = "#2e75b6", "#c55a11"
    models = ["e0_pre", "e0_post", "v0", "v2", "v2o"]
    mlab = {"e0_pre": "e0\npre-fixA", "e0_post": "e0\npost-fixA", "v0": "GRU-v0", "v2": "GRU-v2", "v2o": "GRU-v2o"}

    # ---------- fig5: 分层柱状 ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (axes[0, 0], "rollout", "rmse_main", "Rollout RMSE, main steam (°C)", None),
        (axes[0, 1], "rollout", "band_viol_frac", "Rollout band-violation rate", None),
        (axes[1, 0], "windowed", "win60_rmse_main", "Window-60 RMSE, main steam (°C)", None),
        (axes[1, 1], "windowed", "first_rmse_main", "First-step RMSE, main steam (°C)", None),
    ]
    for ax, scope, key, title, _ in panels:
        x = np.arange(len(models))
        for k, mode in enumerate(("wet", "dry")):
            vals, errs = [], []
            for v in models:
                d = summ[scope].get(v, {}).get(f"agg_{mode}")
                if d is None:
                    vals.append(np.nan)
                    errs.append(0)
                    continue
                vals.append(d[key])
                seed_vals = []
                vd = summ[scope].get(v, {})
                if str(0) in vd:
                    seed_vals = [vd[str(s)][mode][key] for s in (0, 1, 2)
                                 if vd.get(str(s), {}).get(mode) is not None]
                errs.append(np.std(seed_vals) if len(seed_vals) > 1 else 0)
            ax.bar(x + (k - 0.5) * 0.36, vals, 0.32, yerr=errs, label=mode,
                   color=[C_WET, C_DRY][k], alpha=0.9, capsize=3)
        if key == "rmse_main" and scope == "rollout":
            ax.axhline(GATES["P1_rollout_rmse_main_C"], color="crimson", ls="--", lw=1,
                       label=f"P1 gate {GATES['P1_rollout_rmse_main_C']}")
        ax.set_xticks(x)
        ax.set_xticklabels([mlab[v] for v in models], fontsize=9)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.set_title(f"({ 'abcd'[panels.index((ax, scope, key, title, None))] }) {title} — wet vs dry")
    fig.tight_layout()
    fig.savefig(os.path.join(t02.OUT, "figs", "fig5_strat_bars.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig5_strat_bars.png", flush=True)

    # ---------- fig6: 漂移曲线 ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    steps = np.arange(t02.SEQ)
    for row, (var, vlab) in enumerate((("main", "main steam"), ("sh1in", "sh1-in"))):
        for col, mode in enumerate(("wet", "dry")):
            ax = axes[row, col]
            for v, c, ls in (("e0_post", "#1f4e79", "-"), ("v2", "#2e8b57", "--"), ("v2o", "#8b008b", "-.")):
                if v not in drift_out or mode not in drift_out[v]:
                    continue
                m = np.array(drift_out[v][mode][f"{var}_mean"])
                s = np.array(drift_out[v][mode][f"{var}_std"])
                nw = drift_out[v][mode]["n_win"]
                ax.plot(steps, m, c, lw=1.6, ls=ls, label=f"{v} (n={nw})")
                ax.fill_between(steps, m - s / np.sqrt(nw), m + s / np.sqrt(nw), color=c, alpha=0.12)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{mode} windows — {vlab} error vs step", fontsize=10)
            ax.set_ylabel("err (°C)")
            if row == 1:
                ax.set_xlabel("steps since window start (×10 s)")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(t02.OUT, "figs", "fig6_strat_drift.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig6_strat_drift.png", flush=True)

    # ---------- fig7: 长轨迹 e0-post vs v2 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for row, v in enumerate(("e0_post", "v2")):
        npz_tag = "e0" if v == "e0_post" else v
        d = np.load(os.path.join(t02.OUT, f"rollout_{npz_tag}_seed0.npz"))
        pred, truth = d["preds"][:, 4], d["truths"][:, 4]
        ax, axe = axes[row]
        wet_mask = pm_roll <= P_CRIT
        ax.fill_between(t_axis, truth.min() - 2, truth.max() + 2, where=wet_mask,
                        color="steelblue", alpha=0.12, label="wet (pm ≤ 22.064)")
        ax.plot(t_axis, truth, color="0.35", lw=1.0, label="truth")
        ax.plot(t_axis, pred, color=[C_DRY, C_WET][row], lw=1.2, alpha=0.95,
                label=f"{v} pred (rmse={np.sqrt(np.mean((pred-truth)**2)):.1f}°C)")
        ax.set_ylabel("main steam (°C)")
        ax.set_title(f"{v} — 1800-step rollout (seed 0)")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        err = pred - truth
        axe.plot(t_axis, err, color=[C_DRY, C_WET][row], lw=0.9)
        axe.fill_between(t_axis, err, 0, where=np.abs(err) > 5, color="crimson", alpha=0.25)
        axe.axhline(0, color="k", lw=0.8)
        axe.axhline(5, color="crimson", ls=":", lw=0.8)
        axe.axhline(-5, color="crimson", ls=":", lw=0.8)
        axe.set_ylabel("err (°C)")
        axe.set_xlabel("time (min)")
    fig.tight_layout()
    fig.savefig(os.path.join(t02.OUT, "figs", "fig7_strat_rollout.png"), dpi=150)
    plt.close(fig)
    print("[fig] fig7_strat_rollout.png", flush=True)


if __name__ == "__main__":
    main()
