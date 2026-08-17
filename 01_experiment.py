#!/usr/bin/env python3
"""ad-hoc 实验 01：PINN 拆解迁移验证 — 物理特征 + 顺序不变量损失 vs 纯数据基线。
用法: python 01_experiment.py --variant v0|v1|v2
独立于主线实验；数据只读；结果写 out/。
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data/mainT/A侧主汽温全数据_cleaned_10s.csv"
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

WIN_START, WIN, SEQ = 70686, 50000, 60
TRAIN_N, VAL_N = 30000, 10000
ROLL_STEPS = 1800

OUTPUTS = ["一级减温器入口温度", "一级减温器出口温度", "二级减温器入口温度",
           "二级减温器出口温度", "末级过热器出口汽温"]
EXO = ["机组负荷", "主蒸汽压力", "主蒸汽流量", "主给水流量", "总风量指令", "总二次风量",
       "燃料主控输出", "未校正总煤量", "水煤比", "减温水总流量",
       "一级减温喷水调节门指令", "二级减温喷水调节门指令",
       "省煤器出口给水温度", "分离器出口温度", "分离器出口压力", "AGC指令", "机组负荷变化率"]
EXO_EXTRA = ["过热器出口温度升速率"]
T_BAND = [557.75, 572.13]  # 主汽温 p1/p99 (°C)

def parse_variant(v):
    """... / v2xb v1x+band(无顺序) ..."""
    phys = v in ("v1", "v1x", "v2", "v2x", "v2xb", "v2b", "v2o")
    band = v in ("v2", "v2x", "v2xb", "v2b", "v0b")
    order = v in ("v2", "v2x", "v2o")
    return phys, band, order

AUG_COLS = ["二级减温调节阀设定", "二级减温中间设定值", "一级减温副调设定值", "一级减温温度设定偏值",
            "末级过热器出口汽温_B", "一级减温喷水调节门指令_B", "二级减温喷水调节门指令_B",
            "再热出口汽温", "再热器减温水总流量", "高压缸排汽至再热器温度",
            "再热器一级减温入口汽温", "立式低温再热器入口烟气温度", "水平低温再热器入口烟气温度"]

class Net(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.gru = nn.GRU(F, 32, batch_first=True)
        self.fc = nn.Linear(32, 5)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

def physics_features(df):
    w = df["减温水总流量"].to_numpy(); feco = df["省煤器出口给水温度"].to_numpy()
    sh1i = df["一级减温器入口温度"].to_numpy(); sh1o = df["一级减温器出口温度"].to_numpy()
    sh2i = df["二级减温器入口温度"].to_numpy(); sh2o = df["二级减温器出口温度"].to_numpy()
    main = df["末级过热器出口汽温"].to_numpy(); sep = df["分离器出口温度"].to_numpy()
    steam = df["主蒸汽流量"].to_numpy(); coal = df["未校正总煤量"].to_numpy()
    f = pd.DataFrame({
        "spray_cool_1": w * (sh1i - feco),
        "spray_cool_2": w * (sh2i - feco),
        "adv_1": steam * (sh1i - sep),
        "adv_2": steam * (sh2i - sh1o),
        "adv_3": steam * (main - sh2o),
        "heat_intensity": coal / (steam + 1.0),
    })
    f["dspray_6"] = pd.Series(df["二级减温喷水调节门指令"]).diff(6).fillna(0.0).to_numpy()
    return f

class DS:
    def __init__(self, X, Y, exo_idx):
        self.X, self.Y, self.exo_idx = X, Y, exo_idx

def build_data(phys):
    df = pd.read_csv(CSV, usecols=EXO + EXO_EXTRA + OUTPUTS, dtype=np.float32)
    if VARIANT in ("v1x", "v2x", "v2xb"):
        aug = pd.read_csv(os.path.join(BASE, "data", "merged_aug.csv"))
        aug = aug.drop(columns=["date"]).astype(np.float32)
        assert len(aug) == len(df), f"aug 行数 {len(aug)} != 主表 {len(df)}"
        df = pd.concat([df, aug], axis=1)
    df = df.iloc[WIN_START: WIN_START + WIN].ffill().bfill().reset_index(drop=True)
    feats = physics_features(df) if phys else None
    exo_cols = EXO + (EXO_EXTRA if phys else []) + (AUG_COLS if VARIANT in ("v1x", "v2x", "v2xb") else [])
    phy_cols = list(feats.columns) if feats is not None else []
    frames = [df[exo_cols], df[OUTPUTS]] + ([feats] if feats is not None else [])
    raw = pd.concat(frames, axis=1)
    mu, sd = raw.iloc[:TRAIN_N].mean(), raw.iloc[:TRAIN_N].std().replace(0, 1.0)
    Z = ((raw - mu) / sd).to_numpy(np.float32)
    F = Z.shape[1]
    N = len(df)
    # 训练用 stride 5 采样窗口；val/test 用 stride 1
    def windows(lo, hi, stride):
        idx = np.arange(lo + SEQ - 1, hi, stride)
        X = np.stack([Z[i - SEQ + 1: i + 1] for i in idx])
        Y = Z[idx + 1][:, -5:] if not phys else Z[idx + 1][:, F - len(phy_cols) - 5: F - len(phy_cols)]
        return X, Y
    out_start = F - len(phy_cols) - 5 if phys else F - 5
    Xtr, Ytr = windows(0, TRAIN_N, 5)
    Xva, Yva = windows(TRAIN_N, TRAIN_N + VAL_N, 1)
    Xte, Yte = windows(TRAIN_N + VAL_N, N - 1, 1)
    info = {"mu": mu.to_dict(), "sd": sd.to_dict(), "F": F, "out_start": out_start,
            "exo_cols": exo_cols, "phy_cols": phy_cols, "n_train": len(Xtr)}
    return Xtr, Ytr, Xva, Yva, Xte, Yte, df, info

def model_out(yz, info):
    """z-score 输出 → 物理温度 (B,5)"""
    cols = OUTPUTS
    mu = torch.tensor([info["mu"][c] for c in cols], device=DEVICE, dtype=torch.float32)
    sd = torch.tensor([info["sd"][c] for c in cols], device=DEVICE, dtype=torch.float32)
    return yz * sd + mu

def order_loss(phys, t_sep, use_order):
    if not use_order:
        return torch.zeros((), device=DEVICE)
    s1i, s1o, s2i, s2o, main = phys[:, 0], phys[:, 1], phys[:, 2], phys[:, 3], phys[:, 4]
    L = torch.relu(t_sep - s1i + 0.5)                      # 100%: sh1_in > sep_out
    L = L + torch.relu(s1o - s2i + 0.5)                    # 100%: sh2_in > sh1_out
    L = L + torch.relu(s2o - main + 0.5)                   # 100%: main > sh2_out
    L = L + 0.75 * torch.relu(s2i - s2o + 0.5)             # 91%
    L = L + 0.3 * torch.relu(s1o - s1i + 0.5)              # 66%
    return L.mean()

def band_loss(phys, use_band):
    if not use_band:
        return torch.zeros((), device=DEVICE)
    main = phys[:, 4]
    return (torch.relu(main - T_BAND[1]) + torch.relu(T_BAND[0] - main)).mean()

def train(Xtr, Ytr, Xva, Yva, info, use_band, use_order):
    F = info["F"]
    sep_idx = info["exo_cols"].index("分离器出口温度")
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE); Ytr_t = torch.from_numpy(Ytr).to(DEVICE)
    Xva_t = torch.from_numpy(Xva).to(DEVICE); Yva_t = torch.from_numpy(Yva).to(DEVICE)
    mu_sep = info["mu"]["分离器出口温度"]; sd_sep = info["sd"]["分离器出口温度"]
    model = Net(F).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=DEVICE)
    best_va, best_state, patience = 1e9, None, 0
    band_act = order_act = 0.0
    n_batches_total = 0
    n_batch = len(Xtr_t) // 256
    for ep in range(60):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        for b in range(n_batch):
            i = perm[b * 256: (b + 1) * 256]
            yz = model(Xtr_t[i])
            phys = model_out(yz, info)
            t_sep = Xtr_t[i][:, -1, sep_idx] * sd_sep + mu_sep
            mse = ((yz - Ytr_t[i]) ** 2 * w).mean()
            lo = order_loss(phys, t_sep, use_order)
            lb = band_loss(phys, use_band)
            band_act += float((lb > 0).float().mean())
            order_act += float((lo > 0).float().mean())
            n_batches_total += 1
            loss = mse + 0.05 * lo + 0.05 * lb
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            yz = model(Xva_t)
            va = ((yz - Yva_t) ** 2 * w).mean().item()
        if va < best_va:
            best_va, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, f"model_{VARIANT}.pt"))
    return model, best_va, band_act / max(n_batches_total, 1), order_act / max(n_batches_total, 1)

def eval_single(model, Xte, Yte):
    model.eval()
    with torch.no_grad():
        yz = model(torch.from_numpy(Xte).to(DEVICE))
        err = (yz.cpu().numpy() - Yte)  # z 空间
    return err

def rollout(model, df, info, phys):
    """1800 步递归 rollout：5 温度自回归，外生用真实值"""
    use_phys = phys
    exo_cols = info["exo_cols"]; phy_cols = info["phy_cols"]
    cols = exo_cols + OUTPUTS + (phy_cols if use_phys else [])
    feats = physics_features(df) if use_phys else None
    full = pd.concat([df[exo_cols], df[OUTPUTS]] + ([feats] if feats is not None else []), axis=1)
    mu = pd.Series(info["mu"]); sd = pd.Series(info["sd"])
    start = TRAIN_N + VAL_N
    window = full.iloc[start - SEQ: start].reset_index(drop=True).copy()
    preds, truths = [], []
    model.eval()
    with torch.no_grad():
        for t in range(ROLL_STEPS):
            i = start + t
            true_row = df.iloc[i]
            z = np.ascontiguousarray(((window[cols] - mu[cols]) / sd[cols]).to_numpy(np.float32))
            yz = model(torch.from_numpy(z[None]).to(DEVICE))[0]
            pred_T = model_out(yz.unsqueeze(0), info)[0].cpu().numpy()
            preds.append(pred_T)
            truths.append([true_row[c] for c in OUTPUTS])
            # 构造下一行: 外生真实值, 温度用预测
            new = pd.Series(0.0, index=cols)
            for c in exo_cols:
                new[c] = true_row[c]
            for j, c in enumerate(OUTPUTS):
                new[c] = pred_T[j]
            if use_phys:
                for k, c in enumerate(phy_cols):
                    if c == "dspray_6":
                        prev = df.iloc[i - 6]["二级减温喷水调节门指令"]
                        new[c] = true_row["二级减温喷水调节门指令"] - prev
                    else:
                        new[c] = 0.0  # 下面用公式重算
                new = recompute_phys(new)
            window = pd.concat([window.iloc[1:], new.to_frame().T], ignore_index=True)
    preds = np.array(preds); truths = np.array(truths)
    main_p, main_t = preds[:, 4], truths[:, 4]
    r = {
        "rmse_main": float(np.sqrt(np.mean((main_p - main_t) ** 2))),
        "maxerr_main": float(np.max(np.abs(main_p - main_t))),
        "rmse_all": float(np.sqrt(np.mean((preds - truths) ** 2))),
        "band_viol_frac": float(np.mean((main_p > T_BAND[1]) | (main_p < T_BAND[0]))),
        "order_viol_frac": float(np.mean(preds[:, 3] >= preds[:, 4])),  # main > sh2_out
    }
    # 输入分布漂移: 预测温度相对训练分布的 |z|
    mu_o = np.array([info["mu"][c] for c in OUTPUTS], dtype=np.float32)
    sd_o = np.array([info["sd"][c] for c in OUTPUTS], dtype=np.float32)
    z_all = np.abs((preds - mu_o) / sd_o)
    r["drift_main_mean_z"] = round(float(z_all[:, 4].mean()), 3)
    r["drift_main_max_z"] = round(float(z_all[:, 4].max()), 3)
    r["drift_all_mean_z"] = round(float(z_all.mean()), 3)
    np.savez(os.path.join(OUT, f"rollout_{VARIANT}.npz"), preds=preds, truths=truths)
    return r, preds, truths

def recompute_phys(row):
    w = row["减温水总流量"]; feco = row["省煤器出口给水温度"]
    sh1i = row["一级减温器入口温度"]; sh1o = row["一级减温器出口温度"]
    sh2i = row["二级减温器入口温度"]; sh2o = row["二级减温器出口温度"]
    main = row["末级过热器出口汽温"]; sep = row["分离器出口温度"]
    steam = row["主蒸汽流量"]; coal = row["未校正总煤量"]
    row["spray_cool_1"] = w * (sh1i - feco)
    row["spray_cool_2"] = w * (sh2i - feco)
    row["adv_1"] = steam * (sh1i - sep)
    row["adv_2"] = steam * (sh2i - sh1o)
    row["adv_3"] = steam * (main - sh2o)
    row["heat_intensity"] = coal / (steam + 1.0)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v0", "v1", "v1x", "v2x", "v2xb", "v2", "v0b", "v2b", "v2o"],
                    default=None, help="None=跑全部拆解变体")
    args = ap.parse_args()
    global VARIANT
    variants = [args.variant] if args.variant else ["v0", "v1", "v1x", "v0b", "v2o", "v2b", "v2"]
    summary = {}
    for v in variants:
        VARIANT = v
        phys, band, order = parse_variant(v)
        run_one(v, phys, band, order, summary)
    if not args.variant:
        with open(os.path.join(OUT, "summary_ablation.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n=== SUMMARY ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

def run_one(v, phys, band, order, summary):
    t0 = time.time()
    print(f"[{v}] phys={phys} band={band} order={order} building ...", flush=True)
    Xtr, Ytr, Xva, Yva, Xte, Yte, df, info = build_data(phys)
    print(f"[{v}] train={len(Xtr)} val={len(Xva)} test={len(Xte)} F={info['F']} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    model, va, band_act, order_act = train(Xtr, Ytr, Xva, Yva, info, band, order)
    print(f"[{v}] trained, val mse(z)={va:.4f} band_act={band_act:.3f} order_act={order_act:.3f} ({time.time()-t0:.0f}s)", flush=True)
    err = eval_single(model, Xte, Yte)
    single_rmse = float(np.sqrt(np.mean(err[:, 4] ** 2))) * info["sd"]["末级过热器出口汽温"]
    t0 = time.time()
    r, preds, truths = rollout(model, df, info, phys)
    r["variant"] = v; r["single_rmse_main_C"] = round(single_rmse, 3)
    r["val_mse_z"] = round(va, 4); r["rollout_seconds"] = round(time.time() - t0, 1)
    r["band_act"] = round(band_act, 4); r["order_act"] = round(order_act, 4)
    with open(os.path.join(OUT, f"results_{v}.json"), "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
    summary[v] = {k: r[k] for k in ["single_rmse_main_C", "rmse_main", "maxerr_main",
                                    "band_viol_frac", "drift_main_mean_z", "drift_main_max_z",
                                    "val_mse_z", "band_act", "order_act"]}

if __name__ == "__main__":
    main()
