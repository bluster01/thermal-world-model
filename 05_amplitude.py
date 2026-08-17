#!/usr/bin/env python3
"""05: shape/幅度指标 + v1 残差候选驱动筛查
- 形状: Pearson corr, centered RMSE (去均值偏差后)
- 幅度: bias, std 比
- 残差筛查: v1 rollout 残差 vs 再热侧/烟气侧/B侧/SP 候选变量
"""
import os, json
import numpy as np
import pandas as pd

DATA = "/home/bluster/Desktop/AI/时序预测/TIME-all-model/TCN-Improved-GRU/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
WIN_START, TRAIN_N, VAL_N = 70686, 30000, 10000
ABS0 = WIN_START + TRAIN_N + VAL_N   # rollout 起点绝对行号
ROLL = 1800

# ---------- 1) shape/幅度指标 ----------
print("== shape/幅度指标 (rollout 主汽温) ==")
print(f"{'variant':8} {'corr':>6} {'centeredRMSE':>12} {'bias':>7} {'std_ratio':>9} {'rmse':>7}")
metrics = {}
for v in ["v0", "v1", "v0b", "v2o", "v2b", "v2"]:
    d = np.load(os.path.join(OUT, f"rollout_{v}.npz"))
    p, t = d["preds"][:, 4], d["truths"][:, 4]
    corr = np.corrcoef(p, t)[0, 1]
    pc, tc = p - p.mean(), t - t.mean()
    crmse = float(np.sqrt(np.mean((pc - tc) ** 2)))
    bias = float(p.mean() - t.mean())
    sr = float(p.std() / t.std())
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    metrics[v] = dict(corr=round(corr, 3), crmse=round(crmse, 2), bias=round(bias, 2),
                      std_ratio=round(sr, 3), rmse=round(rmse, 2))
    print(f"{v:8} {corr:6.3f} {crmse:12.2f} {bias:7.2f} {sr:9.3f} {rmse:7.2f}")

# ---------- 2) 残差筛查 ----------
print("\n== v1 残差候选驱动筛查 ==")
A = pd.read_csv(f"{DATA}/mainT/A侧主汽温全数据_cleaned_10s.csv")
B = pd.read_csv(f"{DATA}/mainT/B侧主汽温全数据_cleaned_10s.csv")
RH = pd.read_csv(f"{DATA}/reheat/A侧再热汽温全数据_cleaned_10s.csv")
for f in (A, B, RH):
    f["date"] = f["date"].astype(str)
RH2 = RH[["date", "再热器减温水总流量", "再热出口汽温", "再热器入口蒸汽温度",
          "立式低温再热器入口烟气温度", "水平低温再热器入口烟气温度",
          "再热减中间设定值", "高压缸排汽至再热器温度", "再热器减温器后蒸汽温度",
          "再热器一级减温入口汽温", "再热蒸汽压力(DEH)", "再热冷段蒸汽压力(DEH)"]]
B2 = B[["date", "末级过热器出口汽温", "一级减温喷水调节门指令", "二级减温喷水调节门指令",
        "二级减温中间设定值", "一级减温副调设定值", "一级减温温度设定偏值"]]
B2.columns = ["date"] + [c + "_B" for c in B2.columns[1:]]
A2 = A[["date", "二级减温中间设定值", "一级减温副调设定值", "一级减温温度设定偏值",
        "二级减温调节阀设定", "机组负荷变化率", "AGC指令", "过热器出口温度升速率"]]
merged = A[["date"]].merge(A2, on="date").merge(B2, on="date").merge(RH2, on="date")
print(f"merge ok: {len(merged)} rows (A侧 {len(A)}, B侧 {len(B)}, 再热 {len(RH)})")

sub = merged.iloc[ABS0: ABS0 + ROLL].reset_index(drop=True)
d = np.load(os.path.join(OUT, "rollout_v1.npz"))
res = d["preds"][:, 4] - d["truths"][:, 4]   # v1 残差 (预测-真实)

cands = [c for c in sub.columns if c != "date"]
rows = []
for c in cands:
    x = sub[c].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x)
    r_level = np.corrcoef(x, res)[0, 1]
    dx = np.diff(x)
    r_diff = np.corrcoef(dx, np.diff(res))[0, 1]
    rows.append((abs(r_level), c, r_level, r_diff))
rows.sort(reverse=True)
print(f"{'candidate':26s} {'corr_level':>10} {'corr_diff6':>10}")
for a, c, rl, rd in rows[:12]:
    print(f"{c:26s} {rl:10.3f} {rd:10.3f}")

# 残差与真实主汽温偏离设定值的关系 (幅度误差是否随偏离增大)
truth = sub["末级过热器出口汽温"].to_numpy() if "末级过热器出口汽温" in sub else None
# A 侧主汽温在 merged 里没带, 补上
mA = A[["date", "末级过热器出口汽温"]].rename(columns={"末级过热器出口汽温": "Tmain_A"})
sub2 = sub.merge(mA, on="date", how="left")
tmain = sub2["Tmain_A"].to_numpy()
dev = np.abs(tmain - tmain.mean())
print(f"\n残差幅度 vs 真实主汽温偏离均值: corr(|res|,|dev|) = {np.corrcoef(np.abs(res), dev)[0,1]:.3f}")
print(f"残差幅度 vs 再热出口汽温偏离均值: corr(|res|,|RH_out-dev|) = "
      f"{np.corrcoef(np.abs(res), np.abs(sub['再热出口汽温'].to_numpy()-sub['再热出口汽温'].mean()))[0,1]:.3f}")
