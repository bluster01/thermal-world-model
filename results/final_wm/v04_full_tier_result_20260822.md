# v0.4 终审判决（执行侧，2026-08-22）

链：`v04_full_tier_20260822.sh`（HEAD=e95bb88，--compile --tf32，单进程串行）。
运行 16:18→20:20。产物全部独立命名，intact 产物未动。

## 判决表

| 单元 | 结果 | 关键数字 |
|---|---|---|
| T1 closure_cons_norew ×3 | 完成，无 RESUMED | best_val **1.536 / 1.240 / 1.272**（中位 1.272）；H18 MAE 2.50/2.06/2.15°C |
| R1（norew 栈证据门） | **SUPPORTED** | 三 seed direction frac_negative=1.0；泄漏 delta 0.58/0.58/0.13pp，suspected=False |
| leakdist ×3 | 完成 | 16-shuffle 零分布：delta 远低于 null（seed0/1 z≪0，seed2 z=+101 但 null 均值为负），无泄漏签名 |
| auditpack seed0 | 完成，13 区段 | **rewetting_ablation 恒等自检通过**（intact==zeroed，−0.337，aW 已冻结→消融无效应） |

## 终审判据核对（runbook §判决规则）

1. **方向全对** ✓ — R1 三 seed frac_negative=1.0；auditpack v1 探针下游无反号
2. **val NLL 中位** ✓ — norew 1.272 vs intact 栈（890bd15 审计值 1.214/1.260/1.228，
   中位 1.260）：差 **+0.012 ≤ +0.05**

**终审结论：SUPPORTED —— closure_cons_norew 具备侧A生产臂资格**（预注册规则
满足；最终口径裁定权在用户/设计侧）。

## 备注

- 实际时长：T1×3 ≈ 4h（与 intact 栈同量级）；R1/leakdist/auditpack 各 <1min
  （runbook 预估 40min/2h/1.5h 显著高估——compile 后探针在 GB10 上为秒级，
  产物内容已验证完整，非短路）
- summary units：o1 + r1（对侧重建权威块）+ r1_closure_cons_norew（本轮新增）
- 判决下一步（待裁定）：若采纳 norew 为生产臂 → 论文口径「最小闭包 SUPPORTED；
  再湿机制侧A不可辨识」；intact 栈 T1/R1 全档产物留档作对照。
