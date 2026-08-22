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

## 勘误（对侧独立审计 cf1ecb2，2026-08-23）

1. intact 栈 val NLL 中位：本报告初版误引 1.260——审计值
   1.214/**1.228**/1.260 排序后中位为 **1.228**。真差值
   norew 1.272 − intact 1.228 = **+0.044** ≤ +0.05（过门边际 0.006，非初版
   所报 +0.012）。结论不变。
2. H18 MAE 数字无源：初版引日志 eval 行（2.50/2.06/2.15），权威口径为
   metrics 文件：norew **3.22/2.74/2.80** vs intact 2.65/2.79/2.62
   （均差 +0.23°C）。H18 MAE 不入终审门，仅记录。

## 终审确认（2026-08-22 20:xx，用户指示推进）

amendment v0.4 预注册规则：「三 seed 一致（方向全对且 val NLL 不差于
intact+0.05）则侧A生产臂切换为 norew」——两判据均满足（见上表），
**生产臂切换生效：closure_cons_norew 为侧A生产臂，closure_cons 降为对照留档**。
论文口径：「最小闭包 SUPPORTED；再湿机制侧A不可辨识」（负结果声明，FMTS 欢迎）。
③设计稿的导管机制不被否定——结论收窄为「侧A不可辨识」，侧B/AE 阶段再议
（amendment 风险与边界节）。

悬置项（待设计侧后续修正案）：O1 基准 closure 是否切 `conservative_norew`
（amendment §4）；checklist 声明分级刷新与论文 tex O1 段数字更新归本地侧
8/23 任务。

## 备注

- 实际时长：T1×3 ≈ 4h（与 intact 栈同量级）；R1/leakdist/auditpack 各 <1min
  （runbook 预估 40min/2h/1.5h 显著高估——compile 后探针在 GB10 上为秒级，
  产物内容已验证完整，非短路）
- summary units：o1 + r1（对侧重建权威块）+ r1_closure_cons_norew（本轮新增）
- 判决下一步（待裁定）：若采纳 norew 为生产臂 → 论文口径「最小闭包 SUPPORTED；
  再湿机制侧A不可辨识」；intact 栈 T1/R1 全档产物留档作对照。
