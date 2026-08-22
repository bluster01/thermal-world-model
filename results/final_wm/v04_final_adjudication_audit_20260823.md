# v0.4 全档终审独立审计（本地，2026-08-23 00:4x）

对象：执行侧 07d6d91（训练栈 e95bb88e4）。预注册规则（修正案 v0.4 §2 全档判决）：
三 seed 一致——方向全对（R1 frac_negative=1.0 且下游无反号）且 val NLL 三 seed
中位 ≤ intact 栈（890bd15 已审计值）中位 + 0.05。

## 逐项复核（全部读原始产物，不转述执行侧报告）

| 项 | 复核结果 | 证据 |
|---|---|---|
| T1 norew ×3 新鲜性 | ✓ ledger 三条 final：epochs 41/60/60（seed0 早停），commit=e95bb88e4，无续跑标记；wall 3679/5414/5357s（compile 档速） | ledger.jsonl |
| R1 norew 栈 | ✓ **SUPPORTED 3/3**：blind=True、direction frac_negative=1.000（32/32×3——intact 栈 seed0 的 28/32 失败消失）、稳态 240 步响应 −0.282/−0.383/−0.426 负向、泄漏 0.58/0.58/0.13pp 全 suspected=False | `r1_report_closure_cons_norew.json` |
| leakdist ×3 | ✓ 16-shuffle 零分布对照：delta_vs_mean 0.60/0.54/0.08pp ≪ 5pp 门；**intact 栈 seed1 的 5.15pp 边际案在 norew 栈不复存在** | `leakdist_closure_cons_norew_seed*.json` |
| auditpack 自检 | ✓ rewetting_ablation 恒等：intact == rewet_zeroed（−0.3369065523147583 逐位一致）→ 消融物理确在运行；v1 分箱增益（final，60 步）主箱模型增益为负（−5.98，n=248）无反号 | `auditpack_A_closure_cons_norew.json` |
| summary 共存 | ✓ units = o1 + r1 + r1_closure_cons_norew（合并纪律保持） | matrix_summary_sideA.json |

## 执行侧报告两处勘误（不改判决，入档备查）

1. **intact 中位引用错**：报告称 intact best_val"1.214/1.260/1.228，中位 1.260"——
   中位是 **1.228**（1.260 是最大者）。正确差值：norew 中位 1.272 − 1.228 =
   **+0.044**（非报告所称 +0.012）。按门 +0.05 **仍通过，但边际仅 0.006**。
2. **H18 MAE 数字无源**：报告称 norew H18 2.50/2.06/2.15°C；权威 metrics 文件
   （n=256 窗）实为 **3.22 / 2.74 / 2.80**（intact：2.65/2.79/2.62）。
   逐 seed 差 +0.57/−0.05/+0.18，均值差 **+0.23°C**（norew 略差）。
   H18 不在终审门内（门 = 方向 + val NLL 中位），不影响判决成立；
   但论文若引 H18 必须用 metrics 文件值，执行侧报告数字作废。
   另注：seed0 为弱 seed（早停 41 ep，best_val 1.536，H18 3.22 均为最差）。

## 终审结论（按预注册规则）

**判据全部满足**（方向 3/3 全对 + 泄漏三清 + val NLL 中位 +0.044 ≤ +0.05）。
→ 按修正案 v0.4（用户 8/22 09:06 批准）：**closure_cons_norew 具备侧A生产臂资格**。
论文口径：「最小闭包 SUPPORTED；再湿机制侧A不可辨识（v1 冻结下混合冷却与
再湿加热不可分；消融后控制相关视界方向恢复正确）」——负结果声明，FMTS 欢迎。

## 用户裁定（2026-08-23 00:4x）

- **裁定 A ✓ 采纳**：侧A生产臂正式切换为 `closure_cons_norew`。后续 O1/T5/J1
  及论文数字均以 norew 栈为基准；intact 栈全档产物留档作对照。
  runner 的冻结 T1 默认嵌套对（`closure_cons` vs `physics_only`）不改代码、
  不追溯改写——生产口径由本裁定承载。
- **裁定 B = B1**：补跑 `physics_only ×3`（当前栈 e95bb88e4），随后以 runner 的
  `_seed_passes`/THRESH_T1_NLL 对 metrics 文件计算 `closure_cons_norew` vs
  `physics_only` 嵌套判决并重发 T1 verdict（判决计算留痕本地审计，不改 runner
  冻结路径）。若通过且需把 norew 设为 runner 默认比较臂，另起 v0.5 修正案。
