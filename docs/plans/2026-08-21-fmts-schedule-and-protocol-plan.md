# FMTS 2026 投稿排期与协议对齐计划（2026-08-21）

依据：CFP（https://fmts-workshop.github.io/cfp.html）+ 证件清单
`results/final_wm/world_model_credential_checklist_20260821.md`。

## 1. 管网关键约束（已核实）

- **截止：2026-08-30 23:59 UTC = 北京 08-31 07:59**（AoE 8/29，无延期）
- **正文 ≤ 4 页**（不含参考文献与附录；评审无义务读附录）
- 双盲；non-archival（可与期刊版并行）
- 明确欢迎 work-in-progress / preliminary / **negative findings**
- 评审四维：技术正确性；**评估严谨性（基线、泄漏意识、统计处理）**；
  **可复现性（产物质量与文档）**；相关性（含负结果与复现）
- 命中主题：leakage-aware evaluation ✓（我们的 shuffle 对照）、
  calibration and conformal UQ ✓（D1）、long-horizon simulation consistency ✓、
  physical evaluation suites ✓

## 2. 倒排排期（北京时）

| 日期 | 执行侧（Hermes） | 本地（Cascade） | 门禁 |
|---|---|---|---|
| 8/21（今） | R1 链：closure_cons×3 串行（~5h，命令在回执追加4） | CF-1/3/4 + D1 探针+测试（已完成，见 §3） | — |
| 8/22 | 产物回传；auditpack 带新探针重跑（--checkpoint seed0/1/2 各一次，每次 ~20min） | 审计 R1 三 seed 判决；CF/D1 证据读数 | **G1：R1 判决落档** |
| 8/23 | 待命 | 判决审计 + 证件清单状态刷新；修复①设计定稿（五点锚定+压力反演） | **决策点 D1：R1 全过？修复①是否进本期** |
| 8/24-8/25 | （若 ① 激活）重跑 closure_cons×3 ~5h + R1 | 论文修订稿：4 页压缩、O1 段修正、R1 段更新、摘要 D 句处理（§4） | — |
| 8/26-8/27 | 待命 | 论文定稿：图表更新（fig3 换修复批后数据）、zh/en 同步、参考文献 | **G2：论文内部全审** |
| 8/28 | — | 缓冲（回归/补数） | — |
| 8/29 | — | 最终审计 + 匿名化检查 + OpenReview 提交演练 | **G3：提交版冻结** |
| 8/30 | — | 提交（北京 20:00 前） | 截止 8/31 07:59 |

**缓冲策略**：修复①是最大风险项（设计+重跑+重判 ~2 天）。若 8/23 决策点 R1 未全过，
①不激活，论文按现状写（R1 方向门结果如实报告 + v0.3 规则修订说明）——CFP 明示
negative findings 在范围内，诚实的资格矩阵本身就是卖点。

## 3. 本期已落地的协议增补（evaluation-only，无指纹影响）

| 探针 | 证件 | 接入点 | 测试 |
|---|---|---|---|
| CF-1 `counterfactual_fidelity_synthetic` | B4 反事实轨迹精度（合成孪生，replay/observer 两种 abduction） | run_dsyn 每 seed 自动产出 | replay 同一性归零、确定性、schema |
| CF-3 `position_binned_gain` | B3 局部增益-开度曲线（数据事件 + 模型同箱对照） | auditpack 双阀 | 手搭事件分箱正确性、空箱退化 |
| CF-4 `constraint_checks` | B6 单调性 + 零喷水干漂 | auditpack | 先验模型两约束硬通过 |
| D1 `calibration_coverage` | D1 区间覆盖率（3 档×3 horizon×5 通道） | auditpack | 解析正确性（理想 σ 恢复名义水平） |

全部 evidence-only：不改 spec/模型/预算，不进判决门（待有数据后再议定标）。
七项新测试全过，全套 118+7=125。

## 4. 论文红线处理方案（对应清单 §3）

- **摘要"知道自己何时不知道"**：D1 探针已就位，8/22 auditpack 出数后二选一——
  覆盖率达标则保留并引数；不达标则改为"校准缺口作为诊断结果如实报告"
  （负结果在 CFP 范围内，不丢人）；不允许无证据保留原句；
- **O1 段**：以产物真值（learned MIXED / hybrid REJECTED）为准重写，
  早先"写反"记录以本次真值表复核；
- **R1 段**：等三 seed 判决；若过，方向门写 v0.3 修订后口径 + shuffle 对照 leakage；
- **4 页压缩**：现稿正文超 4 页，压缩策略——方法节压到半页（引附录）、
  矩阵判决改表格、讨论节合并部署段；
- **匿名化**：检查 tex/图/产物链接无双盲泄露（机组名"伊敏"需脱敏为
  "a 660 MW supercritical unit"——正文已是，图注再核）。

## 5. 协议对齐确认

- 新探针均为评估侧，**config_fingerprint 不含它们**——不触发重训，不扰动在跑产物；
- CF-2（A/B 双侧近反事实）维持"R1 判决后立项"，不在本期排期；
- 侧 B、J1 重判、参数 MLP、MPC：维持冻结/延后，不进 8/30 范围；
- 执行侧命令不变（回执追加4 的单命令），auditpack 重跑是新增执行项（8/22）。
