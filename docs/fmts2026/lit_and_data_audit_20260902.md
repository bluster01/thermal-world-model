# FMTS 稿件 v2 全量复审报告（文献 + 数据口径）

> 复审对象: fmts_main_v2.tex（456 行，commit 94c5107）
> 复审方法: critical-literature-review skill（Fernandez 5 产出）+ v0.6/v0.7 协议口径对照（PROTOCOL_AUDIT_2026-09-01 + runbook + credential checklist）
> 日期: 2026-09-02

## A. 文献维度（skill 复审）

上轮修复均落地：wan2026 失实引用 3 处全清 ✅；ding2024/ha2018world/runge2020pcmci 补引 ✅；F5 区别句 ✅；ghosh v2 标题 ✅。
遗留（待裁定）：F6 单机组主动表述；F7 chronos/timesfm 未引；F9 Sensors iTransformer 主汽温。
5 产出复查：证明知识 13/15 bib 被引 ✅；gap/问题/定位/理论 ✅。文献维度无新红线。

## B. 数据口径审计（🔴 严重 / 🟠 警告 / 🟡 待核）

### 🔴 B1. 摘要红线违例："must know when it cannot"
- 摘要 29–30 行保留 "it must know when it cannot"（知道自己何时不知道）
- credential checklist §3（8/30 红线）：D1/D2 无证，**"当前必须删改或软化该句"**；稿件 Discussion 自己承认 <D1> 待验（"the sentence stays only if the interval coverage checks out"）
- 处置：投稿版必须删或软化为 "we probe whether it knows when it cannot"（探针性表述）

### 🔴 B2. Table 1 verdicts 全部是 SUPERSEDED 旧批次
- O1（learned MIXED −30.1/−31.4/+13.3%；hybrid REJECTED −0.3/−1.6/+3.7%）= **v0.2 旧判决**；credential checklist 明确：8/22 修复①栈重跑后 learned/hybrid 已 **SUPPORTED 3/3**，旧判决仅作历史对照
- T1（+5.7/+6.9/+2.6 等）、B1（CRPS 1.7–2.0）、J1（+13.5/+16.4/+33.3%）、D-SYN 全部为 pre-v07 批次
- v07 协议审计裁定：**v0.2–v0.6 旧矩阵结果 HISTORICAL / SUPERSEDED**；README 红线："test 与论文 verdict 在回传独立审计前保持锁定"
- 处置：Table 1 整表改为 `<PENDING-V07>` 占位（或标注 superseded-historical），v07 双侧回传+独立审计后解锁

### 🔴 B3. R1 叙事与 v0.7 正式 R1 协议不符
- v07 R1 协议（协议§3）：复用 closure_cons_norew 的 T1 checkpoint、双阀 H18/H60、day-block CI、方向占比、leakage、**逐样本支持域**（越界→INCOMPLETE，不得改外推判决）
- 稿件 R1（blindness 3/3、shuffle leakage Δ0.6%、direction v0.2 REJECTED→repair seed0 100%）= 旧审计叙事（不同栈、不同协议）
- 处置：R1 段整体标注历史或替换为 v07 R1 占位；"R1 direction" 的 19–34%→100% 属旧批次

### 🟠 B4. Line 1/Line 2 数值无 v07 口径对应
- Line 1：0.712/0.927 MAE（旧预测器评测）；Line 2：3.9°C "under the amended training budget"（182–183）——v07 只有 120/20 一种预算，无 "amended" 概念；v07 T1 尚未跑完，该数无对应口径
- 0.808°C（zeroed action pathway）来自旧诊断
- 处置：v07 回传后替换（Line 2 的 MAE 应来自 v07 T1 closure_cons_norew 臂）

### 🟠 B5. "initial vs amended budget" 对比（247–249）
- "+2.1→+5.4°C 塌缩至 ±0.2°C"是旧批次预算诊断——它恰是 v0.6"旧预算欠训"裁定的原始证据（保留价值：证明 120/20 必要性）
- 处置：可保留，但**必须标注历史批次性质**（"pre-v07 protocol diagnostic"），不得作为 v07 结果

### 🟡 B6. 判决规则表述与 v0.7 合同未对齐
- 稿件 262："paired bootstrap confidence intervals and a 2/3 seed rule"
- v0.7 合同核心：固定 validation anchors（每 run 复用，非 10000+epoch）、同窗 paired ΔNLL、fail-closed；校验 anchors 未被稿件提及
- 处置：更新为 v0.7 术语；"2/3 seed rule" 与 v0.7 fail-closed 关系需在文中明确

### 🟡 B7. 数据锚定参考的版本待核
- 27.8/70.0 t/h per full travel；0.21–0.59/0.53–1.48°C per +2%（190–194）基于现场回归——需确认数据版本（v1/v2.0/v2.2）与生成脚本
- 处置：找参考生成脚本，绑定 canonical v2.2 或标注

### 🟡 B8. 图件数据源待核
- fig1（accuracy-action plane，10–30×）、fig3（事件曲线、±0.04、5–13°C）基于旧事件分析/旧矩阵
- 处置：figs/ 下生成脚本需绑定数据版本（v2.2 SHA）；或图标注 pre-v07

### ✅ 已一致
82 天/10s/双侧 ✓（v2.2 n=707709≈82d）；dry operation ✓；seeds 0,1,2 ✓；5 温度+2 阀交叉接线 ✓；single-side verdicts ✓（A only）；"支持域/逐样本"概念稿件未误用 ✓

## C. 统一协议管理建议（"同一协议下管理数据"）

1. **给稿件建 Material Passport 表**（仿协议审计§Material Passport）：每个数值 → 来源批次（v07 / pre-v07 superseded）+ 状态（LOCKED / PENDING-V07 / HISTORICAL）
2. 数值规则：
   - v07 产物（verdicts、MAE/NLL、R1、图）→ **v07 回传 + audit_manifest 独立审计后解锁**（runbook §3.4）
   - pre-v07 只允许作为历史对照/诊断叙事出现，显式标注
   - 图/表生成脚本绑定 canonical v2.2 内容寻址（manifest 哈希）
3. 稿件当前可锁定的部分（不受 v07 影响）：问题定义、设计空间叙事、三层诊断、判别矩阵**方法学**、Limitations、文献定位
4. 不可锁定（必须占位/标注）：Table 1 全部 verdict、R1 全部、Line 1/2 数值、图 fig1/fig3、预算对比叙事

## D. 建议执行顺序

1. **B1 立即修**（摘要删/软化一句，不依赖任何数据）
2. Table 1 + R1 + B4/B5 改为 `<PENDING-V07>` 占位 + 保留历史标注（等 v07 回传填补）
3. B6 判决规则表述对齐 v0.7 合同（加 fixed validation anchors 描述）
4. B7/B8 待核（找脚本/数据版本）
5. F6/F7/F9 按版面定
