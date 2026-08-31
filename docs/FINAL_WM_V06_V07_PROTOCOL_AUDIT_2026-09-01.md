# Final WM v0.6 / v0.7 协议谱系审计与解决方案（2026-09-01）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-01
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 1. 审计结论

v0.6 与 v0.7 不是两套可被当作独立证据的训练矩阵。最终有效协议应合并为一次全量重发：

- **v0.6 是执行底座**：corrected canonical v2.2、一级同侧/二级交叉动作接线、原始质量门、显式 `epochs=120 / patience=20`、P1 位一致提速；
- **v0.7 是可信度合同**：required-evidence fail-closed、同窗 paired `Delta NLL`、统一 `t -> t+1`、逐窗口反事实支持域、D-SYN teacher 扰动可观测性、固定 validation anchors、内容寻址 manifest；
- **正式模型仍使用 canonical 的 7 通道 base view**。v2.2 新增水煤比和负荷只用于数据质量/历史 A5 诊断；A5 已按预注册 `REJECT`，不得进入正式 v0.7 模型；
- **正式 R1 栈是 `closure_cons_norew`**。这是 v0.4 双侧终审后唯一同时保持方向证书和精度平价/收益的生产栈；旧 runner 的 `closure_cons` 默认值是历史残留，不能继续作为新重发默认值；
- 双侧分别报告，禁止聚合；只用 validation，test 保持锁定。

因此不重复执行一套“v0.6 结果”和一套“v0.7 结果”。同一批结果同时绑定 v0.6 的数据/训练底座和 v0.7 的判决/追溯合同，避免重复训练被误读为独立复现。

## 2. 历史修订逐项裁定

| 来源 | 最终处置 | 理由 |
|---|---|---|
| canonical v2.0 | SUPERSEDED | 继承 v1 一级阀错侧 |
| canonical v2.1 | 保留为接线修复来源 | 一级同侧、二级交叉已裁决并验证 |
| canonical v2.2 | **正式数据制品版本** | 在 v2.1 上仅附加 A5 诊断通道；base view 不变 |
| v0.6 120/20 | **纳入所有正式训练臂** | 后续预算探针证明旧预算可能欠训；必须显式写入 spec |
| v0.6 常数锚定能力 | 实现保留、正式矩阵不启用 | 无预注册且内容固定的外部 anchor；`norew` 与锚定合同互斥，强行启用会改变消融语义 |
| P2/P3 加速 | 排除 | v0.6 修正案明确不进首训，且不应在可信度重发时改数值口径 |
| v0.6 原 A1-A4 高层草案 | 不混入 v0.7 | 未形成可执行冻结合同；后续 A1/A2/A4 名称已被不同机制探针复用，直接合并会发生 estimand 漂移 |
| A5 / LPV / zcond / JEPA-B | 历史探索证据 | 均有独立预注册和停止裁定，不是 O1/B1/T1/R1/J1 正式矩阵臂 |
| v0.2-v0.6 旧矩阵结果 | HISTORICAL / SUPERSEDED | 可信度审计已判定需 corrected-record v0.7 重发 |

## 3. 唯一全量矩阵

每侧固定执行以下训练单元，seeds 均为 `0,1,2`：

| 单元 | 臂 | 训练数 | 正式判决 |
|---|---|---:|---|
| D-SYN | student | 3 | teacher 扰动非空且可观测；同型可解性至少 2/3 |
| O1 | steady / learned / hybrid | 9 | H6/H18 paired NLL + state continuity |
| T1 | physics_only / closure_cons / closure_steam / latent4 / closure_cons_norew | 15 | H1/H6/H18 paired NLL + H60 稳定性；含 norew vs intact |
| B1 | GRU boundary | 3 | H6/H18/H36 + downstream forecast-vs-oracle |
| J1 | joint / staged main / staged boundary | 9 | 同窗 H1/H6/H18 + H36 稳定性 |
| R1 | 复用 `closure_cons_norew` | 0 | 双阀 H18/H60、day-block CI、方向占比、leakage、逐样本支持域 |

总计每侧 39 个训练 run；R1 只读复用 T1 checkpoint。quick、partial seed、arm-filter 均不得生成正式判决或 authoritative manifest。

## 4. 必须修复后才可发 Linux

1. D-SYN 当前按 `raw_` 前缀匹配参数，但真实名字为 `raw.<name>`，实际扰动数为零；改为直接遍历 `teacher.transition.raw.parameters()`，保存数量和 L2 距离，no-op fail-closed。
2. canonical v2 当前先 clip 再算 range violation，`clip == range` 时原始越界会被掩盖；派生通道又会把源缺失填零后误报为满覆盖。改为 raw/source quality 先过门，再做模型输入后处理。
   两路喷水流量沿用 v2 首建已审计的负零漂容忍：原始范围分别 `[-10,400]` 与
   `[-2,400]`，通过后才 clip 到 `[0,400]`，不把已知零漂误判成新数据事故。
3. validation 每 epoch 使用 `10000 + epoch`，checkpoint 实际在不断变化的验证样本上选择。改为每 run 固定 validation seed/anchors，所有 epoch 完全复用。
4. resume fingerprint 只绑定 spec 和代码树，没有绑定 canonical、properties、init/anchor checkpoint 内容。补 SHA-256 内容身份。
5. quick/full 目前依赖命名习惯隔离。增加 tier marker，禁止同一输出目录混用。
6. 正式 full run 仅在 clean commit、完整 units、完整 seeds、无 filter、D-SYN PASS 时生成 manifest；manifest 绑定代码、命令、矩阵、record、properties、checkpoint、metrics、summary 和 ledger 哈希。

## 5. 停止与审计规则

- Linux 只执行冻结命令，不改阈值、seed、数据、代码或输出；失败原样回传，不自动重试。
- 任一 manifest 项缺失、哈希不符、unit/seed 不全或 worktree 非 clean，整批为 `INCOMPLETE`。
- 任一 R1 反事实步骤越出逐样本支持域，R1 为 `INCOMPLETE`，不以外推结果补判。
- 结果回传后仅本地独立复算；不自动访问 test，不升级论文 verdict，不把 action sensitivity 写成现场 `do(valve)` 因果效应。
