# FMTS 2026 四页骨架 v2（叙事驱动版，2026-08-21）

> 依据：`docs/PROJECT_TIMELINE_NARRATIVE_2026-08-21.md`（全史）+ 
> `results/final_wm/world_model_credential_checklist_20260821.md`（证件清单，判决真值）
> + `docs/plans/2026-08-21-fmts-schedule-and-protocol-plan.md`（排期）。
> 状态：**骨架讨论稿，非正文**。IRON RULE：用户确认本 Phase-0 记录后才写 tex 正文。

## Phase-0 论文配置记录

| 项 | 值 |
|---|---|
| Venue | FMTS 2026（NeurIPS 2026 Workshop），non-archival，双盲，≤4 页正文 |
| 截止 | 2026-08-30 23:59 UTC（北京 08-31 07:59） |
| 学科/形态 | 时序系统/世界模型资格评测；**negative findings 欢迎**（CFP 明示） |
| 语言 | 英文正文（中文版同步，EN 权威） |
| 写作纪律 | 数字全部可溯源到审计文档章节；撤回数字死刑；无 AI 腔；匿名化（装置/现场/控制系统细节脱敏） |
| 主张形态 | 单主张：预测精度不构成世界模型资格；张力实证 + 判别矩阵 + 修复循环实录 |
| 判决真值源 | `matrix_summary_sideA.json`（v0.2）+ `r1_report.json`（修复批 seed0）+ 证件清单 §0 修正表 |

## 0. 现稿病根诊断

现稿 Introduction 有正确框架（forecaster vs world model、tension、discrimination matrix），
但**没有任何过程特定性**——换成任何工业过程都成立。缺三样东西：

1. **现场钩子**：本文的动机是一个已投运系统暴露的具体异常（预测接入前馈控制后，
   阀门动作几乎不改变预测），不是抽象的"资格问题"；
2. **失败弧**：诊断链（表示→时标→混杂→tracking illusion）与两条死路
   （纯黑箱方向漂移 / 纯物理误差大）——"发现问题→尝试解决→收敛"的弧；
3. **负结果叙事**：O1/B1 被拒、R1 被拒后修复、leakage 门误触后协议化——诚实的
   资格矩阵本身就是卖点（CFP 明示 negative findings 在范围内）。

## 1. 页预算与节映射（保持现五节结构）

| 页 | 节 | 页占比 | 叙事功能 |
|---|---|---|---|
| P1 | §1 Introduction | ~0.75 | **完整故事弧**（钩子→异常→诊断→死路→收敛→贡献） |
| P1 | §2 Testbed and Two Model Lines | ~0.25 起 | 机组事实 + 两条线定义 |
| P2 | §2 续 | ~0.5 | 两线规格、训练口径、审计纪律 |
| P2 | §3 The Tension, Measured Three Ways | ~0.5 起 | 测量一：动作响应（functionally blind） |
| P3 | §3 续 | ~0.9 | 测量二：反事实相干；测量三：精度代价定位 |
| P4 | §4 A Discrimination Matrix | ~0.5 | 矩阵定义 + 判决表 + 修复循环 |
| P4 | §5 Discussion | ~0.5 | 结论、边界、两级架构立场 |

## 2. §1 Introduction 叙事节拍（核心改动，逐拍）

**拍 1 · 现场钩子（2 句）**
一个已投运的机组上，运行侧依赖前视预测做监盘与超温预警，该预测还接入外回路——
预测值修正温度设定值、以微分前馈实现超前控制。模型做的是 180s 前视预测，但 DCS
通道只允许接一个点位，实际接入的是 90s 预测值——这个点是预测 horizon 在 MAE 与
动作响应幅度上的甜点位（此机制只提一嘴，不展开）。系统预测得很好。但接入控制后
暴露了一个不能忽视的事实：**阀门动作对预测结果几乎没有影响，方向也常常不对**。
*（来源：wiki 现场期 + phase1_report 触发段 + 用户 08-21 口述 DCS 单点约束；
匿名化：不出现机组名/DCS 步号/电厂身份；"预测得很好"须锚定到拍 2 的 0.7–0.9°C）*

**拍 2 · 异常定量（2 句，数字=audited）**
把动作路径单独拿出来审计：零动作响应的模型照样拿到 rollout MAE 0.808
（exp_011，phase1_report §0.2）；换绝对阀位表示后响应 ×32–130，但量级仍小、
方向不稳。单步指标看不见 120s+ 的物理滞后；闭环 PID 的反馈内生性让数据里
"开阀↔升温"成为统计事实——模型学到的是混杂，不是因果。
*（motivational 数字不超 3 个，全部注明出处；此段为故事背景，不上升为贡献主张）*

**拍 3 · 最深一层的发现：tracking illusion（1-2 句）**
把"方向正确"的灰箱约束模型放到 1s 数据下复核：SP 变化时方向正确率 84%，
但外源阀位信号仅占 0.2%——**模型学到的是闭环跟踪（SP↑→PID 关阀→升温），
不是 plant 的物理因果**。理论上 SP 变、喷水流量与各级温度都应连锁变化，
模型对此是盲的。
*（PHASE35_DESIGN §0；这是全文最强的叙事转折，也是"为什么必须做物理结构"的论据）*
**因果桥（必须显式写出）**：正因为"方向正确"曾被闭环跟踪伪装过一次，论文 Line 2
（A1 物理分支的升级独立版，见拍 4）的每个资格主张都由判别矩阵逐门审计——
*"direction was faked once, therefore Line 2 is gated by the matrix"*。

**拍 4 · 两条线各自作为世界模型候选失败（2 句）**
Line 1 是高容量深预报器（Direct-WM v2，带动作通道）：direct H18 MAE 0.71–0.93°C
尚可，但双阀响应方向随 fold/阀门/seed 漂移（审计表）——保留为 forecasting 对照线，
不能当世界模型。Line 2 是纯物理 ODE（Fan2020-inspired 集总焓 + 残差闭包）——
它的血统正是拍 3 揭穿的 A1phys 物理分支的升级独立版（adhoc2）：响应真实存在，
但瞬时、误差大、依赖初态——现场缺流量/壁温/焓真值，养不起白箱。
**精度与响应两条路各自失败，才有融合。**
*（ADHOC_DIRECT_WM_V2_SUPERVISOR_AUDIT §3 + fan2020 EVIDENCE_CHAIN E0–E7）*

**拍 5 · 收敛与方法（2 句）**
灰箱融合：物理状态转移（守恒结构 + 可学习参数 + IAPWS 可微物性）+
权限边界的神经组件（observer/boundary/action-blind closure）。资格用**判别矩阵**
逐项审计——方向/泄漏（含打乱对照）/量级，判决与产物全部入仓可重放。

**拍 6 · 贡献（保留现稿三条，措辞对齐证件清单）**
(i) 现场双阀数据上的张力实证：预测精度不转移到动作资格；
(ii) 可复用的资格矩阵（可辨识性门、分层消融、动作盲视与泄漏审计——含
shuffle 对照的泄漏门修正）；
(iii) 精度代价的定位（误差地板锚定）与"收敛诊断重排优化议程"的方法论。

## 3. §2–§5 节拍与证据映射（主张 → 证据 → 图）

| 节 | 主张 | 证据（来源） | 图 |
|---|---|---|---|
| §2.1 过程 | 机组事实（正文不写 MW 级与参数等级，真值=660MW 亚临界，用户 08-21 确认）：主汽温窄带 + 双喷水阀 + 多扰动 | 机组事实（README/数据口径） | fig1 过程示意图 |
| §2.2 线1 | 深预报器（Direct-WM v2，带动作通道）：18 步 direct head，0.7–0.9°C MAE | DirectWM 审计 §2 | — |
| §2.3 线2 | 纯物理 + 残差闭包（fan2020 集总焓状态空间 = A1 物理分支的升级独立版 + 可微物性 + 参数化闭包）；结构增强链四臂作为矩阵消融 | pipeline 设计 + T1 | fig2 架构 |
| §3.1 动作响应 | 预报器功能盲：阶跃响应 0.00004–0.040°C、方向随 fold 漂移；世界模型 −0.228°C/+5% 且修复批后 100% 方向一致 | DirectWM 审计表 + R1 seed0 | fig3a 阶跃曲线 |
| §3.2 反事实相干 | R1 三门：盲视 ✓ / 方向（v0.2 失败→修复批暂定过）/ 泄漏（shuffle 对照 Δ=0.64%） | r1_report + action_signal_analysis | fig3b 泄漏对照 |
| §3.3 精度代价 | 3.5°C vs 0.7–0.9°C；代价定位：sh1_in 上游段 9.4°C=38×persistence、final 近 persistence；分箱偏差在 60ep 后塌缩（训练充分性，非参数自由度） | roadmap v1 + v0.2 eval | fig3c per-channel |
| §4 矩阵 | 五单元判决表（**修正版**：O1 learned MIXED / hybrid REJECTED；T1 closure_cons SUPPORTED 2/3、其余臂 REJECTED；B1 REJECTED 3/3；J1 SUPPORTED；R1 方向门失败→修复批 seed0 暂定 PASS）+ 修复循环实录（leakage 门误触→协议化） | 证件清单 §0 + matrix_summary | table1 判决矩阵 |
| §5 Discussion | 资格先于精度；两级架构（learned proposer + certifiable ROM）；限制：teacher-forced 口径、单装置、A 侧单边判决、B1/O1 负结果如实报告 | 证据阶梯 + 证件清单 G | — |

## 4. 占位符与待判决清单（写作时必须诚实标注）

| 占位符 | 内容 | 落定时点 |
|---|---|---|
| `<R1-3SEED>` | R1 三 seed 正式判决（在跑 ~8/21 晚） | 8/22 回传审计 |
| `<CF1>` `<CF3>` `<CF4>` | 合成孪生反事实精度 / 分箱局部增益 / 约束一致性（对侧已落地） | 8/22 读数 |
| `<D1>` | σ 校准覆盖率（摘要"knows when it doesn't know"的证件）——**用户裁定：等探针（8/22 读数）再定摘要该句去留** | 8/22 读数后决策 |
| `<FIX1-BRANCH>` | 修复①（五点锚定+压力反演）是否进本期 → sh1_in 9.4°C 与 C1/C3 证件 | 8/23 决策点 D1 |
| `<O1-B1-WRITEUP>` | O1 hybrid REJECTED / B1 REJECTED 的负结果写法 | 8/24-25 修订稿 |

## 5. 匿名策略（双盲硬约束）

- **repo 不提供**（用户 08-21 裁定）：论文不出现仓库链接/URL；可复现性承诺改为
  "a frozen runner protocol with in-repo replay, available on request to qualified
  reviewers"（Limitations 明示）；
- 机组：不出现"伊敏 6 号机"厂名级识别；**正文不写 MW 级与蒸汽参数等级**（真值=
  660MW 亚临界，用户 08-21 确认；phase1_report 所写"1000MW 超超临界"为误，
  已修正 timeline Act 0 并留更正记录）——需要参数语境时写
  "a once-through boiler"或按用户口径；
- 现场系统：不出现 DCS 步号（step9）、控制器参数（Kp/Ti）、前馈回路的具体命名——
  用 "a deployed feedforward loop that corrects the temperature setpoint with the
  90s-ahead prediction"；"deployed" 改写为 "a plant operator's deployed system"
  以降低"作者=运行方"的识别性；
- 执行体系：不出现 Hermes/Codex/双机审计的内部称谓——写 "an audited experiment
  protocol with a frozen runner and in-repo artifacts"；
- 作者单位：匿名块；`\usepackage[dblblindworkshop]{neurips_2026}`。

## 6. 与现稿的差异清单（改什么）

1. §1 整段重写为六拍叙事（上）；
2. **摘要整体重写为修复循环版**（预审 TOP-1）：现稿"inverts sign at 60 steps / zeroing
   restores direction"是 v0.2 旧叙事，与修复批后 seed0 direction 100% 直接冲突；
   新摘要按"initial inversion → contracted repair → 100% direction（seed 0；seeds 1–2
   on completion）"写；"knows when it cannot"句留删由 D1 探针决定（用户裁定：等探针）；
3. §3.1 用修复批后数字（−0.228°C/+5%、稳态 −0.194°C/2%），**量级缺口 2.7–7.6× 如实
   写入并进 Limitations**（混合参考=名义线性上界；等百分比阀非线性+窄区间微调）；
4. §4 判决表换修正版（O1/B1 负结果如实入表）；**修复循环实录挑 2 个 case**
   （leakage 门误触→打乱对照证伪→协议化；收敛诊断废掉参数 MLP 立项→路线图重排），
   每个 2-3 行；
5. **拍 6(ii) 核心措辞改为 matrix-as-process**："the matrix plus the
   protocolized verdict-repair loop it produced"（防"评测套件而已"攻击）；
6. **页预算重排（预审 TOP-3）**：intro 文献锚 9→3（wan2026forecast/ghosh2026rom/
   forssell1999）；§3.2 压缩为 §4 R1 前导段；seed 级数字全收 table1；§5 压缩；
   **恢复 fig:tension**（精度-动作资格平面，删独立过程示意图，图总数 ≤3）；
7. 加 related-work 定位句（§5）："positioned against capability checklists; our
   contribution is the verdict-repair loop, not the checklist itself"。
