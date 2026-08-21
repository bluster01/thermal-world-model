# 火电主汽温世界模型：项目全史与叙事主线（2026-08-21 整理）

> 本文档从仓库 git 历史（105 commits，起点 08-10）、docs/ 文档化石、results/ 实验索引、
> physical_models/ 资产与 wiki 现场记录中独立重建项目演变时间线。
> git 历史之前的早期工作没有提交记录（旧史以 d3f8586 等散见 commit 编号存在于 session 文档中），
> 本文以文档化石为准，commit 只作 08-10 后的锚点。

---

## Act 0 · 现场应用期（2025.04 – 2026.07）

**来源**：~/wiki（04-30 起会议纪要 ingest）、`docs/implementation_roadmap.md`、DCS 解析产物。

1. **主汽温预测模型**落地伊敏 6 号机（1000MW 超超临界，10s 采样）：先做**监盘**（预测值实时显示），
   再做**超温预警**与**预测性超温控制**——"阀门快开慢关"，在磨煤机启动等易超温工况压制温度峰值。
2. **外回路前馈控制**（已实现）：1 分钟预测值作为前馈信号引入外回路，带 FX 折线修正、惯性环节、
   通讯质量监测、数据自保持。2025.10 将前馈信号从"真实值与预测值误差"改为
   "真实值与目标值的差值"，对齐运行目标。
3. **step9 微分前馈**（用户口述，无仓库存档）：90s 预测值做微分前馈叠加到当前设定值，实现超前控制。
   现场 DCS 逻辑解析（历史会话产物，档案未保留，数字待重验）确认二级减温 A 侧控制器为纯 PI 结构。
4. 同期沉淀：智能操作评价（LLM 阀门操作评价）、煤质在线测量、DCS 智能问数窗口等。

## Act 1 · Phase 1：动作通道独立审计（2026.08.01–08.03）

**来源**：`docs/phase1_report.md`、`docs/phase1_conclusions_audit.md`、`results/README.md`。

**触发**：接入控制后暴露核心异常——**阀门动作变化对预测结果几乎无影响，方向准确率也差**。
把动作路径单独拿出来系统研究（exp_003–013），五个既有结论被逐一推翻：

1. exp_008/009 作废（MLP encoder 实现错误 + 评测误用 train_data）——消融必须复用同一训练管线；
2. **rollout MAE 陷阱**（exp_011）：动作响应为零的模型（±0.0001°C）仍达 rollout MAE 0.808——
   单步/rollout 指标不区分"学到动力学"与"背下状态转移"；
3. **差分阀位不可学**：Δ 值域 ~10⁻⁴ 经 RevIN 后能量占比 ~1%；四种架构改造（scale×10 / bypass RevIN /
   FiLM / decoder-only）全部失败；换**绝对阀位**后响应提升 32–130×（±0.03~0.13°C）——
   信号表示比架构更关键；
4. **单步预测物理上看不见延迟效应**：喷水阀→过热器出口时间常数远超 10s 采样；事件研究修正：
   开阀后前 90s 主汽温微升 +0.3°C，120s+ 才转降，10min 达 −3.4°C；单步符号正则不仅无效而且有害
   （把 120s 物理滞后压缩成 10s 伪响应）；
5. **闭环数据因果混杂**：PID 在升温时开大喷水 → 数据里"开阀↔升温"正相关 → 模型学到统计关联
   （∂T/∂valve > 0），物理因果是负方向。世界模型需要显式干预/先验约束。

## Act 2 · 架构探索期：DirectWM / M9DSP / A1phys（08.04–08.06）

**来源**：`docs/supervisory_mode.md`（08-04）、`docs/varattn_causality_analysis.md`（08-04）、
`docs/session_2026-08-05_causal_arch_eval.md`、`docs/session_2026-08-06_review_v2.md`。

1. **监督模式决策**（08-04）：现场是串级结构（外回路主调 + 内回路副调），"直接控阀（路线A）"
   与现场对不上；论文思路对齐为**预测驱动 + 监督模式**。
2. **架构事实核查**（08-04）：动作从不经过 VarAttn；M4 消融（−VarAttn）强度被高估
   （只移除 49,984 参数的单层）。动作注入方式（展平 vs cross-attention）被定位为因果保真度的
   决定性因素：展平注入长时程衰减 33–48%，动作 cross-attn 单调增长。
3. **因果架构评测 L4–L7**（08-05）：7 变体 DiD 口径对决，**A1phys（物理分支）胜出**——
   SHAPE +1.00（唯一完美形状匹配）、TTP_err +0（零时延）、180s GAIN 0.651（短程响应最强）；
   CFI 是 600s 主导的复合指标会掩盖这些。结论：**简单物理架构约束（二阶惯性灰箱先验）
   实现了 SP→设定值方向一致的动作预测**。
4. 论文叙事 v2（`docs/narrative_restructure.md`，08-05）：转向因果主线（M9DSP 动作 cross-attention
   180s 方向 89%）——**后因证据链不达标准撤回**，作为历史叙事保留。

## Act 3 · Phase 2：MPC 方法论探索（08.02–08.06）

**来源**：`docs/phase2_results.md`、`docs/narrative_restructure.md` §V。

DWM-MPC 仿真对比（S1–S6，50 轨迹多步执行）给出三个负面结论：
- RMSE 不显著（p=0.25–0.82；`docs/phase2_results.md`）：动作通道弱因果、工况主导，测不出控制效果；
- TV 降低归因于 MPC 框架而非深度模型（线性 MPC 的 TV 更低，S6）；
- S3"因果安全"结论无效（判定符号反 + 2/3 反演）。
机制：**objective mismatch**（Lambert 2020）——训练数据是运行员自然轨迹，MPC 规划动作
超出分布即退化（小幅动作方向正确、大幅持续阶跃方向反演）；且仿真对象与预测模型同构
（同构 plant 缺陷），闭环对比无外推有效性。**开环精度 ≠ 闭环效用** 成为可守结论。

## Act 4 · Phase 3.5：estimand 修正——"方向正确"的真相揭穿（08.08–08.09）

**来源**：`docs/PHASE35_DESIGN.md`（08-08）、`docs/DIRECTION_DECISION_2026-08-09.md`、
`docs/PHASE35_CLOSEDLOOP_RESPONSE_2026-08-09.md`。

**这是全项目最关键的转折**。用户质疑："设定值变了，阀门的控制动作没有变，A1phys 的方向
正确是怎么来的？"数据验证：
- 10s 数据：SP 变化仅占 0.76% timestep，阀位变化占 89.23%；SP 变化时阀位反向正确 64.1%；
- 1s 数据（70M 行）：方向正确率 84%，但外源 valve 信号仅占 0.2%，**混杂占 99.8%**；
- 传递函数辨识：A1phys 的 `g_phys` 拟合的是 **闭环传递函数 G_cl = dT/dSP**（控制器跟踪特性），
  不是 plant 传递函数 G_p = dT/dvalve。**"方向正确"是因为 SP↑→PID 关阀→减少喷水→温度↑，
  是闭环跟踪，不是 g_phys 学到了物理因果**。

这正是"理论上 SP 变，减温水流量、各级中间温度都该连锁变"而模型没学到的根源：
**物理响应（阀门→喷水流量→温度）作为输入特征从未被模型真正使用**。

随后的方向决策（08-09）：
- 交叉喷水（A阀→B侧温、B阀→A侧温）以现场知识为准，不再花算力验证；
- **核心目标转向闭环物理响应识别**：在闭环 PID 数据上可靠识别"阀位→温度"的方向/时标/幅度/剂量；
- 已知约束：导前区 FOPDT 可识别（K≈−2.1°C/%、T≈16s、τ≈40s），惰性区被共因污染，
  严格门禁下 held-step 事件不足（A=7/B=6）；
- 物理闭环响应基准：SP 外生干预 → 串级闭环响应（SP↑→T2↑→Tm↑ 方向率 70–86%），
  阀位内生性坐实，双侧 SP 联动限制单侧归因。

Phase 3.5 设计定稿：`T̂ = f_free(history, context) + g_response(context, a, r)`，
`g_response(·, 0, 0) ≡ 0`；实际阀位是有效喷水作用代理，不是喷水质量流量。

## Act 5 · Phase 3.5-MS 主线验证链（08.09–08.18）

**来源**：`docs/PHASE35_MAINLINE_CONTEXT.md`、docs/PHASE35_MS*_*.md、`results/README.md`、
`docs/PROJECT_STATUS.md`。git 锚点：`aedf1be`（08-10）起。

| 层级 | 结果 |
|---|---|
| MS0 结构合同 | 各路线共享同一 estimand、零参考、时间因果、递推接口 |
| MS1 最小可解性 | 六种表示在同型二阶 truth 上达噪声区（正对照） |
| MS2-V/C/J 模块辨识 | learned-monotone / scheduled K·τ / joint 逐级双层通过；staged 内部拆分失败 |
| MS2-D1 纯迟延 | 阴性关闭：test CI 下界 17.2–18.8% 未达 20%，delay kernel 未唯一恢复 |
| MS2-D2 三阶惯性 | CONFIRMED_SYNTHETIC：三阶 vs 二阶点改善 23.7–25.4%，CI 下界 19.9–21.2% |
| MS2-D3 扰动压力 | VALIDATION_STRESS_PASS（按预算不追加 test） |
| MS5 完整耦合 | **joint 胜出**：total-only 下 free/response 加法分解可辨识；staged 协议 11.1–14.1× 拒绝；free-only 证明"预测准 ≠ 组件恢复" |
| MS3 真实适配 | **asymmetric FAIL**：B 侧 3/3 PASS（0.043–0.049°C），A 侧 0/3 non-collapse FAIL（0.0066–0.0085°C，约 B 的 1/6） |
| MS3-D 不对称诊断 | B 阀位执行更持久；模型 A 侧 response attenuation **未获现场热链路支持**（不能写成吸收已识别） |

MS3 的失败推动 MS3-R 本地设计系列（`SP→阀位→局部温降→末温` 分段监督、MIMO、双阀双输出）：
- Gate A 点位可辨识性（08-11）：两阀创新相关 0.1765、条件数 1.43、common/differential energy ratio 0.70 → 双输入满秩机器证据；
- Gate B 点位闭合（08-11）：11/11 产物；
- Gate C（08-11~08-12）：RM0/RM1 归因、RM2 54-run 并行验证（`c095279`）；
- RM3 48 单元正交响应+公平预测（`2213000`，08-12）：P0 oracle 0.661 / P1 predicted valve 0.948 / P5 hybrid joint latent 0.973；
- RM3-A 容量匹配消融（`ba7a8f1`，08-13）：双向容量匹配 P5 方向成立，无 composite champion；
- RM3-AV 架构验证（`b4ffad1`，08-14）：Q01–Q33 为 30 SUPPORTED / 3 MIXED，P5 bypass 主导、阀位过平滑、shape 不可分，**无冠军**；
- RM3-B1 成对筛查（`e02c0c9`，08-14 → 08-18 收口）：22/22，仅"对角响应"获两折支持，5 混合、2 拒绝（1-pole、bypass add-on）→ **模型扩展路线关闭**。

旧 E1–E5（Δ阀位基准/非线性速率/真实阀位事件/A1phys 复现/SP 未执行）全部废弃，价值只在于
记录为什么闭环阀位事件不能当外生干预（反馈内生性、开关 support 不平衡、低基率 no-execution）。

## Act 6 · 物理路线：Fan2020-UDE（08.17–08.18 收编）

**来源**：`physical_models/fan2020_ude/README.md`、`evidence/EVIDENCE_CHAIN.md`、
`docs/ADHOC_DIRECT_WM_V2_SUPERVISOR_AUDIT_2026-08-18.md`。

孤立分支 `adhoc/lumped-enthalpy` 的纯物理探索（Fan2020-inspired 集总焓 ODE/DAE）：
- 显式部分：质量/能量关系、金属蓄热、燃料滞后、喷水混合、蒸发/干燥候选状态；
- 证据链 E0–E7：**E2 结构失配**（单纯 Fan20 参数适配不足，漂移与湿/干差异）、**E3 蒸发/干燥修复**
  （湿态局部偏差显著缩小）、**E4 残差权限**（读动作后代/状态反馈的残差可压缩或翻转响应——
  残差必须动作隔离）、**E5 动态局部化**（六工况局部线性稳定）、**E7 归因分离**。
- 用户经验总结：**响应有了（物理机制真实存在），但是瞬时的（时滞环节可修），
  误差太大且依赖初始状态**——现场缺可靠喷水流量/壁温/焓真值，不是真白箱。
- Direct-WM v2 审计（黑箱端点对照）：direct H18 F0 0.712 / F1 0.927°C（预测尚可），
  但动作响应弱且**方向随 fold/阀门/seed 漂移**（F0 一级阀 −0.00215/+0.00542、F1 一级阀 −0.01395/+0.03972
  方向正确 0-1/3）→ **DUAL-ACTION PHYSICAL RESPONSE NOT SUPPORTED**；"弱 25–450×"表述撤回
  （物理参考口径冲突），只保留"远小于候选物理对象响应且不稳定"。

**证据主线五条**（EVIDENCE_CHAIN §3）：
1. 纯物理先验有用但不充分（提供能量状态与动作路径，真实迁移暴露结构失配）；
2. 纯预测精度也不充分（黑箱可预测自然轨迹却对动作替换几乎不响应）；
3. 任意残差耦合会破坏物理语义（闭环数据的精度捷径可抵消热惯性或重写动作响应）；
4. 因此需要**统一而有权限边界的生成模型**：概率 observer（初态）+ boundary model（外生条件）+
   物理 transition（动作条件状态转移）+ action-blind closure（只补未测扰动）；
5. 快速模型（A1/LPV/Koopman）是母模型的降阶/蒸馏产物，不从秩亏数据凭空恢复。

## Act 7 · final_wm：灰箱融合管线 + 判别矩阵（08.18 起）

**来源**：`docs/plans/2026-08-18-final-world-model-pipeline-design.md`、
`experiments/final_wm/README.md`、`docs/WORLD_MODEL_EVIDENCE_LADDER.md`。
git 锚点：`a00f402`（08-18 架构定义）→ `37f3f94`（矩阵 v0.1 冻结）→ `e9af3cc`（v0.2）。

- **融合方法论**：Fan2020-inspired 物理状态转移（守恒方程结构 + 可学习参数 + IAPWS 可微代理）+
  神经观测/闭包 + 概率输出；五模块接口（observer/boundary/transition/closure/observation）。
- **D0 数据审计**（`548fa2f`→`cff0a59`）：40 列映射全部溯源既有代码（E0/PINN/RM3 schema），
  14/14 HIGH 无缺失；机组事实：默认干态运行、两相仅存在于减温器喷水液滴蒸发过程。
- **判别矩阵**：O1（观测）/ T1（结构增强链四臂：physics_only → closure_cons → closure_steam →
  latent4）/ B1（边界）/ J1（联合）/ R1（动作可靠性三探针：盲视/方向/泄漏）。
- **审计化探针**（auditpack）：阶跃响应、喷雾灵敏度、再湿消融、误差地板三锚点
  （fast_sigma / within_bin_sigma / 分箱偏差）、事件研究。
- **双机执行协议**：Hermes（本机 DGX）执行冻结命令、Codex 监督审计；产物强制入仓、
  ledger 逐 epoch、checkpoint 指纹校验、test 单次访问门。

## Act 8 · 判决-修复循环（08.19–08.21 至今）

**来源**：`4d742e1`（08-20 阻塞回传）→ `83f55cf`（08-21 修复）→ `6ab8f13`（perf）→
`d2bfa19`（worker）→ `e4da297`（动作信号分析）→ `51adf52`/`503fc73`（审计裁定）。
以及 `results/final_wm/optimization_roadmap.md`、`docs/FMTS2026_EVIDENCE_ALIGNMENT_AUDIT_2026-08-20.md`、
`docs/plans/2026-08-20-repair-batch-234-design.md`、`docs/fmts2026/paper/`。

1. **v0.2 首跑被阻**：resume 指纹不含模型结构 → 12 个 T1 臂全部 RESUMED 旧权重（`4d742e1`）；
   对侧修复为结构感知 + 代码树哈希指纹（`83f55cf`）。
2. **v0.2 全量判决**（`6305b50`）：DirectWM 动作弱且方向漂移；**R1 被拒在 direction**；
   误差结构：sh1_in H1 9.4°C = 38× persistence（模型已放弃该通道）、final 已近 persistence；
   单调负荷偏差在 60-epoch 预算下消失 → **"参数 MLP"立项证据作废**，路线图优先级重排
   （五点锚定 + 压力分段反演 > 喷雾动力学 > 参数 MLP）。
3. **论文证据对齐审计**（08-20）：数字口径冲突全部修正（"25-450×"撤回）、v0.3 修正案冻结
   （R1 规则、物理修复批②③④、side B 延后）、FMTS 论文 v1 快照（`0025c56`）+ 中文版（`ef92ef7`）。
4. **修复批②③④**（`8b15ff5`/`56a4c45`）：喷水→混合链路时滞（状态 9→11）、再湿项符号-量级硬契约、
   喷水灵敏度先验锚定（锚到 adhoc2 learned-lag 证据）。
5. **08-21 重跑 + 速度工程**：物理子步 torch.compile（`6ab8f13`，1.15×）、tf32（平价门 0.0017% 通过）、
   4 路并行实测 0.9× 串行 → **并行 runbook 撤回**（GB10 多进程争用）；per-phase 计时入 ledger。
6. **seed0 判决**：**direction 修复确认**（100% 一致，−0.228°C/+5%）——v0.2 的 DirectWM 症状消失；
   leakage 门 23.9% 触发 → 执行侧打乱 null（Δ=0.64%≈0）+ 欠拟合机制证明**探针伪影**
   （`e4da297`）→ 对侧协议化**打乱控制门**（`51adf52`）→ R1 seed0 暂定 PASS。
7. **增益缺口判读修正**（`503fc73`，用户裁定）：稳态复算 −0.194°C/2% vs 混合参考 −0.53~−1.48
   → 缺口 2.7–7.6×；混合参考是名义线性上界（零延迟+线性阀位+固定 Δh/cp），真实阀门等百分比非线性 +
   窄区间微调激励（|Δcmd|>0.1% 仅 0.89% 步）→ **th2 学到 0.44× 可解读为拟合工作点局部增益**。
8. **T1 减臂裁定**（用户批准）：只训 closure_cons × 3 seeds（省 75% 资源），三 seed 正式判决进行中。

## 贯穿主线：论文叙事骨架

1. **起点**：现场预测模型（监盘/超温预警）接入外回路前馈（step9 微分前馈超前控制）；
2. **异常**：动作通道对预测影响小、方向不可靠——预测准 ≠ 动作可信；
3. **诊断链**（Phase 1→3.5）：表示（差分/绝对 32–130×）→ 时标（120s+ 滞后）→ 混杂
   （闭环 PID 内生）→ **A1phys"方向正确"是闭环跟踪而非物理因果**；
4. **路线分叉**：纯物理（Fan2020）响应真实但瞬时、误差大、依赖初态；纯黑箱（DirectWM）
   精度高但动作方向漂移——**两头都不充分**；
5. **收敛**：灰箱融合世界模型（物理状态转移 + 权限边界神经组件）+ 判别矩阵审计 +
   打乱控制探针 + 误差地板锚定；
6. **主题**：*Forecasting Is Not World Modeling* —— 验证性（可审计的动作方向/幅度/时标），
   而非预测精度，才是世界模型晋级控制的合格属性。

## 附：史料索引

- 现场期：`~/wiki/concepts/火电主汽温预测控制.md`、`~/wiki/concepts/主汽温智能体.md`
- Phase 1：`docs/phase1_report.md`（08-01，含五教训与事件研究修正）
- 架构期：`docs/session_2026-08-05_causal_arch_eval.md`（A1phys 胜出判决）
- estimand 转折：`docs/PHASE35_DESIGN.md`（08-08，supervisor tracking 揭穿）
- MS 主线：`docs/PHASE35_MAINLINE_CONTEXT.md`（验证链 + 不能声称清单）
- 物理路线：`physical_models/fan2020_ude/evidence/EVIDENCE_CHAIN.md`（E0–E7 + 五条主线）
- DirectWM 判决：`docs/ADHOC_DIRECT_WM_V2_SUPERVISOR_AUDIT_2026-08-18.md`
- 融合管线：`docs/plans/2026-08-18-final-world-model-pipeline-design.md`
- 修复循环：`results/final_wm/optimization_roadmap.md`、`docs/plans/2026-08-20-repair-batch-234-design.md`、
  `results/final_wm/rerun_failure_response_20260820.md`（追加 1–4）、`results/final_wm/action_signal_analysis_20260821.md`
- 论文：`docs/fmts2026/paper/fmts_main.tex`（"Forecasting Is Not World Modeling"）
