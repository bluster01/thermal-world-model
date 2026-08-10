# TODO — Phase 3.5-MS 完整模型验证

> 更新：2026-08-11。本文是项目唯一人工任务队列；机器状态见 `configs/phase3_5/experiment_registry.json`。MS2-D2 one-shot test 已确认并关闭，当前唯一授权是 MS2-D3 colored-disturbance validation。旧 E1–E5 已废弃；Phase 4 仍暂停。

## 当前目标

先完成 Phase 3.5-MS 全系列：MS2-D 结构压力、MS5 完整 `free+response` 耦合、MS3 A/B 真实适配和 MS4 串级闭环物理响应；完成后再选择最终架构并进入论文。喷水流量不作真值，主动作是实际二级减温阀反馈开度代理。完整恢复入口见 [上下文快照](docs/PHASE35_CONTEXT_SNAPSHOT.md)。

当前状态：**MS0、MS1、MS2-V/C/J、MS2-D1 和 MS2-D2 已完成；MS2-D3 的设计、矩阵、代码和本地测试已冻结，等待 Linux validation。**D2 独立 test 中 oracle clean NMAE 为 0.0211–0.0255，三阶主模型为 0.0444–0.0465，相对二阶点改善 23.74%–25.36%，冻结 bootstrap 95% CI 下界为 19.90%–21.22%，逐 seed高于 10%。这只确认 frozen known-truth 下的三阶响应优势；二极点+迟延与 DeepONet 仍能在有限 horizon 逼近，不支持现场唯一阶次或机制。

边界保持不变：synthetic 阳性只能证明方法可解与结构压力下的响应恢复，不能替代现场物理响应，不能写成“完全物理响应”或“反事实世界模型已成立”。最终缺口与 W0–W6 门禁见 [主汽温世界模型证据阶梯](docs/WORLD_MODEL_EVIDENCE_LADDER.md)。

### 当前判决点

MS2-D3 当前判决点是：在 D2 三阶、R50、context-scheduled、无 pure delay 真值上，加入 response operator 不可观察的 action-independent stationary AR(1) output disturbance（`sigma=0.03 °C`、`tau=120 s`）后，oracle 是否逐 seed clean NMAE `<0.05`、三阶主模型是否逐 seed `<0.10`，且三阶相对同预算二阶的配对/profile 分层 bootstrap 95% CI 下界是否逐 seed `>=10%`。D3 validation 只作 screening；扰动拟合、tau/迟延、profile/horizon、D2→D3 漂移和 secondary 路线均不参与主门或选冠军，test 继续锁定。

## Phase 3.5-MS — 多步动作响应可解性

统一模型为 `T_hat = f_free(context) + g_response(context, action_path, reference_path)`。未来动作不得进入 `f_free`；所有 `g_response` 在结构上满足参考路径零响应和时间因果性。第一批只跑 synthetic known-truth，不读取 A/B 模型 test，也不恢复旧 E4。

| ID | 任务 | 核心问题 | 产物/门禁 | 状态 |
|---|---|---|---|---|
| MS0 合同 | 冻结统一输入、参考干预、状态与诊断接口 | 四种方法是否在比较同一 estimand？ | exact identity；future action leakage=0 | ✓ 代码与测试完成 |
| MS1 已知真值 | 二阶惯性下的 hold/step/pulse/ramp/multi-step | 架构是否至少能恢复一个可解的多步系统？ | 18/18；参数恢复；结构门禁；单次 synthetic test | ✓ PASS：只支持同型可解性，不设路线冠军 |
| MS2-V 阀门非线性 | R50 真值下 identity/oracle/learned monotone 与灵活算子 | 单调模块能否恢复支持域内增量响应？ | 6 candidates×3 seeds；clean NMAE；独立榜 | ✅ validation+test 双层 PASS（test: monotone vs identity CI下界 0.859–0.884 >> 20%，3seed×256ep×10k bootstrap；oracle 0.0043 复现；`K/phi` 补偿使真实曲线不可单独辨识） |
| MS2-C 工况调度 | 增益/时间常数随 context 变化 | 多步 A1phys 参数调度能否辨识？ | 5 candidates×3 seeds；clean NMAE；独立榜 | ✅ validation+test 双层 PASS（test: scheduled vs global CI下界 0.884–0.891 >> 20%，3seed×256ep×10k bootstrap；K/τ 调度相关性高） |
| MS2-J 联合耦合 | 同一真值同时含 R50 非线性与 context 调度 | 双模块能否共同收敛？staged 是否比 joint 更稳定？ | 9 candidates×3 seeds；双模块 joint/staged、单模块消融、灵活路线；一次性 test | ✅ validation+test 双层：联合模块 PASS（test CI下界 0.73–0.89 >> 20%）；staged FAIL 复现（test ratio 1.14–1.20，CI上界 1.09–1.32 > 1.10）；oracle 0.0225 复现 |
| MS2-D1 纯迟延 | R50+调度 truth 加 20 s delay | delay module 是否必要？响应和参数能否分别恢复？ | 6 candidates×3 seeds；oracle；learned vs no-delay；参数诊断分列 | ⚠️ validation PASS / **test 未确认**：oracle 0.021<0.05 双层复现；test CI 下界 17.2–18.8% < 20%（点估计 20.4–22.5% 方向一致）；E[d] 精确但分布弥散 → capacity 有效、真实迟延未唯一恢复；按冻结解释不重试不调阈值 |
| MS2-D2 | 无 pure delay 的三阶惯性 truth | 三极点能否优于同预算二极点？遗漏阶次会否被误读为 delay？ | 7 candidates×3 seeds；oracle、绝对 NMAE、三阶 vs 二阶；参数/虚假 delay 单列 | ✅ test 确认并关闭；仅限 frozen known-truth response，不证明现场唯一阶次 |
| MS2-D3 | 未建模扰动 | 结论是否跨 action-independent colored disturbance 成立？ | D2 truth + stationary AR(1) output nuisance；7 candidates×3 seeds；clean 主门与扰动诊断分离 | ▶ **当前唯一 Linux 任务：21-run validation** |
| MS5 完整耦合 | free 预训练/冻结/联合与 joint 对照 | response 是否被 free head 吸收？ | stage checkpoint、梯度、参数漂移、动作敏感性 | ◻ MS2-D 后、MS3 前 |
| MS3 真实数据适配 | 复用 A/B causal cache，交叉链按现场冻结 | 合成可解性能否迁移到观测预测？ | validation-only；A/B 分榜；不称因果 | ◻ 等待 MS5 |
| MS4 闭环物理响应 | SP→串级 PID→阀位→T2/Tm | 完整模型是否复现现场闭环响应？ | held-step、方向、时标、跨块；不称开环 do(valve) | ◻ 等待 MS3 |

MS1 的客观复核见 [MS1 Supervisor Review](docs/PHASE35_MS1_REVIEW_2026-08-10.md)，MS2 validation/test 见 [MS2 Validation Review](docs/PHASE35_MS2_VALIDATION_REVIEW_2026-08-10.md) 与 [MS2 Test Review](docs/PHASE35_MS2_TEST_REVIEW_2026-08-10.md)。MS2-J 只做联合模块和 response 内部训练策略，不把 synthetic PASS 升级成现场物理辨识，也不外推完整 `free+response` 的 MS5 staging。

## 历史 E1–E5（废弃归档）

| ID | 核心问题 | 对照 | 论文判据 | 当前状态 |
|---|---|---|---|---|
| E1 动作表征 | Δ阀位差是因丢失基准而失败吗？ | `delta_no_baseline` / `delta_with_baseline` / `absolute_identity` | 绝对阀位或带基准重建预测非劣，动作响应不退化 | 正对照通过；无优越性证据 |
| E2 阀门非线性 | 阀位到有效喷水作用是否需要非线性映射？ | identity / monotone / monotone+rate | 预测非劣，IRF-WMAE 或剂量一致性改善；否则回退 identity | INCONCLUSIVE；无稳定改善 |
| E3 真实物理响应 | 开/关阀后，二减出口与主汽温是否出现方向、迟延和剂量响应？ | 隔离阀门事件 vs 处理前匹配 quiet controls | A/B 分报；方向率、时标、匹配事件数与日级 block-bootstrap CI | INCONCLUSIVE；caliper 后 A=93/93 开阀，B=121 开/1 关，且 balance 未过 |
| E4 模型响应 | A1phys 的 constant-valve 反事实能否复现 E3？ | logged valve vs onset 前恒定阀位；`free_only` 负基线 | direction、lag error、IRF-WMAE、dose monotonicity | BLOCKED；E3 无效且参数塌缩 |
| E5 SP 未执行负对照 | SP 变但 600 s 内阀门不变时，能否区分“监督信号”和“实际动作”？ | executed / no-execution / delayed-or-ambiguous | no-execution 动作分支严格近零；同时报告真实温度变化，不把 SP 当 plant action | INCONCLUSIVE；A/B no-execution=4/2 |

E1–E5 不再属于当前 Gate 或候选选择；表格只保留失败机制和审计追溯。当前现场验证由 MS3/MS4 重新设计。

## 四条并行管理线

| 工作线 | 当前任务 | 放行条件 | 下一步 |
|---|---|---|---|
| A. 方法与架构 | D3 三阶响应的有色扰动稳健性 | oracle `<0.05`；三阶 `<0.10`；配对 bootstrap CI 下界逐 seed `≥10%` | 只判 clean response 稳健性，不按 secondary 排名挑结论 |
| B. 训练与计算 | Linux 执行冻结的 21-run D3 validation | 唯一授权、干净 commit、21/21 manifest/history/checkpoint/episode/log | 原样回传；不改阈值/seed/split，不访问 test，不启动 MS5 |
| C. 统计与审计 | 本地复算 D3 validation episode 与门禁 | clean 主门与 disturbance/tau/delay/profile/horizon diagnostic 分离 | 只形成 screening 判决，再决定是否授权 one-shot test |
| D. 数据与现场 | 维护 MS3/MS4 需求，不提前执行 | A/B 分侧、现场交叉链冻结、SP held-step、稳态/动态分层、新时间块 | 等 MS5 完成后进入 MS3，再进入 MS4 |

## 当前 D3 执行清单

| 顺序 | 任务 | 负责人 | 完成标准 | 状态 |
|---:|---|---|---|---|
| 1 | 独立复核 D2 one-shot test 并关闭 Gate | 本地 | canonical 重算、50k bootstrap、manifest/ledger/pin/日志审计 | ✓ `CONFIRMED_SYNTHETIC_ORDER_RESPONSE` |
| 2 | 冻结 D3 单一问题与 nuisance 生成式 | 本地 | D2 clean truth 不变；stationary AR(1) output disturbance；不混 MS5 | ✓ |
| 3 | D3 matrix、runner 与 fail-closed summary | 本地 | 21-run；clean 主门；诊断隔离；test artifact 拒绝 | ✓ |
| 4 | D3 本地代码与协议回归 | 本地 | generator determinism/stationarity；matrix drift；episode pairing；status | ✓ |
| 5 | D3 validation 正式运行 | Linux | 21/21、日志、summary、checkpoint archive 原样回传 | ▶ **当前唯一远端任务** |
| 6 | D3 validation 独立复算与审计 | 本地 | checkpoint/history/hash/结构门、episode bootstrap 与反例审计 | ◻ 等回传；test/MS5 冻结 |

## 本地与 Linux 分工

| 本地 / Codex | Linux 远端 |
|---|---|
| 设计协议、写模型/脚本/测试、冻结矩阵、审计日志、复算统计、写论文 | 在指定 commit 上只执行 README 命令，训练并原样回传所有产物 |
| 可以修代码，但每次修复产生新 commit 和新 manifest；只有本地可改注册表/TODO/审计结论 | 不改阈值、模型、seed、split，不挑 run，不改权威文档/状态，不自行写审计或结论 |

Linux 的唯一运行说明见 [experiments/phase3_5/README.md](experiments/phase3_5/README.md)。

## 当前已完成

- [x] 建立机器可校验的实验注册表、关键脚本状态和上下文恢复快照；E 系列标记为 `deprecated`。
- [x] MS0 合同、MS1 同型可解性、MS2-V/C 失配与 MS2-J 联合耦合完成 validation+test 审计。
- [x] MS2-J 确认 joint 主训练；staged 作为失败消融保留，不外推 MS5。
- [x] MS2-D1 pure-delay runner、汇总器、冻结矩阵和本地回归测试完成。
- [x] MS2-D1 validation 的 summary/checkpoint/history/archive 独立复核完成；降级为 screening PASS，参数辨识诊断 FAIL。
- [x] 冻结并执行 D1 一次性 synthetic test authorization、访问 ledger、paired episode bootstrap 与 fail-closed 汇总器。
- [x] 独立审计 D1 test：oracle 通过，但主对比 CI 下界 17.2–18.8% 未达 20%；D1 阴性关闭，不重试。
- [x] 冻结 D2 三阶惯性设计、21-run matrix、runner、fail-closed summary 与本地测试。
- [x] 独立审计 D2 validation：21/21 结构与主门通过；确认其为 screening PASS，并记录三阶/迟延机制补偿边界。
- [x] 冻结 D2 one-shot test 的 content pins、配对 bootstrap 主门、执行器、汇总器和访问 ledger；只允许原 21 checkpoints 一次推理。
- [x] 独立审计 D2 test：21/21、内容 pin、episode 重算与独立 50k bootstrap 一致；按 `CONFIRMED_SYNTHETIC_ORDER_RESPONSE` 关闭，同时保留阶次/迟延不可唯一辨识边界。
- [x] 冻结 D3 的 stationary AR(1) output disturbance 设计、21-run matrix、runner、fail-closed summary、episode 统计和本地专项测试；当前只授权 validation。
- [x] 审计旧 `exp_106/112` 的 test-selection、CFI fallback 和 ΔSP estimand 问题。
- [x] 拉取并审计 exp_201：将固定 `R=50` 的方向信号降级为 pilot，并纳入 E2 正式对照。
- [x] 将 Phase 4 暂停，并把所有论文补充实验收归 Phase 3.5。
- [x] 建立 `src/phase35`、训练/评估/汇总脚本和 42-run 冻结矩阵。
- [x] 约束有效开度单调、长期开阀增益非正、constant-valve 干预严格为零。
- [x] validation/test 完全分离；test 需显式授权且写 access ledger。
- [x] A/B 分侧、喷水流量不作真值、SP 600 s 未执行负对照、UTC 日块 bootstrap。
- [x] 42/42 development runs 与 validation 结果已回传；预测消融可保留为开发结果。
- [x] Supervisor 已判 E3 INCONCLUSIVE、E4 BLOCKED、E5 INCONCLUSIVE，当前无 test 候选。
- [x] 原 SP-IV/gain truth 已撤回；现有 SP 事件只作动态闭环不可辨识诊断。
- [x] 修复新增 1 s 脚本的 3s/30s 索引、时间单位和 A/B 参数化。
- [x] 本地修复 split/test-lock/min-gap/provenance；决定当前批不为补结果重跑，旧 v2 JSON 只作探索证据，未来新时间块使用修复版协议。
- [x] 明确 τ 是 step 并换算 seconds，补 rate gain；G3 维持 FAIL（参数塌缩）。
- [x] 登记 A/B 两侧 event test 均已被 exploratory 访问；模型 test 尚未访问，未来正式事件证据改用新时间块。

## 暂停项

Phase 4 的 Fan20 守恒骨架、Fan17 金属蓄热、Fan21 宽负荷 mismatch，以及与 Fan 状态耦合的 Neural ODE / Controlled Koopman / 时变灰箱路线继续冻结。Phase 3.5-MS 当前只验证低维动作增量响应表示；完整 MS 系列结束前不启动 Fan 路线，也不提前确定最终模型或论文主结论。
