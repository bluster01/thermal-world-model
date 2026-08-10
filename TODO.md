# TODO — Phase 3.5 论文核心验证

> 更新：2026-08-10。本文是项目唯一活任务队列。Phase 3.5 原 42-run 批次已收口；Phase 3.5-MS2-V/C validation+test 已独立复核并收口。MS2-J validation 得到“联合模块 PASS、staged 非劣 FAIL”的混合结果，现只放行一次性 synthetic test 复核该结论；Phase 4 仍暂停，Fan2017/2020/2021 不进入当前训练计划。

## 当前目标

以当前审计边界完成 Phase 3 文章：用现场 E1–E5 说明闭环预测与真实动作响应识别的差异，用 synthetic MS0–MS2-J 证明结构化多步响应架构在 known-truth 下可解，再用 fail-closed 门禁阻止两条证据臂互相替代。喷水流量传感器不准，因此它只作诊断；主动作是实际二级减温阀绝对开度代理，不把阀位伪装成质量流量。完整主线见 [Phase 3.5 主线实验上下文](docs/PHASE35_MAINLINE_CONTEXT.md)。

当前状态：**42/42 development runs 已完成且现场模型 checkpoint test 尚未评估；validation 审计判定 E3 不可识别、E4 被阻断、E5 样本不足、G3 参数塌缩，当前没有现场候选。A/B 两侧 1 s SP JSON 均已写入 test 事件及 `dT_post_600`，因此两侧 event test 都不再是盲 lockbox；未来正式事件证据必须使用新时间块。**详见 [Linux 增量整体审计](docs/PHASE3_5_LINUX_RETURN_AUDIT_2026-08-09.md)。

最新补充：分段辨识 pilot 的数字已在本地精确复现，但旧脚本丢失阀门阶跃符号、未实施负荷/压力/主汽温稳态门禁，且所谓脉冲响应只是未预白化滞后相关。因此 85%/74% 和 0 s 峰值不得升级为物理确认；修正门禁见 [分段辨识审计与验证协议](docs/PHASE35_SEGMENTED_IDENTIFICATION_REVIEW_2026-08-09.md)。

收口决定：原 42-run 数据和配置不再补 seed、不打开模型 test，也不在旧协议内扩大 A1phys 矩阵。新增 Phase 3.5-MS 先用已知真值合成系统回答“多步响应架构是否可辨识且稳定”，其阳性结果也不能替代现场 E3，不能写成“完全物理响应”或“反事实世界模型已成立”。最终目标的缺口与后续 W0–W6 门禁见 [主汽温世界模型证据阶梯](docs/WORLD_MODEL_EVIDENCE_LADDER.md)。

### 当前唯一判决点

只执行并审计一次 MS2-J synthetic test。若联合模块复现，停止扩大 synthetic 矩阵并进入论文表图/claim ledger；若不复现，原样报告 validation/test 不一致。staged 无论结果如何都不能事后改写成路线冠军。MS2-D、MS3、MS4、MS5 与 Phase 4 均保持 HOLD，除非论文收口后重新授权。

## Phase 3.5-MS — 多步动作响应可解性

统一模型为 `T_hat = f_free(context) + g_response(context, action_path, reference_path)`。未来动作不得进入 `f_free`；所有 `g_response` 在结构上满足参考路径零响应和时间因果性。第一批只跑 synthetic known-truth，不读取 A/B 模型 test，也不恢复旧 E4。

| ID | 任务 | 论文问题 | 产物/门禁 | 状态 |
|---|---|---|---|---|
| MS0 合同 | 冻结统一输入、参考干预、状态与诊断接口 | 四种方法是否在比较同一 estimand？ | exact identity；future action leakage=0 | ✓ 代码与测试完成 |
| MS1 已知真值 | 二阶惯性下的 hold/step/pulse/ramp/multi-step | 架构是否至少能恢复一个可解的多步系统？ | 18/18；参数恢复；结构门禁；单次 synthetic test | ✓ PASS：只支持同型可解性，不设路线冠军 |
| MS2-V 阀门非线性 | R50 真值下 identity/oracle/learned monotone 与灵活算子 | 单调模块能否恢复支持域内增量响应？ | 6 candidates×3 seeds；clean NMAE；独立榜 | ✅ validation+test 双层 PASS（test: monotone vs identity CI下界 0.859–0.884 >> 20%，3seed×256ep×10k bootstrap；oracle 0.0043 复现；`K/phi` 补偿使真实曲线不可单独辨识） |
| MS2-C 工况调度 | 增益/时间常数随 context 变化 | 多步 A1phys 参数调度能否辨识？ | 5 candidates×3 seeds；clean NMAE；独立榜 | ✅ validation+test 双层 PASS（test: scheduled vs global CI下界 0.884–0.891 >> 20%，3seed×256ep×10k bootstrap；K/τ 调度相关性高） |
| MS2-J 联合耦合 | 同一真值同时含 R50 非线性与 context 调度 | 双模块能否共同收敛？staged 是否比 joint 更稳定？ | 9 candidates×3 seeds；双模块 joint/staged、单模块消融、灵活路线；一次性 test | ✅ validation+test 双层：联合模块 PASS（test CI下界 0.73–0.89 >> 20%）；staged FAIL 复现（test ratio 1.14–1.20，CI上界 1.09–1.32 > 1.10）；oracle 0.0225 复现 |
| MS2-D 后续压力 | 纯迟延、阶次扩展、未建模扰动 | 结论是否跨更强失配成立？ | MS2-V/C 收口后再决定 | ◻ HOLD，当前不铺开 |
| MS3 真实数据适配 | 复用 A/B causal cache，交叉阀位已按现场映射 | 合成可解性能否迁移到观测预测？ | validation-only；A/B 分榜；不称因果 | ◻ HOLD |
| MS4 经验响应校准 | 仅在未来新时间块 E3 通过后连接真实 IRF | 模型响应是否复现可识别物理响应？ | common support、稳态/动态双 estimand | ⛔ 等待新数据证据 |
| MS5 分段训练 | free 预训练 → 短冻结 response → 小学习率联合 | 完整世界模型中 response 是否会被 free head 吸收？ | stage checkpoints、梯度/参数健康、joint 非退化 | ◻ 完整 free+response 耦合仍 HOLD；MS2-J 只先验证 response 内部 staging |

MS1 的客观复核见 [MS1 Supervisor Review](docs/PHASE35_MS1_REVIEW_2026-08-10.md)，MS2 validation/test 见 [MS2 Validation Review](docs/PHASE35_MS2_VALIDATION_REVIEW_2026-08-10.md) 与 [MS2 Test Review](docs/PHASE35_MS2_TEST_REVIEW_2026-08-10.md)。MS2-J 只做联合模块和 response 内部训练策略，不把 synthetic PASS 升级成现场物理辨识，也不外推完整 `free+response` 的 MS5 staging。

## 五组核心实验

| ID | 核心问题 | 对照 | 论文判据 | 当前状态 |
|---|---|---|---|---|
| E1 动作表征 | Δ阀位差是因丢失基准而失败吗？ | `delta_no_baseline` / `delta_with_baseline` / `absolute_identity` | 绝对阀位或带基准重建预测非劣，动作响应不退化 | 正对照通过；无优越性证据 |
| E2 阀门非线性 | 阀位到有效喷水作用是否需要非线性映射？ | identity / monotone / monotone+rate | 预测非劣，IRF-WMAE 或剂量一致性改善；否则回退 identity | INCONCLUSIVE；无稳定改善 |
| E3 真实物理响应 | 开/关阀后，二减出口与主汽温是否出现方向、迟延和剂量响应？ | 隔离阀门事件 vs 处理前匹配 quiet controls | A/B 分报；方向率、时标、匹配事件数与日级 block-bootstrap CI | INCONCLUSIVE；caliper 后 A=93/93 开阀，B=121 开/1 关，且 balance 未过 |
| E4 模型响应 | A1phys 的 constant-valve 反事实能否复现 E3？ | logged valve vs onset 前恒定阀位；`free_only` 负基线 | direction、lag error、IRF-WMAE、dose monotonicity | BLOCKED；E3 无效且参数塌缩 |
| E5 SP 未执行负对照 | SP 变但 600 s 内阀门不变时，能否区分“监督信号”和“实际动作”？ | executed / no-execution / delayed-or-ambiguous | no-execution 动作分支严格近零；同时报告真实温度变化，不把 SP 当 plant action | INCONCLUSIVE；A/B no-execution=4/2 |

E1–E5 全部属于 Phase 3.5 主文证据，不是 Phase 4，也不是附录补跑。详细公式和 claim 边界见 [Phase 3.5 实验设计](docs/PHASE3_5_EXPERIMENT_DESIGN.md)。

## 四条线性工作线

| 工作线 | 必须按顺序完成 | 当前停点 | 放行产物 |
|---|---|---|---|
| A. 数据与事件 | A0 原始 A/B hash → A1 因果 10 s cache/staleness → A2 split 内窗口 → A3 validation 事件/匹配/balance | A0–A2 完成；A3 软件 fail-closed 已修，科学事件定义仍未闭合 | A/B cache manifest、validation event manifests |
| B. 模型与训练 | B0 constant-valve identity → B1 7 配置×A/B×3 seeds → B2 validation 选模 → B3 候选补到 5 seeds | B0–B1 完成；B2 无候选，B3 HOLD | 每侧最多 2 个 test 候选及 canonical checkpoints |
| C. 评测与统计 | C0 预测指标 → C1 E3/E4 IRF → C2 E5 负对照 → C3 日级 block bootstrap → C4 test 一次访问 | C0 完成；C1–C3 维持 INCONCLUSIVE/BLOCKED；C4 HOLD | `summary_validation.*`、冻结候选表 |
| D. 论文与审计 | D0 claim ledger → D1 validation 表图 → D2 单次 test → D3 复算/反例 → D4 主文 | 已形成负面辨识审计；主结果仍 HOLD | Phase 3.5 主文核心表图与可守结论 |

## 收口后的执行清单

| 顺序 | 任务 | 谁执行 | 完成标准 | 状态 |
|---:|---|---|---|---|
| 1 | 复核 A/B CSV 表头、时间范围、关键 tag 与单位 | Linux 只读；本地审计 | A/B 必需列齐全；确认 SP、指令、反馈阀位层级 | ✓ |
| 2 | 生成 causal LOCF cache、staleness 和 SHA256 manifest | Linux | A/B `.npz` 与 `.manifest.json`；补 cache artifact SHA/生成 SHA | △ source 证据有，cache SHA 待补 |
| 3 | dry-run 42 个开发命令并保存环境/git SHA | Linux | 命令数、config、side、seed 与矩阵一致 | ✓ |
| 4 | 运行 42 个开发训练并仅评估 validation | Linux | 7 configs × 2 sides × 3 seeds；模型评估未访问 test | ✓ |
| 5 | 汇总 validation，审计事件数、日块数、balance/pretrend 与参数塌缩 | 本地 / Codex | E1–E5 均可判 PASS/FAIL/INCONCLUSIVE；缺证据不强判 | ✓ 已判；无候选 |
| 5A | 修 1 s 事件 horizon/split/provenance，A/B 只跑 validation | 本地写代码；Linux 执行 | 显式多 horizon；split 真过滤；test 默认锁定；600 s gap/hold 可审计 | ◼ 代码完成；当前批不重跑，旧 JSON 仅作探索材料 |
| 5B | 修参数健康摘要 | 本地写代码；Linux 执行 | τ 两 stage/真实秒、rate gain、完整 checkpoint/cache/anchor hash、free-only 显式排除 | ◼ 代码完成；当前塌缩结论足以阻断，不为“补好看数字”重跑 |
| 5C | 冻结未来正式 E3 双 estimand | 本地设计 | 稳态 held-step 主分析；动态 trajectory 次分析；开/关均满足 common support | ✅ 已冻结 (2026-08-09)：双 estimand 设计、配对/门禁/判定/样本量/placebo 全预注册，见 [E3 estimand 冻结设计](docs/PHASE35_E3_ESTIMAND_FROZEN_2026-08-09.md)；现有数据不达门槛，未来新时间块或现场试验按此执行；未通过前不得恢复 E3/E4 强结论 |
| 5D | 修正分段辨识并执行 V0–V4 | 本地冻结协议/审计；Linux 改实现并回传 | 保留有符号剂量；完整稳态/hold 门禁；blocked 验证；2×2 与 placebo；不得使用伪脉冲响应 | ▶ Linux 执行完毕回传 (2026-08-09)：V0–V4 全 INCONCLUSIVE/NOT PASSED（A=7/B=6 事件不足、12 open/1 close 无双向 support、held-step=2）；85%/74% 降级为 exploratory pilot；脚本已代码化，未来新时间块可原样重跑 |
| 6 | 每侧冻结最多 2 个候选，补 seed 3/4 | Linux | 候选选择和 seed 清单写入版本化 manifest | ⛔ 当前批取消；无候选 |
| 7 | 一次批量打开 test，评估冻结候选 | Linux | 每 run 生成 `access_ledger.json`，不得按 test 回调模型 | ⛔ 当前批取消；模型 test 保持未访问 |
| 8 | 制作 validation 表图、更新 claim ledger 和论文 | 本地 / Codex | 同时报预测、经验响应、模型响应、CI 和失败边界 | ▶ 当前唯一文章任务；结果保持阴性/不确定 |

## 放行门禁

| Gate | 必须满足 | 失败处理 |
|---|---|---|
| G0 代码 | Phase 3.5 单测、compile 和 CPU smoke 通过；全仓收集失败单独登记；future action 不影响 earlier response | 不交 Linux |
| G1 数据 | SHA256、列、时间、缺失/staleness、A/B 分侧和 split 边界可审计 | 返回数据治理 |
| G2 事件 | validation 至少 30 个 matched events、开关各 10 个和 10 个独立日块；balance/pretrend 合格，H60 方向化 CI 排除零 | E3–E5 记 inconclusive，不伪造物理结论 |
| G3 模型 | constant-valve effect=0、开阀长期增益≤0、参数有限且未塌缩 | 淘汰该配置 |
| G4 候选 | validation 预测非劣且 E4 通过；每侧最多保留 2 个 | 不看 test，简化或停止 |
| G5 论文 | A/B 方向一致或差异有工程解释；test 仅一次；结果经本地复算 | 只写限定性/阴性结论 |

## 本地与 Linux 分工

| 本地 / Codex | Linux 远端 |
|---|---|
| 设计协议、写模型/脚本/测试、冻结矩阵、审计日志、复算统计、写论文 | 在指定 commit 上只执行 README 命令，训练并原样回传所有产物 |
| 可以修代码，但每次修复产生新 commit 和新 manifest | 不改阈值、模型、seed、split，不挑好看的 run，不自行写结论 |

Linux 的唯一运行说明见 [experiments/phase3_5/README.md](experiments/phase3_5/README.md)。

## 当前已完成

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

Phase 4 的 Fan20 守恒骨架、Fan17 金属蓄热、Fan21 宽负荷 mismatch，以及与 Fan 状态耦合的 Neural ODE / Controlled Koopman / 时变灰箱路线继续冻结。Phase 3.5-MS 只实现低维动作增量响应表示，用于方法可解性和后续 A1phys-MS 消融；它不启用 Fan 路线，也不改变原 42-run 的候选选择或论文当前现场结论。
