# TODO — Phase 3.5 论文核心验证

> 更新：2026-08-09。本文是项目唯一活任务队列。Phase 4 已暂停；Fan2017/2020/2021 与三条可微路线不进入当前训练计划。

## 当前目标

完成 Phase 3 文章所需的核心证据链：先证明现场阀门确有可辨认的温度响应，再检验 A1phys 是否能复现该响应，同时不牺牲预测精度。喷水流量传感器不准，因此它只作诊断；主动作是实际二级减温阀绝对开度，模型学习 A/B 侧独立的单调非线性有效开度。

当前状态：**框架已实现并通过本地单测，尚无 Phase 3.5 真实数据训练结果，暂不打开 test。**

## 五组核心实验

| ID | 核心问题 | 对照 | 论文判据 | 当前状态 |
|---|---|---|---|---|
| E1 动作表征 | Δ阀位差是因丢失基准而失败吗？ | `delta_no_baseline` / `delta_with_baseline` / `absolute_identity` | 绝对阀位或带基准重建预测非劣，动作响应不退化 | implemented |
| E2 阀门非线性 | 阀位到有效喷水作用是否需要非线性映射？ | identity / monotone / monotone+rate | 预测非劣，IRF-WMAE 或剂量一致性改善；否则回退 identity | implemented |
| E3 真实物理响应 | 开/关阀后，二减出口与主汽温是否出现方向、迟延和剂量响应？ | 隔离阀门事件 vs 处理前匹配 quiet controls | A/B 分报；方向率、时标、匹配事件数与日级 block-bootstrap CI | implemented，待数据审计 |
| E4 模型响应 | A1phys 的 constant-valve 反事实能否复现 E3？ | logged valve vs onset 前恒定阀位；`free_only` 负基线 | direction、lag error、IRF-WMAE、dose monotonicity | implemented，待训练 |
| E5 SP 未执行负对照 | SP 变但 600 s 内阀门不变时，能否区分“监督信号”和“实际动作”？ | executed / no-execution / delayed-or-ambiguous | no-execution 动作分支严格近零；同时报告真实温度变化，不把 SP 当 plant action | implemented，待数据审计 |

E1–E5 全部属于 Phase 3.5 主文证据，不是 Phase 4，也不是附录补跑。详细公式和 claim 边界见 [Phase 3.5 实验设计](docs/PHASE3_5_EXPERIMENT_DESIGN.md)。

## 四条线性工作线

| 工作线 | 必须按顺序完成 | 当前停点 | 放行产物 |
|---|---|---|---|
| A. 数据与事件 | A0 原始 A/B hash → A1 因果 10 s cache/staleness → A2 split 内窗口 → A3 validation 事件/匹配/balance | ▶ A0 Linux 数据预处理 | A/B cache manifest、validation event manifests |
| B. 模型与训练 | B0 constant-valve identity → B1 7 配置×A/B×3 seeds → B2 validation 选模 → B3 候选补到 5 seeds | B0 已通过；▶ B1 | 每侧最多 2 个 test 候选及 canonical checkpoints |
| C. 评测与统计 | C0 预测指标 → C1 E3/E4 IRF → C2 E5 负对照 → C3 日级 block bootstrap → C4 test 一次访问 | C0–C3 已实现；▶ validation 审计 | `summary_validation.*`、冻结候选表 |
| D. 论文与审计 | D0 claim ledger → D1 validation 表图 → D2 单次 test → D3 复算/反例 → D4 主文 | ▶ D0 | Phase 3.5 主文核心表图与可守结论 |

## 现在的执行清单

| 顺序 | 任务 | 谁执行 | 完成标准 | 状态 |
|---:|---|---|---|---|
| 1 | 复核 A/B CSV 表头、时间范围、关键 tag 与单位 | Linux 只读；本地审计 | A/B 必需列齐全；确认 SP、指令、反馈阀位层级 | □ |
| 2 | 生成 causal LOCF cache、staleness 和 SHA256 manifest | Linux | A/B `.npz` 与 `.manifest.json`；无未来回填 | □ |
| 3 | dry-run 42 个开发命令并保存环境/git SHA | Linux | 命令数、config、side、seed 与矩阵一致 | □ |
| 4 | 运行 42 个开发训练并仅评估 validation | Linux | 7 configs × 2 sides × 3 seeds；test 未访问 | □ |
| 5 | 汇总 validation，审计事件数、日块数、balance/pretrend 与参数塌缩 | 本地 / Codex | E1–E5 均可判 PASS/FAIL/INCONCLUSIVE；缺证据不强判 | □ |
| 6 | 每侧冻结最多 2 个候选，补 seed 3/4 | Linux | 候选选择和 seed 清单写入版本化 manifest | × 等 5 |
| 7 | 一次批量打开 test，评估冻结的 5-seed 候选和 `free_only` | Linux | 每 run 生成 `access_ledger.json`，不得按 test 回调模型 | × 等 6 |
| 8 | 本地复算 test、制作表图、更新 claim ledger 和论文 | 本地 / Codex | 同时报预测、经验响应、模型响应、CI 和失败边界 | × 等 7 |

## 放行门禁

| Gate | 必须满足 | 失败处理 |
|---|---|---|
| G0 代码 | Phase 3.5 单测、compile 和 CPU smoke 通过；future action 不影响 earlier response | 不交 Linux |
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
- [x] Phase 3.5 本地测试通过；正式结果仍为空。

## 暂停项

Phase 4 的 Fan20 守恒骨架、Fan17 金属蓄热、Fan21 宽负荷 mismatch，以及 Neural ODE / Controlled Koopman / 时变灰箱三路线全部冻结为未来路线图。它们不影响 Phase 3.5 的代码、预算、候选选择或论文当前结论。
