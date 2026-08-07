# Phase 4 实验计划 — 主汽温火电世界模型

> 版本：2026-08-07；状态：Supervisor pre-registration draft。
> 本文定义“做什么、何时停止、如何判决”。活任务见 [`TODO.md`](../TODO.md)，问题证据见 [`SUPERVISOR_REVIEW_2026-08-07.md`](SUPERVISOR_REVIEW_2026-08-07.md)，代码落地步骤见 [`plans/2026-08-07-phase4-implementation.md`](plans/2026-08-07-phase4-implementation.md)。

## 1. Phase 4 的目的与边界

Phase 4 不是继续寻找一个更低 MAE 的预测头，而是回答以下问题：

1. 主汽温对象、动作层级和可识别输入是否已经定义清楚；
2. Fan20 的两级喷水—焓值—主汽温链能否在当前 10 s DCS 数据上闭合并稳定辨识；
3. Fan17 的显式金属蓄热与 Fan21 的宽负荷能量不匹配是否提供可复现的增量信息；
4. 固定相同物理内容后，structured ODE、fixed-operator controlled Koopman 与 time-varying gray-box 三种表示/closure 路线，哪条在预测、事件响应、物理一致性和计算成本上形成更好的 Pareto 解；
5. 结论能否跨时间、负荷、动作幅值和运行 episode 保持，而不是来自重复滑窗或 test 选模。

Phase 4 **不包含**：正式 MPC 优越性结论、现场闭环部署结论、随机干预因果结论、概率预测主张。MPC 只有在 plant model 锁定并使用独立 evaluation plant 后才另立阶段。

## 2. Supervisor 的路线选择

| 方案 | 优点 | 主要风险 | 判决 |
|---|---|---|---|
| A. 直接强化 A1phys | 成本低，已有代码 | 动作是监督层 ΔSP，物理链不闭合，论文主张上限低 | 仅保留为 baseline |
| B. Fan17/20/21 × 三路线全笛卡尔赛马 | 覆盖全面 | 计算量大，混淆“物理内容”和“动态表达”，多重比较严重 | 不采用 |
| C. 分阶段 physics-first | 每一 Gate 回答一个问题，可提前停止，证据链清楚 | 前期工程与协议成本较高 | **采用** |

推荐路径：`协议与语义 → Fan20 闭合 → 嵌套物理组件 → 三种表示公平筛选 → rolling robustness → locked final / internal-final`。

## 3. 两个任务、两个 estimand

现有代码把监督层设定值和物理喷水动作放进了同一叙事。Phase 4 必须拆开：

```mermaid
flowchart LR
    S["完整 SP 轨迹（ΔSP 只定义事件 exposure）"] --> C["温控器与执行机构"]
    C --> V["阀位 / 阀门指令"]
    V --> F["喷水质量流量或有效代理"]
    F --> P["Fan20 plant: 焓值与蓄热状态"]
    P --> Y["主汽温 Tst"]
```

### Task P — plant-level world model（论文主任务）

- 输入动作：实测喷水质量流量；若缺失，则另立经工程核验的 valve-position proxy 分支。
- 输出：主汽温与可观测/可重构物理状态。
- 目标：估计 logged-action 与预先给定扰动场景下的条件动态响应。
- 限制：闭环 DCS 动作由温度/扰动共同驱动。没有安全激励、已知控制器、充分状态、sequential exchangeability 与 support 时，只能声称 conditional dynamics / observational response consistency，不能声称 plant causal intervention effect。
- 分支：flow 分支可审计喷水质量/能量守恒；只有阀位时不得把代理标成 `kg/s`，也不计算或声称喷水质量守恒。

### Task S — supervisory closed-loop response（辅助任务）

- 输入：核实后的**完整 SP level trajectory**；`ΔSP` 只用于事件 onset、方向和 exposure 大小定义。
- 对象：controller + actuator + plant 的闭环组合。
- A1phys 属于这一任务；它不是 Fan20 plant model 的同层竞争者。
- 目标：预测观察到的闭环响应并审计动作敏感性，不把观测事件称作随机因果效应。

如需连接两者，必须显式学习或标定 `controller/actuator: SP → valve/spray`，再串联 Task P；禁止用一个未标注的 end-to-end head 隐式跨越两个 estimand。

## 4. 预注册研究问题与假设

| ID | 研究问题 | 可证伪假设 | 主要判据 |
|---|---|---|---|
| RQ1 | 动作链能否从 DCS 数据中区分？ | 经 tag、单位、时延和方向核验后，Task P 与 Task S 能形成不重叠的 action manifest | Gate 0 schema/action audit |
| RQ2 | Fan20-SST 在 10 s 数据上能否闭合？ | 受约束 Fan20 core 能稳定积分、通过物性表点，并在 validation 上同时满足预测与物理门槛 | H18/H60 MAE、balance residual、solver failures |
| RQ3 | Fan17/21 机制是否有增量？ | 显式金属蓄热、load scheduling 或无双计的 mismatch 只在预声明工况改善对应指标，且跨 seed/fold 同向 | paired episode bootstrap + 消融 |
| RQ4 | 三种动态表示/closure 路线哪条更合适？ | 在相同状态、输入、参数/搜索预算下，至少一条路线形成非支配解 | validation Pareto set |
| RQ5 | 结论是否稳健？ | 候选在 rolling folds、负荷/action 子组和最终冻结批次上不发生工程上不可接受的退化 | worst-group + locked final；无新数据时 internal-final |

以下量必须在查看 Phase 4 模型结果前由工程负责人签字并写入配置：预测非劣界 `delta_pred`、事件曲线非劣界 `delta_event`、组件最小有意义增益 `delta_gain_pred/event`、允许的物理残差、状态边界、solver failure 上限、动作方向、near-zero dose floor 和事件 response deadband。缺任一项，结果只作 exploratory。

## 5. Gate 0 — 协议、数据与代码地基

### 5.1 数据合同

为每一列保存：原始 tag、中文含义、单位、采样周期、测点位置、传感器范围、缺失/冻结率、是否控制量、是否结果的下游量、Fan 变量映射及置信度。必须回答：

- `二级减温调节阀设定` 是温度副回路 SP、阀门 demand，还是其他信号；
- 一/二级喷水是否有质量流量、阀前后压差或阀门特性；
- 主给水、主蒸汽流量和压力测点的准确位置；
- `T3` 是否与 Fan20 对应过热器段同位置；
- 是否存在可作为 `Dst` 的实测主蒸汽流量；
- 启停、干态/湿态、旁路和传感器维护区间如何标注。

每次远端运行绑定 `data_manifest.json`，至少含文件 SHA256、行数、时间范围、列 schema、采样间隔统计和预处理版本。

### 5.2 episode 与缺失处理

- 按停机、启机、时间断点、长缺测和运行方式变化切分连续 episode。
- 窗口不得跨 episode、split 或不可用数据段。
- 禁止 `NaN → 0`。插值上限与方法按列声明，并同时提供 missingness mask；超限片段剔除。
- 所有单位在物性计算前统一为 SI，输出层再转换为工程单位。
- `ut` 不用负荷变化率冒充。SST 子系统可把历史实测 `Dst` 当状态/扰动；未来 `Dst/Ne` 若使用实测值，只能称 conditional scenario forecast。要估计喷水动作的总响应，未来量必须由模型内生传播，或来自在结果之前冻结且独立给定的场景/预测器，不能使用 post-treatment 实测轨迹抬高指标。

### 5.3 split 与 lockbox

优先级如下：

1. 新采集的未来连续时段；
2. 未被开发查看的另一台机组；
3. 若两者都不存在，使用 nested blocked temporal CV，并把论文结论限定为内部验证。

开发期至少采用三个 rolling outer folds。每个 fold 内为 `train → validation` 的时间顺序，边界 purge/embargo 不小于 `W + H`，事件评测还要覆盖完整 pre/post window。不得从 test 读取 loss、early stopping、checkpoint、阈值、校准系数或图例选择信息。

数据产物：

- `split_manifest.json`：episode IDs、边界、purge、用途和 hash；
- `window_manifest.parquet`：每个窗口的 episode、起点、horizon、split；
- `events_plant_dev.json` / `events_supervisory_dev.json`：只包含 development folds 的 onset、动作层级、幅值、方向、pre/post、匹配组和 split；两个 task 不共用 event manifest。

Gate 0 只冻结 final event builder 的代码、配置和 hash，不生成或查看 lockbox event manifest。最终事件由独立 steward 或 `final_evaluate` 在冻结后盲生成。lockbox 的“一次访问”指一次预注册批量访问：同一命令评估所有冻结模型和 seeds，完成前不逐模型回传结果。若访问失败，只能修复与模型无关的运行错误并完整记录；不得据结果调参。

### 5.4 事件 reference 的定位

现有 DiD/CFE 改名为 **matched observational closed-loop event-response reference**。它是闭环观测基准，不是因果 ground truth。

- event onset 仅由 action 定义；匹配只能用处理前变量；
- 匹配至少包含负荷、温度水平/趋势、燃料/给水、主汽流量、压力和运行方式；
- 报告 overlap、标准化差异、pre-trend、placebo onset、control reuse 和未匹配率；
- control 默认不跨事件重复使用；必须复用时用 cluster-aware uncertainty；
- development folds 分别独立构造；final 事件只在冻结后盲生成，事件与 controls 不跨 split；
- metric source、event IDs 或长度不一致时 fail closed，禁止 fallback 到另一个同名 CFI。

### 5.5 Gate 0 放行条件

- `pytest` 可收集且新增协议测试通过；
- schema/action/split 与 task-qualified **development event** manifests 通过验证；final 只验证 event-builder config/hash；
- PID 方向、工作点、导数项测试修复；
- 任一 direct multi-horizon 模型通过未来动作 Jacobian 因果掩码测试；
- train 进程不能导入 test 数组；
- canonical checkpoint 只由 validation 产生；
- 工程阈值、唯一 primary metric/contrast family 与公式 hash 完成预注册；
- 统计每 fold 的独立 episode/time-block 数，使用不含候选模型结果的 pilot 差值方差计算双侧 `alpha=0.05`、power `>=0.80` 下的 MDE。若 MDE 大于工程非劣/优效界，研究只报告 effect + CI / inconclusive，不释放“优于”判决。

Gate 0 未通过：**不释放正式 Linux 多 seed 实验**。

## 6. Gate 1 — Fan20-SST 物理闭合

### 6.1 物理主干

Fan20 是三篇中唯一直接覆盖两级喷水与主汽温焓值链的 central skeleton **候选**；只有通过本 Gate 才冻结为后续物理主干。实现范围：

- 两级喷水混合的质量/能量关系；
- 分段蒸汽焓和对应蓄热状态；
- `Tst = f(pst, hst)` 的物性转换；
- 运行负荷、燃料/给水、`Dst` 等已核实外生量；
- 正值、范围、守恒和可辨识性约束。

使用 IAPWS-IF97 或同等级经过验证的水蒸气物性库。主汽压不能作为给水压力的无条件替代；若缺给水压力，必须给出范围敏感性并降低结论等级。

### 6.2 分层验证

1. **物性单测**：标准表点、单位转换、相区边界、梯度有限性；
2. **独立方程核验**：手算守恒例、原论文表点/公开数据（若可获得）或第二套独立实现；与被测代码共用同一 RHS 的 synthetic generator 只能证明自洽，不能发现转写错误；
3. **合成恢复**：已知参数、无噪/加噪、不同采样率，检查参数与轨迹恢复；
4. **数值收敛**：Euler、RK4、参考高精度 solver 对比，10 s 步长误差；
5. **可辨识性**：profile likelihood/Fisher 或 bootstrap 参数分布；不可辨识参数固定、合并或从论文 claim 删除；
6. **真实数据 smoke**：seed 0 小样本，检查 finite、边界、长 rollout 和梯度；
7. **真实 validation**：plain Fan20 与统计基线对比。

### 6.3 Gate 1 判决

只有同时满足以下条件才进入 Gate 2：

- 物性表点和解析合成测试通过；
- solver failure、NaN/Inf 和非法状态不超过预注册阈值；
- 主要参数不长期贴边，或贴边已被解释并重参数化；
- flow 分支的 mass/energy residual 达标；valve-position proxy 分支不计算/宣称喷水质量守恒，只通过方向、稳定性、状态范围和 effective-dynamics residual；
- H18/H60 validation 预测没有超过 `delta_pred` 的灾难性退化；
- logged-action scenario 的响应方向与工程方向相符，且不存在未来动作泄漏。

若阀位无法转换成可信喷水代理，Fan20 只能标为 **effective gray-box**，不能作严格质量/能量守恒模型。

## 7. Gate 2 — 嵌套物理组件，而非三篇论文赛马

固定同一数值表达，按以下最小增量顺序做消融：

1. `Fan20-core`；
2. `Fan20 + Fan17 explicit-metal-storage`：Fan20 已有 `uB→rB` 制粉迟延/惯性，不重复添加；新增 `Tj` 前必须重推分段能量方程，且关闭组件时严格恢复 Fan20 core；
3. `Fan20 + Fan21 load-scheduling`：负荷相关低维参数；
4. `Fan20 + Fan21 mismatch`：Fan21 `Q1` 是整炉 closure；必须预注册它替换 Fan20 的哪个热项、如何在 `k11/k12/k13` 三段分配、分配权重和总能量恒等式，禁止直接相加造成双计；
5. Fan21 throttle-loss 只进入保留 `Ne/ut/turbine` 的完整 CCS 扩展；`Dst` 外生的 SST Task P 不测试该项；
6. Phase 4 不做组件组合搜索。若多个单组件通过，按预注册 validation 规则冻结一个；组合留给新的 protocol amendment。

组件通过条件：

- 对预声明工况有机制对应的改善，例如慢蓄热改善长 horizon，负荷调度改善跨负荷子组；
- 三 seed 的 episode-paired 差值方向一致；
- 改善超过独立的最小有意义增益 `delta_gain_pred` 或 `delta_gain_event`，同时不越过 `delta_pred/delta_event` 非劣界，且物理 residual 不恶化；
- 新增参数可辨识、不过度贴边，增益不能仅来自 unrestricted residual bypass。

不通过的组件作为负结果保存，不继续带入三路线比较。

## 8. Gate 3 — 三种动态表示 / closure 路线的公平比较

Gate 2 先冻结一个 physical content specification；随后三种表示/closure 路线共享状态定义、输入、输出、观测器、数据、loss 项、参数/超参数预算和 checkpoint 规则。

### Route A — Fan-structured Neural ODE

- Fan 方程是主干；小网络只能修正未建模闭合项或低维系数。
- 正值参数用 softplus/log 参数化，状态范围可微约束。
- 禁止高容量 output residual 直接绕过积分器。

### Route B — Fan-state Controlled Koopman

- 编码状态后显式使用固定算子 `z[k+1] = A z[k] + B u[k]`；负荷/工况作为已声明状态或外生输入，不用内生 `rB` 调度 `A/B`。若未来改为 `A(rho),B(rho)`，必须把 `rho` 定义为已核实外生调度量并改称 LPV/load-scheduled Koopman。
- 约束离散稳定性或连续谱实部，保留复共轭动态所需表达。
- decoder 回到相同物理状态/输出，并计算同一守恒 residual。
- exp_112 的 diagonal nonlinear free-head 只作历史负面对照，不代表本路线。

### Route C — Fan20/21 time-varying gray-box

- 直接积分 Fan 结构，只有少数参数随负荷/工况平滑变化。
- 调度函数维度、单调性/平滑性和范围预注册。
- 不允许 unrestricted neural output residual。

### 共同预算

- 中央开发 fold：每路线 3 seeds；相同 trial 数、optimizer updates/data passes、batch exposure 和早停规则；wall-clock 只作安全上限并作为计算指标报告，不把慢 solver 的提前超时当同等训练预算；
- 另报告一条预注册 compute frontier（固定 wall-clock/能耗预算），与等更新次数的科学主比较分开；
- 参数量同时报告，但不强求完全相等；额外给出训练/推理耗时、峰值显存和 solver 调用数；
- 超参只在 validation 搜索；每 seed 产生一个 canonical checkpoint；
- 先过 physics/event feasibility gates，再在可行集合内按 validation integrated MAE 选择；禁止用 CFI 单标量选模。

## 9. Baselines 与排行榜边界

### Task P 排行榜

- Persistence / seasonal persistence；
- ARX 与 N4SID/state-space；
- 在新 split 上从头重训的 M7/DirectWM；
- plain Fan20 与通过 Gate 2 的嵌套物理版本；
- Route A/B/C。

### Task S 排行榜

- Persistence / ARX closed-loop response；
- 重训 M7；
- A1phys；
- 显式 `controller/actuator + Task P` 串联系统（若 Gate 0 完成）。

两个排行榜可以共享预测单位，但不能把 action estimand 不同的分数放入同一“冠军”列。

## 10. 指标与判决规则

### 10.1 Primary metrics

| 维度 | 指标 | 报告方式 |
|---|---|---|
| 预测 | `ForecastScore`；horizon-wise 与 H18/H60 integrated MAE | 每 fold/seed、episode cluster CI、负荷/action 子组 |
| 事件 | event-response-curve WMAE（ERC-WMAE）、gain bias、方向一致率 | episode-cluster CI，raw units 与 valid-dose normalized 同报 |
| 物理 | mass/energy residual、非法状态率、solver failure、参数贴边率 | 总体 + worst episode |
| 鲁棒 | 时间 fold、负荷区间、动作幅值/方向、启停/稳态组 | worst-group 与整体并列 |

唯一 confirmatory forecast metric 定义为：

```text
MAE_short(e) = mean over frozen origins and k=1..18 of |y_hat - y|
MAE_long(e)  = mean over frozen origins and k=19..60 of |y_hat - y|
ForecastScore(e) = 0.5*MAE_short(e) + 0.5*MAE_long(e)
ForecastScore = equal-weight mean over independent episodes/time blocks
```

同时报告 `MAE_H18 = mean(k=1..18)`、`MAE_H60 = mean(k=1..60)` 和 duration-weighted sensitivity，但不据它们另选模型。若工程负责人要求改变 0.5/0.5 或 horizon，必须在 R1 前写入 `metrics.yaml` 并冻结 hash。

事件曲线不用 `IRF` 一词。对 event `i`：

```text
ERC_WMAE(i) = sum_k w_event[k] * |delta_y_hat[i,k] - delta_y_ref[i,k]|
               / sum_k w_event[k]
```

`w_event`、响应窗口和 baseline 在结果前冻结。gain/dose 归一化只用于 `|dose| >= epsilon_dose` 的事件，`epsilon_dose` 来自传感器分辨率/工程 deadband；near-zero 事件不做除法。方向一致率只在参考曲线 CI 排除 0 或绝对响应超过预注册 `response_deadband` 的时点计算，其余记为 not-informative。Task P/S 分别使用自己的 action units、events 和 metric manifest。

### 10.2 Secondary/diagnostic metrics

- RMSE、最大误差、到峰时间、shape correlation；
- 如概率头完成独立校准：CRPS、coverage、interval width；否则不报告概率结论；
- CFI 仅保留为分解后的诊断面板，不形成单一选模分数；
- 线性、对称、零动作、未来动作 Jacobian 等不变量；
- 参数量、训练时间、推理延迟、显存和数值步数。

### 10.3 Canonical selection

每个 seed 的候选按以下固定顺序筛选：

1. numerical/physics hard gates；
2. validation event catastrophic-failure gate；
3. validation prediction non-inferiority；
4. 可行集合内最小 validation `ForecastScore`；
5. 若在 `delta_pred` 内等价，依次选择物理 residual 更低、结构更简单、计算更低者。

只保存一个 canonical checkpoint。其他 epoch 可保存训练曲线，不得在表格中拼接不同 checkpoint 的最好数字。

## 11. 统计设计

- 重叠窗口不作为独立样本；顶层重采样单位是连续运行 episode，episode 过少时使用预先定义且不重叠的时间 block。
- 模型比较使用相同顶层 cluster 的 paired bootstrap；抽中一个 episode/block 时携带其中全部嵌套窗口和 events，不把 event 当独立顶层样本。
- seed 变异与数据不确定性分开报告，不能把 seeds 当独立现场重复。
- central screening：3 seeds；rolling robustness：3 outer folds，筛到最多 2 条路线；locked final：5 seeds。
- seeds 只刻画优化变异，不提供统计功效。R1 前以独立 episode/block 数和 baseline pilot paired SD 计算 80% power 的 MDE；功效不足时只作 estimation/inconclusive。
- multiplicity family 分开且预注册：Gate 2 的三个“组件 vs Fan20-core”只对 `ForecastScore` 做 3 项 Holm；Gate 3 的 A/B/C 三个 pairwise `ForecastScore` 做 3 项 Holm；final 只做 `winner vs Fan20` 与 `winner vs M7` 两项 Holm。事件/物理指标是 feasibility gates 和 secondary estimates，不另造“显著冠军”。
- 同时报 effect size、95% CI 和工程阈值；CI 跨越非劣/优效边界时结论为 inconclusive，不以均值大小强判。
- 参数、公式、聚类算法、指标权重和阈值必须在远端首轮正式运行前冻结到带 hash 的配置。

## 12. Gate 4 — rolling robustness 与最小消融

Task P 中央 fold 只保留 Pareto 集最多两条路线，加上 M7-plant 与 plain Fan20 进入 rolling folds；Task S 如获准启动，使用独立的 M7-supervisory/ARX-S、A1phys 和 cascade 配置与排行榜。检查：

- 不同时间 fold 的排名方向；
- 低/中/高负荷和负荷变化工况；
- 正/负动作、幅值区间和稀有动作；
- 10 s 下 fast-state aliasing 敏感性；
- 缺失输入代理、给水压力范围、valve-to-flow 标定的敏感性；
- 固定三项路线消融：`physics constraint/loss off`、`observer alternative`、`neural residual capacity zero/selected`。Fan17/21 已在 Gate 2 单独筛选，不重复计入本批。

若候选在任一安全关键子组出现方向反转、非法状态或 solver failure，则不进入 final，即使平均 MAE 最低。

rolling folds 完成后、打开 final 前，用冻结的 feasibility gates → mean outer-fold `ForecastScore` → 物理 residual → 简洁度顺序，从最多两条路线冻结**一个** Task P winner。若没有路线通过，不访问 lockbox，结论为 no-go。Task S 独立冻结自己的候选，不参与该 winner 选择。

## 13. Gate 5 — locked final evaluation

进入 final 前冻结：commit、容器/依赖、数据与 split hash、模型结构、超参、5 seeds、canonical checkpoint 规则、metrics、统计脚本和图表模板。

Task P final 批量比较固定为：

- Phase 4 winner；
- plain Fan20；
- M7/DirectWM；

Task S 如果已单独通过 Gate 4，则另开一份 final manifest，比较 A1phys、最佳 closed-loop statistical baseline 与显式 controller/actuator + Task P cascade；不与 Task P 合表。

若存在新时段/另一机组 lockbox，则一次预注册批量访问同时运行所有冻结模型和 5 seeds，输出所有指标和失败案例。不得因结果难看而更换 seed、过滤 episode、重定义事件或改图。final 不通过则报告不通过，原 lockbox 不再恢复为未见数据。

若没有真正未见的新时段/机组，则本节改名为 **nested blocked internal final evaluation**：执行相同冻结流程，但不得称独立 lockbox、external validation 或 independent test，Phase 4 完成定义也相应限定为内部验证。

## 14. Linux 远端运行矩阵与预算上限

本地 L0 负责单元测试与 CPU synthetic smoke，不计远端 run；Linux R0 在目标环境重复数值 smoke 并记录 CUDA/solver 产物。每批返回审计并冻结下一 Gate 后，才生成下一批 manifest；不得用一个 `matrix_r1` 跨越 Gate 1–3。

### Task P — 固定、可复算的远端预算

| 批次 | Gate / 固定配置 | 算式 | 上限 |
|---|---|---:|---:|
| R0-P | Fan20、Fan17-metal、Fan21-mismatch 数值 smoke，seed 0 | 3 × 1 | 3 smoke |
| R1-P | Gate 1：plain Fan20 与 M7-plant central validation | 2 configs × 3 seeds | 6 |
| R2-P | Gate 2：`+metal`、`+load-scheduling`、`+mismatch`；core 复用 R1-P | 3 configs × 3 seeds | 9 |
| R3-P | Gate 3：冻结同一 physical spec 后 Route A/B/C | 3 configs × 3 seeds | 9 |
| R4-P | 固定三项路线消融：physics、observer、residual capacity | 3 configs × 3 seeds | 9 |
| R5-P | Gate 4：最多 2 routes + Fan20 + M7，2 个额外 folds | 4 configs × 2 folds × 3 seeds | 24 |
| R6-P | Gate 5：winner + Fan20 + M7，一次批量 final | 3 configs × 5 seeds | 15 |
|  | **Task P 正式上限** |  | **72 + 3 smoke** |

Persistence、ARX/N4SID 等确定性 CPU baselines 每 fold 仍须产出 manifest，但不计 GPU seeded-run 上限。R2-P 不做组件组合；任何新增 config 必须在看到下一批结果前提交 protocol amendment、重新计算预算并生成新 hash，不能临时追加。

### Task S — 条件触发、完全独立的可选预算

只有 supervisory tag 与 controller/actuator 链通过 Gate 0，且 Task P 已形成可串联候选时才启动：

| 批次 | 固定配置 | 算式 | 上限 |
|---|---|---:|---:|
| S1 | ARX/M7-supervisory、A1phys、explicit cascade central | 3 × 3 seeds | 9 |
| S2 | 最多 2 个 Task S 候选，2 个额外 folds | 2 × 2 × 3 seeds | 12 |
| S3 | 三个冻结 Task S configs，一次批量 final/internal-final | 3 × 5 seeds | 15 |
|  | **Task S 可选正式上限** |  | **36** |

Task P 与 Task S 总上限为 `108 formal + 3 remote smoke`，但后者是条件分支，不是默认要跑满。每个 Gate 可以停止后续分支；不得在远端边看 final 边追加“最后一个变体”。

## 15. 远端产物合同

每个 run 必须保存在不可覆盖目录：

```text
results/phase4/<task>/<experiment>/<config_hash>/fold_<k>/seed_<s>/
├── manifest.json
├── resolved_config.yaml
├── environment.txt
├── stdout.log
├── stderr.log
├── metrics_validation.json
├── metrics_final.json               # 仅 final 批次；development 不生成
├── lockbox_access_ledger.json        # 仅 final 批次；internal-final 也记录
├── checkpoint.pt
├── checkpoint.sha256
├── predictions.parquet
├── event_predictions.parquet
├── physics_diagnostics.parquet
└── status.json
```

`manifest.json` 至少记录：git commit/dirty diff hash、命令、主机/GPU、Python/CUDA/依赖、data/split/event hash、开始结束时间、退出码和所有输入产物 hash。汇总器只读取完成且 hash 验证通过的 run；缺文件直接标记 failed，不做静默跳过。

## 16. 结论语言约束

| 证据状态 | 允许表述 |
|---|---|
| Gate 0–2 | “实现/数值/机制可行性” |
| central validation | “开发集筛选信号” |
| rolling folds | “内部时序稳健性” |
| 新未来时段 untouched final | “在该机组、该时间范围的独立时间验证” |
| 无新数据的 nested blocked final | “内部时序验证”，不得称 independent/lockbox |
| 新机组/新时段外部数据 | “外部/前瞻验证”，需说明范围 |
| 观测闭环事件 | “matched observational response consistency” |

任何阶段都不能把 `g(x,0)=0` 写成因果识别，把阀位代理写成质量流量守恒，把单机组观测结果写成现场普适效果，或把同构 neural plant 上的 MPC 仿真写成真实控制收益。

## 17. Phase 4 完成定义

Phase 4 只有在以下产物同时存在时才完成：

- 可复核且 task-qualified 的 data/action/split/dev-event manifests，以及冻结的 final event builder；
- 通过测试的 Fan20 物性与动力学实现；
- Fan17/21 嵌套组件的正/负证据；
- 同物理内容、同预算的三种表示/closure 路线比较；
- rolling robustness 与预注册统计报告；
- 有新数据时：一次批量 locked-final 产物与 access ledger；无新数据时：明确标注的 nested blocked internal-final 产物；
- claim–evidence ledger、失败案例和可复现命令；
- Supervisor 签署的 go / revise / stop 判决。

在此之前，项目状态应保持：**主模型未定性；Phase 4 protocol/build stage**。
