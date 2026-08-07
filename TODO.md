# TODO — Phase 4 多线性实验看板

> 更新：2026-08-07。本文是项目根目录下唯一的活任务队列。详细论证见 [Supervisor Review](docs/SUPERVISOR_REVIEW_2026-08-07.md)，完整协议见 [Phase 4 实验计划](docs/PHASE4_EXPERIMENT_PLAN.md)。

## 一句话目标

在不预设 Neural ODE、Koopman 或时变灰箱胜出的前提下，先验证 **Fan20-SST 是否能成为可信物理骨架**，再比较三种动态表示，最后用时序外推检验能否形成主气温火电世界模型。

当前判决：**只进入本地 L0 / Gate 0；暂不释放 Linux 正式训练。**

## 真正的起手顺序

现在先只启动三件事，其他表格均是后续路线图：

| 顺序 | 现在做什么 | 谁做 | 做完得到什么 |
|---:|---|---|---|
| 1 | 核实关键 DCS tag，尤其“二级减温调节阀设定”到底是哪一层信号 | 项目方 / 现场 | 动作链和 Task P/S 不再含糊 |
| 2 | 修复测试收集、test 选模、未来动作泄漏与结果 manifest | 本地 / Codex | 一套不会泄漏、可复现的实验壳 |
| 3 | 让 Linux 只生成 schema、episode 和 split 审计产物，不训练模型 | Linux 数据侧 | Gate 0 可以正式判决 |

三件事审计通过后，第四件事才是实现并验证最小 `plain Fan20-SST`。此时仍不同时开发三条模型路线。

## 怎么看这张表

- 六条工作线可以并行，但每条线只能从左到右推进。
- `▶` 是现在应做的任务，`□` 是后续任务，`×` 是尚未满足前置条件。
- 本地负责设计、代码、测试、smoke 和审计；Linux 只运行已经冻结的 manifest。
- 任一 Gate 不通过，可以停止后续实验，不要求把预算跑满。

状态统一使用：

```text
designed → implemented → smoke_passed → ready_for_remote
         → remote_running → results_returned → audited → concluded
```

## 总览：六条并行、内部线性的工作线

| 工作线 | 要回答的问题 | 线性步骤 | 当前停点 | 最终产物 |
|---|---|---|---|---|
| A. 数据与研究对象 | 我们到底在预测什么、动作是什么？ | A0 tag → A1 schema → A2 episode → A3 split → A4 Task P/S events | `▶ A0` | 可审计的数据、动作和 split manifests |
| B. 物理骨架 | Fan20-SST 方程在本项目数据上是否闭合？ | B0 变量映射 → B1 物性 → B2 方程 → B3 synthetic → B4 real validation | `▶ B0` | 通过 Gate 1 的 plain Fan20-SST |
| C. 模型实验 | 哪种物理内容、哪种动态表示有效？ | C0 baselines → C1 单组件 → C2 冻结物理内容 → C3 三路线 → C4 消融 | `× 等 A/B/D/E` | 公平比较后的最多两条候选路线 |
| D. 评测与统计 | 分数是否可比较、结论是否有统计含义？ | D0 指标 → D1 checkpoint → D2 event reference → D3 CI/MDE → D4 final rule | `▶ D0` | 冻结的 metrics/statistics protocol |
| E. 代码与远端 | 实验是否能复现、是否会泄漏或覆盖结果？ | E0 tests → E1 manifests → E2 local smoke → E3 remote smoke → E4 batches | `▶ E0` | 可复现代码和不可覆盖的远端产物 |
| F. 论文与证据 | 哪些话现在能说、哪些必须等实验？ | F0 claim ledger → F1 方法冻结 → F2 负结果 → F3 figures → F4 manuscript | `▶ F0` | claim–evidence 对齐的论文材料 |

## 当前唯一工作包：L0 / Gate 0

这一包只解决“实验能不能开始”，不训练候选冠军。完成以下 8 项后，再做一次 Supervisor 放行审查。

| ID | 任务 | 负责人 / 场地 | 完成标准 | 状态 |
|---|---|---|---|---|
| A0 | 核实 `二级减温调节阀设定`、阀位、喷水流量、主汽温等 DCS tag 的物理含义、单位和时序 | 项目方 + 现场/DCS | 形成签字版 `action_schema`；明确 SP、阀指令、阀位、流量的层级 | ▶ |
| A1 | 生成 40 列数据 schema 和 SHA256 | Linux 数据侧；本地审计 | 每列有单位、采样率、缺失率、范围、tag 类型、Fan 映射 | ▶ |
| A2 | 按连续运行段切 episode，审计时间戳、缺测、冻结和启停 | Linux 数据侧；本地审计 | 不再 NaN→0；窗口不跨 episode | □ |
| A3 | 生成带 purge/embargo 的 temporal split | 本地代码；Linux 数据 | `split_manifest.json` 可复算，训练代码不可访问 final | □ |
| D0 | 冻结唯一 ForecastScore、事件曲线误差、物理门禁和工程阈值 | 本地 / Codex | `metrics.yaml`、公式 hash、Task P/S 口径齐全 | ▶ |
| D1 | 改为 validation-only canonical checkpoint | 本地 / Codex | test 不参与逐 epoch 选模；缺事件 reference 时 fail closed | ▶ |
| E0 | 修复 pytest 收集、PID、未来动作泄漏、split 和 RevIN 测试 | 本地 / Codex | `pytest` 全部可收集且协议测试通过 | ▶ |
| E1 | 建立不可覆盖目录、run manifest、环境/数据/checkpoint hash | 本地 / Codex | 任一 smoke 可从 manifest 单命令复现 | □ |

### Gate 0 放行条件

| 必须同时满足 | 不满足时怎么做 |
|---|---|
| Task P 与 Task S 的动作层级已分开 | 不跑动作响应实验，只保留普通预测 |
| schema、episode、split 和 development events 可审计 | 返回数据治理，不训练模型 |
| test 不参与 checkpoint、超参或事件阈值选择 | 修复协议后重新 smoke |
| 测试通过，未来动作不能影响过去输出 | 不释放远端 |
| 独立 episode/block 数、paired SD、80% power MDE 已计算 | 功效不足则只报告 effect + CI / inconclusive |

## A 线 — 数据与研究对象

| ID | 顺序任务 | 关键产物 | 放行到下一步的条件 |
|---|---|---|---|
| A0 | 核实动作和测点语义 | `action_schema.yaml` | `u_supervisory → controller/actuator → valve → spray → Tst` 层级明确 |
| A1 | 建立列级 schema | `data_schema.json` | 单位、范围、缺失、时间语义和数据 hash 齐全 |
| A2 | 切连续运行 episode | `episode_manifest.json` | 异常时间段有标记，窗口不跨段 |
| A3 | temporal split + purge | `split_manifest.json` | purge 至少覆盖 `W+H`，无窗口/事件交叉 |
| A4 | 分别构造 Task P/S development events | `events_plant_dev.json`、`events_supervisory_dev.json` | 只用处理前变量匹配；final events 未生成、不可见 |

任务定义固定为：

- **Task P**：logged spray/valve 条件下的 plant dynamics；主榜。
- **Task S**：完整 SP trajectory 的闭环响应；只有 controller/actuator 链核实后才启动。
- `ΔSP` 只用于定义事件 onset/exposure，不直接等于 Fan20 的喷水质量流量。

## B 线 — Fan20-SST 物理骨架

| ID | 顺序任务 | 实验 | 通过标准 |
|---|---|---|---|
| B0 | 核实 Fan20 变量映射 | `rB/h3/Dsw1/Dsw2/Dst/pst/hst/Tst` 对应表 | 不确定代理单独标记，不把阀位写成 kg/s |
| B1 | 验证水蒸气物性 | IAPWS 标准表点、单位、相区、梯度测试 | 标准点误差与非法状态率低于冻结阈值 |
| B2 | 实现 plain Fan20-SST | 两级喷水、分段焓值、状态和输出方程 | 质量/能量 residual 可审计，无高容量 bypass |
| B3 | synthetic 与 solver 审计 | 手算例、独立实现、参数恢复、Euler/RK4/高精度对照 | 能发现转写错误；10 s 数值误差可接受 |
| B4 | 真实数据 validation | plain Fan20 vs M7-plant，3 seeds | 无灾难性 H18/H60 退化，状态/方向/solver 门禁通过 |

若 B4 失败：停止 Fan20 主干路线，先回查变量映射、代理和结构，不直接进入三路线赛马。

## C 线 — 模型实验

### C1：先筛物理组件

固定 plain Fan20 表达，每次只增加一个组件，不做组合搜索。

| 配置 | 要验证的机制 | 保留条件 |
|---|---|---|
| `Fan20 + Fan17-metal` | 显式金属蓄热是否改善慢时标 | 长 horizon 改善、物理 residual 不恶化、关闭后严格恢复 core |
| `Fan20 + Fan21-scheduling` | 低维负荷调度是否改善宽负荷区间 | 跨负荷子组改善、参数可辨识、无 residual bypass |
| `Fan20 + Fan21-mismatch` | 整炉能量 mismatch 是否必要 | 预先定义替换/分配规则，不与 `k11/k12/k13` 双计 |

Fan20 已含制粉动态，不重复添加 Fan17 制粉迟延；Fan21 throttle-loss 暂不进入 Task P。

### C2：再比三种动态表示

先从 C1 冻结同一个 physical content specification，再公平比较：

| 路线 | 最小定义 | 禁止项 |
|---|---|---|
| Route A | Fan-structured Neural ODE；小网络只修正闭合项/低维系数 | unrestricted output residual |
| Route B | fixed-operator controlled Koopman：`z+=Az+Bu` | 复用 exp_112 free-head；内生变量偷偷调度算子 |
| Route C | Fan time-varying gray-box；少量参数平滑随外生工况变化 | 高容量网络绕过积分器 |

共同 baseline：Persistence、ARX/N4SID、重训 M7-plant、plain Fan20。A1phys 只进入 Task S，不与 Task P 混榜。

## D 线 — 评测与统计

| ID | 顺序任务 | 冻结内容 | 主要红线 |
|---|---|---|---|
| D0 | 主预测指标 | `ForecastScore = 0.5×MAE(k1..18)+0.5×MAE(k19..60)`，episode/block 等权 | 不按结果改 horizon/权重 |
| D1 | checkpoint | 每 seed 一个 validation canonical checkpoint | 不在 test 上逐 epoch 取最好值 |
| D2 | 事件评测 | matched observational response reference、ERC-WMAE、dose floor、deadband | 不称 causal ground truth；不允许 CFI fallback |
| D3 | 不确定性 | paired episode/block cluster bootstrap；seed 与数据变异分报 | 不把重叠窗口、事件或 seeds 当现场独立重复 |
| D4 | 判决 | 非劣界、最小有意义增益、MDE、Holm family、失败门禁 | 单项分数不能直接产生“冠军” |

CFI 只保留分解诊断，不选 checkpoint、不跨历史实验比较。

## E 线 — 代码、smoke 与 Linux 批次

### 本地与远端分工

| 场地 | 做什么 | 不做什么 |
|---|---|---|
| 本地 / Codex | 设计、实现、测试、CPU synthetic smoke、manifest、结果复算与审计 | 不把本地 smoke 当正式实验 |
| Linux | 固定 commit/data/config 上跑真实数据或 GPU 批次，原样回传产物 | 不临时改配置、补跑“更好看的 seed”或自行下结论 |

### 远端批次顺序

| 批次 | 固定问题 | GPU seeded-run 上限 | 前置条件 |
|---|---|---:|---|
| R0-P | Fan20、Fan17-metal、Fan21-mismatch 数值 smoke | 3 smoke | Gate 0 + 本地 L0 smoke |
| R1-P | plain Fan20 vs M7-plant | 6 | R0 审计通过 |
| R2-P | 三个单独物理组件 | 9 | Gate 1 通过 |
| R3-P | Route A/B/C | 9 | 冻结同一 physical spec |
| R4-P | physics、observer、residual capacity 三项消融 | 9 | R3 候选已审计 |
| R5-P | 最多两条路线 + Fan20 + M7，两个额外 folds | 24 | 中央 validation 完成 |
| R6-P | winner + Fan20 + M7，5 seeds 单次 final 批量 | 15 | 唯一 winner 和完整协议冻结 |

Task P 正式上限为 `72 + 3 smoke`，不是默认必须跑满。Task S 是独立条件分支，只有 A0 和 controller/actuator 链核实后另建 manifest。

## F 线 — 论文与证据

| ID | 顺序任务 | 可以写什么 | 仍不能写什么 |
|---|---|---|---|
| F0 | 维护 claim–evidence ledger | 已审计的历史事实、明确的负面预实验信号 | “最终模型”“独立测试已通过” |
| F1 | Gate 0–2 | 实现、数值和机制可行性 | 因果效果、路线胜出 |
| F2 | central validation | 开发集筛选信号和负结果 | 泛化结论 |
| F3 | rolling folds | 内部时序稳健性 | 外部验证 |
| F4 | final | 有真新时段时写独立时间验证；否则写内部时序验证 | 无新数据时称 lockbox/external validation |

当前统一口径：Fan20 是 central skeleton **候选**；Fan17/21 是嵌套组件来源；Neural ODE、Controlled Koopman 和时变灰箱都是待验证路线，不预设赢家。

## 每轮只看这张交接表

| 轮次 | 本地先完成 | Linux 再执行 | 返回后 Supervisor 只做一个判决 |
|---|---|---|---|
| 0 | A0–A4、D0–D4、E0–E2 | R0-P | 数据/协议/数值地基是否可用？ |
| 1 | B0–B3 | R1-P | Fan20 是否可作为物理主干？ |
| 2 | 写死三个单组件配置 | R2-P | 哪一个物理组件保留？ |
| 3 | 冻结 physical spec 和公平预算 | R3-P + R4-P | 哪些表示路线进入 rolling？ |
| 4 | 冻结 folds 与统计脚本 | R5-P | 是否有唯一 winner 可进 final？ |
| 5 | 冻结 commit、5 seeds、图表模板 | R6-P 或 internal-final | go / revise / stop |

## 已完成

- [x] README、项目状态和文档地图按保守口径重整。
- [x] 完成代码、历史结果、Fan2017/2020/2021 和方法论总审。
- [x] 将 exp_112 降级为探索性证据，识别 test-selection 与 CFI fallback 问题。
- [x] 区分 Task P / Task S、本地开发 / Linux 正式运行。
- [x] 将 Phase 4 收敛为六条并行、各自线性推进的实验工作线。
