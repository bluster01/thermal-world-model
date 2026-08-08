# Phase 3.5 Linux 回传审查（2026-08-09）

> 审查对象：`187d614..61601c8` 的 9 个 Linux 回传 commit  
> 当前 HEAD：`61601c8cfe1c2ab0f85e5613f7eb60ae43d871d0`  
> 审查身份：Supervisor / protocol audit  
> 总结：**工程方向有价值，正式准备工作大体同口径；gain 改善可作为 calibration pilot，但当前不能升级为“独立物理响应验证”或论文定论。**

---

## 1. Material Passport

| 项 | 内容 |
|---|---|
| 材料类型 | Git commit、Python 实验脚本、JSON 结果、PNG 图、Linux 环境与 dry-run 日志、原始 CSV 审计摘要 |
| 研究对象 | A/B 侧主汽温数据；A1phys 阀位 action 路径；Phase 3.5 正式 42-run 协议 |
| 用户指定目标 | 改善 A1phys 的 gain 效果，同时为 Phase 3.5 正式训练准备 A/B 数据与命令矩阵 |
| 可复核范围 | Git 中的代码、结果 JSON、命令矩阵、审计摘要和图；本地单元测试与静态编译 |
| 不可复核范围 | Linux 远端 cache 本体、cache manifest 内容、训练 checkpoint、原始 6.58/6.69 GB CSV、完整训练 stdout/stderr |
| 本次验证级别 | **ANALYZED，部分结构 VERIFIED；不是完整实验复现** |

---

## 2. Supervisor 总评

这批工作应拆成两条线评价，不能混成一个“Phase 3.5 已完成”的结论。

| 工作线 | 客观评价 | 与当前口径是否一致 | 放行状态 |
|---|---|---|---|
| 正式 Phase 3.5：A/B 原始数据审计 | 表头、关键列、时间范围、行数和量程审计完整；主动报告 A 侧 SP=296.1°C、负阀位和低负荷异常 | **基本一致**。但只能说表头/量程契约通过，不能说原始文件自带的单位语义已被验证 | **PASS with reservations** |
| 正式 Phase 3.5：causal LOCF cache | commit 声称 A/B grid 均为 714,087 且已记录 SHA256 | **目标一致，证据未闭合**。仓库没有 `.manifest.json`、具体 SHA256、生成命令和日志 | **INCONCLUSIVE** |
| 正式 Phase 3.5：42-run dry-run | 7 configs × A/B × 3 seeds，共 42 条 train 和 42 条 validation evaluate；未发现 test 命令 | **一致**，遵守 validation-only 和锁 test 口径 | **PASS** |
| `exp_201` gain 改善 | λ=0.2 能把模型内部 180 s 扰动 gain 从约 -1.5 拉到结果文件中的均值 -77.18 m°C/%；说明 gain loss 有优化作用 | **实验目的完全一致**，这是用户明确安排的 calibration study | **工程 pilot 有效** |
| `exp_201` 物理/因果结论 | 当前代码把该 pilot 写成“SP-IV 真值”“结构性保证”“方向与幅度全绿” | **与保守论文口径不一致**。调到目标不等于独立验证目标正确，更不等于完全物理响应 | **论文证据暂不放行** |

一句话评价：**Linux 把“gain 能否被校准”做出了正面工程信号，也守住了正式 42-run 的 test；问题不在做这个实验，而在结果文档把 calibration success 提前写成了 causal/physical validation success。**

---

## 3. Linux 实际完成了什么

### 3.1 正式 Phase 3.5 准备

1. 新增只读 CSV 审计脚本 `experiments/phase3_5/audit_raw_csv.py`。
2. 审计 A/B 两个稀疏 historian CSV：
   - A：70,020,906 行，6.58 GB；
   - B：71,204,795 行，6.69 GB；
   - 时间跨度均为 2025-12-23 至 2026-03-15，约 82.6 天；
   - 12 个关键 tag 均存在。
3. 保存 Linux 运行环境：Python 3.11.15、NumPy 2.3.5、Pandas 3.0.2、PyTorch 2.11.0+cu130、NVIDIA GB10。
4. 生成正式矩阵 dry-run：
   - 42 条训练命令；
   - 42 条 validation 评估命令；
   - 7 个冻结 config、A/B 两侧、seed 0/1/2；
   - 0 条 test/final-evaluate 命令。
5. **尚未运行正式 42 个训练**，仓库不存在 `results/phase3_5/` 真实结果。

### 3.2 用户指定的 gain 改善 pilot

1. 为旧 `exp_201` flow 路径增加 180 s gain loss：目标 `GAIN_TARGET_180=-0.1°C/%`。
2. 扫描 λ=0.1/0.15/0.2/0.5，并对 λ=0.2 跑 3 seeds。
3. 增加 `best_gain` checkpoint、SP-event gain 脚本、分层 gain 图和 K/τ 图。
4. 结果 JSON 中 λ=0.2 三个 seed 的 final gain 为：
   - seed 0：-96.18 m°C/%；
   - seed 1：-50.11 m°C/%；
   - seed 2：-85.26 m°C/%；
   - 均值：**-77.18 m°C/%**。

这个结果足以支持一个有限结论：**在该 A 侧旧协议 pilot 中，显式 gain regularization 能显著改变模型内部 action sensitivity，并把均值推向预设目标量级。**

---

## 4. 必须先修的 P0 问题

### P0-1：flow 模式的 Jacobian/gain 评估发生 split offset 错位

`exp_201_valve_action.py` 中：

- 状态窗口来自 `test_raw[i]`，其中 `i` 是 test split 内的相对索引；
- 但 flow 模式的 `eval_jacobian()` 和 `eval_gain_180()` 读取阀位时使用全局 `raw[i]`，没有加 `n_val_end`；
- 因而测试状态与 action 扰动基线来自两个不同时间段。

同样的问题出现在 `exp_201_gain_diag.py`：状态来自 `test_raw[i]`，阀位分层和扰动来自全局 `raw[i]`。

影响边界：

- `eval_mae()` 通过 `build_valve_action(test_raw, ...)` 取 action，未发现同一错位；
- 但所有 flow Jacobian、gain 幅值和分层图都不能按当前数值作为正确对齐的 test response；
- 负方向可能仍反映 K(x) 的符号及单调输入变换，但必须修复索引后重跑，不能把现有 100% 当正式物理验证。

修复要求：评估期只从同一个 split-qualified array 取得 `x`、阀位基线和 action；加一个测试，令 test offset 非零并核对窗口时间戳/原始行号一致。

### P0-2：“SP-IV 真值”的工具变量假设没有被识别

脚本直接声明：SP 是外生的，且只通过 PID→阀位→喷水→温度影响目标。但当前数据没有证明：

1. **相关性/强度**：缺 first-stage 系数、置信区间和弱工具诊断；
2. **独立性**：运行人员改 SP 可能同时响应负荷、煤水比、温度趋势和运行工况；
3. **排除限制**：SP 变化可能伴随其他控制回路或操作，不一定只经所选阀位通道；
4. **单调性**：代码用 `dv30*dSP<0` 事后筛选“按预期动作”的事件，相当于用 post-treatment compliance 选样本；
5. **估计量**：目前是事件级 `response/dV` 后求均值/中位数，不是预注册的 Wald/2SLS，也没有 denominator floor 或 weak-IV robust interval。

因此 P2 的 79 个 val+test 匹配事件最多是**观测响应 reference**，不是独立 plant truth。它还混合了开发集与旧 test，更不能直接进入训练后再作为独立验证集。

允许的表述是“SP-event-derived observational gain reference”；暂不允许“SP-IV truth”或“plant causal ground truth”。

这里不应把所有非稳态 SP 阶跃一律丢弃，而应预注册为两层事件：

1. **S 层（quasi-steady identification events）**：事件前负荷、主蒸汽压力、主汽温及关键协变量均满足稳定性门槛，且没有同期大幅煤量、给水、其他阀门或控制模式切换。该层用于估计 primary gain/IRF，也是最接近现场阶跃试验的观测样本。
2. **D 层（dynamic-regime events）**：负荷、压力或温度仍在变化。该层不和 S 层合并产生单一“真值”，但可用于检验模型在变负荷工况下的条件响应、方向保持、异质性与失效边界。

S/D 的划分必须只使用事件前窗口和预先冻结的阈值，不能根据事件后的温度响应好坏重新分类。动态事件“难识别”不等于“没有信息”：如果在局部匹配、趋势调整和协变量条件化后仍有足够重叠，它们可以提供 secondary evidence；若重叠或 first stage 不足，则报告 inconclusive。

此外，稳定工况下出现“SP 变、阀门不变”的事件应单列为 **weak/zero first-stage**：它能揭示 PID 死区、手自动状态、限幅、执行器迟滞或 tag 语义问题，但不能进入 `ΔT/ΔV` gain 的分母，也不能被解释成“阀门对温度没有物理作用”。

#### 现有 SP 事件稳态审计（补充复核）

对当前代码和 P2 的 79 个 A 侧事件进一步核对后，确认**现有事件集没有完成稳态/动态工况识别**：

- 旧 `causal_eval.select_events()` 只检查 onset 前后约 200 s 内负荷的**最大相邻 10 s 跳变**，P2 阈值为 5 MW；它不是窗口负荷 range、斜率或方差条件，因此平滑升降负荷可以通过；
- P2 匹配只使用负荷水平和 60 s 主汽温趋势，没有主蒸汽压力条件；主汽温趋势只是匹配变量，也不是稳态硬门槛；
- 新 `src/phase35/events.py::detect_sp_execution_events()` 只依据 SP 阶跃/保持和实际阀位响应划分 executed/no-execution/ambiguous，完全没有负荷、主蒸汽压力和主汽温稳定性判断。

用 P2 `onsets18` 对齐本地 `A侧主汽温全数据03_cleaned_10s.csv` 后，79 个事件在事件前窗口的实际 range 为：

| 事件前窗口 | 指标 | Q25 | 中位数 | Q75 | P90 | 最大值 |
|---|---|---:|---:|---:|---:|---:|
| 60 s | 负荷 / MW | 1.03 | 1.71 | 2.69 | 3.68 | 6.80 |
| 60 s | 主蒸汽压力 / MPa | 0.036 | 0.056 | 0.084 | 0.121 | 0.281 |
| 60 s | 主汽温 / °C | 0.132 | 0.240 | 0.359 | 0.599 | 1.318 |
| 600 s | 负荷 / MW | 4.78 | 7.54 | 16.74 | 23.93 | 56.89 |
| 600 s | 主蒸汽压力 / MPa | 0.266 | 0.441 | 0.639 | 1.098 | 2.697 |
| 600 s | 主汽温 / °C | 1.665 | 2.420 | 3.234 | 4.216 | 5.462 |
| 960 s | 负荷 / MW | 5.68 | 10.08 | 24.02 | 44.04 | 92.61 |
| 960 s | 主蒸汽压力 / MPa | 0.376 | 0.581 | 0.861 | 1.352 | 3.863 |
| 960 s | 主汽温 / °C | 2.396 | 3.162 | 4.537 | 5.706 | 9.552 |

这说明短至 60 s 的事件前片段可能看起来平稳，但在与热惯性和模型历史窗更相称的 600–960 s 口径下，当前 79 个事件明显混合了动态工况。作为敏感性示例（**不是已经冻结的门槛**），在 600 s 内同时要求负荷 range≤5 MW、压力 range≤0.2 MPa、主汽温 range≤1.0°C 时仅 1/79 通过；960 s 下为 0/79。

因此当前 -90~-130 m°C/% reference 不能称为“稳态阶跃 gain”。下一版应先输出全部事件及其事件前 stability features，再由工程门槛冻结 S/D 标签；若严格 S 层样本过少，应如实报告稳态识别不足，而不是放宽到把动态事件继续称为稳态事件。

### P0-3：校准和验证使用同一目标，不能构成独立 gain 证明

训练 loss 明确把 180 s batch-average gain 拉向 -0.1°C/%；随后又以接近该数值说明“物理 gain 验证通过”。这是 calibration target recovery，不是 out-of-sample identification。

λ=0.1/0.15/0.2/0.5、freeze/no-freeze、多个 checkpoint 口径也已经根据同一批反馈迭代。该过程适合开发超参，但必须与最终验证 reference 分开。

正确证据链应是：

1. 只在 training/validation 上选 gain prior、λ 和 checkpoint；
2. 冻结后在 B 侧、后续时间块或一次性 locked test 上验证；
3. 报告未用于 loss 的 event-level IRF/gain、置信区间、pretrend、negative control；
4. 如果没有独立 reference，只能声称“internal sensitivity calibrated”，不能声称“physical gain recovered”。

---

## 5. P1 方法与报告问题

### P1-1：旧 SP-IV v1 的时间单位写错

数据网格为 10 s，但 `t+600` 被文档解释为 600 s；实际是 6,000 s。`t+3` 才是 30 s，`t+30` 是 300 s。v1 的 600 s 结果不能使用。v2 使用现成 `r18/r60` 时仍需明确数组步长与 horizon 定义。

### P1-2：`best_gain` 不是 validation checkpoint

`best_gain` 按当前随机 training mini-batch 的 `loss_gain` 最小值保存；它既不是固定 validation panel，也不是跨 epoch 可比的稳定估计。应建立固定 validation gain probe，或者仍以 validation MAE 选 canonical checkpoint，把 gain 作为门禁而不是 checkpoint 排序分数。

### P1-3：R=50 是工程先验，不是已测喷水流量

固定等百分比变换可以解释“绝对阀位优于 Δ阀位”的一个合理机制，也适合作为候选 action representation；但现场没有可靠喷水流量，R=50 也没有用伊敏实际阀特性标定。因此应称 `equal-percentage valve proxy`，不能称真实 flow。

`fig_gain_diag.png` 的 “model learns the nonlinearity” 也不准确：曲线形状主要由固定 R=50 输入变换给定，文档 §5.2 已承认这是输入变换伪影。

### P1-4：K(x) 物理单位换算不可复核

`exp_201_kt_plot.py` 在循环结束后读取一次 `m.revin._std`，这是最后一个窗口留下的状态，却用于全部 300 个 K 值；采样还覆盖 full raw 而非冻结 split。当前 “K mean -82 m°C/%” 不可作为物理参数估计。

此外 τ 饱和在下界，说明时间常数没有被当前目标可靠识别；这与“完全物理响应”相冲突，应作为负面诊断报告。

### P1-5：MAE “+43%”不是同口径比较

按仓库 JSON 的 `final` 字段复算：

| 组 | seed 0 | seed 1 | seed 2 | 均值 |
|---|---:|---:|---:|---:|
| flow no-gain final MAE | 1.3654 | 1.1394 | 1.3328 | **1.2792°C** |
| flow λ=0.2 final MAE | 1.3588 | 1.4711 | 1.2989 | **1.3763°C** |

在当前各自 final checkpoint 定义下，均值增加约 **7.59%**，不是 43%。但两组 checkpoint 选择规则仍不同（base 为 test-CFI，calibrated 为 training-batch gain），所以这也不是最终公平估计。

同时预设 MAE 门槛 `<1.0°C` 未通过，结果 1.38°C；不能写“三项全绿”。

### P1-6：旧协议仍逐 epoch 访问 test

`exp_201` 每 5 epoch 计算 test MAE/Jacobian，并保存 `best_mae`/`best_cfi`；base final checkpoint 因而是 test-selected。它可以保留为历史 pilot，但不能进入正式 Phase 3.5 的 canonical leaderboard。

---

## 6. 11 项统计谬误扫描

| 检查项 | 本批风险 | 结论 |
|---|---|---|
| Simpson's paradox | A 侧总体均值可能掩盖负荷/开度/时段层差异；现有分层又受 offset bug 影响 | CAUTION |
| Ecological fallacy | 用 batch/window 平均 gain 推到单事件和 plant response | CAUTION |
| Berkson/selection bias | 只保留 SP 后阀位按预期响应事件 | RED FLAG |
| Collider bias | 按 post-SP valve compliance 筛选可能条件化共同结果 | RED FLAG |
| Base-rate neglect | 方向百分比未同时报告各工况覆盖率与异常/停机基数 | CAUTION |
| Regression to the mean | 阶跃/异常事件选择及前趋势外推缺固定 placebo 审计 | CAUTION |
| Survivorship bias | λ=0.5 崩溃、不同 checkpoint/未完成路径未进入统一比较 | CAUTION |
| Look-elsewhere effect | 多 action 表示、ff、λ、checkpoint 和分层反复查看同一反馈 | RED FLAG（若当 confirmatory） |
| Garden of forking paths | R=50、SP-IV、λ=0.2、best_gain 均为迭代后确定 | RED FLAG（若当 confirmatory） |
| Correlation ≠ causation | 闭环观测响应被称为 plant truth/causal separation | RED FLAG |
| Reverse causality | 温度→PID→阀位的闭环反向路径仍存在，表示变换不能自动消除 | RED FLAG |

覆盖率：**11/11**。这些风险不否定 gain regularization 的工程价值，只限定因果和物理结论的证据等级。

---

## 7. 可复现性审计

本地验证结果：

- `python -m pytest tests/phase35 -q`：**25 passed**；
- `python -m compileall -q experiments/phase3_5 src/phase35`：通过；
- dry-run 静态复算：**42 train + 42 validation evaluate + 0 test access**；
- 当前 Git 状态在 pull 后为 clean。

仍缺：

1. A/B `.manifest.json` 的实际内容与 source/cache SHA256；
2. cache 生成的完整命令、stdout/stderr、exit code；
3. Linux 上执行 25 项 Phase3.5 tests 的日志；
4. `exp_201` checkpoint 和训练日志，导致本地只能复算 JSON，不能重放训练；
5. 每次 gain run 的完整环境/config snapshot 与明确 data split manifest。

因此：

- 正式 dry-run 的结构：**VERIFIED**；
- 原始 CSV 审计摘要：**ANALYZED**；
- cache 内容与 freshness：**CANNOT VERIFY**；
- gain 结果算术：**VERIFIED FROM JSON**；
- gain 训练和物理结论：**CANNOT REPRODUCE / NOT VALIDATED**。

---

## 8. Gate 决策与下一步

| Gate | 当前决定 | 放行条件 |
|---|---|---|
| 正式 Task 1：raw audit | PASS with reservations | 冻结 A-SP 异常、负阀位和低负荷的处理规则；文案改为“header/range contract” |
| 正式 Task 2：cache | HOLD | 回传 A/B manifest、source/cache SHA256、staleness 配置、生成命令和日志 |
| 正式 Task 3：dry-run | PASS | Linux 再补 `pytest tests/phase35` 日志即可闭环环境证据 |
| 正式 Task 4：42-run training | 暂缓 | Task 2 证据闭合后启动；仍只评 validation |
| gain-calibration pilot | CONDITIONAL PASS | 修 offset、固定 validation probe、重跑 A/B/3 seeds；保留为 pilot |
| “完全物理响应/真实 gain”论文 claim | FAIL/HOLD | 独立 split、未参与训练的 gain reference、事件级 CI、pretrend/negative control、A/B 一致性 |

推荐执行顺序：

1. **先补 cache manifests**，不需要重做 cache，只需把可审计证据回传。
2. **修复 exp_201 flow 评估 offset**，加入 split-offset 单测；撤下现有图中的 `truth` 和 `learns the nonlinearity`。
3. 把 gain 试验迁入正式 Phase 3.5 口径：只在 validation 调 λ/R/monotone representation，不再逐 epoch 看 test。
4. 对 SP-event reference 先按事件前负荷/主蒸汽压力/主汽温稳定性冻结 S/D 分层，再做 first-stage、balance、pretrend、placebo 和 weak-denominator 审计；S 层用于 primary identification，D 层只作条件响应和适用范围证据。若 IV 假设不能闭合，统一称 observational encouragement response。
5. 再运行正式 42-run；validation 审计后冻结候选，最后一次批量打开 test。

---

## 9. 论文可用与不可用表述

当前可以写：

> In an A-side development pilot, an explicit gain regularizer shifted the model's internal valve-conditioned sensitivity toward a prespecified observational reference range, while retaining negative action-response direction under the tested representation. This result motivates preregistered A/B validation under the locked Phase 3.5 protocol.

当前不能写：

- “SP 阶跃给出了 plant causal truth”；
- “表示变换使混杂与因果完成分离”；
- “方向 100% 是结构性保证”；
- “gain 已恢复到真实物理值”；
- “三项指标全绿”；
- “模型已做到完全物理响应”。

最终 Supervisor 结论：**认可 Linux 完成的 gain 改善探索，它回答了‘模型能不能被校准到目标 gain’；不认可把它提前解释成‘目标 gain 已被因果识别且模型已通过真实物理响应验证’。正式 Phase 3.5 主线没有被污染，补齐 cache 证据并修复评估错位后，可以继续。**
