# Phase 3.5 validation 回传审计（2026-08-09）

> 审查对象：Linux commit `4f8d89a`，42 个 development runs 及 validation 产物
>
> 训练代码：`61601c8cfe1c2ab0f85e5613f7eb60ae43d871d0`
>
> 审查结论：**执行完整、test 未打开；预测消融可保留，但当前事件 reference 不可识别，因此没有候选可进入 seed 3/4 或 locked test。**

---

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-09
- Verification Status: ANALYZED
- Version Label: phase35_validation_audit_v1
- Source: `results/phase3_5/`、`src/phase35/`、`experiments/phase3_5/`、A/B 本地 cleaned 10 s 数据代理复核
- Reproducibility Boundary: 可独立复算 JSON 聚合与门禁；仓库没有 cache 本体、cache 内容 SHA、checkpoint 和参数摘要，不能重放训练或审计参数塌缩

---

## 1. Supervisor 决定

| 项目 | 决定 | 含义 |
|---|---|---|
| 42-run 执行完整性 | PASS | 7 configs × A/B × seeds 0/1/2 全部存在 |
| validation-only / test lock | PASS | 42 个 manifest 均为 `test_accessed=false`；无 `access_ledger.json` |
| 预测消融 | 可保留为 development result | A/B 各配置预测差异很小，不能证明 action 表示优越 |
| E3 经验物理响应 | **INCONCLUSIVE** | 匹配严重失衡、control 极端复用、事件未分稳态/动态；不能解释成物理方向失败 |
| E4 模型响应 | **BLOCKED** | E3 reference 无效时，不能独立判 A PASS/B FAIL |
| E5 SP 负对照 | **INCONCLUSIVE** | no-execution 仅 A=4、B=2，小于预注册每组 10 个 |
| G3 参数未塌缩 | **CANNOT VERIFY** | 未回传 gain/τ/map slopes 参数摘要或 checkpoint |
| seed 3/4 与 locked test | **HOLD** | 先修事件协议和门禁逻辑，只重评 validation |

Linux 将 E3 标为 operational `FAIL` 是诚实的质量告警；论文层的科学解释必须是 **“当前数据/匹配协议不足以识别经验响应”**，而不是“阀门对温度没有物理响应”或“物理方向被数据证伪”。

---

## 2. 执行与产物完整性

独立扫描结果：

- 42 个唯一 `(side, config, seed)`；A/B 各 21 个；
- 7 个配置、seed 0/1/2 均完整；
- 所有训练 manifest 的 Git SHA 均为 `61601c8`；
- 所有 checkpoint selector 均声明为 `validation_integrated_mae`；
- 所有事件文件均为 `split=validation`；
- 同侧 event manifest 在全部 config/seed 间字节一致，说明事件集没有按模型结果变化；
- A/B raw source SHA256 在 42 个 manifest 内各自一致；
- 本地 `pytest tests/phase35 -q` 为 **25 passed**，`compileall` 通过；
- 从 42 个 run JSON 独立调用 `summarize()`，数值聚合和门禁与回传一致；保存的 summary JSON 只比本地对象少内嵌 markdown 字段。

仍缺：

1. cache 内容 SHA256 和独立 `.manifest.json`；
2. 42 个 canonical checkpoint 或至少 checkpoint SHA256；
3. gain、τ、monotone knot/slopes、rate gain 的 validation 参数摘要；
4. Linux 上 25 项测试的实际日志；
5. 原始候选事件数（当前只保存随机截断后的 1,000 个）。

任务 5 原要求包含“参数塌缩审计”，现有产物没有参数字段，因此不能称 Task 5 已完整完成。

---

## 3. 预测结果的正确解读

### 3.1 复算表

| Side | 配置 | Integrated MAE mean±SD | 相对 free_only |
|---|---|---:|---:|
| A | free_only | 0.7136±0.0103 | baseline |
| A | delta_no_baseline | 0.7139±0.0118 | +0.044% |
| A | delta_with_baseline | 0.7167±0.0096 | +0.436% |
| A | absolute_identity | 0.7167±0.0096 | +0.436% |
| A | equal_percentage_r50 | 0.7132±0.0096 | -0.053% |
| A | learned monotone | 0.7179±0.0084 | +0.605% |
| A | learned monotone+rate | 0.7170±0.0086 | +0.480% |
| B | free_only | 0.9625±0.0012 | baseline |
| B | delta_no_baseline | 0.9670±0.0030 | +0.469% |
| B | delta_with_baseline | 0.9743±0.0043 | +1.230% |
| B | absolute_identity | 0.9743±0.0043 | +1.230% |
| B | equal_percentage_r50 | 0.9651±0.0026 | +0.275% |
| B | learned monotone | 0.9723±0.0052 | +1.022% |
| B | learned monotone+rate | 0.9725±0.0048 | +1.036% |

### 3.2 可以成立的结论

1. `delta_with_baseline` 和 `absolute_identity` 在两侧数值上近乎完全相同，正对照通过：只要保留 baseline，Δ表示可以重建绝对轨迹。
2. 固定 R=50、learned monotone 和 rate branch 都没有显示稳定的额外预测价值。
3. 预测曲线几乎重合；A 侧最佳相对 free-only 仅改善 0.053%，B 侧 free-only 反而最好。当前不能声称“绝对阀位/非线性 action 提高了预测精度”。
4. 这不否定 action 对 counterfactual response 的价值，但说明 forecast MAE 本身无法证明物理分支被正确识别。

### 3.3 checkpoint seed 口径问题

训练时 `val_anchors` 使用 `10000+seed` 抽取，因此三个 seed 的 checkpoint 不只包含优化随机性，也使用了不同 validation 子样本。最终回传指标又在固定 seed 的共同 4,096 个 validation windows 上重评。

共同重评保证了 leaderboard 可比，但 checkpoint 选择仍混入 validation sampling variation，与“seed 只衡量优化波动、日块 bootstrap 衡量数据不确定性”的预注册口径不完全一致。下一轮应固定 checkpoint-selection validation anchors；若保留现状，需明确 seed SD 同时包含训练和 validation 子抽样波动。

---

## 4. E3 为什么不能用于物理结论

### 4.1 匹配不是有效匹配

| Side | 截断后事件 | open/close | UTC day blocks | max\|SMD\| | 唯一 controls | 最大单 control 复用 |
|---|---:|---:|---:|---:|---:|---:|
| A | 1,000 | 296/704 | 17 | 2.028 | 475 | 415 次 |
| B | 1,000 | 629/371 | 18 | 1.959 | 294 | 650 次 |

`match_quiet_controls()` 总会选最近的 5 个 control，没有 caliper、overlap gate 或最大复用限制。因此只要 controls 数量足够，几乎每个事件都会被记为“matched”，即使距离非常远。A 的 baseline valve SMD=2.028；B 的关键协变量 SMD 同样接近 2，远高于预注册 0.20。

5,000 次 control assignment 在 A/B 仅落到 475/294 个唯一 control；B 最极端的一个 control 被用于 650 个处理事件。当前 day-block bootstrap 没有表达这种共享 control 依赖，CI 不能视为充分校准。

### 4.2 报告的方向率口径混合

summary 的 A=0.394、B=0.477 是 open 与 close 两组方向率的**非加权平均**。事件级 `all_oriented` 方向率实际为：

- A：0.381；
- B：0.389。

B 侧 open/close 数量为 629/371，非加权平均明显抬高了总体方向率。门禁若声明 event-level rate，应直接使用 `all_oriented`；若要等权两方向，必须明确写成 subgroup-balanced rate。

### 4.3 稳态/动态尚未识别

正式检测只要求阀位隔离、60 s dose 和 60 s 首尾负荷差≤10 MW；没有事件前负荷 range/斜率、主蒸汽压力或主汽温稳定门槛。SP executed/no-execution 检测则完全不检查这三个工况变量。

使用本地 A/B cleaned 10 s 数据按正式 cache 时间轴做代理复核，formal event baseline valve 与代理数据的中位绝对差约 0.01%（但 P95 约 1.5%，因此只作定性审计）得到：

| Side/event | 事件前 600 s 负荷 range 中位 | 压力 range 中位 | 主汽温 range 中位 | 示例 moderate 稳态数 |
|---|---:|---:|---:|---:|
| A valve, n=1000 | 6.51 MW | 0.394 MPa | 2.324°C | 16/1000 |
| B valve, n=1000 | 5.14 MW | 0.349 MPa | 2.420°C | 9/1000 |
| A SP, n=146 | 12.09 MW | 0.551 MPa | 2.968°C | 0/146 |
| B SP, n=144 | 12.07 MW | 0.569 MPa | 3.570°C | 0/144 |

示例 moderate 仅用于敏感性检查：600 s 内负荷 range≤5 MW、压力≤0.2 MPa、主汽温≤1.0°C；它不是已冻结工程门槛。结果足以说明当前事件主要混合动态工况，但正式 S/D 数量必须在 Linux causal cache 上用冻结阈值重新生成。

### 4.4 当前经验响应互相矛盾

- A `all_oriented` H60：+0.460°C，95% day-block CI [-0.043, +0.824]；
- B `all_oriented` H60：+0.543°C，95% CI [+0.179, +0.856]；
- 物理预期为阀门开大后温度降低，即方向化后应为负。

这更符合闭环反向因果、动态工况和匹配失败的共同表现，不能据此否定 plant physics。

---

## 5. E4 的 A PASS 目前是伪放行

当前 E4 只要求 model direction≥0.60 且 IRF-WMAE≤1.0°C，没有检查：

1. E3 empirical reference 是否通过 balance/overlap；
2. 是否优于 `free_only` 的零响应基线；
3. 模型响应幅度是否非平凡；
4. gain/τ 是否塌缩到边界。

实际 open/close 平均 IRF-WMAE：

| Side | free_only | absolute_identity | 结果 |
|---|---:|---:|---|
| A | 0.4854°C | 0.4992°C | action 模型略差于零响应 |
| B | 1.6725°C | 1.6821°C | action 模型略差于零响应 |

absolute identity 的 all-oriented H60 模型 effect 均值仅约 A=-0.052°C、B=-0.087°C；方向率较高主要来自 `gain=-softplus(...)` 与 monotone map 的结构约束，而非从有效经验 IRF 中识别出的证据。

因此正确状态是：

- E4 指标已计算；
- A 的数值通过了当前宽松实现阈值；
- **科学门禁应被 E3 阻断，A/B 都记为 INCONCLUSIVE/BLOCKED**。

正式 42-run matrix 也没有纳入 `exp_201` 的 gain-calibration loss；本轮不能回答“gain 校准后是否通过正式 A/B 验证”。在稳态 reference 未建立前，也不应把当前混合事件 gain 用作新训练目标。

---

## 6. E5 应为 INCONCLUSIVE，不是 FAIL

预注册要求 executed/no-execution 每组至少 10 个。实际为：

- A：no-execution=4，executed=134；
- B：no-execution=2，executed=136。

缺少 no-execution 样本属于证据不足。当前代码把 `enough=False` 直接合并到 `FAIL`，与 TODO 的“缺证据不强判”和 G2 的“E3–E5 记 inconclusive”冲突。

此外 executed/no-execution ratio 达到 `1.1e9`/`2.5e4`，是因为 no-execution effect 接近结构性零后使用 `max(denominator,1e-12)`；这个比值没有可解释性。恒定阀位 effect=0 是架构恒等式，可作软件 sanity check，但不能替代真实 SP→阀执行链验证。

---

## 7. 指标实现问题

### 7.1 dose monotonicity 的 ties 处理错误

`evaluation._ranks()` 给相同值按数组顺序分配不同秩，而不是平均秩。`free_only` 的 model effect 全为零，却得到 A=+0.076、B=-0.098 的 dose monotonicity，证明该指标受事件顺序伪相关影响。

E2 的 dose gain 当前不可作为有效证据。应改用 tie-aware Spearman rank，并在 response 方差近零时返回 undefined/inconclusive。

### 7.2 事件截断前总数丢失

`evaluate.py` 在检测后随机截为 1,000 个，再把 `valve_events_detected` 写成 1,000。产物没有保存截断前候选数、抽样比例和拒绝原因，无法审计事件漏斗。应分别记录 detected_raw、passed_stability、sampled_for_matching、matched_within_caliper。

### 7.3 经验指标在三个 seed 中重复计数

同侧 empirical event metrics 对所有 seed 完全相同，但 aggregate JSON 将其记录为 `n=3`、SD=0。它们不是三个独立经验估计。汇总时应把 empirical reference 每侧只保存一次；seed 聚合只用于 model-derived metrics。

---

## 8. 图表审计

- `fig1_forecast_mae` 标题仍写“35/42 runs completed”，且缺 B 侧 learned nonlinear/rate 两柱；是中途预览，不得进入论文。
- `fig3_event_metrics` 的 direction panel 展示 model direction，却没有在图面明确区分结构约束方向和 empirical direction，容易被误读为现场物理响应正确率。
- forecast 图误差条是 seed SD；event 图的模型 seed SD 与经验 day-block CI 属于不同不确定性来源，不能用相同视觉语义混画。

---

## 9. 修正后的 gate 账本

| Gate | Linux summary | Supervisor 复核 | 理由 |
|---|---|---|---|
| E1 action representation | PASS | **正对照 PASS；优越性未证明** | baseline reconstruction 等价成立，但 action 未优于 free-only |
| E2 nonlinear opening | INCONCLUSIVE | **INCONCLUSIVE** | 无预测/IRF/dose 改善；dose metric 还有 ties bug |
| E3 empirical response | FAIL | **OPERATIONAL FAIL / SCIENTIFIC INCONCLUSIVE** | balance、overlap、稳定工况均未闭合 |
| E4 model response A | PASS | **BLOCKED** | reference 无效，且未优于 free-only |
| E4 model response B | FAIL | **BLOCKED** | reference 无效；数值本身也未过阈值 |
| E5 SP negative control | FAIL | **INCONCLUSIVE** | no-execution n=4/2，小于门槛 |
| G3 parameter health | 未报告 | **CANNOT VERIFY** | 缺 gain/τ/map 参数摘要 |

当前**没有任何候选**满足“validation 预测非劣 + 有效 E4 + G3 参数健康”。不得进入 seed 3/4 或 test。

---

## 10. 下一轮只需重做什么

暂时不需要重跑 42 个训练。Linux 应保留 canonical checkpoints，先只修 validation evaluation：

1. 在 causal cache 上计算事件前 600/960 s 的负荷、压力、主汽温 range/斜率，冻结 S/D 标签；S 为 primary，D 为 secondary robustness。
2. 匹配增加主蒸汽压力、稳定性特征、caliper/common-support；不合格事件应真正 unmatched。
3. 输出 control reuse、unique controls、matching weights/ESS，并在 bootstrap 中处理共享 control 依赖。
4. 使用 `all_oriented` event-level direction，修复 tie-aware Spearman。
5. E3 balance/overlap 不过时，将 E4 自动标为 BLOCKED；E5 样本不足标 INCONCLUSIVE。
6. 回传每个 checkpoint 的 gain/τ/map/rate 参数分布和边界命中率，完成 G3。
7. 用现有 checkpoint 重跑 validation event evaluation 和 summary；仍不打开 test。

若事件 reference 修复后有效，再决定是否在正式管线增加 gain calibration。gain target 必须来自 training-only S 层或独立工程先验，validation S 层只评估，不能同一批事件既定标又验收。

---

## 11. Statistical fallacy scan

- Coverage: **11/11 checked**

| Fallacy | Severity | 本批表现 |
|---|---|---|
| Simpson's paradox | CAUTION | open/close 数量不平衡，非加权组均值与总体方向率差异明显，尤其 B=0.477 vs 0.389 |
| Ecological fallacy | CAUTION | 日均 IRF 被推广为统一 plant response，未先建立工况同质性 |
| Berkson/selection bias | CAUTION | 隔离阀事件、quiet controls 和 1,000-event 截断共同形成选择样本 |
| Collider bias | CAUTION | 按 SP 后实际阀位 execution 分组属于 post-treatment 分层，因果解释需限制 |
| Base-rate neglect | CAUTION | 报 subgroup-balanced direction 未同时强调 open/close 基数 |
| Regression to the mean | CAUTION | 阀位突变事件与温度偏差处于闭环极端状态，匹配失败时易混入回归均值 |
| Survivorship bias | NOTE | 42/42 run 完整；但只保存截断后事件且未报告完整事件漏斗 |
| Look-elsewhere effect | NOTE | 正式矩阵已冻结，风险较低；中途 35/42 图不得作最终选择依据 |
| Garden of forking paths | CAUTION | 正式路径较好，但旧 exp_201 pilot 不能反向修改本轮 confirmatory 解释 |
| Correlation ≠ causation | RED FLAG | matched closed-loop events 在 balance/overlap 失败时不能称 causal/physical truth |
| Reverse causality | RED FLAG | 温度/工况→控制器→阀位路径仍主导，经验方向率约 0.38–0.39 |

---

## 12. 最终结论

本次 Linux 执行质量总体合格：完整跑完 42 个 development runs、正确锁住 test、如实输出失败门禁，没有挑选好看的正式结果。

真正的问题在 validation 方法：当前事件没有稳态分层，匹配缺乏 overlap/caliper，controls 极端复用；门禁代码又在 E3 无效时继续允许 E4 A PASS，并把 E5 样本不足写成 FAIL。故这批结果能支持预测消融和数据治理诊断，**不能支持 A1phys 已实现真实物理响应，也不能支持 A/B 物理能力差异。**
