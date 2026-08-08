# Phase 3.5 增量回传审计（2026-08-09）

> 审查范围：`aaadbf1..271fdf9`，即 cache 证据、exp_201 offset 修复、A 侧 1 s SP 事件诊断与 42-run 参数健康摘要。
>
> Supervisor 判决：**叙事降级正确，SP-IV 撤回成立；新增结果强化了“当前闭环数据不能识别 gain”的负结论，但没有完成真正物理响应验证。A 侧事件 lockbox 已被诊断性访问，Phase 3.5 仍不得进入 seed 3/4 或最终 test。**

---

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-09
- Verification Status: ANALYZED
- Version Label: phase35_incremental_review_v1
- Source: commits `aaadbf1`, `aedc6f0`, `2221bf7`, `5b5212a`, `271fdf9`；新增代码与 JSON 产物；既有 Phase 3.5 协议
- Reproducibility Boundary: 可复算仓库 JSON、检查代码和运行本地轻量测试；本地没有 Linux cache/checkpoint，不能重放 42-run 参数抽样或扫描 6.5 GB 原始 CSV

---

## 1. 本轮真正解决了什么

| 项 | 判决 | 说明 |
|---|---|---|
| 原 SP-IV / “gain truth” | **撤回成立** | 全样本 SP→阀位线性关联很弱，严格稳态样本几乎不存在；原参考不能再作为独立真值或校准依据 |
| exp_201 flow split offset | **代码修复通过，仍只属历史 pilot** | `eval_jacobian`/`eval_gain_180` 已使用 `n_val_end+i` 的同源阀位路径；重评估仍全部以负方向为主，但 checkpoint 仍是旧 test/reference-selected 产物 |
| cache 证据 | **CONDITIONAL PASS** | A/B 原始文件 SHA、行数、时间范围、重采样参数和 staleness 已回传；仍缺实际 cache `.npz` 的 SHA256 与机器可读生成代码 SHA |
| G3 参数健康 | **负面诊断成立，摘要未闭合** | action 分支大面积 near-zero gain 且时间常数贴上界，足以维持 E4 BLOCKED；但 τ 单位、rate 分支和跨 map gain 比较仍有错误 |
| “真正物理响应验证” | **未完成** | 新分析仍是动态闭环中的选择性时序关联；没有合格的稳态事件 reference、B 侧复现或独立 lockbox |

这与 Linux 的总体科学口径基本一致：他们没有继续把 SP 事件包装成 IV，也明确把 gain 写为不可识别。需要修正的是若干实现与协议细节，而不是把结论重新拔高。

---

## 2. P0：会改变数字或证据资格的问题

### P0-1：所谓 30 s 阀位响应实际取的是 3 s

`experiments/phase3_5/sp_events_1s.py:109` 在 1 s 网格上计算：

```python
valve[n_pre + 3] - valve[n_pre - 1]
```

但字段名和下游脚本都写成 `valve_dv_30s`。因此：

- “41.6% 事件 30 s 内阀位不动”实际是 **3 s 内** `|Δvalve|≤0.1%`；
- `strong-compliance n=82` 实际按 `|Δvalve@3s|>1%` 且与 SP 反向筛选；
- 该子集的 80.5% 温度方向率也是 **3 s action-selected subset** 的描述，不是 30 s 口径。

必须显式保存 `dv_3s/dv_10s/dv_30s/dv_60s/dv_180s/dv_600s`，用合成时间轴单测逐个核对索引，再重跑所有相关数字。当前 365-event JSON 和 82-event subset 不得继续按 30 s 引用。

### P0-2：A 侧事件脚本绕过 split，事件 lockbox 已被打开

`sp_events_1s.py` 硬编码扫描 A 侧整段原始 CSV，没有 `--split`、split bounds 或 access ledger。按正式 60/20/20 时间边界复算 365 个事件：

| split | 全部 SP 事件 | strong-compliance subset |
|---|---:|---:|
| train | 279 | 68 |
| validation | 32 | 7 |
| test | **54** | **7** |

因此，“42 个训练 run 没有读取 test”仍然为真，但 **A 侧事件方法与叙事已经看过 test 时间块**。后续不能再称 A 侧该末 20% 为一次性独立 lockbox。

处理方式：

1. 将本次完整 A 侧 1 s 分析登记为 exploratory access；
2. A 侧正式验证改用未来时间块或另行冻结的未见区间；
3. B 侧 1 s 事件尚未回传，可在代码与阈值完全冻结后保留为一次性侧向复现，但 A/B 同属一台锅炉，不能包装成独立机组外部验证；
4. 所有事件命令必须要求 `--side --split --input --output`，并写 access ledger。

### P0-3：τ 被标成秒，模型里实际是 10 s 采样步数

`src/phase35/model.py:189` 使用 `alpha=1/tau` 每个离散步更新；Phase 3.5 cache 的一个步长是 10 s。`param_summary.py:87` 却直接打印 `tau=...s`，commit 也将 107–119 写为秒。

当前最自然的解释是 `tau_steps`：

- action 配置的平均 107–119 应报告为约 **1070–1190 s**；
- 两级级联的两个 τ 还应分开报告，不能先混在一起取均值；
- 上界贴合结论仍成立，而且说明多数动力学被推到 600 s 预测窗之外。

若设计本意真是 `tau_seconds`，则递推必须显式使用 `dt=10 s`（例如离散化 `alpha=1-exp(-dt/tau_s)`）并重新训练。先冻结语义，再决定是否重跑；不能继续用错误单位制作物理参数图。

---

## 3. P1：实现完整性与解释边界

### 3.1 参数摘要没有真正报告 rate gain

`param_summary.py:71-72` 从模型输出字典读取 `rate_gain`，但 `A1PhysValveWM.forward()` 没有返回该字段。因此 42 个条目的 `rate_mean` 全部为 `null`，包括 6 个 `absolute_nonlinear_rate` runs。G3 尚未完整通过。

同时应修正：

- 排除 `free_only` 的未训练 physics 参数；其 gain=-0.05、τ=18 只是初始化，不是辨识结果；
- 输出 `tau1/tau2`、rate gain、每个参数的分位数/贴边率、checkpoint SHA、anchor manifest/hash；
- R50、identity、learned map 的 raw gain 单位是“每 effective-opening %”，局部剂量尺度不同，不能以 K 的绝对大小声称“仅 R50 非平凡”；应比较同一真实阀位扰动下的 physical-unit IRF。

现有 JSON 的可靠负面信号是：除 `free_only` 外，A/B 多数 action 配置有约 55%–99% validation windows 的 `|gain|<1e-3`，且大量 τ 贴上界。它支持“参数不可辨识/分支被 free head 绕过”，不支持 R50 gain 被识别。

### 3.2 新 offset 单测没有覆盖被修函数，且不可移植

`tests/phase3_feedforward/test_split_offset.py` 验证 `data_all[n_val_end+i] == test_raw[i]`，但没有调用或 spy `eval_jacobian()` / `eval_gain_180()`。即使以后函数重新写回 `raw[i]`，该测试仍可能通过。

本地复核结果：

- `pytest tests/phase35 -q`：**25 passed**；
- 新 split-offset test：collection 阶段因硬编码 legacy cleaned CSV 路径不存在而失败；
- `compileall`：通过。

应把索引换算抽为纯函数，或对 raw/test arrays 与 `valve_to_flow` 注入合成 fixture，直接断言两个评估函数实际消费的全局行。

### 3.3 “时序优先”不是 plant gain 识别

`sp1s_temporal_ident.py` 没有按温度结果筛选事件，这一点正确；但 `|dv|>1% 且 dv·dsp<0` 是对中介/执行结果的选择。控制器可能同时响应事件前温度趋势、负荷、压力和人工操作，3 s 阀位先于 600 s 温度并不能阻断这些 backdoor paths。

此外：

- 脚本称做 pretrend 检查，实际只有事件前温度 range 分层，没有 slope、placebo、负荷/压力调整或 balance；
- 82/365 只占 22.5%，80.5% 不能外推到全部 SP 事件；
- 该子集 `dT600/dv` 中位约 -1327 m°C/%；作者已正确标为闭环/小分母下不可识别，不应进入 calibration target；
- 365 个事件分布在 70 个 UTC 日，36 对相邻事件间隔不足 600 s；正式 CI 必须做日/运行 episode 顶层聚类并去除重叠响应窗。

### 3.4 事件提取仍缺少工程闭合

- 只处理 A 侧；脚本注释的“B 线”指分析路线 B，不是 B 侧数据，命名很容易误读；
- `t0_ns` 实际存的是 epoch microseconds，因为时间戳先除以 1000；字段单位错误；
- SP 日志更新间隔中位约 61.2 s，但 hold 窗仅 60 s 且要求至少一个后续更新，可能选择性保留更新较快事件；hold 逻辑也只比较窗末值，不能排除中间反向跳变；
- 结果没有 source SHA、side、split、生成 commit、拒绝原因漏斗或 event config hash；
- 当前严格 S 层在 600 s 门槛下仅 1/365，且该事件位于 train；validation/test 都没有严格 S 事件。动态 D 层可以用于描述鲁棒性，但不能替代 primary steady-step identification。

### 3.5 文档状态仍不一致

`docs/PHASE35_DESIGN.md:250` 仍把已经修复/重评的 offset 写成“修复中”；执行顺序仍有多项“修复中”。同时文中“B 侧/未来块独立验证”尚是计划，不是本轮结果。应在下一次回传时只更新事实状态，不把计划写成已完成证据。

---

## 4. 修正后的 gate 账本

| Gate | 本轮复核 | Supervisor 状态 |
|---|---|---|
| G0 代码/测试 | Phase35 25 tests 通过；新增 event horizon 与 offset test 不合格 | **CONDITIONAL** |
| G1 数据证据 | source SHA/staleness 已有；cache artifact SHA/生成 SHA 缺失 | **CONDITIONAL PASS** |
| E1/E2 预测消融 | 旧 validation 结论不变 | 正对照可保留；无 action/nonlinear 优越性 |
| E3 真实响应 | 严格 S validation n=0；动态事件仍混杂 | **INCONCLUSIVE** |
| E4 模型响应 | reference 无效；action 分支参数塌缩 | **BLOCKED** |
| E5 SP 负对照 | 原 n=4/2；新增 strong-compliance 又是选择子集 | **INCONCLUSIVE** |
| G3 参数健康 | gain near-zero、τ 上界；rate 缺失 | **FAIL / INCOMPLETE** |
| G4 seed 3/4 | 无候选 | **HOLD** |
| G5 final test | A event test 已污染；B event test 尚可冻结后一次使用 | **HOLD** |

`exp_201` offset 重评不改变这个账本。其 10 个历史 checkpoint 中 9 个 Jacobian 方向率为 100%，1 个为 95%；这证明符号约束/pilot 信号在修复后仍存在，不是独立物理验证。

---

## 5. 下一轮 Linux 只做七件事

1. 修复并单测 `3/10/30/60/180/600 s` event horizon；事件脚本参数化 side/split/path，写 provenance 和 access ledger。
2. 明确 `tau_steps` 还是 `tau_seconds`；按两个 stage、真实秒和边界分别回传，补齐 rate gain。
3. 在 **A train/validation** 上重做 SP→command→valve first-stage panel；A test 只登记为 exploratory，不再用于门禁。
4. 用冻结阈值分别生成 S/D；S primary 要同时满足负荷、主蒸汽压力、主汽温的 range + slope，D 只作 secondary。若 S validation 仍不足，直接报告 insufficient evidence。
5. 修正式 E3 matching：common support/caliper、control reuse/ESS、重叠事件去除、日/episode cluster CI；不过门时 E4 自动 BLOCKED。
6. 参数摘要排除 free-only 未训练分支，并以固定真实阀位 perturbation 报告 action IRF，禁止跨 opening map 直接比较 raw K。
7. 代码/指标完全冻结后才运行 B 侧 1 s validation；是否一次打开 B test 由 Supervisor 另行放行。暂不重跑 42 个模型、暂不补 seed 3/4。

真正可发表的“完全物理响应”仍需新的准稳态现场阶跃、可靠喷水流量/阀特性标定，或一个此前未见的未来时间块。仅靠当前动态闭环历史数据，最多能写“结构上保证正确符号、但 plant-level gain 在观测数据中不可辨识”。

---

## 6. Statistical fallacy scan

- Coverage: **11/11 checked**

| Fallacy | Severity | 本批表现 |
|---|---|---|
| Simpson's paradox | CAUTION | 全样本、3 s compliance、温度 range 分层的方向率不同，不能只报 80.5% |
| Ecological fallacy | CAUTION | 总体 R²/方向率不能外推为每个负荷与阀位工况的 plant response |
| Berkson/selection bias | RED FLAG | SP hold、完整窗口与 strong-compliance 共同筛选出 82/365 选择样本 |
| Collider bias | RED FLAG | 按 SP 后阀位执行情况分层，可能在控制器动作这个中介上开启混杂路径 |
| Base-rate neglect | CAUTION | 80.5% 未同时突出 subset 只占 22.5%，严格 S 仅 1 个 |
| Regression to the mean | CAUTION | 控制器通常在温度/工况偏离时动作，600 s 变化可能含自然回归 |
| Survivorship bias | CAUTION | 要求后续 SP 更新、完整 960/600 s 窗和有限 tag，未报告完整拒绝漏斗 |
| Look-elsewhere effect | CAUTION | 3/30/600 s 多窗口与多个稳定阈值需要预注册 family/secondary 口径 |
| Garden of forking paths | CAUTION | 在看过完整 A 数据后继续改 horizon/阈值，A test 已不能恢复为盲测 |
| Correlation ≠ causation | RED FLAG | `dT600/dv` 是闭环选择性比值，不是流量→温度因果增益 |
| Reverse causality | RED FLAG | 温度趋势/工况→控制器→阀位路径仍未被设计阻断 |

---

## 7. 最终结论

Linux 的科学态度与 Supervisor 口径总体一致：他们接受了原审计，撤回 SP-IV 和“完全物理响应”，并如实暴露参数塌缩。这一点应保留。

但实现上新增了三个必须先修的硬问题：3 s 被标成 30 s、A 全时段绕过 split、τ 步数被标成秒；参数摘要还漏掉 rate gain。故本轮是一次有价值的**负面辨识诊断**，不是 Phase 3.5 核心验证完成，也不能据此恢复 gain calibration 或打开 final test。
