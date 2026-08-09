# Phase 3.5 Linux 增量整体审计（2026-08-09）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-09
- Verification Status: ANALYZED
- Version Label: phase35_linux_return_audit_v1
- Audited range: `02c3883..b75a9df`
- Local remediation base: `b75a9df`

## 1. Supervisor 判决

本批 Linux 回传有两项可靠进展：参数摘要确认了物理支路的 gain/τ/rate 塌缩；
1 s SP 数据补齐了 A/B 与 3/10/30/60/180/600 s 多时标响应。它们都支持一个保守结论：
当前 A1Phys 只有结构符号约束，尚未复现可定量验证的 plant-level 物理响应。

但本批不能把 E3 升级为“协议合格后的可信 FAIL”，也不能写“B 侧阀门没有响应”。
caliper 后事件几乎只剩开阀，balance 未过；B 侧在 3–30 s 实际存在清楚的反向阀位响应，
只是到 600 s 已被闭环回调、工况变化或其他控制作用覆盖。正式状态维持：

| Gate | A | B | 原因 |
|---|---|---|---|
| E1 | development PASS | development PASS | 仍受负阀位预处理不一致限制，不能解释为物理优越性 |
| E2 | INCONCLUSIVE | INCONCLUSIVE | 非线性映射无稳定增益 |
| E3 | **INCONCLUSIVE** | **INCONCLUSIVE** | A 93 个全为开阀；B 121 开/1 关；SMD=0.302/0.709 |
| E4 | **BLOCKED** | **BLOCKED** | 没有合格 E3 empirical reference |
| E5 | INCONCLUSIVE | INCONCLUSIVE | no-execution 仅 4/2 个 |
| G3 | FAIL | FAIL | τ 接近上界、gain/rate 塌缩 |

没有候选可以补 seed 3/4，也不得打开模型 test。

## 2. 拉取范围和结果谱系

本次 fast-forward 从 `02c3883` 到 `b75a9df`，包含 8 个提交、149 个文件。
主要变更为 matching/caliper、42-run validation 重评估、SP v2 脚本与 A/B JSON、
参数健康摘要及三份 Linux 结论文档。

### 2.1 可复核的正向事实

- A/B 原始 CSV SHA256 与既有 manifest 一致。
- 42 个 run tuple 完整且唯一；训练 manifest 都指向 `61601c8`，`test_accessed=false`。
- 新参数 JSON 含 42 个条目和 42 个不同 checkpoint 短 hash；其中 36 个 action 配置才有已训练物理分支。
- 非 free-only 36 runs：τ1/τ2 均值约 1139 s；gain near-zero 中位数 74.2%；
  6 个 rate 分支的 `|rate_mean|<5e-6`；固定 +5% 阀位 IRF 位于约 `[-0.184, 0] °C`。
- 以上足以判 G3 参数健康 FAIL；不需要把它升级为真实 gain 估计。

### 2.2 provenance 断点

`phase35_sp1s_events_v2*.json` 内记录的生成 SHA 是 `d0dc879`，但
`sp_events_1s_v2.py` 到 `2ea4f23` 才进入 Git。说明这些 JSON 是从未提交的 dirty tree 生成；
单一 `git_sha` 不能重建当时脚本。旧 evaluator 产物也没有保存 caliper、bootstrap、
evaluator SHA、checkpoint SHA 或 cache SHA。

本地已把这些字段写入 evaluator provenance，并令正式 SP/参数脚本默认拒绝未提交的非结果文件
（允许 `results/` 中刚生成的产物）；
旧 JSON 仍只能作为 exploratory artifact，不能因补了一个 SHA 字段就升级为可复现证据。

## 3. E3 matching 审计

### 3.1 caliper 后不是双向 common support

| 侧 | 随机截断后 detected | matched | matched 开/关 | 日块 | max SMD | reuse ratio |
|---|---:|---:|---:|---:|---:|---:|
| A | 1000（296 开/704 关） | 93 | **93/0** | 11 | **0.302** | 1.42 |
| B | 1000（629 开/371 关） | 122 | **121/1** | 12 | **0.709** | 3.35 |

预注册要求每方向至少 10 个事件、`max|SMD|≤0.20`。两侧都未过。
因此方向率 A=0.323、B=0.057 只描述被 caliper 选择后的近单向子样本，不能外推为
“阀位通道没有物理响应”。原 reporting 把样本不足/balance 失败写成 FAIL，又允许 A 的
E4 PASS；本地已改为 E3 INCONCLUSIVE → E4 BLOCKED。

### 3.2 caliper 仍是探索性选择

`q=0.02` 是在 validation 上扫描 `0.01–0.3` 后按 matched 数/SMD 选择的，且定义为
全部 event-control pair 距离的分位数，不是独立预注册的物理 common-support 阈值。
它可以用于探索，但不能称确认性协议。代码仍未限制 control 最大复用次数。

### 3.3 E3 事件本身仍不是 600 s 隔离阶跃

`detect_valve_events()` 只检查动作前 60 s 安静和 60 s dose，未要求 60–600 s 阀位保持。
原始复算显示 A/B 分别有 989/996 个事件在后续 60–600 s 再次动作；quiet controls 却要求
整个 horizon 安静。处理组与对照组 estimand 不对称。下一步必须先定义：

1. 真实 held-step：60 s 成形后阀位在 600 s 内保持，且负荷/压力稳定；或
2. 闭环 trajectory response：承认后续动作，使用时变处理方法，不再称 step IRF。

在二者选定前，不再重跑/解释 E3。

## 4. SP v2 代码和数据审计

### 4.1 已发现的实现错误

1. `--split` 原实现只过滤 DataFrame 的 `n`，payload 的 `events` 仍写未过滤 `rows`。
2. 默认 `split=all` 且没有 test unlock；A/B JSON 均提交了 test outcome。
3. 文档称有 600 s minimum gap，但代码未实现；当前 A/B 事件分别有 36/30 个相邻间隔小于 600 s。
4. 60 s hold 只比较窗口末端，窗口中间可偏离后返回。
5. 1 s grid 先把 `t0` 向下取整到整秒，`n_pre` 并不严格对应真实事件时刻，且 float 运算会损失 ns 精度。
6. 新增 3 个测试复制了索引表达式，没有调用筛选/序列化主路径，因此无法发现 1–5。

本地已经修复 1–5，并新增 split/test-lock/精确事件网格回归测试；同时记录 `sp_max_dev_600` 和
`sp_held_600`，以区分 60 s onset diagnostics 与真正 600 s held step。Linux 需要在新 commit
上只重跑 validation，旧 A/B v2 JSON 不覆盖、不伪装成正式结果。

### 4.2 B 侧“阀门没响应”结论不成立

Linux 文档的 A/B 表使用的是 `dv_600s`，但表头没有声明 horizon。按同一批 train+validation、
`|ΔSP|≤3°C` 事件复算：

| 层 | 侧 | n | SP–阀位异号率 3 s | 10 s | 30 s | 600 s |
|---|---|---:|---:|---:|---:|---:|
| 60 s strict | A | 143 | 63.6% | 86.7% | 88.1% | 63.6% |
| 60 s strict | B | 143 | **84.6%** | **90.2%** | 76.9% | 49.7% |
| 60sV | A | 45 | 57.8% | 88.9% | 91.1% | 73.3% |
| 60sV | B | 63 | **90.5%** | **92.1%** | 71.4% | 50.8% |
| 60sV∩180sV | A | 18 | 61.1% | 77.8% | 83.3% | 77.8% |
| 60sV∩180sV | B | 20 | **95.0%** | **95.0%** | 60.0% | 45.0% |

B 侧不是“不执行”，而是早期执行明显、600 s 净变化回到随机水平。可能原因包括 PID 回调、
SP 再变化、模式切换或其他通道；仅凭这些数据不能区分。可守表述是“B 侧长期净阀位变化弱，
且时标结构与 A 侧不同”，不能写“温度不是通过二级阀实现”。

### 4.3 所谓 60/180 s 交叉验证不是独立验证

两层来自同一批事件，阈值是在查看通过数和方向率后选择，交集又进一步富集方向一致事件。
“83.3% 显著高于”没有独立样本、配对检验或预注册，属于 robustness 描述，不是 cross-validation。

## 5. SP 模型对照审计

`sp_model_contrast.py` 依赖未版本化的 `/tmp/a_split_bounds.json`、只跑 A/s0、没有结果 JSON，
且 45 个 60sV 事件中 train=44、validation=1。它不能作为 validation 物理复现。

文档的“约 70 倍”还混用了模型中位数与经验均值：

| 同口径比较 | 模型 | 经验 | 比值 |
|---|---:|---:|---:|
| mean absolute response | 0.253°C | 3.527°C | 13.9× |
| median absolute response | 0.051°C | 2.707°C | 53.1× |

无论采用哪一口径，模型幅度偏小的诊断仍成立；但 logged-valve model effect 与观测总温变
不是相同 causal estimand，不能把该比值称 plant gain bias。

## 6. canonical summary 与测试状态

Linux 批量改写了 42 个 run JSON，却没有更新 `summary_validation.json/.md`；旧 summary 仍显示
matched=1000 和 A E4 PASS。本地已从 42 个现有产物重新汇总，并实施 fail-closed gate：
E3 两侧 INCONCLUSIVE、E4 BLOCKED、E5 INCONCLUSIVE。

- `pytest tests/phase35 -q`: 36 passed（本地修复后）
- 全仓 `pytest tests -q`: collection 仍失败：旧 eval protocol 缺 `TimeXerWM`，另一个测试导入时读取硬编码 CSV
- 新增对抗覆盖：constant response 的 dose monotonicity 不再错误返回 1.0；
  零方差但均值不同的 matching 不再错误返回 `max|SMD|=0`

因此 G0 只能写“Phase3.5 局部回归通过；全仓回归未闭合”。

## 7. 本地修复清单

本次不修改 checkpoint，不伪造 Linux 结果；只修框架和 canonical reporting：

- average-rank ties 与零方差 SMD fail-closed；
- E3 样本/平衡不足 → INCONCLUSIVE，E4 自动 BLOCKED，E5 小样本 → INCONCLUSIVE；
- checkpoint/cache side 检查，TOUT2 独立 event-id cluster；
- evaluator 保存完整参数与 checkpoint/cache/evaluator provenance；
- matrix 冻结并显式传递 exploratory `caliper_quantile=0.02`；
- SP split 真过滤、validation 默认、test/all 显式解锁、dirty-tree 拒绝、600 s gap 与 hold 字段；
- 参数脚本 CLI 化，free-only 显式标记未训练，补完整 checkpoint/cache/anchor-input hashes；
- 重建 `summary_validation.*` 和唯一 TODO。

## 8. 下一轮 Linux 仅执行

1. checkout 本地审计 commit，确认 clean tree；
2. 运行 `pytest tests/phase35 -q` 与 compileall；
3. 用新 `sp_events_1s_v2.py --split validation` 分别生成 A/B validation JSON，不得使用 all/test；
4. 用新 `param_summary.py --cache-a ... --cache-b ...` 重跑参数 JSON；
5. 回传命令、环境、stdout、Git SHA 和新 JSON；
6. 暂不重训 42 runs，也不再重跑 E3，等待 held-step/trajectory estimand 冻结。

## 9. 统计谬误扫描

覆盖：11/11。

| 类型 | 判定 |
|---|---|
| Simpson's paradox | CAUTION：A/B 和 3–600 s horizon 聚合会遮蔽方向反转 |
| Ecological fallacy | CAUTION：选中事件结果不能外推整台机组/所有模式 |
| Berkson's paradox | RED FLAG：hold、steady、caliper、60sV 共同选择样本 |
| Collider bias | RED FLAG：按 PID 输出/一级阀安静筛选会受温度与控制器共同影响 |
| Base-rate neglect | CAUTION：E5 no-execution 基率仅 4/2；close common support 0/1 |
| Regression to mean | NOTE：SP 阈值事件可能富集极端偏差，尚无独立排除证据 |
| Survivorship bias | CAUTION：旧 funnel 和 matched-only 汇总隐藏大量拒绝事件 |
| Look-elsewhere effect | RED FLAG：多 threshold/horizon/caliper 扫描后引用最好数字 |
| Garden of forking paths | RED FLAG：60/180/V/intersection 与 q=0.02 均为 validation 后选择 |
| Correlation != causation | RED FLAG：闭环观测 response 仍非随机物理因果效应 |
| Reverse causality | RED FLAG：温度偏差驱动 PID 阀位，future valve 也是反馈路径结果 |

## 10. 可写入论文的证据上限

当前只可称：模型具有预设的方向约束，预测性能在开发集上与自由模型接近；但事件 common
support、不受混杂的阀位阶跃 reference 和 plant-level gain 都没有建立，且学习到的物理支路
参数明显塌缩。因此“完全物理响应”“真实物理响应复现”“A/B 外部复现”均不能成立。
