# 最终世界模型结果可信度审计（2026-08-27）

> 审计基线：`76cf7d7bb824c36577214f10572334134fbd548f`（`main == origin/main`）
> 判决：**AUDITED / HOLD FOR CREDIBILITY REPAIR / HISTORICAL VERDICTS SUPERSEDED PENDING REISSUE / VALIDATION ONLY / TEST LOCKED / NO LINUX AUTHORIZATION**

## 1. 审计目标与边界

本审计只回答一个问题：当前代码和产物是否足以支撑可复算、可追溯、适合论文引用的世界模型结论。

处理原则如下：

1. 只有会改变判决、引入未来信息或时序错位、破坏数据/权重/结果绑定的缺陷才进入修复主线；
2. 已有结果不删除，统一降级为历史探索证据，等待 corrected canonical v2.1 上的正式重发；
3. 不因代码风格、仓库体积、依赖管理或假设性风险改动模型；
4. 不访问 test，不把 validation 或 synthetic 结果升级为现场因果、闭环可用或论文最终结论；
5. 本轮只做源码、协议、元数据和已回传产物审计，没有启动训练或改动实现。

审计对象包括：

- `src/final_wm/`、`experiments/final_wm/`、`tests/final_wm/`；
- 冻结矩阵及 v0.3–v0.6 修正记录；
- `artifacts/final_wm*`、`results/final_wm/` 与实验状态注册表；
- corrected canonical v2.1 的构建说明、元数据和现有单种子探针结果。

## 2. 总判决

当前实现可以继续作为**研究开发原型**，但当前 `SUPPORTED / REJECTED / PASS` 不能作为完整冻结协议的正式判决直接写入论文。原因不是模型表现好坏，而是以下四条证据链尚未闭合：

- 冻结协议与实际判决代码不一致；
- 泄漏、反事实和 D-SYN 的关键探针存在可改变结论的逻辑缺陷；
- checkpoint 选择和 NLL 判决统计口径不足以支持现有百分比阈值；
- corrected v2.1 数据、权重、配置和汇总没有形成唯一、内容寻址的正式证据包。

因此：

- v0.2–v0.6 既有结果保留为 `HISTORICAL / EXPLORATORY`；
- corrected v2.1 上最近的 H18、锚定和 LPV 数字保留为 `SINGLE-SEED PROBE`；
- 在修复、完整 3 seeds 重跑和独立复算前，不发布新的正式路线冠军或反事实证书；
- 当前状态为 `active=true`、`ready_for_linux=false`、`results_returned=true`、`audited=true`，Linux 授权为空。

## 3. 必须修复的问题

### C1. 判决器没有执行冻结协议的全部门禁

**证据**

- O1 冻结规则要求 H6/H18 NLL 与相邻窗口 state continuity；实际判决只计算 H18 NLL：
  `docs/plans/2026-08-18-final-wm-discrimination-matrix.md:68-72`，
  `experiments/final_wm/run_matrix.py:510-523`。
- B1 要求 7 通道 H6/H18/H36 NLL/CRPS 和 forecast 相对 oracle 的下游退化；实际只用聚合 H18 CRPS：
  矩阵 `:77-80`，runner `:549-570`。
- T1 要求 60 步有界性/收敛性；J1 要求 H36 稳定性；实际正式判决均只用 H18 NLL：
  矩阵 `:88-94`、`:116-118`，runner `:526-546`、`:573-614`。
- R1 v0.3 要求 H18/H60 的均值方向、UTC-day block bootstrap CI 和正确方向占比 ≥0.60；runner 仍按默认 valve2、单档探针、无 CI，并以 100% 正确方向为门：
  矩阵 `:170-174`，runner `:631-684`。
- `experiments/final_wm/matrix_spec.py:16` 仍声明 `MATRIX_VERSION = "0.2"`。

**影响**

当前汇总文件中的状态不是冻结协议的完整执行结果。缺失门禁可能把本应为 `INCOMPLETE/MIXED/REJECTED` 的单元写成 `SUPPORTED`，属于正式判决阻断项。

**审计处置**

必须修。新版本不得在任一必需证据缺失时产生方向性判决；缺证据只能写 `INCOMPLETE`。

### C2. 泄漏探针和 CF replay 存在时序错位

**证据**

- `src/final_wm/diagnostics.py:67-74` 用 future step 0 推进一次状态，却直接用 future boundary/action step 1 生成预测并对齐 `future_obs[:, 1]`，少了一次 transition；因此残差、特征和动作不在同一时刻。
- `src/final_wm/evaluation.py:334-339` 用 history 第一个 boundary/action 与 history 最后一个 observation 构造初态，再把整个 history 重放，初态锚点自身不共时。

**影响**

现有“leakage clean”和 CF-1 delta trajectory 指标没有测到它们声明的目标，相关证书不能引用。

**审计处置**

必须修。先用人工可计算的一步/两步 teacher 轨迹锁定 `state_t -> action_t,boundary_t -> observation_t` 语义，再恢复探针。

### C3. 反事实支持域门禁没有按样本生效，正式探针还绕过门禁

**证据**

- `src/final_wm/contracts.py:384-393` 把 batch 和 history time 全部压平为一个全局 min/max box，一个样本可借用另一个样本的动作范围。
- `ActionSupport.contains()` 在 `src/final_wm/contracts.py:372-381` 把边界张量固定建在 CPU，CUDA action 会发生设备不一致。
- `src/final_wm/evaluation.py:217-251` 和 `:295-379` 直接调用 transition/integrate，没有走 `FinalWorldModel.counterfactual()`，也不报告 `in_support`。
- `tests/final_wm/test_model.py:96-130` 复现的是当前全 batch 支持盒，未覆盖逐样本隔离。

**影响**

当前 CF probe 可能把无支持的动作替换计入正式指标，反事实证书的适用域不可审计。

**审计处置**

必须修，但不引入密度模型或复杂 OOD 检测。最小方案是逐样本 history min/max、同 device 比较、所有正式 CF 路径统一走 `counterfactual()`，并把支持率与外推率写入结果。

### C4. D-SYN teacher 的计划扰动实际没有发生

**证据**

`experiments/final_wm/run_matrix.py:167-169` 只扰动名字以 `raw_` 开头的参数；transition 的 `ParameterDict` 参数名为 `raw.<key>`（`src/final_wm/transition.py:137-143`）。循环命中数为零。

**影响**

D-SYN PASS 没有证明 student 能从偏离默认先验的同型 teacher 恢复，只证明了更接近默认骨架的任务可拟合。该门禁当前失去预注册意义。

**审计处置**

必须修。除修正匹配条件外，结果中必须记录扰动参数数、扰动前后距离和非零断言，避免再次静默 no-op。

### C5. canonical v1 的一级阀错侧已确认，但 corrected v2.1 尚未形成正式重发证据包

**证据**

- `results/final_wm/known_defect_v1_valve1_20260826.md` 已确认 v1 一级阀双侧错配；一级阀 A/B 相关约 0.8，只能解释为什么方向性结论可能定性存活，不能恢复幅度与 CF 定量可信度。
- `docs/plans/2026-08-26-v06-training-protocol-amendment.md` 记录 v2.1 已按“一级同侧、二级交叉”重建并通过连续性门。
- 仓内有 v2 meta 和 v1 内容哈希，但 corrected record、正式多种子 checkpoint 与统一汇总没有组成可重放包；最近 corrected H18/LPV 结果仍是单种子或不完整种子返回。

**影响**

v1 上的幅度、反事实和模型比较数字不能迁移到 v2.1；当前 corrected 数字也不能替代完整矩阵。

**审计处置**

数据构建逻辑不重做。正式重发必须绑定 corrected record SHA256、mapping SHA256、物性表 SHA256、代码 commit、每个 checkpoint SHA256 和完整 seed 列表。大文件可继续不入 Git，但哈希与生成命令必须入仓。

### C6. v2 质量门在 clip 后检查范围，部分生产通道的越界门会恒为零

**证据**

`src/final_wm/data_v2.py:379-386` 先 `np.clip()`，随后才计算 `range_violation`。`configs/final_wm/channel_mapping_v2.json` 中多个通道的 `clip` 与 `range` 相同，因此原始越界值会被抹平后再计数。

派生通道还有源缺失被填零后覆盖率显示正常的风险；现有测试使用的简化 mapping 没有覆盖生产 `clip == range` 情形。

**影响**

meta 中“range gate 通过”不能证明原始数据确实在范围内，属于数据质量声明缺口。当前 canonical 的主数组没有无效行，但这不能替代扩展通道原始质量门。

**审计处置**

必须修。只把统计顺序改为“raw finite/range/coverage -> gate -> clip for model input”，并增加一个生产 mapping 回归测试；不重写数据管线。

### C7. checkpoint 选择每个 epoch 使用不同 validation 窗口

**证据**

`src/final_wm/training.py:234-240` 的 validation seed 为 `10_000 + epoch`。因此每个 epoch 的 `val_nll`、`best_val`、patience 和 `val_tail` 来自不同随机窗口。

**影响**

早停和 best checkpoint 同时受模型变化与采样变化影响，最优值有“挑中较容易验证子样本”的偏差；不同臂之间也难以把训练收敛与抽样噪声分开。

**审计处置**

必须修。每个 run 在开始时冻结同一组 validation anchors，所有 epoch 和最终选择器复用；训练 batch 仍可随机。

### C8. NLL 百分比改善不是稳定的判决尺度

**证据**

- `src/final_wm/model.py:303-304` 与 `src/final_wm/evaluation.py:72-73` 使用省略高斯常数的 NLL。
- `src/final_wm/evaluation.py:107-140` 对逐日 NLL 做 `(baseline-arm)/baseline`，并把分母 clamp 到 `1e-9`。

**影响**

NLL 可因测量单位、加性常数和较小/负基线改变“百分比改善”，2%/3%/5% 门不是统计不变量；当日 NLL 非正时，clamp 会产生失真的巨大比例。现有 NLL 百分比及其 CI 不适合作为正式门。

**审计处置**

必须修统计判决，不需要改模型训练目标。正式门改为 paired day-block `ΔNLL = NLL_arm - NLL_baseline` 的点估计与 CI；实用幅度用 CRPS/MAE 相对改善报告。历史百分比仅保留说明，不沿用为新判决。

### C9. run 身份与产物命名不足以唯一绑定一次正式执行

**证据**

- `src/final_wm/training.py:174` 的 `run_id` 只有 unit/arm/seed；quick/full、record、物性、anchor 均不在名字中。
- `config_fingerprint()`（`:318-329`）覆盖 spec、结构和 HEAD 树哈希，但不覆盖 canonical 内容、物性表内容、init/anchor checkpoint 内容和 dirty worktree。
- `experiments/final_wm/run_matrix.py:463-465` 允许任意 seed 子集；正式 verdict 只屏蔽 quick/arm-filter，两个 seed 已可能满足 `MIN_SEED_PASSES=2`。
- quick summary 分名，但 checkpoint、metrics 和 ledger 仍共享 run_id 命名空间。
- 当前侧 A/B summary 不含全部 O1/T1/B1/J1/R1，且旧 `verdict_audit_sideA.json` 与较新的 summary 口径不一致，缺少单一权威索引。

**影响**

即使数值本身正确，也无法仅从 verdict 唯一证明它来自哪份数据、物性、权重和执行模式；部分执行或 quick 产物可能污染正式命名空间。

**审计处置**

必须修，但只增加一个内容寻址 manifest：正式 verdict 要求固定完整 seeds、clean commit 和全部必需哈希；partial/quick 永远写 `INCOMPLETE/SMOKE` 且使用独立目录。不要引入数据库或新服务。

## 4. 本轮明确不改的事项

下列问题存在，但在当前数据和论文可信度主线上没有足够证据证明会改变结论，因此暂不进入修复批：

| 事项 | 当前判断 | 处置 |
|---|---|---|
| `valid` mask 未被 loader/sampler 使用 | 当前 canonical A/B 复核无 invalid row | 暂不改；若新 record 出现 invalid row，优先 fail-closed，不先设计复杂 masked sampler |
| stuck ratio 对长平台段的计数偏低 | 按完整 run 重算仍低于 5% 门 | 记录但不改，不触发重建 |
| `torch.load(weights_only=False)` | 是安全/供应链风险，不改变当前受控本地产物的科学判决 | 不进入本批；若接收外部 checkpoint 再单独处理 |
| Git 体积、未使用 LFS、缺依赖锁、README 测试数过时 | 影响工程便利性，不直接改变当前科学结果 | 不动 |
| LPV、closure、controller、observer 架构 | 最新 LPV 仍为单种子 `INCONCLUSIVE`，没有足够证据支持继续调结构 | 冻结，不借审计扩展模型 |
| 轴对齐支持盒是否应升级为密度/OOD 模型 | 当前根因是逐样本隔离和门禁绕过 | 先做最小修复；只有支持盒经实证仍明显误判时再立项 |

## 5. 结果可引用范围

修复和重发前，可以保留的表述：

- 项目已经实现概率物理状态世界模型原型及 train/validation 评测工具链；
- canonical v1 一级阀错侧缺陷已定位，v2.1 构建规格已修正；
- 单种子探针可用于提出假设、估算预算和关闭明显无效方向；
- synthetic、validation 和物理先验结果均不得外推为现场因果或闭环能力。

暂时不能保留的表述：

- 当前 O1/T1/B1/J1/R1 的正式 `SUPPORTED/REJECTED` 是完整冻结协议判决；
- 当前 leakage probe 已证明无未来信息泄漏；
- 当前 CF-1/CF probe 已证明支持域内反事实保真；
- v1 上的喷水响应幅度或 CF 定量结果可代表 corrected v2.1；
- 单种子 corrected/LPV 数字代表稳定模型改进。

## 6. 审计限制与闭合条件

本轮审计阶段没有动态执行训练或测试，只完成源码路径核对、Git/产物/JSON 静态检查和结果一致性审计。2026-08-28 执行前复核已确认本机具备 `torch 2.5.1`、`pytest 8.3.4` 和 CUDA；后续修复按方案逐项跑失败测试、最小实现和回归测试。

解除 `HOLD` 需要同时满足：

1. C1–C9 对应的最小修复和回归测试全部通过；
2. 冻结新的机器可执行矩阵版本，文档与 runner 必需证据逐项一致；
3. corrected v2.1、完整固定 seeds、validation-only 的正式重跑返回；
4. 独立脚本从 manifest、原始 metrics 和 checkpoint 哈希复算所有 verdict；
5. 人工审计快照与 `configs/phase3_5/experiment_registry.json` 同步登记为 `audited`；
6. 用户另行决定是否解冻论文和是否授权 test，二者都不自动发生。

逐项修复顺序和验收证据见：
`docs/plans/2026-08-27-final-wm-credibility-repair-plan.md`。
