# Linux 远端实验协议

> 本协议区分实验研发、算力执行与结果审计，避免把“代码可运行”误写成“模型已验证”。

> 当前活动批次是 Phase 3.5，精确命令以 [`experiments/phase3_5/README.md`](../experiments/phase3_5/README.md) 为准；活状态只更新根目录 [`TODO.md`](../TODO.md)。Phase 4 已暂停。

## 职责边界

### 本地工作区

负责：

1. 审查已有证据、数据口径和潜在泄漏。
2. 写清假设、对照组、主要指标、失败条件和停止规则。
3. 实现模型、训练/评测脚本、测试与 smoke 模式。
4. 生成唯一可复现的远端运行说明。
5. 接收结果后复算关键数字、审查异常、比较 seed，并更新状态和结论。

### Linux 远端

负责：

1. 检出指定 commit，不使用未提交代码。
2. 在真实数据与 GPU 环境中按原命令运行。
3. 保存完整日志、结构化结果和运行元数据。
4. 原样回传失败信息，不在远端临时修改模型或评测口径。

Linux 的版本化写入范围严格限定为：

- 当前授权 Gate 的 `results/<active_gate>/**` 或运行手册明确给出的等价结果目录；
- runner 按协议生成/更新的 manifest、metrics、ledger、checkpoint hash；
- 若运行手册明确要求，可新增一个 `docs/REMOTE_RETURN_<gate>_<date>.md`，且顶部必须标记 `UNVERIFIED_REMOTE_REPORT`。

Linux **不得**修改 `configs/`、`src/`、`experiments/`、`tests/`、根 `TODO.md`、根 `README.md`、`docs/PROJECT_STATUS.md`、`docs/PHASE35_CONTEXT_SNAPSHOT.md`、机器注册表，或任何 `SUPERVISOR_AUDIT/REVIEW` 文档；不得自行把状态改为 `audited`、`closed` 或放行下一 Gate。只有本地 Supervisor 在复算后更新权威状态、结论和下一批授权。

## 实验状态

| 状态 | 含义 | 可否写入结论 |
|---|---|---|
| `planned` / `implementation` | 设计或代码尚未完成 | 否 |
| `local_verified` | 本地测试、smoke、dry-run 已通过，尚未放行 | 否 |
| `ready_for_linux` | 已提交、运行清单完整且是唯一 Linux 授权 Gate | 否 |
| `linux_running` | Linux 按冻结版本执行中 | 否 |
| `results_returned` | 结果已回传，尚未审计 | 否 |
| `audited` | 指标、日志、协议和异常已由本地复核 | 可写带范围的结果 |
| `test_authorized` / `test_completed` | 仅对明确 Gate 开放一次 test / 已执行待审计 | 未审计前否 |
| `closed` | Gate 已按预注册规则完成最终判决 | 可更新带范围的项目结论 |
| `paused` / `deprecated` | 暂停或废弃路线 | 否 |

机器状态词以 `configs/phase3_5/experiment_registry.json` 的 `status_vocabulary` 为准；Linux 不得依据旧文档中的近义状态自行迁移。

## 本地交付给 Linux 的运行包

每次正式实验至少提供：

```yaml
task: valve_level_phase3_5
experiment_id: phase3_5_development
required_git_tag: <supervisor-provided tag>
git_commit: <runtime verifies and records 40-char SHA>
script: experiments/phase3_5/run_matrix.py
command: python experiments/phase3_5/run_matrix.py --cache-a <A> --cache-b <B> --device cuda --execute --evaluate-validation
working_directory: <repo-root>
data_id: <A/B raw SHA256 and cache manifests>
seeds: [0, 1, 2]
expected_output: results/phase3_5/runs/<side>_<config>_s<seed>/
primary_metric: validation_integrated_mae
diagnostic_metrics: [IRF-WMAE, direction, lag, dose_monotonicity, SP_negative_control]
stop_rule: <predeclared failure/early-stop rule>
estimated_runtime: <estimate>
```

同时说明：

- 依赖或 Conda 环境；
- 必需的环境变量；
- GPU/显存要求；
- smoke 命令与预期输出；
- 结果文件不得覆盖的既有目录。

## Linux 回传包

至少包含：

```text
git_commit.txt          # git rev-parse HEAD
command.txt             # 实际执行的完整命令
environment.txt         # Python/PyTorch/CUDA/GPU 信息
data_fingerprint.txt    # 数据版本或 hash
stdout.log
stderr.log
exit_code.txt
results/...             # JSON/CSV/NPZ 等结构化结果
```

checkpoint 体积过大时可不提交 Git，但应保留文件名、大小、SHA256 和保存位置。正式结果 JSON 必须入库或提供可追溯归档。

远端报告中的数值摘要默认状态是 `UNVERIFIED_REMOTE_REPORT`。它可以说明命令是否完成、产物位置和原始门禁输出，但不能写“独立审计通过”“论文结论成立”或自行部署下一 Gate。

## 本地审计清单

结果返回后依次检查：

1. commit、命令、数据、seed 和预注册配置是否一致。
2. 是否存在 NaN、OOM、早停异常、断点续训或部分 seed 缺失。
3. checkpoint 是否由验证集选择，测试集是否只在最终阶段使用。
4. 关键指标能否从逐事件/逐样本结果重新聚合得到。
5. 是否分别报告预测、观测事件响应、物理、OOD 和计算成本；CFI 不参与 checkpoint 选择。
6. 负面结果是否可能来自数值积分、单位、动作编码或结果覆盖错误。
7. 结论范围是否严格小于等于实验覆盖范围。

审计完成后由本地 Supervisor 更新机器注册表、根 `TODO.md`、`docs/PHASE35_CONTEXT_SNAPSHOT.md`、`docs/PROJECT_STATUS.md` 和 `results/README.md`。`docs/CURRENT_TASKS.md` 是历史任务说明，不再维护。只有改变项目主判断时才更新根 README。

## 失败处理

- 运行环境失败：远端回传日志，本地补环境说明或兼容修复，生成新 commit。
- 代码失败：本地增加最小复现与测试后修复，远端从新 commit 重跑。
- 数据问题：冻结当前结果，不用补丁数据继续同一实验编号；修正后使用新协议后缀或新实验编号。
- 指标/协议问题：旧结果标记为作废或待重算，禁止静默覆盖。
