# Linux 远端实验协议

> 本协议区分实验研发、算力执行与结果审计，避免把“代码可运行”误写成“模型已验证”。

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

## 实验状态

| 状态 | 含义 | 可否写入结论 |
|---|---|---|
| `designed` | 假设、对照和判据已写清 | 否 |
| `implemented` | 代码完成，尚未通过 smoke | 否 |
| `smoke_passed` | 小规模前向、训练和保存链路通过 | 否 |
| `ready_for_remote` | 已提交且远端清单完整 | 否 |
| `remote_running` | Linux 正式实验执行中 | 否 |
| `results_returned` | 结果已回传，尚未审计 | 否 |
| `audited` | 指标、日志、协议和异常已复核 | 可写带范围的结果 |
| `concluded` | 多 seed 与关键消融支持判决 | 可更新项目结论 |

## 本地交付给 Linux 的运行包

每次正式实验至少提供：

```yaml
experiment_id: exp_xxx
git_commit: <40-char SHA>
branch: main
script: experiments/<phase>/exp_xxx_name.py
command: python experiments/<phase>/exp_xxx_name.py <args>
working_directory: <repo-root>
data_id: <dataset name/version/fingerprint>
seeds: [0, 1, 2, 3, 4]
expected_output: results/exp_xxx_name/
primary_metrics: [mae, cfi_agg]
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

## 本地审计清单

结果返回后依次检查：

1. commit、命令、数据、seed 和预注册配置是否一致。
2. 是否存在 NaN、OOM、早停异常、断点续训或部分 seed 缺失。
3. checkpoint 是否由验证集选择，测试集是否只在最终阶段使用。
4. 关键指标能否从逐事件/逐样本结果重新聚合得到。
5. 是否同时报告预测、干预、物理、OOD 和计算成本。
6. 负面结果是否可能来自数值积分、单位、动作编码或结果覆盖错误。
7. 结论范围是否严格小于等于实验覆盖范围。

审计完成后更新 `docs/PROJECT_STATUS.md`、`docs/CURRENT_TASKS.md` 和 `results/README.md`。只有改变项目主判断时才更新根 README。

## 失败处理

- 运行环境失败：远端回传日志，本地补环境说明或兼容修复，生成新 commit。
- 代码失败：本地增加最小复现与测试后修复，远端从新 commit 重跑。
- 数据问题：冻结当前结果，不用补丁数据继续同一实验编号；修正后使用新协议后缀或新实验编号。
- 指标/协议问题：旧结果标记为作废或待重算，禁止静默覆盖。
