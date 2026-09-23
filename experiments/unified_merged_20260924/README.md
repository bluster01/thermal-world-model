# Linux 执行单：使用现有 merged 数据包

实验：`unified_industrial_merged_stage1_20260924`。本批直接使用 Linux 已有、由 `d4a04eb` 回执登记的私有 merged 数据包。数据按该包现有文件 SHA256 原样冻结，执行入口自动核验；直接开始预运行和训练。

## 1. 领取本批

在独立 coordination checkout 中 `fetch origin main`，读取最新任务状态及本批领取回执。已有有效 worker 领取时跟随其状态。首次领取写入：

```text
results/unified_industrial_merged_stage1_20260924/claim.json
```

领取内容使用实际值：

```json
{
  "experiment_id": "unified_industrial_merged_stage1_20260924",
  "source_commit": "<THIS_RELEASE_COMMIT>",
  "claim_id": "<UNIQUE_CLAIM_ID>",
  "worker_alias": "<OPAQUE_WORKER_ALIAS>",
  "status": "CLAIMED",
  "updated_utc": "<CURRENT_UTC>"
}
```

仅提交本批 `claim.json`，通过既有 `origin/main` 普通 fast-forward 流程 push；领取 push 成功后启动。发生冲突则重新 fetch 并核查领取，不 force push。执行代码使用独立 checkout，固定在本批包含 `release_manifest.json` 的发布 commit；协调回执不改变运行中的执行树。

## 2. 直接运行

使用已有 Linux CUDA 环境，并沿用 [依赖要求](../unified_experiments_20260923/requirements.txt)。`--data` 填 Linux 当前 merged 数据包的私有绝对路径；`--out` 填 Git checkout 外的全新私有输出目录。本批输出使用新 experiment ID，不复用 native 实验的输出或 checkpoint。

在固定发布 commit 的代码根目录运行：

```bash
python -m experiments.unified_merged_20260924.run preflight \
  --data '<EXISTING_PRIVATE_MERGED_PACK>' \
  --out '<NEW_PRIVATE_OUT>'

python -m experiments.unified_merged_20260924.run run \
  --data '<EXISTING_PRIVATE_MERGED_PACK>' \
  --out '<NEW_PRIVATE_OUT>' \
  --resume
```

第二条命令用 `--resume` 接续预运行创建的输出目录。后续中断也用同一条 `run ... --resume` 接续。预运行和正式拟合均要求 Linux + CUDA；运行器核验本批源码、数据和续训身份。现有 merged 包直接消费，无需传输原生逐点数据，无需重新准备数据，也无需修改冻结哈希。缺项通过本批公开状态回报简短原因，详细异常留在私有目录。

## 3. 本轮矩阵

| 配置 | 值 |
|---|---|
| 任务 | `scr`、`main_steam`、`reheat_steam` |
| 模型 | `gru_direct`、`ssm_rollout`、`itransformer`、`mechanism`、`mechanism_no_slow` |
| 种子 | 11、22、33，共 **45** 次正式拟合 |
| 历史与训练预测 | 360 步历史，60 步未来；10 秒网格 |
| 延展评价 | 同一权重预测 180 步 |
| 未来输入 | history-only，不读取真实未来动作或边界 |
| 选模 | 验证集选模；45 次拟合全部完成后统一评价历史测试 |

完整训练参数以 [protocol.json](protocol.json) 为准。模型与汇总程序复用已冻结实现；三个任务分别训练权重。此包的有效性、观测标记和年龄按合并表网格语义解释，本轮建立该源口径下的预测对比与慢状态消融结果。

H180 为 H60 训练后的外推：状态模型继续递推，直接模型沿用同一连续时距查询头。热任务使用每测点局部快慢储存交换，SCR 使用非负浓度源/移除代理；本轮结果对应这些实际模块。

## 4. 进度和结果回传

只允许向本批目录回传以下三个公开文件：

```text
results/unified_industrial_merged_stage1_20260924/claim.json
results/unified_industrial_merged_stage1_20260924/public_status.json
results/unified_industrial_merged_stage1_20260924/return_receipt.json
```

将运行输出目录生成的 `public_status.json` 按原内容回传；完成时发布简短 `return_receipt.json`，记录本批身份、实际状态和完成拟合数。仍通过独立 coordination checkout 的 `origin/main` 普通 fast-forward 提交这三个白名单文件。

模型权重、真实指标、轨迹、逐起点统计、详细日志、路径及自动生成的 `summary.json` / `SUMMARY_ZH.md` 保留在 Linux 私有输出。达到 epoch 上限记为 `BUDGET_EXHAUSTED`，保留其曲线；最终 `COMPLETED_PRIVATE_RESULTS_READY` 表示私有评价与汇总已生成。
