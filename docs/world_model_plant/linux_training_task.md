# 1/10 thermal world model：Linux 原型任务

## Material Passport

- Type / stage: ARS experiment-agent / run preparation.
- Task: `thermal_world_model_tenth_train_v1`.
- Status: QUEUED; source and schedule frozen in `release_train_v1.json`.
- Question: 保留真实历史证据是否改善长时推演；注意力是否优于年龄加权读取。
- Evidence: 本地数据准备和软件检查完成；这轮真实数据训练尚未回传。

## 实验

四臂 `shared_age / shared_attention / quota_age / quota_attention`，先跑 seed 11。共用 1/10 连续数据、初始神经参数和抽样计划；物理参数使用固定历史先验。每臂最多请求 250 批，batch 4，history 64，训练 H32；每 50 批评估固定 H128 窗口。配额缓存为 64 个证据事件和 64 个预测事件，共享缓存容量同为 128。

模型使用记录的实际阀位和未来边界条件，比较条件推演能力。五温度锚点要求 cut 时刻全部 present。验证按窗口等权取归一化 free MSE；两条训练分支均需在物性支持域内。越界窗口计入分母，跳过更新且不补抽。暂不据单 seed 宣称稳定优势。

## Linux 执行

先完成 [环境预检](github_handoff.md) 和 [本地数据准备](linux_data_task.md)。数据准备完成后可直接执行本任务，无需再等待论文审查或 SSH 配置。训练输出必须在私有目录。

```bash
export TWM_DATA='/private/local/tenth_v1'
export TWM_PROPERTIES='/private/local/iapws_surrogate.npz'
export TWM_RUN='/private/local/thermal_tenth_train_v1'
```

使用同一 Python 环境运行四臂。已测软件环境为 Python 3.11/3.12、PyTorch 2.5.1、NumPy 1.26.4；实际版本和设备会写入运行记录。以下从仓库根目录执行，`TWM_RUN` 必须尚不存在。

```bash
python -m experiments.world_model_plant.run_matrix \
  --dataset "$TWM_DATA" --properties "$TWM_PROPERTIES" --output "$TWM_RUN" \
  --config experiments/world_model_plant/task_train_v1.json \
  --release experiments/world_model_plant/release_train_v1.json \
  --device cpu
```

有可用 CUDA 时可统一改为 `--device cuda`。顺序执行，CPU 使用 1 个线程。进程每 30 秒更新 `public_matrix_status.json`；每臂硬超时 6 小时，失败或超时停止矩阵并保留记录，不自动重试。可在持久终端或 tmux 内运行。

## GitHub 领取与回传

任务回执分支：`codex/linux-thermal-train-20260909`。目录：`coordination/thermal_world_model_tenth_train_v1/`。领取前检查该分支是否已有运行回执，避免重复作业。记录当前源码 commit 后，从它创建回执分支，在该目录提交 `claim.json`，包含 `task_id`、`status: CLAIMED`、`source_commit`、匿名 `worker_id` 和 `claimed_utc`；推送该分支后开始运行。

允许复制到回执目录的结果仅为：

- 根级 `public_matrix_status.json`；
- 每臂子目录中的 `public_status.json`、`public_validation_log.jsonl`。

这些文件提供进度、有效窗口数和模型归一化误差。至少在每臂结束与任务终止时提交并推送回执。权重、输入计划、NPZ、CSV、私有日志、数组及 manifest 留在 Linux；不要整目录提交。

判读时对照 step 0 与训练后误差、有效窗口覆盖及四臂差值。如果没有完整有效的验证 checkpoint，回传该事实和分母，由结果决定下一次实验，不换样本掩盖失败。
