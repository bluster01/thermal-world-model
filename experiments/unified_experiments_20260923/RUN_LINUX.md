# Linux worker 执行合同

本批实验为 `unified_industrial_stage1_20260923`，共 45 次正式拟合。代码通过项目既有 GitHub 通道领取；训练与工业数据预运行全部在 **Linux CUDA** 执行。本文中的 source commit 指包含本批 `release_manifest.json` 的发布提交，发布前不填写虚构哈希。

## 1. 先领取，再启动

先读取项目公开任务状态及同批所有领取回执。若已有其他 worker 的有效领取，继续跟随该 worker 的状态，不重复启动。不得终止、重用或覆盖旧实验进程及目录。本批只允许通过 `origin/main` 普通 fast-forward 提交以下三个公开回执文件：

```text
results/unified_industrial_stage1_20260923/claim.json
results/unified_industrial_stage1_20260923/public_status.json
results/unified_industrial_stage1_20260923/return_receipt.json
```

回执在独立 coordination checkout 中提交，执行源码在另一个固定 release commit checkout 中运行，回执更新不得改变运行中的执行树。领取前先 `fetch` 并检查远端 `main` 最新状态；只有领取提交 `push` 成功才取得本批执行资格。若 push 冲突，重新 fetch 并核查是否已有其他领取，不 force push，不带着未成功的领取继续启动。

worker 先发布领取回执，再核验环境与启动程序。回执可以包含实验标识、本次 release commit、随机 claim ID、非敏感 worker 别名、UTC 时间和执行状态。例如：

```json
{
  "experiment_id": "unified_industrial_stage1_20260923",
  "source_commit": "<THIS_RELEASE_COMMIT>",
  "claim_id": "<UNIQUE_CLAIM_ID>",
  "worker_alias": "<OPAQUE_WORKER_ALIAS>",
  "status": "CLAIMED",
  "updated_utc": "<CURRENT_UTC>"
}
```

在独立 coordination checkout 中，用实际路径替换变量并核查领取：

```bash
export COORD_ROOT='<ABSOLUTE_SEPARATE_COORDINATION_CHECKOUT>'
export RECEIPT_DIR='results/unified_industrial_stage1_20260923'
git -C "$COORD_ROOT" fetch origin main
git -C "$COORD_ROOT" merge --ff-only origin/main
if git -C "$COORD_ROOT" cat-file -e "origin/main:$RECEIPT_DIR/claim.json" 2>/dev/null; then
  git -C "$COORD_ROOT" show "origin/main:$RECEIPT_DIR/claim.json"
fi
```

确认没有有效领取后，将填好的上述 JSON 写入该 checkout 的 `claim.json`；人工检查仅此白名单文件变更，再提交领取：

```bash
git -C "$COORD_ROOT" add -- "$RECEIPT_DIR/claim.json"
git -C "$COORD_ROOT" commit -m 'Claim unified industrial stage1 Linux execution'
git -C "$COORD_ROOT" push origin HEAD:main
```

不要在公开回执中放原始文件名、机器私有绝对路径、真实数值、详细异常堆栈、预测轨迹或权重。领取状态与运行器生成的进度状态分别保留。`public_status.json` 只有执行进度，可以按本批允许的回执流程回传。

## 2. 准备独立代码目录和私有路径

以下 Bash 变量由实际 Linux worker 填写。`CODE_ROOT` 使用本批独立 checkout；`PRIVATE_ROOT`、`DATA_ROOT`、`OUT_ROOT` 必须位于所有 Git checkout 外，`DATA_ROOT` 与首次使用的 `OUT_ROOT` 为新目录。已有严格匹配的数据包可使用后文的 `--verify-only` 分支。

```bash
export SOURCE_COMMIT='<THIS_RELEASE_COMMIT>'
export CODE_ROOT='<ABSOLUTE_ISOLATED_CODE_CHECKOUT>'
export PRIVATE_ROOT='<ABSOLUTE_PRIVATE_DIRECTORY_OUTSIDE_ALL_GIT_CHECKOUTS>'
export RAW_ROOT='<ABSOLUTE_PRIVATE_ORIGINAL_CSV_DIRECTORY>'
export DATA_ROOT="$PRIVATE_ROOT/unified_industrial_stage1_data"
export OUT_ROOT="$PRIVATE_ROOT/unified_industrial_stage1_runs"
cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
```

先使用项目已有代码领取流程创建或选取上述独立 checkout，检出本次 release commit；不要在正在执行其他实验的 checkout 上切换版本。`run.py` 会核验发布源码清单并记录实际 Git commit。更改模型、协议或清单后不能继续使用本批原身份。

使用独立环境并遵循 [requirements.txt](requirements.txt)：NumPy 1.26.4、pandas 2.1.4、PyTorch >=2.5 且 <3，以及所列 pytest 版本范围。PyTorch 选择适配 worker 驱动的 CUDA 构建，再安装其余依赖。Python、PyTorch、NumPy、CUDA 版本及设备信息被写入运行身份，同一矩阵续训要求签名一致。

```bash
python -m pip install -r experiments/unified_experiments_20260923/requirements.txt
```

私有输出盘至少保留 10 GiB 空闲空间；源 CSV 与重建数组另需相应空间。运行器会检查输出盘门槛。

```bash
python -m experiments.unified_experiments_20260923.prepare_linux --help
python -m experiments.unified_experiments_20260923.run --help
python -c 'import platform, torch, numpy, pandas; assert platform.system() == "Linux"; assert torch.cuda.is_available(); print({"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
```

这里的两个 `--help` 与发布源码实际参数一致。缺少 Linux、CUDA、匹配源数据或必要依赖时，保留已发现信息，在回执中报告 `PREREQUISITE_MISSING` 及非敏感缺项，不静默改到 Windows、CPU 正式拟合或别的数据集。

合成软件检查可以使用 CPU，不接触工业 CSV：

```bash
python -m pytest -q \
  experiments/unified_experiments_20260923/test_models.py \
  experiments/unified_experiments_20260923/test_runner.py \
  experiments/unified_experiments_20260923/test_prepare_linux.py \
  experiments/unified_experiments_20260923/test_summarize.py \
  experiments/unified_experiments_20260923/test_execution.py
```

## 3. 私有数据重建或核验

从 Linux 本地原始 CSV 重建：

```bash
python -m experiments.unified_experiments_20260923.prepare_linux \
  --raw-root "$RAW_ROOT" \
  --data "$DATA_ROOT"
```

可选的 `--point-mapping '<PRIVATE_POINT_MAPPING_CSV>'` 指向本机点位对照表。未提供时，准备器保留源文件身份并标记点位编码尚未核实，不从别名反推仪表编码。

准备器逐别名匹配原 CSV 文件名哈希、文件大小及完整字节 SHA256，验证冻结准备源码，再生成数组和窗口索引。缺少某个字节一致的源文件时停止并报告前置条件不足；不替换成名字相似或新时间段的文件。私有准备日志、源文件映射及日历配置都留在 `DATA_ROOT`。

若 Linux 已有本批相同数据包，令 `DATA_ROOT` 指向它，仅核验：

```bash
python -m experiments.unified_experiments_20260923.prepare_linux \
  --data "$DATA_ROOT" --verify-only
```

数据校验包括数组文件哈希、压缩包内窗口数组、标准化参数、通道顺序、任务配置及源身份。`data_identity.json` 必须为 `PASS`，数值容差为零；运行器每次启动与续训都会重新核验实际数据。核验失败保留私有诊断并报告，不修改 `expected_data.json`。重建命令拒绝覆盖已存在的数据目录；中途失败目录原样保留，重新准备使用新的私有路径。

## 4. Linux CUDA 预运行

确认领取仍属于当前 worker，`OUT_ROOT` 尚不存在，然后执行：

```bash
python -m experiments.unified_experiments_20260923.run preflight \
  --data "$DATA_ROOT" \
  --out "$OUT_ROOT" \
  --workers 2
```

预运行核验数据身份，对三个任务的五种模型逐一检查少量训练窗口的小步优化和 H180 有限输出。结果写入私有 `preflight.json`，公开进度为 `PREFLIGHT_PASSED` 后才进入完整矩阵。预运行结果不进入正式效果排名。

`OUT_ROOT` 由运行器创建；不要为了放日志提前创建它。需后台运行时使用 worker 已有的 tmux、调度器或服务管理方式，日志重定向到 `PRIVATE_ROOT` 中另一个私有文件，而不是尚未创建的 `OUT_ROOT`。

## 5. 完整矩阵与统一历史测试

预运行已经创建同一输出目录，所以正式矩阵明确使用 `--resume`：

```bash
python -m experiments.unified_experiments_20260923.run run \
  --data "$DATA_ROOT" \
  --out "$OUT_ROOT" \
  --workers 2 \
  --resume
```

也可对一个全新的 `OUT_ROOT` 直接执行 `run run` 并省略 `--resume`；运行器会先做预运行，再执行完整矩阵。两种方式不要同时启动。运行器对同一输出目录使用单进程锁；遇到占用时停止第二个启动请求，不删除运行中锁文件、不杀已有实验。公开领取记录用于防止不同输出目录或不同机器重复执行同一批任务。

矩阵顺序为 seed → task → arm，种子 11/22/33，3 个任务、5 个模型，共 45 次拟合。保持冻结协议：H60 训练，最大 80 epoch，最少 20 epoch，验证集提前停止；达到预算上限保留 `BUDGET_EXHAUSTED`。没有按中途排名跳过模型的分支。

首个完整 epoch 后，用私有 `ledger.jsonl` 的实际耗时与样本吞吐测算进度；不同任务和模型分别更新估计，不预先填写未经测量的 GPU 速度或完成时长。

所有 45 次拟合都到达完成状态后，统一加载各自最佳验证权重，执行完整验证集及历史测试集的 H60/H180 评价。历史测试不得提前读取来调参；校验完成标记与产物哈希后，已完成的最终评价不重复计算。H180 是相同 H60 权重的时域延展，不另训模型，不使用已记录未来动作。

## 6. 续训与回传

进程中断后先核实原 worker 已退出、同批没有新的有效领取，再继续原领取或按合同明确移交。使用同一代码、数据及运行时身份和 `OUT_ROOT`，显式运行上一节带 `--resume` 的命令。续训从完整 epoch checkpoint 恢复模型、优化器、调度器和随机状态；已完成拟合保持原状。每次验证改进保存不可变的 `best_epoch_XXX.pt`，`last.pt` 记录它的引用，拟合完成后再生成固定 `best.pt`。没有完整 epoch checkpoint 的失败目录保留并报告。任何改配置的重试需要另行登记身份。

私有输出主要包括：

```text
OUT_ROOT/
  run_identity.json
  protocol.json
  data_identity.json
  preflight.json
  public_status.json
  summary.json
  SUMMARY_ZH.md
  failure.txt                         # 失败时的私有诊断
  runs/<task>__<arm>__s<seed>/
    status.json
    ledger.jsonl
    validation_selection_indices.npy
    best.pt
    best_epoch_XXX.pt
    last.pt
    evaluation_<split>_<profile>.json
    evaluation_<split>_<profile>.npz  # 全部起点误差统计 + 前三个 batch 轨迹预览
```

评价 NPZ 对所有起点保存 `origin_time_ns`、`label_count`、`normalized_sse`、`physical_absolute_error`、`persistence_absolute_error`、`increment_count` 和 `increment_absolute_error`；`preview_*` 字段存前三个 batch 的预测、标签与掩码。NPZ 原子写入完成后，JSON 完成标记和哈希标识这一组评价产物。

自动汇总 `summary.json` 和 `SUMMARY_ZH.md` 包含三种子统计、逐目标指标及慢状态配对差，属于私有结果。详细指标、真实预测、日志、源路径和权重均保留于私有存储。公开只回传允许的领取回执、执行状态、完成拟合数及 `public_status.json` 中既有进度字段。最终状态 `COMPLETED_PRIVATE_RESULTS_READY` 表示私有结果已经生成。
