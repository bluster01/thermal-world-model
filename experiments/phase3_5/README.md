# Phase 3.5 Linux 执行手册

本目录是 Phase 3 论文核心验证的唯一执行入口。Linux 只运行冻结命令并回传产物，不改代码、阈值、配置、seed 或 split。正式运行前记录 `git rev-parse HEAD`，且工作树必须干净。

## 0. 环境与路径

从仓库根目录执行。以下路径仅为示例，按远端挂载修改环境变量，不修改版本化配置：

```bash
export PH35_RAW_A=/data/yimin/A侧主汽温全数据4.csv
export PH35_RAW_B=/data/yimin/B侧主汽温全数据4.csv
export PH35_CACHE_A=/data/thermal-world-model/phase3_5/cache_A.npz
export PH35_CACHE_B=/data/thermal-world-model/phase3_5/cache_B.npz
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

git status --short
git rev-parse HEAD
python --version
python -c "import numpy,pandas,torch; print(numpy.__version__, pandas.__version__, torch.__version__, torch.cuda.is_available())"
```

若 `git status --short` 非空，停止并回传，不在远端热修。

## 1. 生成 A/B 因果缓存

```bash
python experiments/phase3_5/prepare_data.py \
  --input "$PH35_RAW_A" --output "$PH35_CACHE_A" --side A

python experiments/phase3_5/prepare_data.py \
  --input "$PH35_RAW_B" --output "$PH35_CACHE_B" --side B
```

每侧必须回传 `.npz` 同目录下的 `.manifest.json`。manifest 包含原始文件 SHA256、行数、时间范围、tag 更新间隔和 grid staleness。喷水流量不进入缓存契约。

## 2. 本地协议测试与矩阵 dry-run

```bash
python -m pytest tests/phase35 -q
python -m compileall -q src/phase35 experiments/phase3_5

python experiments/phase3_5/run_matrix.py \
  --cache-a "$PH35_CACHE_A" --cache-b "$PH35_CACHE_B"
```

dry-run 必须打印 42 个开发 run：7 configs × A/B × seeds 0/1/2。先保存 stdout，不直接执行。固定 `equal_percentage_r50` 是 exp_201 导出的先验消融，不是流量标定真值。

## 3. 开发训练与 validation 评估

```bash
python experiments/phase3_5/run_matrix.py \
  --cache-a "$PH35_CACHE_A" --cache-b "$PH35_CACHE_B" \
  --device cuda --execute --evaluate-validation --skip-existing

python experiments/phase3_5/summarize.py \
  --split validation --output-dir results/phase3_5
```

`run_matrix.py` 会把矩阵中冻结的 evaluation 参数逐项传给 `evaluate.py`；当前
`caliper_quantile=0.02` 是在首轮 validation 后冻结的探索性参数，不能包装成预注册的
确认性因果门禁。评估产物必须保存 evaluator/checkpoint/cache SHA 和完整参数。

训练期间唯一 checkpoint selector 是 validation integrated MAE。不得运行 `--split test`。每个 run 应至少包含：

```text
manifest.json
history.json
checkpoint_best_val.pt
metrics_validation.json
event_metrics_validation.json
event_manifest_validation.json
```

把整个 `results/phase3_5/`、控制台日志和环境信息原样回传，由本地检查事件数、独立日块数、`max|SMD|`、pretrend、IRF 和模型参数后再冻结候选。

参数健康摘要使用显式 cache 路径运行；`free_only` 保留用于 42-run 闭合，但通过
`physics_parameters_trained=false` 排除于 gain/τ 健康统计：

```bash
python experiments/phase3_5/param_summary.py \
  --cache-a "$PH35_CACHE_A" --cache-b "$PH35_CACHE_B" \
  --device cuda --output results/phase3_5/param_summary_validation.json
```

## 4. 候选补足 5 seeds

本地审计后，每侧最多指定两个 config。以下只示范候选名，必须替换为本地签字清单中的真实值：

```bash
python experiments/phase3_5/run_matrix.py \
  --cache-a "$PH35_CACHE_A" --cache-b "$PH35_CACHE_B" \
  --configs absolute_identity,absolute_nonlinear \
  --seeds 3,4 --device cuda --execute --evaluate-validation
```

`--seeds 3,4` 是显式补充 seed 集，不会改写开发矩阵。补跑后再次汇总 validation 并回传；仍不得访问 test。

## 5. 单次批量 test

只有在候选配置、5-seed checkpoint 清单、代码 commit 和统计脚本全部冻结后才执行。对签字清单中的每个 checkpoint 逐一运行：

```bash
python experiments/phase3_5/evaluate.py \
  --checkpoint results/phase3_5/runs/A_absolute_identity_s0/checkpoint_best_val.pt \
  --cache "$PH35_CACHE_A" --split test --device cuda --allow-test-access
```

替换 side/config/seed 并在同一批次完成全部冻结 checkpoint。每个 run 会生成：

```text
metrics_test.json
event_metrics_test.json
event_manifest_test.json
access_ledger.json
```

存在 `access_ledger.json` 时脚本拒绝再次访问该 run 的 test。不得依据 test 结果重新选择模型或再开新 seed。最后执行：

```bash
python experiments/phase3_5/summarize.py \
  --split test --output-dir results/phase3_5
```

## 6. 回传清单

- Git SHA 与干净工作树证据；
- 完整命令、stdout/stderr、退出码；
- Python/NumPy/Pandas/PyTorch/CUDA 环境；
- A/B cache manifests；
- `results/phase3_5/` 完整目录，不挑选或重命名文件；
- 失败 run 也原样回传，不在远端修复。

Linux 输出只是 `results_returned`。只有本地复算事件、统计量、反例和门禁后，才能写入论文结论。
