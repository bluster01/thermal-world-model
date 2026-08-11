# Phase 3.5 Linux 执行手册

本目录是 Phase 3.5-MS 完整模型验证的唯一执行入口。Linux 只运行注册表已授权的冻结命令并回传产物，不改代码、阈值、配置、seed 或 split。正式运行前先执行 `python experiments/phase3_5/experiment_status.py --check --json`，记录 `git rev-parse HEAD`，且工作树必须干净。历史 42-run/E 系列命令仅供追溯，除非注册表重新授权，不得执行。

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

## 7. Phase 3.5-MS 多步响应可解性批次

该批次与前述 42-run 分开：它不读取 A/B 真实数据，不恢复被阻断的 E3/E4，而是在已知二阶惯性真值下验证 Graybox、Controlled Koopman、PI-ODE 和 Causal DeepONet 能否辨识多步动作响应。阳性结果只记作 `synthetic_method_feasibility`。

执行前必须阅读 [`docs/PHASE35_MS_METHODS_AND_REFERENCES.md`](../../docs/PHASE35_MS_METHODS_AND_REFERENCES.md)；该文档冻结了公式、方法命名、可辨识性和引用边界。Linux 不根据文献或临时结果修改模型。

先在目标 commit 上执行专项验证并展开矩阵：

```bash
python -m pytest tests/phase35/multistep -q
python -m compileall -q src/phase35/multistep experiments/phase3_5/multistep_sysid.py
python experiments/phase3_5/multistep_sysid.py --dry-run
```

dry-run 必须得到 `6 routes × 3 seeds = 18 runs`。本地 CPU 冒烟只验证程序链，不产生论文结果：

```bash
python experiments/phase3_5/multistep_sysid.py \
  --route-id graybox_2p --seed 0 --device cpu \
  --output-root results/phase3_5/multistep_smoke --smoke --execute
```

Linux 正式批次用一个命令顺序执行冻结的 18 runs；`--skip-existing` 只会跳过 manifest、history、checkpoint、validation metrics 齐全，且 Git SHA/config/seed 全部与当前矩阵一致的 run。残缺或混版本目录会直接报错：

```bash
python experiments/phase3_5/multistep_sysid.py \
  --device cuda --output-root results/phase3_5/multistep_synthetic \
  --execute-matrix --skip-existing
```

调试单条 run 时使用下面模板；调试产物不得并入正式目录：

```bash
python experiments/phase3_5/multistep_sysid.py \
  --route-id graybox_2p --seed 0 --device cuda \
  --output-root results/phase3_5/multistep_synthetic --execute
```

训练命令不提供 test 访问。每个 run 回传：

```text
manifest.json
history.json
checkpoint_best_val.pt
metrics_validation.json
```

`metrics_validation.json` 同时记录 H1/H6/H18/H60 响应误差和结构诊断。路线进入 synthetic test 前必须由本地确认：`reference_identity_max_error=0`、`future_action_leakage_max_error=0`、状态/输出有限、稳定路线谱半径小于 1，并冻结候选与 checkpoint 清单。

签字后的单个冻结 checkpoint 使用独立命令一次性打开 synthetic test；命令会原样加载 validation checkpoint，写入 `metrics_test.json` 和 `synthetic_test_access_ledger.json`，重复访问会被拒绝：

```bash
python experiments/phase3_5/multistep_sysid.py \
  --route-id graybox_2p --seed 0 --device cuda \
  --output-root results/phase3_5/multistep_synthetic \
  --evaluate-synthetic-test --allow-synthetic-test
```

该显式开关只授权 synthetic known-truth test，不授权任何真实数据 test 访问。

## 8. Phase 3.5-MS2 结构失配 validation 批次

MS1 已完成；结论和限制见 [`docs/PHASE35_MS1_REVIEW_2026-08-10.md`](../../docs/PHASE35_MS1_REVIEW_2026-08-10.md)。MS2 只运行两个独立 regime：阀门单调非线性 `MS2-V` 与工况调度 `MS2-C`。冻结设计见 [`docs/plans/2026-08-10-phase35-ms2-mismatch-design.md`](../../docs/plans/2026-08-10-phase35-ms2-mismatch-design.md)。

Linux 必须在目标 commit 的干净工作树执行：

```bash
python -m pytest tests/phase35/multistep -q
python -m compileall -q src/phase35/multistep \
  experiments/phase3_5/multistep_mismatch.py \
  experiments/phase3_5/summarize_multistep_mismatch.py
python experiments/phase3_5/multistep_mismatch.py --dry-run
```

dry-run 必须严格得到 `2 regimes / 11 candidates / 33 runs`。本地/远端 smoke 只验证链路，不进入结果目录：

```bash
python experiments/phase3_5/multistep_mismatch.py \
  --candidate-id c_g2_scheduled --seed 0 --device cpu \
  --output-root results/phase3_5/multistep_mismatch_smoke \
  --smoke --execute
```

正式 validation 运行：

```bash
python experiments/phase3_5/multistep_mismatch.py \
  --device cuda \
  --output-root results/phase3_5/multistep_mismatch \
  --execute-matrix --skip-existing

python experiments/phase3_5/summarize_multistep_mismatch.py \
  --output-root results/phase3_5/multistep_mismatch
```

MS2 validation runner **没有 synthetic test 开关**。不得调用 MS1 的 test evaluator 读取 MS2 test，也不得依据 validation 临时新增 candidate、seed 或修改 300-epoch cap。33 个 run 均须原样回传 manifest、history、validation metrics 和 checkpoint。

`checkpoint_best_val.pt` 受 `.gitignore` 排除，因此在 push JSON 之前必须另外归档全部 checkpoint，并记录归档 SHA-256。每个 manifest 自带单文件 `checkpoint_sha256`；归档回传后本地逐项校验。若只有 JSON、没有可校验 checkpoint，状态只能记 `results_analyzed`，不能记 `reproducibility_passed`。

汇总器要求 33 个 checkpoint 与 hash 全部存在，检查未授权 test 产物、manifest/history 一致性和结构门禁，并生成 `summary_validation.json`；任一失败会以非零退出。Linux 只汇报运行状态和该原始聚合，不作路线冠军判定。MS2-V 与 MS2-C 分榜；`clean_effect_nmae` 为主要已知真值诊断，带噪 `effect_mae` 仍是唯一 checkpoint selector。

### 8.1 MS2 synthetic test 单次授权

本地已完成 checkpoint 权重级复算并签字授权。Linux pull 授权 commit 后，先保证 33 个忽略跟踪的 checkpoint 位于原 run 目录；若本地文件已丢失，从已核验归档恢复：

```bash
tar -xf results/phase3_5/archive/ms2_checkpoints_validation.tar \
  -C results/phase3_5/multistep_mismatch
```

执行测试与固定 bootstrap 汇总：

```bash
python -m pytest tests/phase35/multistep -q
python -m compileall -q \
  experiments/phase3_5/multistep_mismatch_test.py \
  experiments/phase3_5/summarize_multistep_mismatch_test.py

python experiments/phase3_5/multistep_mismatch_test.py \
  --device cuda \
  --output-root results/phase3_5/multistep_mismatch \
  --evaluate-test-matrix --allow-synthetic-test --skip-existing

python experiments/phase3_5/summarize_multistep_mismatch_test.py \
  --output-root results/phase3_5/multistep_mismatch
```

每 run 必须新增 `metrics_test.json`、`episode_metrics_test.json`、`synthetic_test_access_ledger.json`，并只把 manifest 的 `test_accessed` 从 false 改为 true。汇总器默认执行按动作类型分层的 10,000 次 paired-episode bootstrap；两个主对比的三个 seed 均须满足 95% CI 下界不低于 20%。

回传所有 test JSON、更新后的 manifest、`summary_test.json`、完整 stdout/stderr 和环境信息。checkpoint 权重不变，不需要重新训练或重新打包；任一 started/partial ledger 必须原样回传，不得删除后重试。该授权仅覆盖 synthetic MS2 test，不授权 A/B 真实数据 test。

## 9. Phase 3.5-MS2-J 联合耦合 validation

MS2-V/C 已收口。MS2-J 只检验同一 synthetic truth 中 R50 非线性与 context 调度能否共同收敛，以及三阶段训练相对 joint-from-scratch 是否稳定。冻结设计见 [`docs/plans/2026-08-10-phase35-ms2j-coupling-design.md`](../../docs/plans/2026-08-10-phase35-ms2j-coupling-design.md)。本批不加入 delay、三阶或扰动，不读取任何 synthetic/真实 test。

Linux 在授权 commit 的干净工作树先执行：

```bash
python -m pytest tests/phase35/multistep -q
python -m compileall -q \
  src/phase35/multistep/staging.py \
  experiments/phase3_5/joint_coupling.py \
  experiments/phase3_5/summarize_joint_coupling.py
python experiments/phase3_5/joint_coupling.py --dry-run
```

dry-run 必须严格得到 `1 regime / 9 candidates / 27 validation runs`。先在正式结果目录外做 staged smoke：

```bash
python experiments/phase3_5/joint_coupling.py \
  --candidate-id j_g2_monotone_scheduled_staged --seed 0 --device cpu \
  --output-root results/phase3_5/joint_coupling_smoke \
  --smoke --execute
```

正式 validation：

```bash
python experiments/phase3_5/joint_coupling.py \
  --device cuda \
  --output-root results/phase3_5/joint_coupling \
  --execute-matrix --skip-existing

python experiments/phase3_5/summarize_joint_coupling.py \
  --output-root results/phase3_5/joint_coupling
```

汇总器检查 27 个 checkpoint/manifest/history/validation metrics、同 seed trajectory hash、环境 provenance 和全部结构门禁。staged 的每个 seed 还必须回传 `checkpoint_stage_a/b/c.pt`、`metrics_stage_a/b/c_validation.json` 和阶段摘要。全部 `.pt` 需归档并记录 SHA：27 个 canonical checkpoint 加 staged 三 seed 的 9 个阶段 checkpoint，共 36 个文件。

若预注册的 20% 联合模块 Gate 或 10% staged 非劣 Gate 失败，汇总器会以 code 2 退出；这属于科学结果，不是运行错误，仍须原样回传全部 artifacts。Linux 不改阈值、不补 seed、不自行重训，也不写路线冠军结论。validation runner 本身没有 test 开关；后续 test 只能走下节独立的一次性授权入口。

## 10. Phase 3.5-MS2-J 一次性 synthetic test

MS2-J validation 已冻结为混合结果：联合模块门禁 PASS，staged 非劣门禁 FAIL。独立 checkpoint/归档/参数审计通过后，test 只用于确认该混合结论，不把 validation 改写成整体 PASS。Linux 先在干净工作树执行无访问预检：

```bash
python experiments/phase3_5/joint_coupling_test.py --dry-run
```

必须返回 27 runs、36 archive members、`test_accessed=false`。随后只允许执行一次完整矩阵，不接受 candidate/seed 过滤：

```bash
python experiments/phase3_5/joint_coupling_test.py \
  --device cuda --evaluate-test-matrix --allow-synthetic-test

python experiments/phase3_5/summarize_joint_coupling_test.py
```

runner 在首次生成 test 前核对 authorization、训练矩阵、validation summary、checkpoint tar 的 SHA256，以及 27 个 manifest、30 个实际读取权重（27 canonical + 3 Stage-A）和冻结训练代码等价性。test 以完整 episode 为统计单位，按 action profile 分层做 10,000 次 paired bootstrap：joint 对两个单模块的改善 CI 下界均须 ≥20%；staged/joint 误差比 CI 上界须 ≤1.10；staged 对 Stage-A 改善 CI 下界须 ≥20%。由于 validation 的 staged 门禁已失败，test 汇总器再次以 code 2 退出是预期科学结果；不得删除 started/partial ledger 后重跑。回传全部新增 JSON、更新后的 manifest、stdout/stderr 和环境信息，不重新训练、不改 checkpoint、不访问 A/B 真实数据 test。

## 11. 已完成：MS2-D1 纯迟延 validation

本节保留已完成 validation 的可复现命令，不再授权重复运行。MS2-D1 在 MS2-J 的 R50 非线性、context 调度和二阶惯性真值上只增加 20 s 纯迟延，回答显式因果迟延模块是否改善多步响应。它不访问 synthetic test 或 A/B 数据，不启动 D2/D3/MS5。

先核对机器状态、工作树和冻结矩阵：

```bash
python experiments/phase3_5/experiment_status.py --check --json
git status --short
git rev-parse HEAD
python -m pytest tests/phase35/multistep tests/phase35/test_experiment_status.py -q
python -m compileall -q src/phase35/multistep \
  experiments/phase3_5/ms2d_delay.py \
  experiments/phase3_5/summarize_ms2d_delay.py
python experiments/phase3_5/ms2d_delay.py --dry-run
```

状态输出必须是 `active_gate=ms2d_d1`、`status=ready_for_linux`、`linux_authorized_gate=ms2d_d1`；工作树必须为空；dry-run 必须严格得到 `1 regime / 6 candidates / 18 validation runs` 且 `test_authorized=false`。任一条件不符即停止回传。

正式运行与汇总：

```bash
python experiments/phase3_5/ms2d_delay.py \
  --device cuda \
  --output-root results/phase3_5/ms2d_delay \
  --execute-matrix --skip-existing

python experiments/phase3_5/summarize_ms2d_delay.py \
  --output-root results/phase3_5/ms2d_delay
```

六个候选为同结构 no-delay 消融、learned-delay 主模型、fixed-delay+R50 oracle 正控，以及 Koopman、PI-ODE、DeepONet 三个次要表示参考。主要判决只包含：18/18 artifact/结构门通过；oracle 每 seed clean NMAE `<0.05`；learned-delay 相对 no-delay 每 seed 改善 `≥20%`。期望迟延误差 `≤1 step` 与真值 ±1 step 邻域质量 `≥0.80` 单列为参数诊断，不与响应恢复混成同一个结论。

汇总器可能因科学门禁失败以 code 2 退出；仍须原样回传 18 个完整运行目录、`summary_validation.json`、命令输出、环境和 Git SHA。不得改阈值、补 seed、删除失败运行、访问 test，或继续 D2。D1 只有经本地复算后才从 `results_returned` 进入 `audited`。

## 12. 已完成：MS2-D1 一次性 synthetic test

本节保留已完成 test 的可复现命令，不再授权重复运行。MS2-D1 最终判决为 `TEST_NOT_CONFIRMED_AT_20PCT_MARGIN`：oracle 通过，改善方向稳定，但逐 seed CI 下界 17.2–18.8% 未达 20%；不重试、不调阈值。

先在干净工作树运行无 test 访问的预检：

```bash
python experiments/phase3_5/experiment_status.py --check --json
git status --short
git rev-parse HEAD
python -m pytest tests/phase35/multistep tests/phase35/test_experiment_status.py -q
python -m compileall -q src/phase35/multistep \
  experiments/phase3_5/ms2d_delay_test.py \
  experiments/phase3_5/summarize_ms2d_delay_test.py
python experiments/phase3_5/ms2d_delay_test.py --dry-run
```

状态必须是 `active_gate=ms2d_d1`、`status=test_authorized`、`linux_authorized_gate=ms2d_d1`；dry-run 必须返回 18 runs、18 archive members、`validation_screening_pass=true`、`delay_parameter_diagnostic_pass=false`、`test_accessed=false`。任一项不符立即停止。

确认无误后只执行一次完整矩阵：

```bash
python experiments/phase3_5/ms2d_delay_test.py \
  --device cuda --evaluate-test-matrix --allow-synthetic-test

python experiments/phase3_5/summarize_ms2d_delay_test.py
```

runner 会在首次 test 访问前校验 authorization、matrix、validation summary 和 checkpoint tar 的 SHA256，核对 18 个 manifest/checkpoint 与冻结训练代码等价性，并先写 root access ledger。汇总器要求同 seed 的全部候选共享完全相同的 test trajectory hash；主对比以 256 个配对 episode 为单位、按 action profile 分层做 10,000 次 bootstrap。oracle 每 seed clean NMAE 必须 `<0.05`；learned-delay 相对 no-delay 的改善 95% CI 下界每 seed必须 `≥0.20`。迟延权重集中度继续只作参数诊断。

汇总器因科学门禁失败返回 code 2 时，仍须原样提交全部新增 JSON、更新后的 manifest、stdout/stderr 和环境信息；不得删除 started/partial ledger、重复访问 test、重训、改门槛或启动 D2。test 结果只证明或否证该 known-truth pure-delay 设计，不支持现场 20 s 迟延、开环阀门因果或完整世界模型结论。

## 13. 已完成归档：MS2-D2 三阶惯性 validation

MS2-D2 是独立的阶次压力诊断，不继承 D1 的 learned-delay 阳性表述。synthetic truth 固定为 R50 非线性、context scheduling、三个惯性极点 `[40,70,210] s`，且 `input_delay_steps=0`。主对比只回答三极点是否优于同预算二极点；二极点+learned-delay 只检查遗漏阶次是否被误读成延迟。

先核对状态、工作树、冻结矩阵和本地测试：

```bash
python experiments/phase3_5/experiment_status.py --check --json
git status --short
git rev-parse HEAD
python -m pytest tests/phase35/multistep tests/phase35/test_experiment_status.py -q
python -m compileall -q src/phase35/multistep \
  experiments/phase3_5/ms2d_order.py \
  experiments/phase3_5/summarize_ms2d_order.py
python experiments/phase3_5/ms2d_order.py --dry-run
```

本节命令只用于复现已完成批次，不再授权重复执行。原运行时状态为 `ready_for_linux`，dry-run 为 `1 regime / 7 candidates / 21 validation runs` 且 `test_authorized=false`。

正式运行与汇总：

```bash
mkdir -p results/phase3_5/ms2d_order/remote_execution
git rev-parse HEAD > results/phase3_5/ms2d_order/remote_execution/git_commit.txt
python --version > results/phase3_5/ms2d_order/remote_execution/environment.txt 2>&1
nvidia-smi >> results/phase3_5/ms2d_order/remote_execution/environment.txt 2>&1

python experiments/phase3_5/ms2d_order.py \
  --device cuda \
  --output-root results/phase3_5/ms2d_order \
  --execute-matrix --skip-existing \
  > results/phase3_5/ms2d_order/remote_execution/train_stdout.log \
  2> results/phase3_5/ms2d_order/remote_execution/train_stderr.log
echo $? > results/phase3_5/ms2d_order/remote_execution/train_exit_code.txt

python experiments/phase3_5/summarize_ms2d_order.py \
  --output-root results/phase3_5/ms2d_order \
  > results/phase3_5/ms2d_order/remote_execution/summary_stdout.log \
  2> results/phase3_5/ms2d_order/remote_execution/summary_stderr.log
echo $? > results/phase3_5/ms2d_order/remote_execution/summary_exit_code.txt
```

冻结矩阵为 7 candidates × 3 seeds：二极点消融、三极点主模型、三极点+R50 oracle、二极点+learned-delay 诊断，以及 Koopman、PI-ODE、DeepONet secondary references。主门禁逐 seed 要求：

1. 21/21 artifact 与结构门禁通过；
2. `d2_g3_oracle_structure` clean NMAE `<0.05`；
3. `d2_g3_three_pole` clean NMAE `<0.10`；
4. 三极点相对二极点 clean NMAE 改善 `≥10%`。

tau 集合的 permutation-invariant log-MAE，以及无迟延 truth 下 learned-delay 的期望步数/零步质量，只是诊断，不改变主门禁。该 validation 已由本地审计为 screening PASS；不要重复训练，也不要用 validation 排名升级 secondary 路线。

## 14. 已完成归档：MS2-D2 one-shot synthetic test

本批已完成并由本地 Supervisor 关闭，不得重复访问。以下命令只保留审计追溯：它曾对已归档的 7 candidates × 3 seeds 共 21 个 validation-selected checkpoints 做一次独立 test 推理，不训练、不调参、不补 seed。

```bash
python experiments/phase3_5/experiment_status.py --check --json
git status --short
git rev-parse HEAD
python -m pytest tests/phase35/multistep/test_ms2d_order_test.py \
  tests/phase35/test_experiment_status.py -q
python -m compileall -q src/phase35/multistep \
  experiments/phase3_5/ms2d_order_test.py \
  experiments/phase3_5/summarize_ms2d_order_test.py
python experiments/phase3_5/ms2d_order_test.py --dry-run
```

状态必须严格为 `active_gate=ms2d_d2`、`active_status=test_authorized`、`linux_authorized_gate=ms2d_d2`；工作树必须为空。dry-run 必须显示 `run_count=21`、`archive_member_count=21`、`validation_screening_pass=true`、`test_accessed=false`；tau 诊断为 true、no-true-delay 诊断为 false 是预期状态，不是 test 失败。任一 content pin、manifest、checkpoint、冻结代码或既有 test artifact 检查失败即停止，不删除产物规避重复访问锁。

一次性执行与汇总：

```bash
mkdir -p results/phase3_5/ms2d_order/remote_test
git rev-parse HEAD > results/phase3_5/ms2d_order/remote_test/git_commit.txt
python --version > results/phase3_5/ms2d_order/remote_test/environment.txt 2>&1
nvidia-smi >> results/phase3_5/ms2d_order/remote_test/environment.txt 2>&1
printf '%s\n' \
  'python experiments/phase3_5/ms2d_order_test.py --device cuda --evaluate-test-matrix --allow-synthetic-test' \
  > results/phase3_5/ms2d_order/remote_test/command.txt

python experiments/phase3_5/ms2d_order_test.py \
  --device cuda --evaluate-test-matrix --allow-synthetic-test \
  > results/phase3_5/ms2d_order/remote_test/test_stdout.log \
  2> results/phase3_5/ms2d_order/remote_test/test_stderr.log
echo $? > results/phase3_5/ms2d_order/remote_test/test_exit_code.txt
```

只有 `test_exit_code.txt` 为 0 才继续汇总：

```bash
python experiments/phase3_5/summarize_ms2d_order_test.py \
  > results/phase3_5/ms2d_order/remote_test/summary_stdout.log \
  2> results/phase3_5/ms2d_order/remote_test/summary_stderr.log
echo $? > results/phase3_5/ms2d_order/remote_test/summary_exit_code.txt
```

确认主门逐 seed为：oracle clean NMAE `<0.05`；三极点 clean NMAE `<0.10`；三极点相对二极点的配对 episode、profile 分层 10,000 次 bootstrap 95% CI 下界 `>=0.10`。tau 恢复和伪迟延仍是非阻断诊断；即使后者失败，不能据此把主门改成 FAIL。汇总器因科学门失败返回 code 2 时，仍须原样提交所有 JSON、ledger、更新后的 manifest 和日志。

Linux 只提交 `results/phase3_5/ms2d_order/**`。不得修改 `configs/`、`src/`、`experiments/`、`tests/`、TODO、README、注册表、PROJECT_STATUS、上下文快照或任何 Supervisor 文档；不得新增自审 review，不得删除 partial ledger、重复 test、重训或启动 D3。远端只报告执行与原始门禁，本地 Supervisor 负责独立复算和状态迁移。

## 15. 已完成归档：MS2-D3 colored-disturbance validation

D3 固定继承 D2 的 R50、context scheduling、三阶 `[40,70,210] s` 和无 pure delay clean truth，只在输出端加入 response operator 不可观察的 stationary AR(1) nuisance：`sigma_d=0.03 °C`、`tau_d=120 s`。这不是现场扰动谱、状态观测器或完整 `free+response` 实验。先在授权 commit 的干净工作树预检：

```bash
python experiments/phase3_5/experiment_status.py --check --json
git status --short
git rev-parse HEAD
python -m pytest tests/phase35/multistep tests/phase35/test_experiment_status.py -q
python -m compileall -q src/phase35/multistep \
  experiments/phase3_5/ms2d_disturbance.py \
  experiments/phase3_5/summarize_ms2d_disturbance.py
python experiments/phase3_5/ms2d_disturbance.py --dry-run
```

原执行时状态必须严格为 `active_gate=ms2d_d3`、`active_status=ready_for_linux`、`linux_authorized_gate=ms2d_d3`；工作树必须为空。dry-run 必须显示 `protocol_version=phase3.5-ms2d-d3-v1`、`run_count=21`、`test_authorized=false`。当前注册表已进入 MS5，不得为复现本节而回退状态。

正式训练与汇总：

```bash
mkdir -p results/phase3_5/ms2d_disturbance/remote_execution
git rev-parse HEAD > results/phase3_5/ms2d_disturbance/remote_execution/git_commit.txt
python --version > results/phase3_5/ms2d_disturbance/remote_execution/environment.txt 2>&1
nvidia-smi >> results/phase3_5/ms2d_disturbance/remote_execution/environment.txt 2>&1
printf '%s\n' \
  'python experiments/phase3_5/ms2d_disturbance.py --device cuda --output-root results/phase3_5/ms2d_disturbance --execute-matrix --skip-existing' \
  > results/phase3_5/ms2d_disturbance/remote_execution/command.txt

python experiments/phase3_5/ms2d_disturbance.py \
  --device cuda \
  --output-root results/phase3_5/ms2d_disturbance \
  --execute-matrix --skip-existing \
  > results/phase3_5/ms2d_disturbance/remote_execution/train_stdout.log \
  2> results/phase3_5/ms2d_disturbance/remote_execution/train_stderr.log
echo $? > results/phase3_5/ms2d_disturbance/remote_execution/train_exit_code.txt

python experiments/phase3_5/summarize_ms2d_disturbance.py \
  --output-root results/phase3_5/ms2d_disturbance \
  > results/phase3_5/ms2d_disturbance/remote_execution/summary_stdout.log \
  2> results/phase3_5/ms2d_disturbance/remote_execution/summary_stderr.log
echo $? > results/phase3_5/ms2d_disturbance/remote_execution/summary_exit_code.txt
```

冻结主门逐 seed为：21/21 artifact 与结构合同闭合；`d3_g3_oracle_structure` clean NMAE `<0.05`；`d3_g3_three_pole` clean NMAE `<0.10`；三阶相对 `d3_g2_two_pole` 的 profile-stratified paired episode bootstrap 95% CI 下界 `>=0.10`。扰动 realization、tau、no-true-delay、profile/horizon、D2→D3 drift 和 secondary 路线全部是非阻断诊断。汇总器因科学门失败返回 code 2 时，也必须原样提交全部 21 个运行目录、`summary_validation.json`、checkpoint archive 和日志。

本批已完成并以 `VALIDATION_STRESS_PASS / NO_TEST_BY_BUDGET_DECISION` 关闭。以上命令只保留追溯，不再授权重复训练或 synthetic test。权威判决见 `docs/PHASE35_MS2D3_SUPERVISOR_AUDIT_2026-08-11.md`。

## 16. 已完成归档：MS5 full free+response coupling validation

MS5 只回答 total-only supervision 下 `free` 分支是否吸收动作响应，以及 joint 或短阶段 staged 哪个满足冻结资格门。它不访问 A/B，不比较 Koopman/PI-ODE/DeepONet/Fan 路线，也不启动 MS3。正式执行前在授权 commit 的干净工作树运行：

```bash
python experiments/phase3_5/experiment_status.py --check --json
git status --short
git rev-parse HEAD
python -m pytest tests/phase35/multistep tests/phase35/test_experiment_status.py -q
python -m compileall -q src/phase35/multistep \
  experiments/phase3_5/ms5_full_coupling.py \
  experiments/phase3_5/summarize_ms5_full_coupling.py
python experiments/phase3_5/ms5_full_coupling.py --dry-run
```

状态必须严格为 `active_gate=ms5`、`active_status=ready_for_linux`、`linux_authorized_gate=ms5`；工作树必须为空。dry-run 必须显示 `protocol_version=phase3.5-ms5-v1`、`run_count=12`、四个冻结 mode、`test_authorized=false`。任一 D3 content pin、矩阵、源码、既有 test artifact 或状态检查失败即停止，不在远端修复。

正式训练与汇总：

```bash
mkdir -p results/phase3_5/ms5_full_coupling/remote_execution
git rev-parse HEAD > results/phase3_5/ms5_full_coupling/remote_execution/git_commit.txt
python --version > results/phase3_5/ms5_full_coupling/remote_execution/environment.txt 2>&1
nvidia-smi >> results/phase3_5/ms5_full_coupling/remote_execution/environment.txt 2>&1
printf '%s\n' \
  'python experiments/phase3_5/ms5_full_coupling.py --device cuda --output-root results/phase3_5/ms5_full_coupling --execute-matrix --skip-existing' \
  > results/phase3_5/ms5_full_coupling/remote_execution/command.txt

python experiments/phase3_5/ms5_full_coupling.py \
  --device cuda \
  --output-root results/phase3_5/ms5_full_coupling \
  --execute-matrix --skip-existing \
  > results/phase3_5/ms5_full_coupling/remote_execution/train_stdout.log \
  2> results/phase3_5/ms5_full_coupling/remote_execution/train_stderr.log
echo $? > results/phase3_5/ms5_full_coupling/remote_execution/train_exit_code.txt

python experiments/phase3_5/summarize_ms5_full_coupling.py \
  --output-root results/phase3_5/ms5_full_coupling \
  > results/phase3_5/ms5_full_coupling/remote_execution/summary_stdout.log \
  2> results/phase3_5/ms5_full_coupling/remote_execution/summary_stderr.log
echo $? > results/phase3_5/ms5_full_coupling/remote_execution/summary_exit_code.txt
```

汇总器以 code 2 退出表示科学门失败，不表示执行产物无效，仍须原样回传。冻结判决顺序为：component-oracle 正控必须通过；joint 全过就选 joint；joint 失败时只有 staged 全过且 staged/joint total error ratio 每 seed `<=1.10` 才选 staged；否则 fail closed。free-only 是 prediction-only 负控，不参与选模。

Linux 只提交 `results/phase3_5/ms5_full_coupling/**`。不得修改 `configs/`、`src/`、`experiments/`、`tests/`、`docs/`、TODO、README 或注册表；不得改阈值、阶段、seed、样本量，不补超参数扫描，不删除失败/partial run，不访问 synthetic test/A/B，不启动 MS3，也不得改写 summary 的 archive path/hash。本地 Supervisor 负责独立复算与状态迁移。

该批现已由本地权重级重放，以 `CLOSED / VALIDATION_ONLY_COMPONENT_RECOVERY_PASS / JOINT_SELECTED / STAGED_PROTOCOL_REJECTED` 关闭，不再授权重复运行。权威判决见 `docs/PHASE35_MS5_SUPERVISOR_AUDIT_2026-08-11.md`。

## 17. 当前唯一授权：MS3 A/B observational validation

MS3 只把 MS5 选中的 joint 三极点架构迁移到真实 A/B 交叉控制回路。它验证条件预测、动作分支非坍缩和 logged-action 时间对齐，不验证 `do(valve)`；test 继续禁止。数据源必须是冻结的 `all_merged_10s.csv`：

```bash
export PH35_ALL_MERGED=/data/yimin/all_merged_10s.csv
export PH35_MS3_CACHE_A=/data/thermal-world-model/phase3_5/ms3_cross_A.npz
export PH35_MS3_CACHE_B=/data/thermal-world-model/phase3_5/ms3_cross_B.npz
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

git pull --ff-only origin main
python experiments/phase3_5/experiment_status.py --check --json
git status --short
git rev-parse HEAD
python -m pytest tests/phase35 -q
python -m compileall -q src/phase35 experiments/phase3_5
python experiments/phase3_5/ms3_real_adaptation.py --dry-run
```

状态必须为 `active_gate=ms3`、`active_status=ready_for_linux`、`linux_authorized_gate=ms3`；工作树必须为空；dry-run 必须为 2 candidates×A/B×3 seeds=`12`、`test_authorized=false`。任一红项立即停止。

先一次扫描 4 GB 源文件，生成写死交叉配对的两个 cache：

```bash
mkdir -p results/phase3_5/ms3_real_adaptation/remote_execution

python experiments/phase3_5/prepare_ms3_cross_data.py \
  --input "$PH35_ALL_MERGED" \
  --output-a "$PH35_MS3_CACHE_A" \
  --output-b "$PH35_MS3_CACHE_B" \
  > results/phase3_5/ms3_real_adaptation/remote_execution/cache_stdout.log \
  2> results/phase3_5/ms3_real_adaptation/remote_execution/cache_stderr.log
echo $? > results/phase3_5/ms3_real_adaptation/remote_execution/cache_exit_code.txt
cp "${PH35_MS3_CACHE_A%.npz}.manifest.json" \
  results/phase3_5/ms3_real_adaptation/cache_A.manifest.json
cp "${PH35_MS3_CACHE_B%.npz}.manifest.json" \
  results/phase3_5/ms3_real_adaptation/cache_B.manifest.json
```

cache manifest 必须显示 source SHA `85a3f926...e4da6`、A=`A_valve_to_right_B_thermal_train`、B=`B_valve_to_left_A_thermal_train`。随后正式训练和汇总：

`phase3.5-ms3-v1.1` 还要求两个 manifest 同时显示：`timestamp_storage_unit=ns`、`grid_rows=1192329`、`grid_start_ns=1766541870000000000`、`grid_end_ns=1778543960000000000`、`irregular_transition_count=282`、`max_transition_seconds=75750.0`。任一项不符不得启动训练。旧 `v1` cache 会被 runner 明确拒绝；重新执行 builder 会覆盖 cache 和 manifest，不需要也不允许复用失败 cache。

```bash
git rev-parse HEAD > results/phase3_5/ms3_real_adaptation/remote_execution/git_commit.txt
python --version > results/phase3_5/ms3_real_adaptation/remote_execution/environment.txt 2>&1
python -c "import numpy,pandas,torch; print(numpy.__version__, pandas.__version__, torch.__version__, torch.cuda.is_available())" >> results/phase3_5/ms3_real_adaptation/remote_execution/environment.txt 2>&1
nvidia-smi >> results/phase3_5/ms3_real_adaptation/remote_execution/environment.txt 2>&1
printf '%s\n' \
  'python experiments/phase3_5/ms3_real_adaptation.py --cache-a "$PH35_MS3_CACHE_A" --cache-b "$PH35_MS3_CACHE_B" --device cuda --output-root results/phase3_5/ms3_real_adaptation --execute-matrix --skip-existing' \
  > results/phase3_5/ms3_real_adaptation/remote_execution/command.txt

python experiments/phase3_5/ms3_real_adaptation.py \
  --cache-a "$PH35_MS3_CACHE_A" --cache-b "$PH35_MS3_CACHE_B" \
  --device cuda --output-root results/phase3_5/ms3_real_adaptation \
  --execute-matrix --skip-existing \
  > results/phase3_5/ms3_real_adaptation/remote_execution/train_stdout.log \
  2> results/phase3_5/ms3_real_adaptation/remote_execution/train_stderr.log
echo $? > results/phase3_5/ms3_real_adaptation/remote_execution/train_exit_code.txt

python experiments/phase3_5/summarize_ms3_real_adaptation.py \
  --output-root results/phase3_5/ms3_real_adaptation \
  > results/phase3_5/ms3_real_adaptation/remote_execution/summary_stdout.log \
  2> results/phase3_5/ms3_real_adaptation/remote_execution/summary_stderr.log
echo $? > results/phase3_5/ms3_real_adaptation/remote_execution/summary_exit_code.txt
```

summary exit 2 表示冻结科学门未通过，不是无效运行；12 个目录、summary、archive、两个 cache manifest 和全部日志仍须原样提交。Linux 只提交 `results/phase3_5/ms3_real_adaptation/**` 以及其中复制的 cache manifests；不提交 cache `.npz`，不改代码/配置/文档/状态，不补跑、不改阈值、不访问 test、不启动 MS4。
