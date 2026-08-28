# Final World Model Credibility Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** 用最小改动让世界模型的 validation 判决与冻结协议一致、统计口径可靠、数据/权重/结果可唯一追溯，并在 corrected canonical v2.1 上重发可供论文审计的完整结果。

**Architecture:** 不改世界模型主体架构。先让所有正式 verdict fail-closed，再依次修正时序、反事实支持域、D-SYN、数据质量、验证选择和运行身份；每项均以一个能在旧代码失败的小测试开场。全部本地测试和审计通过后才冻结 Linux 命令，Linux 只执行并回传，最终由本地独立复算。

**Tech Stack:** Python、PyTorch、NumPy、pytest、JSON/JSONL、Git SHA256；不增加第三方依赖。

---

> 状态：`TASK 1–2 LOCAL VERIFIED / TASK 3 NEXT / HOLD / TEST LOCKED`
> 依据：`docs/FINAL_WM_CREDIBILITY_AUDIT_2026-08-27.md`
> 执行原则：一次只完成一个任务；测试通过、人工核对 diff 后再进入下一项；不自动启动 Linux、test 或论文改写。

## Task 1: 冻结 v0.7 可执行判决合同并 fail-closed

> 完成记录（2026-08-28）：v0.7 required-evidence 合同、协议指标和 fail-closed tier 已落地；
> 复审修正了条件稳定性初态/工况口径与 pandas 兼容点，全量回归 `154 passed`。
> 未启动训练、Linux 或 test split。复审稿见 `docs/FINAL_WM_TASK1_POST_AUDIT_2026-08-28.md`。

**Files:**

- Modify: `experiments/final_wm/matrix_spec.py`
- Modify: `experiments/final_wm/run_matrix.py`
- Modify: `src/final_wm/evaluation.py`
- Modify: `tests/final_wm/test_matrix_smoke.py`
- Modify: `tests/final_wm/test_evaluation.py`
- Modify: `docs/plans/2026-08-18-final-wm-discrimination-matrix.md`

**Step 1: 写失败测试**

- 断言 O1 缺 continuity、B1 缺 H6/H18/H36 或 downstream degradation、T1 缺 60-step stability、J1 缺 H36 stability、R1 缺 H18/H60+CI 时，正式 verdict 必须为 `INCOMPLETE`；
- 断言 partial seeds、quick 或 arm-filter 不能写 `SUPPORTED/REJECTED`；
- 断言矩阵版本与 summary/manifest 一致。

**Step 2: 运行定向测试，确认旧实现失败**

Run:

```powershell
python -m pytest tests/final_wm/test_matrix_smoke.py tests/final_wm/test_evaluation.py -q
```

Expected: 新增的缺证据和 partial-seed 用例失败。

**Step 3: 最小实现**

- 将新冻结版本登记为 v0.7，不改写 v0.2–v0.6 历史；
- 在 `matrix_spec.py` 为每个单元列出机器可读的 required evidence；
- runner 只有在完整固定 seeds 和 required evidence 全部存在时才调用 verdict；
- 复用已有 `state_continuity`、boundary metrics、constant-condition/step-response 工具补齐指标；缺少实现时输出 `INCOMPLETE`，不得先给方向性判决；
- R1 同报 valve1/valve2、H18/H60、均值、按 UTC-day block bootstrap 的 95% CI 和正确方向占比。

**Step 4: 运行测试**

Run:

```powershell
python -m pytest tests/final_wm/test_matrix_smoke.py tests/final_wm/test_evaluation.py -q
```

Expected: PASS；quick/partial 只产生 `SMOKE/INCOMPLETE`。

**Step 5: 人工验收与提交**

- 对照矩阵文档逐项核对 runner 输出键；
- 不运行训练；
- Commit: `fix(final-wm): fail closed on incomplete matrix evidence`。

## Task 2: 修正 NLL 正式判决统计口径

> 完成记录（2026-08-28）：O1/T1/J1 正式门已改为同窗口、按 UTC 日聚合的
> `ΔNLL = arm - baseline` bootstrap CI；J1 改为同窗口复评，CRPS/MAE 仅报告效应量；
> 定向回归 `21 passed`。未启动训练、Linux 或 test split。

**Files:**

- Modify: `src/final_wm/evaluation.py`
- Modify: `experiments/final_wm/matrix_spec.py`
- Modify: `experiments/final_wm/run_matrix.py`
- Modify: `tests/final_wm/test_evaluation.py`
- Modify: `docs/plans/2026-08-18-final-wm-discrimination-matrix.md`

**Step 1: 写失败测试**

- 构造相同两模型的 NLL，在整体加常数或改变等价单位后，paired `ΔNLL` 判决保持不变；
- 构造负或近零 baseline NLL，确认不发生 `clamp_min(1e-9)` 导致的巨大百分比；
- CRPS/MAE 相对改善仍可按正尺度报告。

**Step 2: 运行并确认旧实现失败**

Run:

```powershell
python -m pytest tests/final_wm/test_evaluation.py -q
```

Expected: NLL 百分比不变性用例失败。

**Step 3: 最小实现**

- 训练损失不动；
- 正式 NLL 门改成逐 UTC day 的 `ΔNLL = arm - baseline`，bootstrap 95% CI 上界 `< 0` 才支持改善；
- 删除 NLL 百分比阈值的判决作用，保留 CRPS/MAE 百分比作为可解释的实用幅度；
- summary 同时报绝对 `ΔNLL`、CI、日块数、CRPS/MAE effect size。

**Step 4: 运行测试与提交**

Run:

```powershell
python -m pytest tests/final_wm/test_evaluation.py tests/final_wm/test_matrix_smoke.py -q
```

Expected: PASS。
Commit: `fix(final-wm): use paired delta nll for verdicts`。

## Task 3: 修正泄漏探针与 CF replay 的时序语义

**Files:**

- Modify: `src/final_wm/diagnostics.py`
- Modify: `src/final_wm/evaluation.py`
- Modify: `tests/final_wm/test_cf_probes.py`
- Modify: `tests/final_wm/test_evaluation.py`

**Step 1: 写失败测试**

- 用确定性 toy transition 给出手算的两步状态和 observation；断言 leakage probe 的 residual、feature、action 都对齐同一 `t+1`；
- 用 history 首末 observation 不同的 teacher，断言 replay 初态只使用共时的 boundary/action/observation；
- identity replay 的 baseline 与 counterfactual delta 在 teacher==student 时逐位一致。

**Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/final_wm/test_cf_probes.py tests/final_wm/test_evaluation.py -q
```

Expected: 新时序断言在旧实现失败。

**Step 3: 最小实现**

- 明确统一语义为 `state_t + boundary_t + action_t -> state_{t+1} -> observation_{t+1}`；
- leakage probe 要么只预测 step 0 对应 observation，要么完成第二次 transition 后再对齐 step 1；不保留半步混合；
- replay 初态使用 history 第一个共时三元组，随后只重放其后的动作/边界，避免重复消费起点。

**Step 4: 回归与提交**

Run:

```powershell
python -m pytest tests/final_wm/test_cf_probes.py tests/final_wm/test_evaluation.py -q
```

Expected: PASS。
Commit: `fix(final-wm): align leakage and replay timesteps`。

## Task 4: 修正逐样本反事实支持域并统一正式 CF 路径

**Files:**

- Modify: `src/final_wm/contracts.py`
- Modify: `src/final_wm/model.py`
- Modify: `src/final_wm/evaluation.py`
- Modify: `tests/final_wm/test_contracts.py`
- Modify: `tests/final_wm/test_model.py`
- Modify: `tests/final_wm/test_cf_probes.py`

**Step 1: 写失败测试**

- 两个 batch 样本具有不相交动作范围，样本 A 不得借用样本 B 的范围；
- CPU/CUDA（可用时）support tensor 与 action 保持同 device；
- 正式 CF probe 离开支持域时 fail-closed，显式允许外推时必须报告逐步 mask 与支持率。

**Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/final_wm/test_contracts.py tests/final_wm/test_model.py tests/final_wm/test_cf_probes.py -q
```

Expected: batch 隔离与正式 probe 门禁用例失败。

**Step 3: 最小实现**

- `ActionSupport` 保存 `(B, 2)` lo/hi tensor，不压平 batch；
- `contains()` 使用输入 action 的 dtype/device；
- `step_response_direction`、CF-1 和其他正式动作替换统一调用 `model.counterfactual()`；
- summary 强制写 `support_rate`、`n_unsupported` 和 `allow_extrapolation`。

**Step 4: 回归与提交**

Run:

```powershell
python -m pytest tests/final_wm/test_contracts.py tests/final_wm/test_model.py tests/final_wm/test_cf_probes.py -q
```

Expected: PASS。
Commit: `fix(final-wm): enforce per-window counterfactual support`。

## Task 5: 让 D-SYN 真实扰动 teacher

**Files:**

- Modify: `experiments/final_wm/run_matrix.py`
- Modify: `tests/final_wm/test_matrix_smoke.py`

**Step 1: 写失败测试**

- 运行 quick D-SYN 前后比较 teacher transition parameters；
- 断言扰动参数数目大于零、raw 参数距离大于零，且报告中保存数量和范数。

**Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/final_wm/test_matrix_smoke.py::test_dsyn_quick_gate_runs -q
```

Expected: 旧实现的扰动数断言失败。

**Step 3: 最小实现**

- 匹配真实参数名 `raw.`，或直接遍历 `teacher.transition.raw.parameters()`；
- no-op 立即抛 `FinalWMProtocolError`；
- D-SYN 结果增加 `n_perturbed`、`parameter_delta_l2`，其余预算和 teacher/student 结构不变。

**Step 4: 回归与提交**

Run:

```powershell
python -m pytest tests/final_wm/test_matrix_smoke.py -q
```

Expected: PASS；quick 仍只标 `SMOKE`。
Commit: `fix(final-wm): make dsyn teacher perturbation observable`。

## Task 6: 修正 canonical v2 原始质量门

**Files:**

- Modify: `src/final_wm/data_v2.py`
- Modify: `tests/final_wm/test_data_v2.py`

**Step 1: 写失败测试**

- 使用生产式 `clip == range` mapping 注入超范围原始值，构建必须 fail-closed；
- 派生通道源数据缺失时，coverage 必须反映原始源缺失而不是填零后的有限性；
- 合法输入构建值保持与旧实现逐位一致。

**Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/final_wm/test_data_v2.py -q
```

Expected: 超范围与源 coverage 用例失败。

**Step 3: 最小实现**

- 对 raw values 先统计 finite coverage 和 range violation；
- 门通过后才 clip/填充为模型输入；
- meta 同时保存 raw quality 与 postprocess 摘要；
- 不改通道定义、接线、split 或现有容忍阈值。

**Step 4: 回归、重建验证与提交**

Run:

```powershell
python -m pytest tests/final_wm/test_data_v2.py tests/final_wm/test_data.py -q
```

Expected: PASS；合法 fixture 数组逐位一致。
Commit: `fix(final-wm): gate v2 channels before clipping`。

## Task 7: 固定 validation anchors 并内容寻址正式 run

**Files:**

- Modify: `src/final_wm/training.py`
- Modify: `experiments/final_wm/run_matrix.py`
- Modify: `tests/final_wm/test_training.py`
- Modify: `tests/final_wm/test_matrix_smoke.py`
- Create: `experiments/final_wm/audit_manifest.py`

**Step 1: 写失败测试**

- 记录连续 epoch 的 validation window indices，断言完全相同；
- 修改 record、properties 或 anchor checkpoint 任一字节，fingerprint 必须变化；
- dirty worktree、quick、partial seeds 不得生成 authoritative manifest；
- quick/full 的 checkpoint、metrics、ledger 路径不得相同；
- manifest 缺任一 unit/seed/hash 时审计器必须失败。

**Step 2: 运行并确认失败**

Run:

```powershell
python -m pytest tests/final_wm/test_training.py tests/final_wm/test_matrix_smoke.py -q
```

Expected: 固定 anchors、内容哈希和命名空间用例失败。

**Step 3: 最小实现**

- 每个 run 用固定 seed 生成一次 validation anchors，并在所有 epoch 复用；
- fingerprint 加入 canonical、properties、init/anchor checkpoint 的 SHA256；
- authoritative run 要求 clean commit、完整固定 seeds 和 `quick=false`；
- quick/partial 使用独立目录，状态固定为 `SMOKE/INCOMPLETE`；
- 生成一个 JSON manifest 绑定 commit、命令、矩阵版本、数据/物性/权重/metrics 哈希和 unit 完整性；
- 审计脚本只做哈希与 summary 重算，不引入数据库。

**Step 4: 回归与提交**

Run:

```powershell
python -m pytest tests/final_wm/test_training.py tests/final_wm/test_matrix_smoke.py -q
```

Expected: PASS。
Commit: `fix(final-wm): bind validation runs to immutable inputs`。

## Task 8: 全量本地验证并冻结 Linux 重发命令

**Files:**

- Modify: `experiments/final_wm/README.md`
- Create: `results/final_wm/v07_reissue_runbook_20260827.md`
- Modify: `configs/phase3_5/experiment_registry.json`

**Step 1: 本地定向回归**

Run:

```powershell
python -m pytest tests/final_wm -q
```

Expected: 全部 PASS；测试总数从 pytest 实际输出登记，不手填旧数字。

**Step 2: quick smoke**

Run 使用 README 中 v0.7 quick 命令。
Expected: 所有单元可执行；产物只进入 quick 目录；没有正式 verdict。

**Step 3: 静态证据审计**

- JSON/JSONL 全部可解析；
- runner 的 required evidence 与矩阵文档逐项相等；
- test split 仍拒绝采样；
- Git diff 仅包含本计划列出的文件。

**Step 4: 冻结 Linux 命令**

- runbook 固定 clean commit、corrected v2.1 record SHA256、properties SHA256、完整 seeds、预算、输出目录和回传清单；
- 注册表改为 `ready_for_linux=true` 前，必须由本地人工复核 runbook；
- Linux 只执行，不修改代码、阈值、seed、数据或命令。

**Step 5: 提交**

Commit: `docs(final-wm): freeze v07 credibility reissue runbook`。

## Task 9: corrected v2.1 正式重发与独立审计

**Files:**

- Create: `results/final_wm/v07_reissue_audit_YYYYMMDD.md`
- Create: `artifacts/final_wm_v07/<side>/manifest.json`（由 runner 生成）
- Modify: `configs/phase3_5/experiment_registry.json`
- Modify only after audit: 论文结果引用文件

**Step 1: Linux 执行**

- 仅在注册表 `ready_for_linux=true` 且用户授权后执行冻结命令；
- validation only、完整固定 seeds；test 保持锁定。

**Step 2: 回传完整证据包**

Expected: manifest、ledger、metrics、summary、checkpoint 哈希和运行环境摘要齐全；大文件可带外保存，但内容哈希必须匹配。

**Step 3: 本地独立复算**

Run:

```powershell
python experiments/final_wm/audit_manifest.py --manifest artifacts/final_wm_v07/sideA/manifest.json
```

Expected: 所有哈希、seed 完整性、required evidence 和 verdict 重算一致；任何缺项均 fail-closed。

**Step 4: 人工审计与状态更新**

- 审计稿逐单元给出 `SUPPORTED/MIXED/REJECTED/INCOMPLETE` 及证据键；
- 注册表依次登记 `results_returned=true`、`audited=true`；
- 只有用户另行决定后才解冻论文或授权 test。

**Step 5: 提交**

Commit: `audit(final-wm): adjudicate v07 corrected-record reissue`。

## 明确排除项

本计划不包含：新模型架构、LPV 继续调参、新 controller、数据扩列、test 访问、论文自动收口、外部 checkpoint 安全改造、Git LFS、依赖管理重构或 README 全面整理。只有某项最小修复的回归测试证明这些排除项成为阻塞时，才另行立项。
