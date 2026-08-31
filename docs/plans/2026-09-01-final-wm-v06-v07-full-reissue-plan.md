# Final WM v0.6 / v0.7 Full Reissue Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** 在不改世界模型主体架构和科学判据的前提下，把 v0.6 数据/训练底座与 v0.7 可信度合同合并成一套可由 Linux 原样执行、可本地独立复算的双侧全量重发包。

**Architecture:** canonical v2.2 继续提供 7 通道正式 base view；runner 对 D-SYN、O1、T1、B1、J1、R1 统一使用 120/20、固定 validation anchors、内容寻址输入和分层输出。完整执行结束后由纯读取 manifest 模块生成并验证 SHA-256 证据包；任何不完整条件 fail-closed。

**Tech Stack:** Python 3、PyTorch、NumPy、pytest、JSON/JSONL、Git、SHA-256；不增加第三方依赖。

---

### Step 1: Task 5 D-SYN teacher 扰动可观测性

- Output: D-SYN 报告含 `n_perturbed`、`parameter_delta_l2`，no-op 抛 `FinalWMProtocolError`。
- Test: `python -m pytest tests/final_wm/test_matrix_smoke.py -q`。

### Step 2: Task 6 canonical v2 raw quality gate

- Output: raw/source coverage 与原始 range violation 先于 clip/fill 判定；meta 同时报 raw 与 postprocess 摘要。
- Test: `python -m pytest tests/final_wm/test_data_v2.py tests/final_wm/test_data.py -q`。

### Step 3: Task 7 固定 validation anchors 与内容身份

- Output: 每 run 固定 validation seed；fingerprint 绑定 record、properties、init/anchor checkpoint；quick/full tier 隔离。
- Test: `python -m pytest tests/final_wm/test_training.py tests/final_wm/test_matrix_smoke.py -q`。

### Step 4: authoritative manifest 与独立审计器

- Output: `experiments/final_wm/audit_manifest.py` 可生成/验证完整 manifest；非 clean、partial、quick、缺文件均拒绝。
- Test: manifest 哈希篡改、缺 unit/seed、dirty/quick/partial 负例均通过。

### Step 5: 冻结最终矩阵与 Linux runbook

- Output: 所有正式训练臂显式 120/20；R1 默认 `closure_cons_norew`；双侧完整命令、输入指纹占位检查和回传清单入库。
- Test: matrix spec、README/runbook/registry 的版本、units、seeds、输出目录和 test lock 一致。

### Step 6: 本地验证、状态登记与推送

- Output: Task 5–8 审计稿、README、TODO、registry 更新；本地 quick 只写 smoke；提交并 push 到 `origin/main`。
- Test: `python -m pytest tests/final_wm -q`、`python -m compileall -q src/final_wm experiments/final_wm`、JSON 全解析、`git diff --check`。
