# Phase 3.5 审计回应③：E3 匹配协议修复 + 全量重跑（2026-08-09）

> **Supervisor 复核：本文件保留为 Linux 回传记录，不再作为最终判决。** caliper 后 A=93/93
> 开阀、B=121 开/1 关，且 SMD=0.302/0.709，未满足双向 common support 或 balance；
> E3 应为 INCONCLUSIVE，E4 BLOCKED。详见 `PHASE3_5_LINUX_RETURN_AUDIT_2026-08-09.md`。

## 回应审计 §10-2~4（validation evaluation 修复）

### 已实现修复

1. **caliper 匹配**（§10-2）：`match_quiet_controls` 增加 scaled-distance 上限
   （`caliper_quantile`，取全部候选距离的分位数）。超过 caliper 的事件
   **真正 unmatched**，不再"总能选到 5 个 control"。默认 0.5，扫描 0.01-0.3。
2. **主蒸汽压力入匹配协变量**（§10-2）：`MATCHING_COVARIATES` 增加
   `main_pressure`，`_pretreatment_covariates` 同步。
3. **control 复用审计**（§10-3）：`matching_diagnostics` 输出
   `n_unique_controls`、`control_reuse_ratio`（总使用次数/唯一 control 数）。

### 修复效果（A_absolute_identity_s0 冒烟）

| 指标 | 修复前 | 修复后 (q=0.02) |
|---|---|---|
| matched events | 1000（全匹配，假 matched） | 93（caliper 内真实匹配） |
| max_abs_SMD | 2.03 | 0.302 |
| control reuse | ~650 次复用 | 1.42 |
| unique controls | ~155 | 327 |

测试：`tests/phase35/test_events.py` 4/4 通过。

### caliper 选择

q=0.02 为最终参数：matched=93（余量足），SMD=0.302（最低），
reuse=1.42。更严（0.01）不改善 SMD 反而损失事件。

## 全量重跑结果（42 runs，validation，caliper=0.02）

42/42 完成，0 失败。事件层指标不依赖模型配置（7 配置×3 seed 一致），
按侧汇总：

| 侧 | matched | max_SMD | reuse | 方向率 | 日块 | pretrend | E3 判定 |
|---|---|---|---|---|---|---|---|
| A | 93（93开/0关） | 0.302 | 1.42 | **0.323** | 11 | 0.045 | **INCONCLUSIVE** |
| B | 122（121开/1关） | 0.710 | 3.35 | **0.057** | 12 | -0.034 | **INCONCLUSIVE** |

### 判定解读

- **A 侧 INCONCLUSIVE**：没有关阀 common support，且 SMD 0.302 超过 0.20。
- **B 侧 INCONCLUSIVE**：仅 1 个关阀 matched event，且 SMD 0.709；方向率不能外推。
- **门禁连锁**：E3 INCONCLUSIVE → E4 BLOCKED → 无候选进入 seed 3/4 或 test。

### 与 SP 事件稳态层对比（论文叙事关键）

| 事件通道 | 方向率 | 性质 |
|---|---|---|
| SP 事件（运行人员干预） | **75-83%**（结果后分层的探索值） | 闭环观测，不能当工具变量 |
| 阀位事件（PID 闭环输出） | 0.06-0.32 | 内生，方向不可识别 |

结论：本批结果只能说明当前阀位 matching 缺乏双向 common support。SP 与阀位是不同层级，
模型 action 仍是实际阀位，不能把阀位模型支路改称 SP 干预支路；E3 仍不合格。

## 产物

- `src/phase35/events.py`：caliper + 压力协变量 + reuse 审计
- `experiments/phase3_5/evaluate.py`：`--caliper-quantile` 参数
- `results/phase3_5/runs/*/event_metrics_validation.json`：42 run 全量重跑
- 本次未动 test split（仍锁定）
