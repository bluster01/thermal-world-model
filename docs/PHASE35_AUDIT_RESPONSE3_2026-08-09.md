# Phase 3.5 审计回应③：E3 匹配协议修复 + 全量重跑（2026-08-09）

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
| A | 93 | 0.302 | 1.42 | **0.323** | 11 | 0.045 | **FAIL** |
| B | 122 | 0.710 | 3.35 | **0.057** | 12 | -0.034 | **FAIL** |

### 判定解读

- **A 侧 FAIL 原因**：方向率 0.323 < 0.60 门槛。协议合格
  （pretrend 0.045 ✅，SMD 0.30 略超 0.20 但为 temp_error 物理不可平衡——
  阀位事件由温度偏差内生触发，quiet controls 无此偏差）。
- **B 侧双重 FAIL**：方向率 0.057（≈零，无方向性响应）+ SMD 0.71
  （匹配不平衡）。
- **门禁连锁**：E3 FAIL → E4 BLOCKED → 无候选进入 seed 3/4 或 test。

### 与 SP 事件稳态层对比（论文叙事关键）

| 事件通道 | 方向率 | 性质 |
|---|---|---|
| SP 事件（运行人员干预） | **75-83%**（60sV∩180sV 交集） | 相对外生，方向干净 |
| 阀位事件（PID 闭环输出） | 0.06-0.32 | 内生，方向不可识别 |

结论：观测数据上只有"运行人员 SP 干预"方向可识别；PID 闭环阀位动作
方向性极弱（B 侧≈0）。模型干预分支（G3 曾显示 R50 非平凡 gain）应对应
SP 干预通道，而非阀位通道。这与"SP 与阀门不可分，唯一离线可验证动作
通道=阀位"的历史结论一致——但 E3 现在给出的是协议合格后的可信证据。

## 产物

- `src/phase35/events.py`：caliper + 压力协变量 + reuse 审计
- `experiments/phase3_5/evaluate.py`：`--caliper-quantile` 参数
- `results/phase3_5/runs/*/event_metrics_validation.json`：42 run 全量重跑
- 本次未动 test split（仍锁定）
