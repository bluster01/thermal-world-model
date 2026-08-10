# Phase 3.5-MS2-J Checkpoint & Parameter Health Audit（2026-08-10）

> 本地审计（Linux），为 MS2-J synthetic test 授权提供前置证据。审计对象：27-run validation 的 checkpoint、阶段日志、参数健康。

## 1. 审计范围与方法

- 审计 27 个 `checkpoint_best_val.pt` + 9 个 staged `checkpoint_stage_{a,b,c}.pt`（共 36 个）
- 逐文件加载（torch.load CPU）+ 逐参数 NaN/Inf 检查 + 参数语义诊断
- 对照 MS2-J 设计文档 §5："test 必须在本地审计 checkpoint、阶段日志和参数健康后另行授权"

## 2. 审计结果

### 2.1 Checkpoint 完整性与可加载性 ✅

| 项目 | 结果 |
|---|---|
| best_val checkpoint | 27/27 存在，全部可加载，结构含 model_state_dict |
| staged A/B/C checkpoint | 9/9 存在（3 runs × 3 阶段），全部可加载 |
| 参数 NaN/Inf | 27 runs 全部参数有限 ✓（零例外） |
| 归档一致性 | tar 内 36/36 与 manifest SHA 匹配（前置验证） |

### 2.2 Staged 阶段日志健康 ✅（训练无故障）

| seed | Stage A | Stage B | Stage C | 总 best_epoch |
|---|---|---|---|---|
| 0 | 120/120, best 118 | 90/90, best 90 | 90/90, best 88 | 298 |
| 1 | 120/120, best 118 | 90/90, best 89 | **38/90**（early stop, best 18） | 228 |
| 2 | 120/120, best 100 | 90/90, best 90 | 90/90, best 89 | 299 |

- 每阶段 optimizer updates 正常（A: 1920, B: 1440, C: 1440 或 early-stop 608）
- Stage A→C 的 effect MAE：0.020→0.016（s0）、0.019→0.016（s1）、0.019→0.016（s2），每阶段单调改善
- seed=1 的 Stage C 38/90 提前停属**正常 early stopping**（patience 触发），非故障
- 阶段边界：Stage B 从 A 的 best 状态继续（boundary=0.0200 == A best），Stage C 从 B best 继续——状态续传正确

### 2.3 参数语义诊断（主模型 vs 对照）

| 参数 | joint 主模型 | staged 主模型 | oracle 正对照 | 解读 |
|---|---|---|---|---|
| raw_gain | -3.253 | -3.240 | **-2.288** | exp(-2.288)=0.101 ≈ 真值 K0=0.10 ✓ oracle 恢复真值增益 |
| opening 斜率 mean | 0.285 | 0.273 | —（真值 R50 注入） | 学习映射与真值不同（K/φ 补偿不可辨识，同 MS2-V 结论） |
| gain_schedule std | 0.324 | 0.287 | 0.324 | joint 与 oracle 调度复杂度一致 |
| tau_schedule std | 0.256 | 0.203 | 0.239 | 同上 |

关键结论：
- **oracle 恢复真值 K0**（exp(-2.288)=0.101 vs 真值 0.10）→ 优化链与数据生成可解性在联合 regime 再次确认
- **joint 与 staged 学到的参数在同一解区域**（raw_gain/opening/schedule 分布接近）→ staged 的差距不是"学到不同解"，而是同一解的精度上限更低（微调阶段无法进一步逼近）
- 学习模型的 K/φ 补偿不可辨识（与 MS2-V 结论一致）：open 映射斜率 0.28 与真值 R50 不同，靠 gain 补偿——这是已知的标识性边界，不构成故障

## 3. 审计判决

**MS2-J validation 产物可信，无训练故障、无参数异常。** 联合模块门禁 PASS、staged 门禁 FAIL 均为真实科学结论：

1. joint-from-scratch 是双模块联合训练的正确方案（staged 不优于它，3 seeds 一致）
2. staged 训练稳定（阶段完整、单调改善、early stop 正常），作为"稳定性对照"成立——论文表述为对照而非主方案
3. oracle 恢复真值增益，优化链可解性第三重确认（MS1→MS2-V/C→MS2-J）

## 4. Synthetic test 授权建议

**建议批准 MS2-J synthetic test 单次访问**，条件（沿用 MS2 协议）：
1. test 与 validation 同生成器、不同 seed；单次访问 + ledger；拒绝重复/部分访问
2. 主对比在 test split 上重算联合模块门禁（≥20% CI 判定）与 staged non-inferiority
3. 9 candidates × 3 seeds 全部署（oracle 作 benchmark 复核）
4. 冻结执行路径代码等价校验（git diff 训练 commit vs HEAD）
5. staged 门禁已 FAIL——test 阶段**不改变预注册判定**，只报告 test 数值是否复现 validation 结论

执行点：Linux（本机）在冻结 commit 上执行；test 授权由 Codex 在远端部署后我 pull 执行。
