# Codex ↔ Linux Handoff（2026-08-10 MS2-J 收口）

> **Supervisor override（同日）**：本文的 MS2-J 数值与产物清单仍有效，但“停止 MS2-D/MS3–MS5、进入论文”的后续决定已被撤销。当前权威状态见 `configs/phase3_5/experiment_registry.json` 与 `PHASE35_CONTEXT_SNAPSHOT.md`；旧 E1–E5 已废弃，完整 MS 系列继续。

> 用途：Codex 上下文丢失后的恢复入口。pull 本 commit 后，以本文为同步基准。本文由 Linux 侧维护，如与本仓库其他文档冲突，以最近 commit 为准。

## 1. 当前状态一句话

**MS2-J 一次性 synthetic test 已完成并双层收口，synthetic 矩阵停止扩展，进入论文表图与 claim ledger 阶段。** 远端 main = `c567279`（本 commit），本地与远端完全同步，工作树干净。

## 2. Commit 链（按时间序）

| Commit | 内容 |
|---|---|
| `e3c6144` | Codex: MS2-J coupling validation 部署（9 candidates × 3 seeds = 27 runs） |
| `b74f9ab` | Linux: MS2-J validation 完成（联合模块 PASS 79–91%、staged FAIL ratio 1.15–1.23）、checkpoint 归档 36 文件 SHA `3005fd4b` |
| `73ec4dd` | Linux: MS2-J 本地审计（36/36 可加载无 NaN、staged 阶段日志健康、oracle 恢复真值 K0） |
| `5fa9769` | Codex: authorize one-shot MS2-J test |
| `1872abd` | Codex: consolidate paper experiment mainline（PHASE35_MAINLINE_CONTEXT.md 等） |
| `78904ba` | Linux: 修正授权文件 validation_summary pin 不同步（见 §4） |
| `5260d3f` | Linux: **MS2-J test 完成**（27/27 单次访问，三门禁判定，review 文档） |
| `c567279` | Linux: 全部主线文档同步 test 完成状态（本文） |

## 3. MS2-J test 结果（三层证据，paired stratified bootstrap 10k reps）

| 门禁 | 结果 | 数值 |
|---|---|---|
| 1. 联合模块增量 | ✅ PASS | vs 两单模块消融 obs 0.77–0.88，CI 下界 0.73–0.89 >> 20%，6/6 |
| 2. staged 非劣 | ❌ FAIL（复现 validation） | ratio 1.14–1.20，CI 上界 1.09–1.32 > 1.10 |
| 3. staged vs Stage A | ✅ PASS | 改善 0.73–0.74，CI 下界 0.68–0.77 ≥ 20% |

- oracle `j_g2_r50_scheduled` 0.0225（val 0.0208，无 degradation）；主模型 joint 0.0452（val 0.0410）
- **结论**：联合模块跨 split 复现 → 主训练方案定 joint；staged 双层 FAIL → 仅作阴性消融。与 validation 无 split 不一致，无需重试/补 seed。
- 关键文件：`docs/PHASE35_MS2J_TEST_REVIEW_2026-08-10.md`、`results/phase3_5/joint_coupling/summary_test.json`（ledger completed）

## 4. 需要 Codex 知悉的两件事

1. **授权文件 pin 修正**：`5fa9769` 的 `configs/phase3_5/joint_coupling_test_authorization.json` 中 validation_summary pin（`788b1bc2...`）与仓库实际文件（`212e53cc...`）不同步，git 对象库无该 blob、枚举全部序列化组合无法复现，判定为写授权时 SHA 计算与 commit 不同步。已在 `78904ba` 修正为实际哈希；修正前后 frozen_validation_status 三项完全一致（all_gates_pass=false, joint=true, staged=false），内容寻址语义不受影响。**建议 Codex 在本地复算确认**。
2. **审计文档措辞修正**（Codex 在 `5fa9769` 已自行处理）：raw_gain 解释应为 `-softplus(raw_gain)` 而非 `exp(raw_gain)`；"joint is the correct scheme"、"same solution region" 已按 evidence-limited 措辞弱化。

## 5. 收口清单进度（PHASE35_MAINLINE_CONTEXT.md §7）

1. ✅ test artifacts 复核（ledger/归档/trajectory pairing/结构门禁/bootstrap）
2. ✅ MS2-J test review（validation/test 与 synthetic/field 口径分离）
3. ⬜ **论文三张核心表**：现场 E1–E5、synthetic MS1–MS2-J、claim/evidence boundary（Codex）
4. ⬜ **两张核心图**：控制层级与 action proxy、free+response 架构及证据流（Codex）
5. ⬜ **论文提纲和主文**（Codex）

## 6. 硬性边界（保持不变）

- 不启动 Phase 4、MS2-D、MS3、MS4、MS5；synthetic 矩阵不扩
- staged 无论结果如何不得改写成路线冠军
- 现场证据必须新时间块 + 已冻结 E3 双 estimand（common support/balance/pre-trend/placebo）
- 论文不得声称：真实阀门曲线/喷水流量已恢复、现场反事实已识别、完整状态闭合 simulator、闭环可用
