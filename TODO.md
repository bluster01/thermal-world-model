# Thermal World Model TODO

> 更新：2026-08-11。本文是项目唯一人工任务队列；机器状态见 `configs/phase3_5/experiment_registry.json`。当前唯一授权为 Phase 3.5-MS5 validation。旧 E1–E5 已废弃，Phase 4 暂停。

## 当前主线

完整顺序保持为：

```text
MS0 → MS1 → MS2-V/C/J → MS2-D1/D2/D3 → MS5 → MS3 → MS4 → 模型选择/论文
```

当前目的不是提前写论文，而是验证完整模型

\[
\widehat T_{1:H}=f_{free}(history,context)+g_{response}(context,a_{1:H},r_{1:H})
\]

在只用总温度监督时，动作响应是否会被 `free` 分支吸收。`free` 不读取未来动作；`g_response(c,r,r)=0`，开阀长期增益非正。喷水流量不作真值，现场实际阀位仅作为有效喷水作用代理。

## Gate 总表

| Gate | 问题 | 当前结论/状态 | 下一动作 |
|---|---|---|---|
| MS0 | 统一多步响应合同 | ✓ CLOSED | 保持冻结 |
| MS1 | 同型 known-truth 可解性 | ✓ CLOSED | 不设路线冠军 |
| MS2-V/C/J | 非线性、工况调度和联合耦合 | ✓ CLOSED | joint 为 response 主训练；不外推 MS5 |
| MS2-D1 | pure-delay 压力 | ✓ 阴性关闭 | 不重试、不传播 delay 结构 |
| MS2-D2 | 三阶惯性压力 | ✓ test 确认 | 仅 frozen known-truth 响应优势 |
| MS2-D3 | colored nuisance 压力 | ✓ validation-only 关闭 | 不补 test；进入 MS5 |
| **MS5** | 完整 `free+response` 动作吸收 | ▶ **READY_FOR_LINUX** | 4 modes×3 seeds=12 validation runs |
| MS3 | A/B 真实数据适配 | ◻ 冻结 | 等 MS5 本地审计关闭 |
| MS4 | SP→阀位→温度闭环响应 | ◻ 冻结 | 等 MS3；不恢复旧 E 匹配 |

## D3 收口

Linux 回传的 21/21 runs 已由本地重放：oracle clean NMAE `0.0357–0.0446`，三阶 `0.0558–0.0633`；三阶相对二阶的冻结 bootstrap 95% CI 下界 `10.8%–14.3%`，逐 seed达到 10% 门槛。独立 50k bootstrap 判决一致。按预算决定标签为：

```text
CLOSED / VALIDATION_STRESS_PASS / NO_TEST_BY_BUDGET_DECISION
```

它不是独立 test，不能支持现场扰动谱、现场唯一阶次或因果反事实。权威审计见 [D3 Supervisor Audit](docs/PHASE35_MS2D3_SUPERVISOR_AUDIT_2026-08-11.md)。

## 当前唯一任务：MS5

### 冻结 12-run 矩阵

| 模式 | 作用 | 选择资格 |
|---|---|---|
| `ms5_free_only` | prediction-only 负控；response 精确为零 | 不参与选择 |
| `ms5_joint_total` | free 与 response 从头联合、只用 total loss | 首选简单策略 |
| `ms5_staged_total` | hold 预训 free → 冻结 free 训 response → 0.2×LR 联合 | joint 失败时的候补 |
| `ms5_component_oracle` | 加 synthetic component loss 的正控 | 必须先通过，否则 fail closed |

每种 3 seeds，共 12 runs；train/validation 为 1024/256，最多 300 epochs。只运行 validation，不访问 synthetic test 或 A/B。

### 判决规则

1. 12/12 manifest/history/checkpoint/episode/结构合同闭合；
2. oracle 每 seed：total/free/response clean NMAE 均 `<0.10`；
3. joint/staged 资格门每 seed：total/free `<0.10`、response `<0.15`、response amplitude ratio `0.80–1.20`；
4. joint 全过就选择 joint，不因 staged 数字更好增加复杂度；
5. joint 失败时，只有 staged 全过且 staged/joint total error ratio 每 seed `<=1.10` 才选择 staged；
6. oracle 失败或两个 total-only 策略均失败：阻断 MS3 的强 component claim，不补超参数扫描；
7. 按负责人决策，MS5 validation 通过即可关闭，不追加 synthetic test。

详细协议见 [MS5 设计](docs/plans/2026-08-11-phase35-ms5-full-coupling-design.md)。

## 本地 / Linux 分工

| 本地 / Codex | Linux 远端 |
|---|---|
| 设计协议、写代码和测试、冻结矩阵、审计日志与重算统计、迁移状态 | 只在指定干净 commit 执行 README 第 16 节命令，原样提交结果 |
| 只有本地可改 TODO、注册表、阈值、结论和 Supervisor 文档 | 不改 `configs/src/experiments/tests/docs`，不挑 seed、不补跑、不后处理 summary |

## MS5 执行清单

| 顺序 | 任务 | 负责人 | 状态 |
|---:|---|---|---|
| 1 | D3 产物、统计和 provenance 独立审计 | 本地 | ✓ |
| 2 | 冻结 MS5 known-truth、4 modes、门禁和停止规则 | 本地 | ✓ |
| 3 | generator、full model training、runner、summary TDD | 本地 | ✓ |
| 4 | 专项测试、compile、dry-run、状态检查 | 本地 | ✓ |
| 5 | 12-run validation 与 fail-closed summary | Linux | ▶ 当前唯一远端任务 |
| 6 | checkpoint/episode/门禁独立复算并关闭 MS5 | 本地 | ◻ 等回传 |
| 7 | 若 MS5 通过，冻结 MS3 A/B validation-only 适配 | 本地 | ◻ 不提前启动 |

Linux 唯一运行说明见 [experiments/phase3_5/README.md](experiments/phase3_5/README.md)。

## 不可提前声称

- MS5 PASS 只证明冻结 synthetic policy correlation 下的 component recovery，不证明现场 free/response 分解；
- 当前不能声称 `do(valve)`、完整状态 simulator、可识别反事实、闭环可用或最终路线冠军；
- Koopman、PI-ODE、DeepONet 与 Fan 路线不在 MS5 当前矩阵中重新赛马；
- MS3、MS4、模型选择与论文在 MS5 本地审计前继续冻结。
