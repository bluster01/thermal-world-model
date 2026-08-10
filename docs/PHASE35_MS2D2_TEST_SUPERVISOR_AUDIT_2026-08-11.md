# Phase 3.5-MS2-D2 One-Shot Test Supervisor Audit（2026-08-11）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-11
- Verification Status: VERIFIED（ledger、manifest、episode metrics、冻结汇总与独立 bootstrap 已复核）
- Overall Confidence: SOLID for the preregistered synthetic response claim；CAUTION for mechanism interpretation
- Version Label: phase35_ms2d2_test_supervisor_audit_v1
- Evaluation Commit: `c22140391906ad86c4d87afd9d60759303e9f649`
- Linux Return Commit: `d97538faf6d3dea8480a29f45862ff691417590a`
- Evidence Scope: `synthetic_order_pressure_test_not_field_causality`

## 1. Supervisor 判决

MS2-D2 以 **`CLOSED / CONFIRMED_SYNTHETIC_ORDER_RESPONSE`** 关闭。该判决只表示：在冻结的 R50、context scheduling、三惯性极点 `[40,70,210] s`、无 pure delay 的 known-truth 设计中，显式三极点 graybox 相对同预算二极点能稳定恢复更好的多步动作响应。

全部预注册确认门通过：

1. 21/21 artifacts、访问 ledger 与结构门禁通过；
2. oracle clean NMAE 逐 seed为 0.0211、0.0255、0.0237，均 `<0.05`；
3. 三极点主模型逐 seed为 0.0444、0.0444、0.0465，均 `<0.10`；
4. 三极点相对二极点的配对、profile 分层 10,000 次 bootstrap 改善为 23.74%、24.31%、25.36%，95% CI 下界为 19.90%、19.98%、21.22%，均 `>=10%`。

这不是“从数据中唯一识别出真实三阶”的证明。候选阶次由实验者预先指定；二极点+learned-delay 与 DeepONet 的 test mean NMAE 分别为 0.04648、0.04644，接近三极点的 0.04509。结果确认的是响应容量优势，不是阶次、迟延或设备状态的唯一机制归因。

## 2. 完整性与可复现性

| 检查 | 本地复核 |
|---|---|
| Linux 写入范围 | 95 个变更全部位于 `results/phase3_5/ms2d_order/**`；未修改代码、配置、测试或权威文档 |
| 命令与版本 | runner/summary exit code 均为 0；evaluation SHA 为授权 commit `c221403` |
| 环境 | Python 3.11.15、PyTorch 2.11.0+cu130、CUDA 13.0、NVIDIA GB10 |
| root/run ledger | root completed；21/21 run ledger completed；执行顺序与冻结 7×3 matrix 一致 |
| test split | 每 run 256 episodes；同 seed 七候选 trajectory hash 完全相同；三个 seed hash 互异 |
| manifest | 21/21 `test_accessed=true`、`test_authorized=true`、checkpoint 与 evaluation SHA 一致 |
| episode 重放 | aggregate metrics 从 episode JSON 重算；最大绝对差 `2.47e-8` |
| 冻结汇总 | 本地 `build_test_summary()` 与回传 `summary_test.json` canonical JSON 完全一致 |
| 独立统计 | NumPy PCG64、50,000 次同层级 bootstrap 的 CI 下界为 19.83%、19.99%、21.29%，判决不变 |

内容 pins：

- authorization SHA256：`20fc3060ec5a1f7800cc1cb9506195a69ee5fa519a4aa9ee8be28461bd2fadcd`
- matrix SHA256：`dfa01ad4124c452f3fd5de2f22b0d384f56041fab4837e7c7e5f05c23a854c26`
- validation summary SHA256：`4061353d9c4ef058a6db3dd969505452a3163a9fe957b13aee32251bd11ce701`
- checkpoint archive SHA256：`e8d6d8064fcef58ecb1da154727379fa781b2d80c63afc0aee02d1e2cc25c43f`
- test summary SHA256：`9c4e9300314379436d9b3e3a3bd004114fb76aeaf406164def86fce95cb71ea0`

## 3. 异质性与 split 稳定性

| seed | 点改善 | 冻结 10k CI | 独立 50k CI | oracle NMAE | three-pole NMAE |
|---:|---:|---:|---:|---:|---:|
| 0 | 23.74% | [19.90%, 27.80%] | [19.83%, 27.85%] | 0.02108 | 0.04441 |
| 1 | 24.31% | [19.98%, 28.85%] | [19.99%, 28.86%] | 0.02551 | 0.04438 |
| 2 | 25.36% | [21.22%, 29.61%] | [21.29%, 29.60%] | 0.02368 | 0.04649 |

所有非 hold action profiles 的改善方向均为正；ramp 的 seed 2 只有 7.06%，低于总体 10% 门槛，但逐 profile 从未被预注册为确认门，不能事后改变总体判决。改善在 H6/H18 最大，H60 仍为 17.46%–24.00%；H1 两者均为精确零响应。三极点 validation→test mean NMAE ratio 为 1.015，未见 selection 后明显退化。

## 4. 非阻断诊断

- three-pole tau-set log-MAE 为 0.151–0.192；oracle 为 0.179–0.243，均低于 0.35。它只验证给定三极点容量的参数集合近似，不证明现场有三个唯一物理状态。
- no-delay truth 下，delay-compensation 仍给出 2.16–2.40 steps 的期望迟延，零步质量 0.241–0.297；诊断继续 FAIL。它证明遗漏阶次可被迟延容量吸收，而非现场存在该迟延。
- secondary representation 的 test 排名不进入主门。D2 不产生 Koopman、PI-ODE 或 DeepONet 的路线冠军。

## 5. Fallacy Scan（11/11）

| Fallacy | Severity | 结论 |
|---|---|---|
| Simpson's paradox | NOTE | 总体与所有非 hold profile 方向一致；幅度异质已保留 |
| Ecological fallacy | CAUTION | 合成 episode 结果不能外推现场机组、负荷段或设备状态 |
| Berkson's paradox | N/A | known-truth generator 无选择入样机制 |
| Collider bias | N/A | 未按中介或结果筛 episode |
| Base-rate neglect | NOTE | 五类 profile 人工近均衡，不代表现场动作基率 |
| Regression to the mean | PASS | independent test 未参与 checkpoint 选择 |
| Survivorship bias | PASS | 21/21 checkpoints 全纳入 |
| Look-elsewhere effect | PASS/NOTE | 三个主门预注册；secondary 排名未升级 |
| Garden of forking paths | PASS | authorization、阈值、seed、bootstrap 与一次访问均冻结 |
| Correlation implies causation | CAUTION | synthetic causal generator 不建立现场 `do(valve)` |
| Reverse causality | N/A | synthetic 作用方向由生成器定义 |

## 6. 下游授权

D2 不再训练、不重复 test。该判决只放行 MS2-D3 的本地设计与实现：增加 action-independent colored disturbance，检验 D2 的响应结论是否在不可预测、时间相关的背景扰动下保持。D3 的本地测试、smoke、矩阵与统计门禁随后已独立完成，因此当前 Linux validation 授权以机器注册表为准；这不扩展 D2 的结论或授权。MS5、MS3、MS4 继续冻结。
