# JEPA-B / B5 Linux 回传独立审计（2026-09-01）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-01
- Verification Status: ANALYZED
- Version Label: validation_v1
- Upstream Dependencies: `jepa-b-series-v1`, `jepa-b5-series-v1`, Linux commits `7791cc4..5d79f24`

## 审计结论

- **预注册裁定：可信。** B1 为 `INCONCLUSIVE_EXPLORATORY_SEED0`；B2/B3/B4/B5 为
  `REJECT_EXPLORATORY_SEED0`；B3-SHUFFLE 仅为负控制。没有候选获得晋级资格，
  `paper_verdict_upgraded=false` 正确。
- **机制叙事：需要降级。** B2→B5 的均值符号变化与“慢态读取动作效果”的解释一致，
  但单种子、一次结构消融不能证明“主因”“与信息源无关”或“机制级因果冲突”。可写成
  mechanism-consistent exploratory evidence，不可写成已识别因果机制。
- **可复现性：PARTIALLY_REPRODUCIBLE。** 两次独立 C0 训练的训练选择量、H18/H36/H60
  evaluation 和四个方向格逐位一致；所有 ledger 内部闭合。Windows 缺少 Linux canonical
  v2.2、IAPWS 产物和 checkpoint，不能从原始数据完整重跑候选臂，因此不升级为 VERIFIED。

## 证据链核对

| 项目 | 核对结果 |
|---|---|
| B v1 matrix | SHA-256 `b664c06272318775ad5aa89cc93c337c09a72806e5b16340552d536c66224751`，匹配预注册 |
| B5 matrix | SHA-256 `28dcb4b6f41aed44184c1ca785916b9e3b6d03499c22f78e7969cbc0ceeed884`，匹配预注册 |
| 数据/物性指纹 | 两批均报告 record `24da7796…d1d0`、properties `9fd7a1db…6e92`；本机无原件，不能独立重算 |
| ledger | 8/8 连续；epoch 数、final、best epoch/best val NLL、arm/commit/matrix 与报告一致 |
| 身份门 | B v1 五机制和 B5 均 `exact=true`；B5 最大温度差 0 |
| test 权限 | matrix 与 record 代码均锁 test；报告为 validation-only、seed0 |
| B5 预注册时序 | `0170d9f` 于结果提交 `5d79f24` 前约 6 h，设计、矩阵和门限先于训练结果入库 |
| 重复 C0 | B v1 与 B5-C0 evaluation、direction 完全相同；best NLL/epoch 也完全相同 |

## 独立复算裁定

| 臂 | H18 相对 C0 | H18 负荷 spread 相对变化 | 方向 4 格 | 复算裁定 |
|---|---:|---:|---|---|
| B1 | +4.39% | +6.80% | 4/4 | INCONCLUSIVE |
| B2 | −5.31% | +0.71% | 2/4 | REJECT |
| B3 | +5.73% | +10.12% | 4/4 | REJECT |
| B3-SHUFFLE | −6.11% | +1.11% | 4/4 | NEGATIVE_CONTROL_ONLY |
| B4 | +5.74% | +4.50% | 4/4 | REJECT |
| B5 | −4.42% | +10.94% | 3/4 | REJECT |

B5 的 valve2-H18 为 mean −0.0104 °C、day-bootstrap CI 上界 +0.0091 °C、
`frac_negative=0.446`，同时未满足 mean/CI/`frac≥0.60` 三项中的后两项。因而即使 H18
精度门只差 0.58 个百分点、spread 门只超 0.94 个百分点，拒绝也不依赖这两个边缘点门限。

B3-SHUFFLE 的 H18 末端 MAE 优于 C0 和 B3，而 B3 的 best validation NLL 优于 SHUFFLE。
这只说明“正确配对影响优化目标/整体似然”，不支持“正确特权配对改善预注册主预测指标”。

## 方向门协议修复审计

原执行 commit `7791cc4` 把每个窗口首个边界/动作重复到全 horizon，并以 horizon=1
抽样；`cde385e` 改为真实 future boundary/action 原轨迹，并按 H18/H60 各自生成有效窗。
这是实质性评估口径修复，发生在看到首批结果之后，应视为 **post-result protocol
correction**，不能静默当作原始预注册执行。

修复前后六臂裁定全部不变，C0 四格均通过，B2 的 valve2 两格均持续失败；因此最终“不晋级”
对两种口径稳健。但当前六份报告仍只写训练 commit `a0495d9`，没有显式记录方向重算 commit，
且方向格没有 anchor SHA；文件虽可由 Git 历史追溯，报告自身 provenance 不完整。

## 发现与解决办法

| 优先级 | 发现 | 影响 | 解决办法 |
|---|---|---|---|
| P1 | B5 已完成但 registry 仍为 `ready_for_linux`、`linux_completed=false`、`results_returned=false` | 运行权限与事实不一致 | 将 B/B5 标为 audited/closed，清空 Linux 授权，恢复可信度修复 pipeline 为 active |
| P1 | B v1 报告被 `cde385e` 覆盖方向值，但只保留训练 commit | 论文证据无法仅凭 report 定位评估代码 | 在 root/六臂报告增加 direction evaluation provenance；保留训练 commit 不变 |
| P1 | B5 结果稿写 `frac≥0.5`，正式门为 0.60；B2 MAE 写 0.6731，权威报告为 0.6056；“9 臂”无对应矩阵 | 数值/计数误引 | 更正为 0.60、0.6056；报告 7 个唯一臂/8 次训练执行，或只数 5 个晋级候选 |
| P1 | “主因”“机制级因果冲突”“与信息源无关”超出单种子消融证据 | 因果过度声称 | 降为“与输入路径贡献/注入通道冲突假说一致”；需要固定多种子和注入点正交消融才能升级 |
| P2 | `--matrix B5 --arm b5` 在 spec 选择前被 argparse 的 B-v1 choices 拒绝 | 单臂恢复/诊断入口不可用；本次 `--queue` 结果不受影响 | argparse 先接受字符串，再在 matrix spec 选定后验证 arm membership，并加 CLI 测试 |
| P2 | B5 spec 的 registry 授权检查未核对 seed/test/paper/retry/result-state 全合同 | fail-closed 强度退化 | 补齐与 B-v1 同等级的授权字段检查 |
| P2 | registry 测试仍断言旧 `jepa_b_series` active gate | 当前仓库专项测试 3/7 失败 | 更新为审计后最终状态，并在 mock 授权测试中显式关闭 B/B5 状态 |

## 统计与方法谬误扫描

- **覆盖：11/11。** 总体置信度：裁定 `SOLID`；机制解释 `CAUTION`。

| 谬误 | 级别 | 审计结论 |
|---|---|---|
| Simpson's paradox | CAUTION | B5 aggregate 改善，但 160–450 MW 两箱恶化、450–700 MW 两箱改善；禁止只报总体值 |
| Ecological fallacy | NOTE | 未发现从负荷箱/日均直接外推单窗口的正式裁定；论文需保持分析单位 |
| Berkson's paradox | CAUTION | A5 和支持域筛选限定了样本；结论只能覆盖该运行支持域 |
| Collider bias | NOTE | 未发现明确 outcome collider 控制；支持域选择仍需作为选择机制披露 |
| Base-rate neglect | N/A | 非诊断分类任务 |
| Regression to mean | N/A | 非极端样本 pre/post 设计 |
| Survivorship bias | CAUTION | H60 方向门仅约 57% 窗口在支持域；通过格不代表全部验证窗 |
| Look-elsewhere effect | NOTE | B5 是看过 B2 后的探索性 follow-up，但在自身结果前预注册；不得并入确认性证据 |
| Garden of forking paths | CAUTION | 首批方向语义在结果后修正；两口径裁定虽一致，仍必须双轨留痕 |
| Correlation ≠ causation | RED_FLAG（仅针对文稿措辞） | B2/B5 单种子结构差异不能证明注入通道是因果主因 |
| Reverse causality | CAUTION | B2 state 同时承载动作后果和热状态；B5 支持吸收假说，但未唯一识别方向 |

## 可用于论文的最强安全表述

“在单种子、validation-only 的预注册探索中，没有 JEPA 状态增强候选同时满足 H18
精度、跨负荷稳健性和双阀方向证书。B2 的精度改善伴随 valve2 方向失败；移除慢态更新中的
物理状态输入后，B5 的平均方向恢复为负，但 H18 方向 CI 与窗口一致率仍未过门，且精度和
负荷稳健门也未通过。该结果与状态表示和动作响应通道之间存在冲突的假说一致，但不构成
机制因果识别或跨机组结论。”

## 审计后修复记录

上述 P1/P2 项已按“不改实验数值、不改判据、不补跑”的边界落地：B/B5 状态关闭并清空
Linux 授权；六臂方向重算 provenance 写回报告；B5 文稿数字、0.60 门限、臂计数及因果
措辞更正；B5 单臂 CLI 和授权合同 fail-closed 修复；专项注册测试同步。复核结果：

- B v1 的 root + 六臂报告去除新增 provenance 后，与 `cde385e` 逐字段一致；
- 两份 matrix SHA-256 保持不变；8 份 ledger 未修改；
- JEPA/registry 定向测试 24/24；`tests/final_wm` 185/185；
  `tests/phase35 --import-mode=importlib` 472/472；
- registry valid，active gate 恢复 `final_world_model_pipeline`，Linux authorization 为 null。
