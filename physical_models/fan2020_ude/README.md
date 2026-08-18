# Fan2020-UDE 物理模型资产

本目录把孤立分支 `adhoc/lumped-enthalpy` 的有效资产收回主仓，定位为最终世界模型的**物理状态转移候选**，不再作为独立 Paper A 仓库或已经验证的物理真值。

## 来源与边界

- 源分支：`adhoc/lumped-enthalpy`
- 冻结提交：`57fa0185f7e2bdbb576aea8c19155d1331da6e64`
- 导入日期：2026-08-18
- 导入方式：选择性快照。源分支与 `main` 没有共同祖先，因此没有执行 merge/subtree，也没有导入其完整历史。
- 已导入：Python 实验代码、冻结配置、设计记录和用于复核结论的小型 JSON 摘要。
- 未导入：模型权重、rollout 数组、图片、日志、缓存和 IAPWS 生成数组。原始产物继续保留在源分支；不得用本目录摘要替代原始产物审计。

## 目录

```text
fan2020_ude/
├── legacy_experiments/   # 原分支最新代码快照；保持原编号和相对路径
│   └── configs/          # 原冻结配置
│   └── tests/            # 原分支合同测试；部分需要历史 checkpoint
├── docs/archive/         # 原设计与推理记录，只作历史追溯
├── evidence/
│   ├── EVIDENCE_CHAIN.md # 经主仓重新分级的证据链
│   └── raw_summaries/    # 小型原始 JSON 摘要
└── README.md
```

`legacy_experiments` 不是最终软件 API。脚本仍按原实验顺序依赖本地 `out/` 中间产物；本次仅保证源码可追溯和可编译，不声称任意后期脚本能在缺少父 checkpoint 时独立执行。最终 pipeline 会从中提取物理状态、状态转移和观测方程到新的正式包，而不是继续叠加编号脚本。

本地验证已确认 49 个脚本可编译、5 个 Direct-WM 协议测试通过；Q32 的无 checkpoint 合同测试在提供 IAPWS 缓存后 4 项通过，依赖历史模型权重的等价性测试保留在源分支执行。该依赖本身是后续抽取正式包时必须消除的技术债。

## 当前模型身份

更准确的名称是 **Fan2020-inspired physics-embedded neural state-space model / UDE candidate**：

- 显式部分包括集总焓、质量/能量关系、金属蓄热、燃料滞后、喷水混合以及蒸发/干燥候选状态；
- 学习部分包括工况参数、阀位代理和受限闭合项；
- 现有数据没有可靠喷水质量流量、壁温和完整焓状态，因此它不是完全白箱，也不是 plant truth；
- 现有闭环数据只支持扰动条件响应和支持域内敏感性，不支持任意 `do(valve)`。

最终世界模型设计见 [最终 pipeline 设计](../../docs/plans/2026-08-18-final-world-model-pipeline-design.md)。现有证据及缺口见 [证据链](evidence/EVIDENCE_CHAIN.md)。
