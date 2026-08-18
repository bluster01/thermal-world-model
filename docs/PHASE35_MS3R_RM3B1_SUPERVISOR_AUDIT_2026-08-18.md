# Phase 3.5 MS3-R RM3-B1 Supervisor Audit

## 最终判决

```text
AUDITED / VALIDATION-ONLY PAIRED SCREEN /
22 OF 22 ARTIFACT-COMPLETE /
NO COMPOSITION CHAMPION /
NO RM3-B2 / NO TEST / NO MS4 /
PHASE3.5 MODEL-EXPANSION TRACK CLOSED FOR FINAL-PIPELINE TRANSITION
```

RM3-B1 完成了最后一轮单模块配对筛查。它提供了结构取舍证据，但没有出现同时闭合预测、动作响应、placebo 和状态仿真的组合，因此不继续 B2/B3 组合搜索。相关模块作为最终世界模型 pipeline 的设计输入保留。

## 1. 产物与复算

- 22/22 runs、每个 8000 optimizer updates、exit 0；
- 110 个 run-level ledger 成员和 3 个 root ledger 成员逐字节 SHA256 一致；
- 22 个 checkpoint SHA 与 manifest 一致；
- episodes 独立复算四任务 MAE，最大绝对差 `9.54e-7`；
- 每 fold 的 11 个候选共享同一 reporting anchors；selector/reporting anchors 与 UTC 日均隔离；
- test 未访问，自动科学 PASS、综合排名和模型冠军均为空；
- 执行提交 `d18d7f8`，22/22 仅一次 attempt，资源记录为 5h47m、peak RSS 3.17 GB。

## 2. 八组成对结论

| Pair | 判决 | 主要证据 |
|---|---|---|
| B03−B01 action shield | MIXED | H60 显式响应约放大 4 倍，wrong/shuffled specificity 两折改善；terminal/local 两折均小幅变差，lead 仍未形成强正间隔 |
| B04−B01 OOF R-loss | MIXED | H60 响应约放大 5 倍且 wrong/shuffled gap 改善；response-off terminal delta 仅约 0.001°C，terminal/local 跨折取舍，不能说响应真正进入末温 |
| B05−B02 valve dynamics loss | MIXED | valve MAE 两折改善，但 `|Δv|` 与 roughness 改善方向跨折不一致，terminal 两折均退化 |
| B06−B02 PI+GRU policy | MIXED | valve MAE、动态幅度和 local MAE 两折改善；terminal 两折退化约 0.025/0.029°C，未满足“不伤末温” |
| B07−B01 diagonal response | SUPPORTED AS SIMPLIFICATION | 四任务误差与响应幅值近乎不变，两个 fold 非劣；支持当前数据下无需 full MIMO 参数化，不支持独立 A/B plant gain |
| B08−B01 one-pole | REJECTED FOR PRIMARY RESPONSE | 预测近乎不变，但 H60 显式响应和 wrong/shuffled specificity 约减半；简单一极点不能作为已验证主响应基底 |
| B09−B01 linear ramp | MIXED / SHAPE-SENSITIVE | 预测近似，响应幅值与 specificity 同样显著改变；证明结果依赖基函数，不允许声明真实阶次 |
| B10−B00 action-invariant bypass | REJECTED AS ADD-ON | 相对容量控制 anchor，terminal 两折均变差 `+0.0158/+0.0087°C`；内部 bypass-off 退化很大只说明该候选依赖旁路，不证明旁路优于 anchor |

`SUPPORTED AS SIMPLIFICATION` 只表示 B07 在这一 validation screen 中可替代更复杂坐标，不是物理或因果 PASS。其他 MIXED 模块可作为最终 pipeline 的消融维度，但不进入自动组合搜索。

## 3. 科学结论

RM3-B1 巩固了三点：

1. action shield/R-loss 能放大显式响应，但预测损失与末温实际使用仍未同时闭合；
2. 更复杂的双侧响应坐标没有获得额外证据，common/diagonal 支持域应优先；
3. terminal bypass、阀位动态和 response basis 都存在明确能力取舍，继续堆叠只会恢复不可归因的架构搜索。

所以 Phase3.5 的最合理输出不是一个冠军，而是最终 pipeline 的权限合同：高容量网络承担概率初始化/边界，Fan2020-UDE 承担动作条件状态转移，closure 动作隔离，Koopman 只在母模型验证后蒸馏。

## 4. 关闭边界

- `RM3-B2/B3` 不生成；
- Phase3.5 test 和 MS4 不授权；
- 旧 RM3 模型不升级为 state-closed simulator；
- 后续新训练只能在最终 pipeline 的新矩阵、新状态注册和独立提交下授权。
