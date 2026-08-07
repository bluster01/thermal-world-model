# 项目保守整理设计

## 背景

仓库已经积累三阶段、百余个实验，但根 README 仍以 2026-08-04 的 M7-MPC 结论为主，未反映随后完成的因果评测、A1phys、Fan 系列论文调研以及 exp_112。与此同时，当前模型尚未定性：exp_112 只否定了将 Koopman 用作 A1phys `free_head` 的具体方案，不能外推为 Koopman、Neural ODE 或 Fan 灰箱模型整体无效。

## 整理目标

1. 让新进入项目的人在根 README 中看懂问题、数据、研究转向和当前任务。
2. 建立唯一的项目状态页与当前任务页，避免从旧实验结论反推现状。
3. 将文档和实验按“当前证据 / 研究参考 / 历史路线”导航，但不移动原文件。
4. 明确 Fan 2017/2020/2021 与三种可微动力学候选路线仍待公平验证。
5. 区分本地实验研发/审计与 Linux 远端正式运行，避免状态混淆。

## 边界

- 不移动、不重命名历史实验、结果和图表。
- 不把 `experiments/phase3_feedforward/` 中的活跃代码迁入 `src/`。
- 不宣布 A1phys、Neural ODE、Koopman 或任何 Fan 路线为最终模型。
- 不修改训练或评测协议；本轮只整理信息架构与事实口径。

## 信息架构

- `README.md`：项目入口、当前判断、目录、快速导航与最小运行说明。
- `docs/PROJECT_STATUS.md`：可信结论、被推翻结论、候选模型与证据等级。
- `docs/CURRENT_TASKS.md`：按依赖排序的研究任务与验收标准。
- `docs/README.md`：文档地图。
- `experiments/README.md`：实验阶段、关键入口、历史与活跃代码的边界。
- `results/README.md`：补充 CFE/A1phys/exp_112 结果入口。
- `docs/REMOTE_EXPERIMENT_PROTOCOL.md`：commit 固定的远端交接与回传规范。

## 关键口径

仓库现有 `exp_020_koopman_vs_gru.py` 的三路线是 GRU、受控 Koopman 和简化 Euler Neural ODE。它使用早期 11 状态框架，验证的是解码器形式，不是 Fan 方程。

后续真正需要比较的是统一数据、统一状态/输入、统一损失和统一 CFE 评测下的可微动力学：Fan-structured Neural ODE、controlled Koopman，以及含时变/残差修正的灰箱模型。Fan 2017、2020、2021分别提供基础状态方程、SST 两级喷水焓值链、宽负荷能量不匹配与时变参数结构。

## 验证

- 所有 Markdown 相对链接能解析到仓库内文件。
- README 不再把 M7-MPC 或 A1phys 写成最终模型。
- 当前任务页明确列出 Fan 数据可观测性、物性计算、三路线公平基线和多 seed 要求。
- `git diff --check` 通过，现有测试不因文档整理受影响。
