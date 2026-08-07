# 文档地图

文档按“当前口径、核心证据、物理路线、历史阶段”阅读。旧文档记录实验发生时的判断，若与 `PROJECT_STATUS.md` 冲突，以后者为准。

## 当前入口

- [PROJECT_STATUS.md](PROJECT_STATUS.md)：可信结论、候选模型、作废结论与下一判决点。
- [CURRENT_TASKS.md](CURRENT_TASKS.md)：Fan 骨架、三类可微路线和统一评测任务。
- [REMOTE_EXPERIMENT_PROTOCOL.md](REMOTE_EXPERIMENT_PROTOCOL.md)：本地研发与 Linux 正式实验的交接协议。
- [plans/2026-08-07-project-reorganization-design.md](plans/2026-08-07-project-reorganization-design.md)：本次保守整理设计。

## 因果评测与当前架构

- [causal_eval_framework.md](causal_eval_framework.md)：L0-L7 因果评测框架。
- [exp_107_review.md](exp_107_review.md)：单点 CFI、事件数和 checkpoint 选择问题。
- [session_2026-08-05_causal_arch_eval.md](session_2026-08-05_causal_arch_eval.md)：A1/A3/B1 第一轮结果。
- [session_2026-08-06_review_v2.md](session_2026-08-06_review_v2.md)：修正后的完整审查与当时任务。

## Fan 与可微动力学

- [Fan_三篇整合_热工控制入门.md](Fan_三篇整合_热工控制入门.md)：三篇物理模型的总体关系。
- [paper_Fan2017_USC直流炉3入3出模型.md](paper_Fan2017_USC直流炉3入3出模型.md)：基础 4 状态非线性 ODE。
- [paper_Fan2020_直流炉动态模型.md](paper_Fan2020_直流炉动态模型.md)：7 状态、两级喷水与 SST 焓值链。
- [paper_Fan2021_宽负荷非线性模型.md](paper_Fan2021_宽负荷非线性模型.md)：能量不匹配、节流损失和时变参数。
- [伊敏40列_vs_Fan模型变量对照.md](伊敏40列_vs_Fan模型变量对照.md)：可观测变量与缺口。
- [Neural_ODE_Koopman_三篇关键论文.md](Neural_ODE_Koopman_三篇关键论文.md)：Neural ODE、Deep Koopman、Koopa 调研。

原文 PDF 和全文转写保存在同目录，仅用于研究核对，不作为项目状态入口。

## Phase 1：预测基线

- [phase1_status.md](phase1_status.md)
- [phase1_report.md](phase1_report.md)
- [phase1_conclusions_audit.md](phase1_conclusions_audit.md)
- [phase1_benchmark_design.md](phase1_benchmark_design.md)
- [phase1_references.md](phase1_references.md)

注意：这些文档中的“最终模型”只表示当时的预测基线收口，不表示当前全项目模型已经定性。

## Phase 2：MPC 探索

- [phase2_plan.md](phase2_plan.md)
- [phase2_results.md](phase2_results.md)
- [phase2_final_audit.md](phase2_final_audit.md)
- [supplementary_experiments.md](supplementary_experiments.md)
- [experiment_audit.md](experiment_audit.md)

优先阅读 `phase2_final_audit.md`。早期“DWM-MPC 优于 PID”的数值已因同构 plant、动作弱因果和协议问题降级。

## 论文叙事与应用边界

- [narrative_restructure.md](narrative_restructure.md)：2026-08-05 的因果主线重构，仍早于 Fan 路线验证。
- [supervisory_mode.md](supervisory_mode.md)：预测驱动与现场监督模式。
- [references.md](references.md)：参考文献索引。
- [papers/README.md](papers/README.md)：外部论文材料。

## 历史文档使用规则

1. 引用数字时同时记录实验编号、数据切分、预测时域和 checkpoint 口径。
2. 带“最终”“定稿”“路线关闭”的旧标题只在其原实验范围内成立。
3. 如果结论被审计文档修正，论文和新 README 只采用修正后的口径。
