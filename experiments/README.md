# 实验地图

本目录保留按时间演化的研究脚本。为了保证历史结果可追溯，本轮不移动或重命名文件。

## `phase1_dynamics/`

预测建模、事件研究、基线与早期动力学表达。

关键入口：

- `exp_025_unified_benchmark.py`：后续大量实验共用的数据、模型与切分入口。
- `exp_020_koopman_vs_gru.py`：早期 GRU / Koopman / 简化 Neural ODE decoder 对比。
- `event_study_valve*.py`：真实阀门事件研究。

`exp_020` 不是 Fan 灰箱模型验证：它不含 Fan 守恒方程、焓值链或宽负荷参数。

## `phase2_mpc/`

MPC、PID、公平协议、鲁棒性和控制方法对照。

关键入口：

- `eval_protocol.py`：统一评测基础设施。
- `exp_S1_fair_comparison.py` 至 `exp_S6_disturbance_scan.py`：审计后的补充实验。
- `exp_086_final_main.py`：旧 Phase 2 收口入口，仅作历史控制探索。

当前不继续扩展 MPC-vs-PID 主线。其外推局限见 `docs/phase2_final_audit.md`。

## `phase3_feedforward/`

SP 通道、监督模式、因果评测和当前灰箱候选。

当前有效模块：

- `causal_eval.py`：动作构造、事件选择、DiD/CFE 指标。
- `causal_arch.py`：A1/A3/B1、`g(x,0)=0`、A1phys 与 Koopman free-head。
- `exp_104_did_groundtruth.py`：DiD 真值。
- `exp_106_causal_arch.py`：因果架构训练。
- `exp_107_did_eval.py`：跨时程 DiD 评测。
- `exp_109_p2_expand.py`：扩展事件集。
- `exp_110_baselines.py`：完整基线汇总。
- `exp_112_koopman_full.py`：Koopman free-head 三 seed 完整对照。

## 新实验规则

1. 新脚本仍放在对应阶段目录，命名保持唯一实验编号。
2. 必须有 `if __name__ == '__main__':` 保护。
3. 训练、验证和测试事件严格隔离。
4. 结果保存到独立目录，配置变化不得覆盖旧结果。
5. checkpoint 用验证集选择，测试集不参与调参。
6. 模型比较同时报告预测、干预、物理、泛化与计算指标。
7. Fan 路线开始实现后，先增加新的独立阶段目录；模型定性后再决定是否迁入 `src/`。
8. 本地负责设计、实现、测试和 smoke；Linux 远端只执行已提交的固定版本，不直接热修。
9. 远端回传结果必须包含 commit、命令、环境、seed、日志、退出状态和结果文件；审计完成前状态只能是 `results_returned`。

当前优先级见 `docs/CURRENT_TASKS.md`。
