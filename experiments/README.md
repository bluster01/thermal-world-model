# 实验地图

本目录保留按时间演化的研究脚本。为了保证历史结果可追溯，不移动或重命名旧文件；当前唯一正式入口是 `phase3_5/`。

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

历史模块（存在 test-selection、fallback 或旧 estimand，不再直接用于正式结论）：

- `causal_eval.py`：动作构造、事件选择、DiD/CFE 指标。
- `causal_arch.py`：A1/A3/B1、`g(x,0)=0`、A1phys 与 Koopman free-head。
- `exp_104_did_groundtruth.py`：DiD 真值。
- `exp_106_causal_arch.py`：因果架构训练。
- `exp_107_did_eval.py`：跨时程 DiD 评测。
- `exp_109_p2_expand.py`：扩展事件集。
- `exp_110_baselines.py`：完整基线汇总。
- `exp_112_koopman_full.py`：Koopman free-head 三 seed 完整对照。
- `exp_201_valve_action.py`：绝对/差分/固定等百分比阀位 pilot；方向信号有启发，但 test-selected，不是正式 Phase 3.5 入口。

## `phase3_5/`

Phase 3 论文核心验证的正式入口，覆盖 E1–E5：动作表征、阀门非线性、真实阀门事件响应、A1phys 反事实响应和 SP 未执行负对照。

- `prepare_data.py`：从异步稀疏 CSV 构造 causal 10 s cache、staleness 与 SHA256 manifest。
- `train.py`：单 run、validation-only checkpoint 训练。
- `evaluate.py`：分离 validation/test 的预测、事件和负对照评估；test 写 access ledger。
- `run_matrix.py`：42-run 开发矩阵 dry-run/执行及候选 seed 补跑。
- `summarize.py`：A/B、seed 和 E1–E5 门禁汇总。
- `README.md`：Linux 唯一执行手册。

模型与评测代码位于 `src/phase35/`，版本化矩阵位于 `configs/phase3_5/experiment_matrix.json`，测试位于 `tests/phase35/`。

## 新实验规则

1. 新脚本仍放在对应阶段目录，命名保持唯一实验编号。
2. 必须有 `if __name__ == '__main__':` 保护。
3. 训练、验证和测试事件严格隔离。
4. 结果保存到独立目录，配置变化不得覆盖旧结果。
5. checkpoint 用验证集选择，测试集不参与调参。
6. 模型比较同时报告预测、干预、物理、泛化与计算指标。
7. Phase 4/Fan 路线当前暂停，不得混入 Phase 3.5 配置或排行榜。
8. 本地负责设计、实现、测试和 smoke；Linux 远端只执行已提交的固定版本，不直接热修。
9. 远端回传结果必须包含 commit、命令、环境、seed、日志、退出状态和结果文件；审计完成前状态只能是 `results_returned`。
10. Linux 只能写当前 Gate 的结果目录和标记为 `UNVERIFIED_REMOTE_REPORT` 的回传记录；注册表、TODO、上下文快照和 Supervisor 文档只由本地更新。

当前优先级见根目录 `TODO.md`；精确命令见 `phase3_5/README.md`。
