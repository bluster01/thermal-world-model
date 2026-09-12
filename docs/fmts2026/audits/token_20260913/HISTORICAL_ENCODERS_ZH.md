# 历史 VariableAttention 编码器与训练记录回溯

日期：2026-09-13。仅查源码与已有训练产物，未训练、未改模型、未 push。

## 结论

用户记忆有依据。最直接对应的是 Phase 1 的
**RevIN → 每变量 patch → 共享参数的 PerVariableTCN → VariableAttention → 保留变量位置的展平读出**。
这条主干后来用于 A1phys；不是本轮的 state-query token observer。
还存在 Phase 3.5 的两种变量 attention 编码器，以及名字相近但实际上用 GRU 的 Gate C 编码器。
不能只按“attention / encoder / hybrid”名字混在一起。

## 1. 能直接对上的 Phase 1 产物

下表逐项读取 `results/exp_025_<ID>/results.json`，不是转抄旧论文表。
数值是历史测试窗口的 `avg_mae_degC`，不是当前 equal-day 验证指标。

| ID | 实现差别 | 历史平均 MAE °C | best_epoch | 参数量 |
|---|---|---:|---:|---:|
| M0 | 完整 Patch + PerVariableTCN + VarAttn | 0.310400 | 7 | 849268 |
| M4 | 去掉 VarAttn，保留逐变量时序编码 | 0.310922 | 7 | 799284 |
| M2 | 去掉 patch，保留逐变量编码 | 0.341288 | 7 | 847476 |
| M3 | 合并变量时序表示的消融 | 0.949693 | 28 | 160308 |
| M7 | 同类主干，固定 β-NLL 策略 | 0.294745 | 9 | 849268 |
| B4 | iTransformer：整段历史投影 + 变量 attention + 目标 token 读出 | 0.359281 | 26 | 58612 |
| M9 | TimeXer：目标 patch + 独立未来动作 cross-attention + flatten head | 0.346955 | 7 | 204340 |

这支持“旧编码器家族曾获得较好的预测表现”，不支持“VarAttn 单层贡献很大”：
M0/M4 接近，而 M3 同时改变了表示与参数容量，不能把差距全部归因于某一层。
旧 `phase1_conclusions_audit.md` 的部分数字已落后于当前结果 JSON（如 M3），此处以产物为准。

代码定位：

- `src/world_model.py:20`：RevIN；`:44`：PatchEmbedding；`:61`：VariableAttention；`:118` 附近：PerVariableTCN。
- `experiments/phase1_dynamics/exp_025_unified_benchmark.py:182`：DirectWM，含组件 flags。
- 同文件 `:100`：TimeXerWM；`:337`：iTransformerBaseline。
- 同文件训练入口：默认 seed 42，AdamW + ReduceLROnPlateau，validation 选择 checkpoint，
  最后 `eval_rollout(model, te, prob)` 生成上述历史 test 数字。

证据限制：这些结果目录当前只有结果汇总，含 best_epoch/训练耗时/18 点误差，
不是完整逐 epoch ledger；本次在主工作区的 M0 目录也未找到 best checkpoint。
脚本可帮助恢复设计，但当前 `src/config.py` 不是每个历史 run 的独立冻结配置。
旧协议 test 曾反复参与开发的审查警告仍然有效，不能把这些数字并入本轮论文对照表。

## 2. 后续确实沿用过，不是未训练的设想

`experiments/phase3_feedforward/causal_arch.py:231` 的 `ResidualCausalWM` / A1phys：
仍用 RevIN + Patch + PerVariableTCN + VarAttn，展平表示接 free head 与干预分支。
`results/exp_106_causal_arch/A1phys_s{0,1,2}_ff10/result.json` 三份记录均存在：

| seed | H | best MAE | best MAE epoch | curve 中 epoch 数 |
|---|---:|---:|---:|---:|
| 0 | 60 | 0.831758 | 21 | 41 |
| 1 | 60 | 0.851398 | 22 | 42 |
| 2 | 60 | 0.857071 | 14 | 34 |

这里保存了逐 epoch loss/MAE 等曲线。它们是不同任务/时域的历史训练记录，
不是当前 H18、23 通道、Fan 状态观察器的复现，也不重新赋予旧 CFI 因果真值地位。

## 3. Phase 3.5 的名称辨析

| 代码 | 实际编码器 | 与当前 token 的区别 |
|---|---|---|
| `src/phase35/model.py:74` HistoryEncoder | 窗内归一化 → 每变量 Linear(W,d) → variable attention → flatten MLP | 没有 state-query 压缩 |
| `src/phase35/multistep/rm3_prediction.py:48` PairedHistoryBackbone | 每变量 Conv1d → 时间均值 → variable attention → 变量 tokens/flatten | 输入用所属预测器的统计量；非 patch-query observer |
| `src/phase35/multistep/gatec_model.py:16` PairedHistoryEncoder | 输入投影 → GRU → LayerNorm | 这里实际上不是 VarAttn |

RM3 的 `P2_m9_future_sp_F0_s0` 有 manifest 与训练指标记录，关联 M9StylePairedPredictor；
它通过 PairedHistoryBackbone 编码历史，再用 future-SP cross-attention 输出基准温度增量。
本次只确认归属和产物存在，不跨协议比较它的分数或重新认证整条 RM3 线。

## 4. 对当前问题的含义

当前 token 是“23×11 个变量-patch tokens → self-attention → 11 个状态 queries →
共享标量 head → tanh 有界修正 → 只保留 4 个状态”，没有 PerVariableTCN，
也没有保留全变量展平读出。旧主干的成功并未在此被原样继承。
另一方面，旧模型直接读出温度，当前仅修正物理初态，所以单纯换回编码器不保证恢复旧分数。

**修正排查优先级：先确认当前观察器的真实历史退化；如需新增候选，优先讨论复用已有
逐变量时序编码 + VarAttn 主干，而不是把新 state-query 设计当作旧方法的代表。**
复用时应保留当前动作/物理接口、9 个新增历史边界、锚点和独立状态读出约束；
如果采用 RevIN，还需保留绝对水平/尺度供物理状态估计使用，不能照搬温度直接预测的反归一化。
以上为回溯时的建议。随后作者明确授权并实现为 [FMTS-M7R1](../../PREREG_M7_FUSION_20260913.md)；尚无新正式结果，不能宣称已经恢复旧分数。
