# Phase 3.5 MS3-R RM3 Orthogonal Response and Fair Backbone Design

## 1. 目的

RM2 已证明局部动作依赖可以稳定复现，但 A1、LPV-Koopman、PI-Neural-ODE 与 causal DeepONet 在预测分数近似时给出相差数倍的 response amplitude。RM3 不继续增加 operator，而先修正两个相互独立的问题：

1. **response identification**：raw logged-future-valve auxiliary 会把闭环策略相关性直接写入显式 response，不能唯一识别 gain；
2. **world-model prediction**：当前小型 Gate C 同时预测 valve、Tin、local 和 terminal，不能据此声称优于 Phase1 M7 或 Phase3 M9DSP。

RM3 仍只使用 train/validation，test、MS4、任意 `do(valve)` 和 operator champion 均冻结。

## 2. 三种方案与决定

| 方案 | 优点 | 缺点 | 决定 |
|---|---|---|---|
| raw logged-action auxiliary | 简单、RM2 已验证稳定 | response 尺度由 loss/parameterization 决定 | 淘汰为 gain 识别目标，仅留历史对照 |
| end-to-end adversarial branch separation | 灵活 | 难审计、仍可能只实现另一种任意分解 | 暂不采用 |
| rolling OOF nuisance + orthogonal R-loss | estimand 清晰、可做 lead/shuffle/rank 负控制 | 只能支持条件局部响应，需单独训练预测 backbone | **采用** |

## 3. RM3-A：正交响应矩

对每个 H60/H180 局部终点，令 (X) 只含 anchor 当时及过去的 history、SP、Tin 与工况。用 expanding rolling folds 分别拟合：

\[
\hat m_U(X)=\widehat E[U\mid X],\qquad
\hat m_Y(X)=\widehat E[Y\mid X].
\]

其中 (U) 是未来阀位相对 anchor 阀位的 dose，(Y) 是未来 `Tin−Tout` 相对 anchor local drop 的变化。只在 out-of-fold 行上形成：

\[
\tilde U=U-\hat m_U(X),\qquad
\tilde Y=Y-\hat m_Y(X).
\]

显式 response 只通过正交矩训练：

\[
L_R=\sum_{h\in\{60,180\}}w_h
\left\|\tilde Y_h-g_\phi(\tilde U_{1:h},c)\right\|_\delta .
\]

禁止把 raw future valve、raw local truth 或 Gate B coefficient 直接作为 response label。Gate B 仅提供符号、时程与数量级诊断，不是真值。nuisance 与 response 的 selector/report anchors 必须分离；seed 只表示优化波动，UTC 日/连续块才是统计单位。

### 主门与负控制

- residual action Gram condition number、common/differential energy；
- H60/H180 每侧 `correct − |wrong|` 与 `future − |lead|` 的逐日配对区间；
- shuffled action residual 必须失效；
- 对已知真值且激励充分的 synthetic DGP，OOF gain 必须恢复；秩亏 DGP 必须拒绝 differential claim；
- A1 scheduled 与 common-only 先进入校准；其它 operator 只有在相同正交目标下才可比较。

若跨 fold gain 尺度仍不收敛，输出部分辨识区间/robust model set，不再通过增大模型容量补救。

## 4. RM3-B：公平预测架构

物理链路不实现为长串点预测头。最终候选采用“高容量 prediction backbone + joint latent dynamics + physical interfaces”：

\[
x_{k+1}=F_\theta(x_k,c_k)+B_T(c_k)T_{in,k}+B_u(c_k)\phi(v_k),
\]

\[
[\hat v,\hat T_{in},\hat T_{out},\hat T_{main}]=C_\theta(x_k).
\]

controller/actuator、local spray response、downstream latent thermal block 是仅保留的三个可观测接口。未测喷水流量、混合与金属壁温合并入稳定 latent，不拆成无真值串行模块。

terminal 允许 action-invariant direct residual bypass：

\[
\hat T_{main}=T_{main}^{latent}+r_{terminal}(history),
\]

其中 bypass 不读取未来动作，动作扰动前后严格不变。oracle/forecast 双模式使用一致性蒸馏；downstream robustness 使用 OOF valve/Tin prediction residual 注入，不使用任意高斯噪声。

## 5. 公平预测表

所有候选使用同一 paired A/B 数据、H60、rolling folds、train-only normalization、相同 reporting anchors。旧 Phase1/3 test 数字只作历史背景，不进入判决。

| ID | 输入权限 | 角色 |
|---|---|---|
| P0 M7 oracle valve | history + true future valve | action-oracle upper bound；不参与部署排名 |
| P1 M7 predicted valve | history + future SP → predicted valve | 高容量 direct baseline |
| P2 M9 future SP | history + future SP | action cross-attention baseline |
| P3 Gate C paired-free | 15-column paired history + future SP | 小模型无显式 response 负控 |
| P4 Gate C A1 scheduled | 同 P3 | RM2 reference |
| P5 hybrid joint latent | M7/M9-capacity backbone + OOF-calibrated explicit local response | 最终候选 |

分别报告 oracle-action、forecast-action、valve/Tin/local/terminal MAE、correct/wrong/lead placebo、gain 跨 fold 稳定性与计算预算。不得用 composite score 掩盖 terminal 或 local 的一致退化。

实现审计说明：RM3 中的 `M7-style` 与 `M9-style` 指在统一 paired 数据合同上重建对应的高容量归纳偏置，不复用旧单侧 40 列 checkpoint，也不声称参数级复现。M7 dense action injection 不满足 prefix causality，只保留为预测基线；M9 adapter 改为逐时刻 causal action attention。六候选的输出监督域不同，禁止使用一个 composite score 横跨 `terminal_only`、`valve_and_terminal` 与 `full_multitask` 三类输出作冠军排名。

冻结真实矩阵 envelope 为：6 prediction candidates × 2 folds × 3 seeds=`36`，A1 scheduled/common-only 两个 response calibration candidates × 2 folds × 3 seeds=`12`，合计 `48` runs。该数字只是封闭预算与产物设计，当前 `linux_authorized=false`。

## 6. 阶段边界

本地本轮只实现：冻结合同、OOF residualization、正交矩/R-loss、synthetic recovery/rank/placebo 测试、矩阵 dry-run 和 joint-latent interface 的结构测试。没有真实长训练授权。代码与 micro smoke 完成并经 Supervisor 检查后，才可另行冻结 Hermes 批次。
