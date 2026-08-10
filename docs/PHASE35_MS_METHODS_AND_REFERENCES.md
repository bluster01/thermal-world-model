# Phase 3.5-MS 多步动作响应：方法、推导与参考文献

> 版本：`phase3.5-ms-v1`，更新至 2026-08-10。本文描述当前代码已经实现的内容，不代表 Linux 正式结果，也不恢复旧 E3/E4 现场因果结论。代码入口为 `src/phase35/multistep/`；当前 Gate 由 `configs/phase3_5/experiment_registry.json` 指向 `ms2d_delay_matrix.json`。

## 1. 研究问题与证据边界

Phase 3.5-MS 首先回答一个方法学问题：当动作—响应真值已知时，同一个模块化架构能否从多种阀位轨迹中恢复稳定的多步温度增量响应。它把完整预测拆成

\[
\widehat{T}^{(a)}_{1:H}
=f_{\mathrm{free}}(c)+g_R(c,a_{1:H},a^{\mathrm{ref}}_{1:H}),
\tag{1}
\]

其中 \(c\) 是处理前历史编码或工况上下文，\(a_{1:H}\) 是候选阀位轨迹，\(a^{\mathrm{ref}}_{1:H}\) 是参考阀位轨迹，\(R\) 表示一种响应表示。`f_free` 的接口不接收未来动作，因而不能直接吸收未来阀位路径；本轮 synthetic benchmark 只训练 \(g_R\)，尚未接入 Phase 1 checkpoint。

模型输出的相对响应为

\[
\widehat{\Delta T}^{,a:a^{\mathrm{ref}}}_{1:H}
=g_R(c,a_{1:H},a^{\mathrm{ref}}_{1:H}).
\tag{2}
\]

式（2）是模型定义的路径差，不自动等于现场因果效应。只有当动作语义、可交换性/common support、处理前平衡、时间顺序和经验响应参考均通过独立设计时，才可能进一步解释为 \(do(a)\) 对比。当前真实数据 E3 仍为 `INCONCLUSIVE`，因此 MS1 的允许表述仅为“已知真值系统上的表示与优化可行性”。

## 2. 动作、单位与参考路径

主动作是现场二级减温阀反馈开度，单位为百分比；它是喷水作用的代理，不是 kg/s 质量流量。A/B 侧按已确认的现场交叉链分开建模，不与监督层 SP 任务混榜。有效开度差定义为

\[
u_k=\phi(a_k)-\phi(a_k^{\mathrm{ref}}),
\tag{3}
\]

其中 \(\phi:[0,100]\rightarrow[0,100]\) 是端点固定的单调映射。代码提供 identity、固定 equal-percentage 先验和正斜率分段线性映射；MS1 冻结为 \(\phi(a)=a\)，避免把阀门非线性和动力学表示同时改变。非线性 \(\phi\) 属于 MS2 失配压力实验。

所有算子从零增量状态开始：\(z_0=0\)。因此参考路径相等时 \(u_k=0\)，但只有递推或差分结构同时正确时，才能保证全时域零响应。

## 3. 共同结构合同

四条路线共享以下合同：

1. **参考恒等式**

\[
g_R(c,a^{\mathrm{ref}},a^{\mathrm{ref}})\equiv0.
\tag{4}
\]

2. **时间因果性**：若两条动作路径在 \(1{:}k\) 相同，则对应输出在 \(1{:}k\) 必须相同：

\[
a_{1:k}=\widetilde a_{1:k}
\Rightarrow
g_R(c,a,a^{\mathrm{ref}})_{1:k}
=g_R(c,\widetilde a,a^{\mathrm{ref}})_{1:k}.
\tag{5}
\]

3. **状态续传**：递推路线在任意切点 \(m\) 分段执行，并把第一段末状态传入第二段，应与整段 rollout 一致。

4. **估计对象一致**：相同上下文、动作/参考路径、样本、loss、optimizer-update 预算和 validation selector；不把旧 Koopman free-head、SP→阀门监督层模型或 Fan plant 模型混入同一冠军表。

式（4）和（5）是架构恒等式/信息流性质，不是从数据学习得到的经验结论。它们可排除一类伪反事实，但不能解决隐藏混杂。

## 4. Stable Graybox-1P/2P/3P

### 4.1 离散动力学

对第 \(i\) 个一阶环节定义

\[
\alpha_i=\exp(-\Delta t/\tau_i),\qquad \tau_i>0.
\tag{6}
\]

一阶模型为

\[
z_{1,k}=\alpha_1z_{1,k-1}+(1-\alpha_1)u_k,
\qquad
\widehat{\Delta T}_k=Kz_{1,k}.
\tag{7}
\]

二阶串联模型增加

\[
z_{2,k}=\alpha_2z_{2,k-1}+(1-\alpha_2)z_{1,k},
\qquad
\widehat{\Delta T}_k=Kz_{2,k}.
\tag{8}
\]

代码使用

\[
K=-\operatorname{softplus}(\kappa),qquad
\tau_i=\tau_{\min}+(\tau_{\max}-\tau_{\min})\sigma(\eta_i),
\tag{9}
\]

从参数化上保证 \(K<0\) 和 \(\tau_i\in(\tau_{\min},\tau_{\max})\)。

MS2-C 的 context-scheduled A1phys-MS 在该全局参数上增加有界对数尺度调度：

\[
K(c)=K_0\exp\{s\tanh(w_K^\top c)\},\qquad
\tau_i(c)=\operatorname{clip}\!\left[
\tau_{i,0}\exp\{s\tanh(w_{\tau_i}^\top c)\},
\tau_{\min},\tau_{\max}
\right],
\tag{9a}
\]

其中 \(s=0.5\)，调度权重零初始化。该形式在全部 context 下保持 \(K(c)<0\)、\(\tau_i(c)>0\)，但单个 \(K/\tau\) 与阀门映射仍可能存在等价补偿，必须结合已知真值参数和响应误差审计。

### 4.2 MS2-D1 因果纯迟延

MS2-D1 在进入惯性环节前引入只依赖当前与过去动作的有限迟延核：

\[
\widetilde u_k=\sum_{d=0}^{D}w_d u_{k-d},\qquad
w_d=\frac{\exp(q_d)}{\sum_{j=0}^{D}\exp(q_j)},
\quad u_{k-d}=0\ (k-d<0).
\tag{9b}
\]

因此 \(w_d\ge0\)、\(\sum_dw_d=1\)，不会读取未来动作；模型状态除惯性状态外还包含长度 \(D\) 的动作缓冲区，分段 rollout 必须续传该缓冲区。fixed-delay 正控将 \(w_2=1\)，对应 \(\Delta t=10\,\mathrm s\) 下的 20 s 真值；learned-delay 使用 \(D=4\)，并单独报告

\[
\widehat L=\Delta t\sum_{d=0}^{D}d\,w_d.
\tag{9c}
\]

\(\widehat L\) 接近真值是参数恢复诊断，不是响应门禁的必要同义条件。为避免 0–4 step 的均匀核在未训练时就恰好得到 2 step 期望值，learned logits 初始化偏向 \(d=0\)；参数诊断还要求真值 ±1 step 邻域质量 \(\sum_{d=1}^{3}w_d\ge0.80\)。迟延核与惯性时间常数仍可能补偿，所以必须把 learned-delay 相对同结构 no-delay 的 clean-response 改善和参数恢复分开报告。

### 4.3 稳定性与方向

由 \(\tau_i>0\) 和 \(\Delta t>0\) 可得 \(0<\alpha_i<1\)，每个一阶环节都是 BIBO 稳定。MS2-C 进一步由有限 \(\tau_{\max}\) 给出统一的 \(\alpha_i(c)<1\) 上界。对固定 context 和常值剂量 \(u_k=u^*\)，稳态满足 \(z_{1,\infty}=z_{2,\infty}=u^*\)，故

\[
\widehat{\Delta T}_{\infty}=Ku^*.
\tag{10}
\]

当开阀使 \(u^*>0\) 时，\(K<0\) 保证长期降温方向。该约束只针对阀位代理的有效开度，不把 \(K\) 解释成喷水质量流量增益。

一/二阶惯性与热工对象常用的 FOPDT/串联惯性描述一致；主汽温系统的导前区/惰性区分段辨识可作为工程先验。MS1 没有显式纯迟延项；MS2-D1 才把纯迟延作为单独的已知真值压力轴（Cao et al., 2021；Brolese, 2021）。

## 5. Stable Controlled Modal Operator（Koopman-family）

### 5.1 当前实现

当前路线采用稳定对角潜状态：

\[
z_{k}=Az_{k-1}+Bu_k,
\qquad
\widehat{\Delta T}_k=Cz_k,
\tag{11}
\]

\[
A=\operatorname{diag}(\lambda_1,\ldots,\lambda_d),
\quad 0<\lambda_i<1,
\tag{12}
\]

并参数化为

\[
\lambda_i=\sigma(\ell_i),\quad
B_i=\frac{1-\lambda_i}{d}\operatorname{softplus}(b_i),\quad
C_i=-\operatorname{softplus}(c_i).
\tag{13}
\]

因此谱半径 \(\rho(A)=\max_i\lambda_i<1\)。对常值正剂量，每个状态的稳态符号非负，\(C_i<0\) 使总稳态响应非正。显式 \(B\) 把外部动作与自主动力学分开，这与 DMD with control 和 controlled Koopman linear predictors 的基本结构一致（Proctor et al., 2016；Korda & Mezić, 2018）。

### 5.2 命名与可辨识性边界

本实现没有从完整 plant state 学习非线性 encoder/decoder，也没有证明潜变量是 Koopman eigenfunctions。更准确的名称是 **stable diagonal controlled modal response operator**；`Koopman-K2/K4` 是矩阵中的方法族简称。Lusch et al.（2018）的 deep Koopman 工作支持“学习坐标后线性演化”的一般思路，但不能为本实现自动赋予 Koopman 谱解释。

式（11）还存在相似变换和 \(B/C\) 缩放非唯一性。除非增加状态锚定、规范化和独立激励，潜状态与单个模态参数不能解释成唯一物理量。当前只把 \(\rho(A)\)、rollout 稳定性和响应误差作为诊断，不把 \(\lambda_i\) 直接宣称为真实锅炉时间常数。

MS1 实现中的 \(A/B/C\) 对全部样本共享，`context` 仅为统一数据接口保留，并不调度算子参数。因此它不是 \(A(c),B(c)\) 型工况调度 Koopman/LPV 模型。若后续让算子依赖工况，必须改称 context-scheduled/LPV controlled operator，并重新审计逐工况谱半径和切换稳定性，不能沿用当前稳定性结论。

## 6. Physics-informed ODE / neural closure

### 6.1 名义模型与闭合项

PI-ODE 使用二阶连续时间名义模型：

\[
\dot z_1=\frac{u-z_1}{\tau_1}+r_1(c,z,u),
\tag{14}
\]

\[
\dot z_2=\frac{z_1-z_2}{\tau_2}+r_2(c,z,u),
\qquad
\widehat{\Delta T}=Kz_2.
\tag{15}
\]

神经闭合写成

\[
r(c,z,u)=s_r\,
\frac{|u|+\|z\|_1}{1+|u|+\|z\|_1}
\tanh\!\left(N_\theta[c,z,u]\right),
\tag{16}
\]

其中 \(s_r\) 是冻结的闭合幅值上限。最后一层零初始化，使训练开始时模型等于名义 ODE。代码用 RK4、每个 10 s 采样间隔两个子步积分。

当 \(a=a^{\mathrm{ref}}\) 时，\(u=0\)、\(z_0=0\)，式（16）的 gate 为零；式（14）—（15）的唯一递推解保持 \(z=0\)，从而得到式（4）。这比只在 loss 中惩罚零响应更强。

### 6.2 与 PINN/Neural ODE 的关系

该路线不是在时空 collocation 点求解 PDE 的经典 PINN。它是“已知名义 ODE＋小型可学习闭合”的 universal differential equation/physics-informed neural ODE；Neural ODE、PINN 和 UDE 文献分别提供可微 ODE、物理残差正则和已知方程与通用逼近器混合的依据（Chen et al., 2018；Raissi et al., 2019；Rackauckas et al., 2020）。

闭合损失为

\[
\mathcal L_{\mathrm{phys}}
=\frac{1}{2B H}\sum_{n,k}\sum_{j\in\{1,2\}}r_{n,k,j}^2.
\tag{17}
\]

闭合项可能改变名义方向，也可能破坏整体稳定性，因此 PI-ODE 在代码中明确标记 `direction_constrained=false`。正时间常数或名义线性块谱半径小于 1，只证明名义块稳定，不构成含神经闭合项的完整 ODE 的 Lyapunov 稳定性证明；必须用正阶跃终值方向、有限时域 rollout、闭合幅度和失配实验单独门禁。PINN 的软约束会导致病态优化或复杂物理学习失败，故本项目不采用“大而全 PINN”，并保留 curriculum/分阶段训练作为后续选项（Krishnapriyan et al., 2021）。

## 7. Causal DeepONet

标准 DeepONet 用 branch net 编码输入函数、trunk net 编码输出坐标（Lu et al., 2021）。如果 branch 一次读取完整动作路径，后段动作可能改变早期输出，不满足式（5）。当前实现将 branch 改为前缀因果 GRU：

\[
b_k=\operatorname{GRU}\!\left(
[\phi(a_k)/100,c],b_{k-1}
\right),
\tag{18}
\]

\[
G_k(c,a_{1:k})
=\frac{s_G}{\sqrt d}\,b_k^\top q(k/H),
\tag{19}
\]

\[
g_{\mathrm{DO}}(c,a,a^{\mathrm{ref}})_k
=G_k(c,a_{1:k})-G_k(c,a^{\mathrm{ref}}_{1:k}).
\tag{20}
\]

式（20）通过共享权重的严格相减保证式（4）；式（18）只接收前缀，保证式（5）。这是项目为控制路径设计的 **Causal DeepONet**，不是 Lu et al.（2021）原始全路径 branch 的原样复现。physics-informed DeepONet 表明可把算子学习与物理残差结合（Wang et al., 2021），但当前 R6 没有加入物理残差；PI-DeepONet 仍是后续消融，不应提前写成已实现。

Causal DeepONet 在当前代码中固定 \(H=60\)，不接受不同 horizon，也不宣称能够无限递推；与三条 stateful 路线的比较限于同一固定时域响应预测。

## 8. Synthetic known-truth protocol

### 8.1 真值生成器

MS1 使用二阶稳定真值：

\[
K^*=-0.04\ {^\circ\mathrm C}/{\%},\qquad
(\tau_1^*,\tau_2^*)=(70,210)\ \mathrm s,
\quad \Delta t=10\ \mathrm s.
\tag{21}
\]

动作包含 hold、step、pulse、ramp 和 multi-step，参考开度在 12%–48% 随机取值，幅值与方向随机，最后裁剪到 0%–100%。训练、validation、test 使用不同确定性 seed offset，不复用动作路径或噪声；观测响应叠加 \(\sigma=0.02\ ^\circ\mathrm C\) 高斯噪声。

Graybox-2P 与真值结构同型，因此 MS1 是正对照并存在 **inverse crime**。Graybox-2P 在 MS1 胜出只能证明代码/优化能恢复同型系统，不能证明它优于 Koopman、PI-ODE 或 DeepONet。公平路线判断必须进入 MS2，至少加入阶次失配、工况变参数、非线性有效开度、纯迟延和未建模扰动。

MS1 真值动力学也不依赖随机 context \(c\)，所以 \(c\) 在这一阶段是干扰变量，而不是宽负荷物理条件。Graybox 与 controlled modal routes 忽略它，PI-ODE 闭合项和 DeepONet branch 可以读取它；这只检验模型是否会错误利用无关 context。MS2 若引入随工况变化的时间常数、增益或迟延，必须显式改写真值并重新冻结数据协议，不能把 MS1 结果表述成“宽负荷验证”。

### 8.2 冻结矩阵

MS1 正式 validation 矩阵为 6 routes × 3 seeds：Graybox-1P/2P、Koopman-K2/K4、PI-ODE、Causal-DeepONet。每个 run 使用 train/validation/test = 1024/256/256 条 synthetic trajectory；训练最多 100 epochs。

MS2-V/C 冻结两个独立失配轴：`valve_nonlinear_r50` 的 6 candidates 与 `context_scheduled_2p` 的 5 candidates，共 11 candidates × 3 seeds = 33 runs；validation+一次性 synthetic test 已完成。两轴的主响应对比均通过，但 learned `phi` 不可单独辨识。MS2-J 随后在同一真值同时启用 R50 非线性和 context 调度，以 9 candidates × 3 seeds 比较双模块 joint/staged、单模块消融及灵活路线；validation+一次性 test 均已完成（`5260d3f`）：联合模块双层 PASS（test CI 下界 0.73–0.89 >> 20%），staged 非劣双层 FAIL（test ratio 1.14–1.20），主训练方案定 joint。

当前 MS2-D1 只在上述联合真值上增加 20 s pure delay，冻结 6 candidates × 3 seeds = 18 validation runs：no-delay 主消融、learned-delay 主模型、fixed-delay+R50 oracle，以及 Koopman/PI-ODE/DeepONet 次要表示参考。主要响应门是 learned-delay 相对 no-delay 每 seed clean NMAE 改善至少 20%；oracle 每 seed clean NMAE 必须小于 0.05；期望迟延误差不超过 1 step 且真值邻域质量不低于 0.80 单列为参数诊断。D1 不提供 test 入口，D2 三阶惯性与 D3 未建模扰动仍等待 D1 审计。完整设计见 [`plans/2026-08-10-phase35-ms2d-pressure-design.md`](plans/2026-08-10-phase35-ms2d-pressure-design.md)。

## 9. 训练目标、选模与评测

数据项使用 \(\beta=0.2\) 的 Huber loss：

\[
\mathcal L_{\mathrm{data}}
=\frac1{BH}\sum_{n,k}
\operatorname{Huber}_{0.2}
(\widehat{\Delta T}_{n,k}-\Delta T_{n,k}).
\tag{22}
\]

总损失为

\[
\mathcal L
=\mathcal L_{\mathrm{data}}
+\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}},
\qquad \lambda_{\mathrm{phys}}=0.01,
\tag{23}
\]

其中非 PI-ODE 路线的 \(\mathcal L_{\mathrm{phys}}=0\)。唯一 checkpoint selector 是 validation effect MAE；训练阶段不能读取 synthetic test。

报告指标包括

\[
\mathrm{MAE}=\frac1{BH}\sum|e_{n,k}|,
\quad
\mathrm{RMSE}=\sqrt{\frac1{BH}\sum e_{n,k}^2},
\tag{24}
\]

H1/H6/H18/H60 MAE、每条轨迹 integrated absolute error 和非零响应方向率。当前 integrated absolute error 是离散绝对误差和 \(\sum_k|e_k|\)，没有乘采样间隔。MS1 的旧方向率使用带噪 target，已在结果审计中降为不可区分诊断；MS2 同时保存无噪声 `clean_effect`，主要报告 clean MAE/NMAE，并只在 \(|\Delta T_{clean}|>0.01\ ^\circ\mathrm C\) 处计算 clean-direction。结构诊断独立报告：

- `reference_identity_max_error`；
- `future_action_leakage_max_error`；
- `post_change_sensitivity_max_c`，防止“完全不看动作”的模型只靠零泄漏过关；
- `positive_step_terminal_effect_max_c`；
- 状态/输出有限性；
- Graybox/Koopman 名义谱半径；
- PI-ODE closure residual。

validation 审计后才冻结候选。synthetic test 使用独立命令原样加载 canonical checkpoint，写一次性 `synthetic_test_access_ledger.json`；重复访问拒绝。这个授权不扩展到 A/B 真实数据 test。

## 10. 可辨识性、可解释性与允许主张

| 层级 | 当前能证明什么 | 当前不能证明什么 |
|---|---|---|
| 代码恒等式 | 零参考响应、prefix causality、递推状态续传 | 现场因果识别 |
| MS1 synthetic | 同型二阶系统上的参数/响应可恢复性 | 路线普遍优越性、真实阀门增益 |
| MS2-V/C mismatch（validation+test 已完成） | 合成真值内非线性响应容量与 context 通道的模块价值 | learned 阀门曲线、联合收敛、纯迟延/扰动、现场 `do(valve)` |
| MS2-J coupling（validation+test 双层：联合模块 PASS、staged 非劣 FAIL，`5260d3f`） | 双模块联合响应可辨识；当前 staged 协议未达到 1.10 非劣界 | 单独恢复 `K/phi`、所有 staged 方案优劣、真实数据迁移与现场因果响应 |
| MS2-D1 pure delay（代码与协议已冻结，validation 待返回） | 可检验显式迟延模块对已知真值响应恢复的增量价值 | 在结果审计前不能声称迟延已恢复；响应 PASS 也不等于迟延核唯一可辨识 |
| MS3 real validation（未实现） | A/B 观测预测与模型敏感性 | 未控制混杂下的反事实效应 |
| MS4 new-time E3/E4 | 若门禁通过，可比较经验响应与模型响应 | 超出数据支持域的闭环安全性 |

特别地：

- 阀位—流量关系未知，\(\phi\) 只能称有效开度映射；
- Koopman latent modes 非唯一，不能直接贴物理状态标签；
- PI-ODE closure 是模型失配吸收项，不是未观测物理机制的证据；
- DeepONet 的 fixed-horizon operator 不能替代 state-closed simulator；
- synthetic test 阳性不能解除真实 E3 的 common-support 阻断。

## 11. 多阶段耦合训练的冻结原则

真实数据阶段拟采用：

1. 单独预训练或加载 `f_free`；
2. 冻结 `f_free`，训练 \(g_R\) 并检查动作敏感性/参数健康；
3. 以较小学习率联合微调，同时保留响应与结构损失；
4. 分阶段保存 checkpoint，不允许只保留联合阶段最优模型。

这是完整 `free+response` 世界模型的待实现设计，不是当前训练代码已有能力。MS2-J 只在 response 内部测试了 base/opening/schedule 三阶段训练，且当前协议未达到 joint 的 1.10 非劣界；它不能代替 MS5 对 free head 吸收动作信号的检验。是否采用短冻结必须由 MS5 的 stage-wise gradient、参数漂移和消融结果决定，不能只凭文献、直觉或 MS2-J 外推。

## 12. 代码—公式—测试追溯

| 内容 | 代码 | 主要测试 |
|---|---|---|
| 合同与式（4）—（5） | `multistep/contracts.py` | `test_all_routes_obey_shape_and_exact_reference_identity`、`test_future_action_cannot_change_earlier_response` |
| 式（6）—（10）与（9a） | `multistep/operators.py::StableGrayboxOperator` | Graybox 方向/时间常数测试、`test_context_scheduled_graybox_varies_parameters_without_losing_direction_or_stability` |
| 式（11）—（13） | `multistep/operators.py::ControlledKoopmanOperator` | `test_controlled_koopman_is_stable_and_not_the_legacy_free_head` |
| 式（14）—（17） | `multistep/operators.py::PhysicsInformedODEOperator` | `test_pi_ode_reports_finite_neural_closure_penalty` |
| 式（18）—（20） | `multistep/operators.py::CausalDeepONetOperator` | fixed-horizon、future-action causality 测试 |
| 式（21） | `multistep/synthetic.py` | deterministic split、split non-reuse 测试 |
| 式（22）—（24）与 test ledger | `multistep/training.py` | CLI smoke、artifact、repeat-test refusal 测试 |
| 递推组合律 | 三条 stateful operator | `test_recursive_routes_preserve_state_across_rollout_chunks` |
| MS2-V/C 真值与 clean metrics | `multistep/synthetic.py`、`multistep/training.py` | nonlinear/context truth、MS2 CLI 与 checkpoint hash 测试 |
| MS2-J coupling 与一次性 test | `joint_coupling.py`、`staging.py`、`joint_coupling_test.py`、`summarize_joint_coupling_test.py` | 27-run freeze、stage A/B/C、content-address、repeat refusal、paired episode gate 测试 |
| MS2-D1 pure delay | `synthetic.py`、`operators.py`、`ms2d_delay.py`、`summarize_ms2d_delay.py` | exact delay timing、simplex、buffer continuation、18-run freeze、artifact/response/parameter 分离门禁 |

## 13. Reference ledger

下列条目已在 2026-08-09 通过出版机构、会议官方页面、DOI 或机构知识库核验。引用只支持其右栏所列方法来源，不自动支持本项目的实验结论。

| ID | 参考文献 | 本项目使用范围 | 核验入口 |
|---|---|---|---|
| R1 | 曹振乾、印江、张津华（2021）. 基于改进粒子群算法的主蒸汽温度系统辨识. *系统仿真学报*, 33(10), 2411–2419. | 主汽温分段/FOPDT 工程辨识背景 | https://doi.org/10.16182/j.issn1004731x.joss.20-0609 |
| R2 | Brolese, L. (2021). *Steam temperature control system for a water tube boiler* [Master's thesis, Politecnico di Milano]. | 减温器/过热器低阶模块化建模背景；非伊敏真值 | https://hdl.handle.net/10589/181714 |
| R3 | Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016). Dynamic mode decomposition with control. *SIAM Journal on Applied Dynamical Systems*, 15(1), 142–161. | 将自主动力学与外部控制输入分开的线性受控表示 | https://doi.org/10.1137/15M1013857 |
| R4 | Korda, M., & Mezić, I. (2018). Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control. *Automatica*, 93, 149–160. | controlled Koopman linear predictor 的方法族来源 | https://doi.org/10.1016/j.automatica.2018.03.046 |
| R5 | Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). Deep learning for universal linear embeddings of nonlinear dynamics. *Nature Communications*, 9, 4950. | 学习坐标后线性演化；用于限定当前路线与完整 deep Koopman 的差别 | https://doi.org/10.1038/s41467-018-07210-0 |
| R6 | Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems*, 31. | 可微连续时间神经动力学 | https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html |
| R7 | Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. | 物理残差作为软约束；当前 PI-ODE 不是其 PDE collocation 原样实现 | https://doi.org/10.1016/j.jcp.2018.10.045 |
| R8 | Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., Skinner, D., Ramadhan, A., & Edelman, A. (2020). Universal differential equations for scientific machine learning. *arXiv preprint* arXiv:2001.04385. | 已知 ODE 与可学习闭合混合；预印本，引用时标明状态 | https://arxiv.org/abs/2001.04385 |
| R9 | Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. *Advances in Neural Information Processing Systems*, 34. | PINN 软约束的病态优化风险与 curriculum/sequence 策略 | https://proceedings.neurips.cc/paper_files/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html |
| R10 | Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3, 218–229. | branch/trunk operator learning来源；当前 prefix-causal 修改是项目设计 | https://doi.org/10.1038/s42256-021-00302-5 |
| R11 | Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. *Science Advances*, 7(40), eabi8605. | 物理残差与 DeepONet 的可组合性；当前 R6 尚未实现该组合 | https://doi.org/10.1126/sciadv.abi8605 |

## 14. 引用审计结论

- 文内 R1–R11 均在 Reference ledger 中出现，无孤儿引用。
- DOI/handle/官方会议 URL 已补齐；R8 明确标为 arXiv preprint。
- 本文没有把文献中的性能数字移植为本项目结论。
- `Causal DeepONet`、subtractive reference identity、稳定对角受控模态参数化和本项目 fail-closed test protocol 是项目设计，应引用本文/代码而不是错误归因给 R3–R11。
- Brolese（2021）是硕士论文，只作工程结构背景；不能单独支撑伊敏现场时间常数或增益。
