# MS3-R Gate C 路线特定算子实施计划

日期：2026-08-11

## 1. 需求与边界

功能要求：四条路线共用 history encoder、SP→valve、Tin boundary、downstream latent、loss 和 selector，只替换局部 `valve → Tin−Tout` 算子。每条路线必须满足 A/B 配对输出、600 s 有限 rollout、前缀因果、恒动作 identity、梯度可达和统一诊断接口。

非功能要求：稳定性由参数化保证而不是训练后碰运气；未知喷水流量、压力和焓不能伪装成白箱真值；真实训练前必须先通过已知真值和秩亏负控制；Linux 只接收后续独立授权的冻结命令。

声明边界：这些路线估计的是 observed-policy 下、给定 measured/forecast/scenario Tin 的阀位条件响应，不是任意 `do(valve)` plant identification。

## 2. 共同输入与开度映射

令归一化阀位为 `s=v/100`。共享单调开度代理为：

`phi(v)=sum_p softmax(a)_p s^p, p in {1,2,3}`

因此 `phi(0)=0`、`phi(100)=1` 且单调。动作剂量为：

`q_k = phi(v_k)-phi(v_0)`

先通过对角占优的非负 2×2 MIMO equilibrium mixing，再变换到 common/differential 坐标：

`m_c=(m_A+m_B)/2, m_d=(m_A-m_B)/2`

该变换用于显式追踪数据支持的子空间；是否允许独立 A/B 声明仍由 excitation rank gate 决定。

## 3. 四条独立路线

### 3.1 A1phys three-pole

每个 common/differential 模态包含三个稳定一阶蓄热极点：

`x_{j,p,k+1}=rho_{j,p} x_{j,p,k} + (1-rho_{j,p}) g_j(c)m_{j,k}`

`e_{j,k}=sum_p softmax(w_j)_p x_{j,p,k}`

`rho=exp(-dt/tau)`，`tau` 被限制在冻结范围内。它是可解、可审计的显式多时常主候选，但不声称真实设备恰好三阶。

### 3.2 Stable LPV-Koopman

使用逐时刻奇函数 lift：

`psi(m)=[m, m|m|, tanh(m)]`

`z_{k+1}=A(c)z_k+(I-A(c))B(c)psi(m_k)`

其中 `A(c)` 是工况调度、谱半径严格小于 1 的对角 LPV operator。它称为 stable LPV representation，不称标准定常线性 Koopman。

### 3.3 PI neural ODE

使用解析稳定离散化的 dissipative closure：

`dz/dt=-lambda(c)z + lambda(c) h(m,c)`

`h(m,c)=m * positive_gate(|m|,c)`

`lambda>0`，每一步采用指数积分而不是不受控 Euler。PI 只表示可验证的稳定、方向和 identity 约束，不伪造质量/焓守恒。

### 3.4 Causal DeepONet response

branch 只处理当前及历史动作前缀；零剂量 branch 输出严格为零。稳定 memory bank 累积 branch 系数，trunk 只依赖当前归一化时间：

`b_{j,p,k}=m_{j,k} softplus(B_p[c,|m_{j,k}|])`

`r_{j,p,k+1}=rho_p r_{j,p,k}+(1-rho_p)b_{j,p,k}`

`e_{j,k}=sum_p softmax(T_p(t_k/H)) r_{j,p,k}`

这保留 operator-learning 对照，同时禁止标准非因果 DeepONet 使用完整未来动作函数修改早期响应。

## 4. ADR：独立类而非 route 条件分支

决定：每条路线使用独立 `nn.Module` 类，并由同一 builder 返回共同字典契约。

理由：可以直接检查类型、参数、pole diagnostics 和 route-specific failure；避免名称不同但共享方程；便于逐路线消融和单元测试。

代价：代码量和测试矩阵增加；部分共享逻辑需放入小型 base/helper，不能通过复制粘贴漂移。

拒绝方案：

- 单一类加 `route_scale`：已经证明只能制造命名差异；
- 无约束 GRU response：稳定性与 identity 只能经验检查；
- 非因果 DeepONet：会产生未来动作泄漏；
- 长期逐模块 freeze：真实数据下容易形成错误解释权分配。

## 5. 验证批次

1. 类型/参数非同构：四条路线必须是不同类，不能只差一个 scalar。
2. 共同结构门：shape、finite、pole bound、constant identity、prefix causality、joint gradient。
3. 支持充分合成：逐路线训练局部算子，报告方向、幅值、时常/rollout 误差，不以单一 seed 宣布冠军。
4. 负控制：共线输入拒绝独立通道；future-Tin leakage 和 response collapse fail-closed；free capacity × residual excitation 用于识别解释权不唯一。
5. 只有四路线真实实现和端到端恢复均通过，才设计真实 validation runner；本计划本身不授权 Linux。
