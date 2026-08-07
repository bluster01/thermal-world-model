# Neural ODE + Deep Koopman + Koopa · 三篇关键论文

> **概念调研，不是路线证据。** Neural ODE、controlled Koopman、time-varying modeling 是动态表达；Fan17/20/21 提供的是不同物理内容。Phase 4 必须先固定同一 Fan20-centered physical specification，再公平比较表达，不能用本文直接决定主模型。

> 来源：微信公众号「时序之心」第13期推荐，结合 thermal-world-model 项目分析

---

## 一、Neural ODE (NeurIPS 2018 Best Paper, 7448 引)

**Chen TQ, Rubanova Y, Bettencourt J, Duvenaud D.**
arXiv: 1806.07366

**核心**：`dz/dt = f(z, t; θ)`，f 是神经网络，前向=调 ODE 求解器积分，反向=伴随法（O(1)内存）。

**对世界模型的意义**：Fan 2017/2020/2021 的 ODE 结构（`dx/dt = A(x) + B(x)·u`）可以直接用 Neural ODE 范式实现——A(x) 和 B(x) 换成神经网络，从数据端到端学习，而不是手工做非线性回归。

**局限**：慢（每步调求解器），纯数据驱动可能学出违反物理的动力学。

---

## 二、Deep Koopman (Nature Comms 2018, 1697 引)

**Lusch B, Kutz JN, Brunton SL.**
DOI: 10.1038/s41467-018-07210-0 | **全文 OA 免费**

**核心**：编码器 z=φ(x) → 隐空间线性演化 z_{t+1}=K·z_t → 解码器 x̂=ψ(z)。K 的特征值直接给出振荡频率和衰减率。

**对世界模型的意义**：
1. 把 Fan 的非线性 ODE 推向线性：在学到的 Koopman 坐标中，预测 = 一次矩阵乘法，比 ODE 求解快几个数量级
2. 频谱可解释：K 的特征值告诉你 SST 的主周期和衰减速率
3. 天然适配控制：线性 LQR/MPC 直接套在 Koopman 隐空间上

**局限**：观测函数维度靠调参，强混沌时线性化失效。

---

## 三、Koopa (NeurIPS 2023)

**Liu Y, Li C, Wang J, Long M.**
arXiv: 2305.18803 | 代码：github.com/thuml/Koopa

**核心**：Fourier Filter 解耦时不变/时变分量 → 各配 Koopman Predictor → 残差堆叠。

**对世界模型的意义**：
1. 非平稳处理：伊敏数据的负荷变化、煤质波动 → 时变 Koopman 算子自适应
2. 效率高：训练快 77%、内存省 76%——DGX Spark 跑得动
3. 残差结构 → 天然兼容 g_phys + g_free + g_koopman 多分支设计

---

## 四、与 Fan 三篇的衔接

| Fan 论文 | 提供什么 | Neural ODE 能做什么 | Koopman 能做什么 |
|---|---|---|---|
| 2017 | 4 状态非线性 ODE | 用 NN 替代手工回归 A(x)/B(x) | 找到 4+ 维线性坐标，预测=矩阵乘法 |
| 2020 | 7 状态 + SST 双重调节 | 端到端学习喷水-温度动力学 | 特征值读出喷水/给水的时间尺度 |
| 2021 | 能量函数 + 时变参数 | 可微分能量函数 θ(x,t) | Koopa 的时变算子 = 天然的时变参数框架 |

---

## 五、下一步行动建议

1. **先跑 Deep Koopman 的最小 demo**（Nature Comms 有 PyTorch 代码，GitHub）——在 Lorenz 或简单 ODE 上验证
2. **把 Fan 2017 的 ODE 写成 torchdiffeq 格式**——验证可微分仿真的可行性
3. **设计 Koopman 分支**：g_koopman(x) = ψ(K·φ(x))，作为 A1phys 的线性化替代
