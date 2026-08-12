# 时序因果表示学习论文阅读包

本目录 `papers/causal_representation/` 保存了与当前 MS3-R 因果审计最相关的论文 PDF。阅读顺序按“非平稳机制变化 → 潜在时序过程 → 瞬时依赖 → 非可逆观测”排列。

## 1. 非平稳数据提供识别信号

### CD-NOD（JMLR 2020）

文件：`papers/causal_representation/CD_NOD_JMLR2020.pdf`

论文：Biwei Huang, Kun Zhang 等，*Causal Discovery from Heterogeneous/Nonstationary Data*。

核心思想是：分布变化不只是 nuisance，而可能暴露局部机制变化；通过检测变化的局部机制、恢复骨架，并利用独立变化确定方向。它还讨论了非平稳性与 soft intervention 的联系。

对本项目的直接启发：不能把 F0/F1 rolling fold 自动叫作 environment。我们需要先定义日期、负荷、压力或燃烧状态环境，再检验哪些条件机制真的发生变化、哪些变化彼此独立。当前 RM3 的跨折 gain 漂移只能说明条件响应时变，尚未完成 CD-NOD 意义上的机制变化识别。

### Nonstationary State-Space（ICML 2019）

文件：`papers/causal_representation/Nonstationary_StateSpace_ICML2019.pdf`

论文：Biwei Huang, Kun Zhang 等，*Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models*。

该工作在特定非线性状态空间模型中允许因果强度和噪声方差变化，并展示非平稳性如何帮助结构识别和预测适应。关键不是“时间切分”，而是明确的时变生成机制及其可识别条件。

对本项目的启发：可将主汽温系统写成 measured-boundary latent state-space model，并分别检查 transition、noise 和 action-response 是否随环境变化；不能只用终端 MAE 或单一 gain 选择模型。

## 2. 潜在时序因果过程

### LEAP（ICLR 2022）

文件：`papers/causal_representation/LEAP_ICLR2022.pdf`

论文：Weiran Yao, Yuewen Sun, Alex Ho, Changyin Sun, Kun Zhang，*Learning Temporally Causal Latent Processes from General Temporal Data*。

目标是从观测序列的非线性混合中恢复有时间滞后的潜在因果变量，并给出非参数非平稳与参数化条件下的识别结果。

对本项目的边界：RM3 的 joint-latent 只能被称为共享动态状态表示。除非补充潜变量独立性、混合可识别性和跨环境恢复实验，否则不能称为 LEAP 意义上的 latent causal process recovery。

### TDRL（NeurIPS 2023）

文件：`papers/causal_representation/TDRL_NeurIPS2023.pdf`

论文：Xiangchen Song, Weiran Yao 等，*Temporally Disentangled Representation Learning under Unknown Nonstationarity*。

该工作研究未知非平稳性下的时序解耦，并利用时间结构和适当条件恢复潜在独立成分及其时间滞后关系。

对本项目的启发：free/residual latent 不能仅凭命名获得燃烧扰动语义。需要验证 latent 是否跨环境稳定、是否具有稀疏转移、是否能解释局部点位而不是只改善 terminal MAE。

### CtrlNS（NeurIPS 2024）

文件：`papers/causal_representation/CtrlNS_NeurIPS2024.pdf`

论文：Xiangchen Song, Zijian Li, Guangyi Chen 等，*Causal Temporal Representation Learning with Nonstationary Sparse Transition*。

核心是非平稳稀疏转移：通过转移变化识别 domain shift，再在条件独立约束下学习潜变量和潜在时序因果关系。

对本项目的启发：如果要借用这条路线，必须把环境变化、转移稀疏性、条件独立性变成可审计指标。当前 joint-latent 结构本身没有提供这些证明。

## 3. 瞬时依赖与观测不可逆

### IDOL（ICLR 2025）

文件：`papers/causal_representation/IDOL_ICLR2025.pdf`

论文：Zijian Li, Yifan Shen 等，*On the Identification of Temporal Causal Representation with Instantaneous Dependence*。

该工作允许潜变量之间同时存在瞬时关系和滞后关系，并通过稀疏影响约束与充分变化条件建立识别结果。

对本项目的直接价值：二级减温闭环天然包含同一采样时刻的控制耦合、双侧同步和传感器混合，不能默认“所有因果关系都有正滞后”。lead/placebo 仍然重要，但还应明确 instantaneous path 与 delayed path 的区分。

### CaRiNG（ICML 2024）

文件：`papers/causal_representation/CaRiNG_ICML2024.pdf`

论文：Guangyi Chen, Yifan Shen 等，*CaRiNG: Learning Temporal Causal Representation under Non-Invertible Generation Process*。

该工作针对观测由潜变量经过非线性、非可逆混合生成的情况，利用时间上下文恢复丢失信息并给出相应识别理论。

对本项目的启发：阀门开度只是喷水流量的非线性代理，测点还存在混合、滞后和未测状态，因此更接近非可逆观测问题。不能把 valve branch 直接解释成 measured spray-flow physics；应使用 measured-boundary latent MIMO 与部分可辨识响应集合。

## 4. 与当前 RM3 的对应结论

| 论文方向 | 当前项目已有证据 | 仍缺少的证据 | 当前可用表述 |
|---|---|---|---|
| CD-NOD / 非平稳机制变化 | rolling folds、跨折 gain 漂移 | 显式环境、机制变化、独立变化检验 | 时变条件响应 |
| LEAP / TDRL | joint latent、历史状态与多任务监督 | latent 独立性、可识别混合、跨环境恢复 | 共享动态 latent |
| CtrlNS | 有状态转移和多点位任务 | 稀疏转移与条件独立审计 | 物理约束预测结构 |
| IDOL | prefix causality、lead/wrong-side placebo 配置 | 瞬时/滞后路径分解 | 时序方向诊断 |
| CaRiNG | 对代理输入、混合和缺测保持保守 | 非可逆观测下的恢复实验 | 部分可辨识响应模型 |

## 5. 论文写作边界

当前最稳妥的主张是：

> disturbance-conditioned, physically constrained closed-loop prediction with partial local-response identification

中文可写为：

> 面向闭环观测数据的扰动条件物理约束预测与局部响应部分辨识。

暂不应写成：已完成因果发现、因果表示学习、非线性 ICA 恢复、唯一 plant gain 识别或任意 `do(valve)` 反事实验证。

下一轮若要升级因果证据，应优先做：显式 environment audit、机制变化检验、瞬时/滞后路径分解、latent 稀疏转移与条件独立检验，以及小幅外生 SP 激励或可信 IV 验证。

## 原始链接

- [LEAP 项目页](https://weirayao.github.io/papers/leap)
- [CD-NOD JMLR](https://jmlr.org/beta/papers/v21/19-232.html)
- [Nonstationary State-Space ICML 2019](https://proceedings.mlr.press/v97/huang19g.html)
- [TDRL 项目页](https://weirayao.github.io/papers/tdrl)
- [CtrlNS NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8cef4e4bcb85f7d4a1005a9db018d6b6-Abstract-Conference.html)
- [IDOL ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5726513facc85f802be4a25e77fb9765-Abstract-Conference.html)
- [CaRiNG arXiv](https://arxiv.org/abs/2401.14535)
