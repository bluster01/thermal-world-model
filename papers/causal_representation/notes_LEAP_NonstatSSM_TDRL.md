# 因果表示学习三篇精读笔记（thermal-world-model 项目）

> 来源目录: `/home/bluster/projectA/thermal-world-model/papers/causal_representation/`
> 提取方式: pymupdf 全文提取, 逐页通读正文+关键附录证明。公式为行内 LaTeX。

---

## 一、LEAP: Learning Temporally Causal Latent Processes from General Temporal Data (ICLR 2022, Yao et al.)

**① 问题设定（数据/假设）**
观测时序 $x_t$ 由隐因果过程经非线性可逆混合生成 $x_t=g(z_t)$，隐变量 $z_t$ 之间存在带时延的因果边；目标是无监督地同时恢复 $z_t$ 及其因果结构。两种设定：(i) 非参数非平稳：$z_{it}=f_i(\{z_{j,t-\tau}|z_{j,t-\tau}\in Pa(z_{it})\},\epsilon_{it})$，噪声分布 $p_{\epsilon_i|u}$ 被观测到的离散工况变量 $u$ 调制；(ii) 参数化：线性 VAR 转移 $z_t=\sum_{\tau=1}^{L}B_\tau z_{t-\tau}+\epsilon_t$。

**② 模型架构（生成过程/图模型）**
VAE 扩展，核心是"因果过程先验网络"。用逆转移函数 $\hat\epsilon_{it}=r_i(\hat z_{it},\{\hat z_{t-\tau}\})$ 把 $[\hat z_{t-L..t-1},\hat z_t]$ 映射到 $[\hat z_{t-L..t-1},\hat\epsilon_t]$，Jacobian 下三角，变量替换得转移先验 $\log p(\hat z_t|\{\hat z_{t-\tau}\},u)=\sum_i\log p(\hat\epsilon_i|u)+\sum_i\log|\partial r_i/\partial\hat z_{it}|$；非平稳噪声用"每工况一份"的神经样条流（spline flow）拟合；推断网络为双向 GRU。

**③ 损失函数与优化**
增广 ELBO：$L_{ELBO}=L_{Recon}-\beta L_{KLD}-\gamma L_{Mask}-\sigma L_{TC}$，其中 $L_{TC}$ 是 FactorVAE 式对比判别器估计的噪声总相关（Total Correlation），强制噪声分量时空独立；结构可视化用软掩码 $L_1$ 正则 + LassoNet 剪枝（识别性理论不依赖稀疏假设）。

**④ 可识别性/理论结论**
定理1（非参数）：$f_i$ 三阶可微、$g$ 可逆、噪声在工况内时空独立（IN 条件）、且存在 $2n+1$ 个工况使 $w(z_t,u)$（各分量对数条件密度的一阶/二阶导数组成的向量）之差线性无关（充分变异性）⇒ $z_t$ 可识别到置换+分量可逆变换。定理2（参数）：噪声服从广义 Laplacian $p_\epsilon\propto e^{-\lambda|\epsilon|^{\alpha}}$（$\alpha<2$）且至少一个 $B_\tau$ 满秩 ⇒ 同样可识别。分量级识别后，条件独立关系完全刻画时延因果图，故因果结构一并可识别。

**⑤ 关键实验与结论**
合成数据（n=8, L=2）MCC≈0.98；消融：因果先验（0.45→0.72）、非平稳流（→0.94）、噪声判别器（→0.98）各自贡献显著；违反假设时：瞬时因果/高斯噪声明显破坏识别，低秩转移仍部分可识别（低维投影被恢复），工况数略少于 $2n+1$ 仍能全识别。真实数据：KiTTiMask（学到的 $B_1$ 近对角，退化为 SlowVAE 特例）、弹簧振子（SHD=0 恢复弹簧连接）、CMU MoCap（无监督恢复 pitch/yaw/roll 三类隐动态与耦合结构）。

**⑥ 作者自述局限**
(1) 假设隐变量间无瞬时因果影响；(2) 因果影响不随工况变化（实验显示可部分泛化到变结构，但无理论保证）。

**⑦ 与我们项目的关联点**
我们的 $\hat T=f_{free}(history)+g_{response}(context,a_{1:H},r_{1:H})$ 可改写为 LEAP 转移先验形式：把 $(a_{1:H},r_{1:H})$ 当作"工况 $u$"进入条件转移，用逆响应函数+下三角 Jacobian 把 $g_{response}$ 显式参数化为带独立噪声的机制模块（而非黑箱残差），转移似然可精确计算。IN 条件可作为分解正确性的正则——响应机制扰动应与自由演化噪声独立。LEAP-VAR 的线性转移矩阵 $B_\tau$ 直接对应阀位→汽温的时滞增益/脉冲响应；"每工况一份 spline flow"可用于按负荷/煤质分段建模非平稳增益。

---

## 二、Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models (ICML 2019, Huang, Zhang, Gong, Glymour)

**① 问题设定（数据/假设）**
观测多元时序 $X_t$，每个变量由瞬时因果父母加权生成：$x_{i,t}=\sum_{x_j\in PA_i}b_{ij,t}x_{j,t}+e_{i,t}$（可扩展时滞项 $\sum_{s=1}^{sl}\sum_{x_k\in PL_i}c^{(s)}_{ik,t}x_{k,t-s}$）。核心设定：因果强度与对数噪声方差各自按 AR 过程演化，$b_{ij,t}=\alpha_{ij,0}+\sum_{p=1}^{pl}\alpha_{ij,p}b_{ij,t-p}+\epsilon_{ij,t}$，$h_{i,t}=\log\sigma^2_{i,t}$ 同理；全图并集 $G=G_1\cup\dots\cup G_T$ 无环。即"非平稳＝时变因果系数"。

**② 模型架构（生成过程/图模型）**
把 $X_t=(I_m-B_t)^{-1}E_t$ 与系数/波动率的 AR 过程合并，视为以 $Z=\{b_{ij}\},\{h_i\}$ 为隐状态、$\theta=\{\alpha_{ij,p}\},\{\beta_{i,q}\},\{w_{ij}\},\{v_i\}$ 为参数的非线性状态空间模型（对隐变量非线性）。

**③ 损失函数与优化**
SAEM（随机逼近 EM）：E 步用带祖先采样的条件粒子滤波 CPF-AS 采样隐状态，以粒子权重蒙特卡洛近似 $\tilde Q_k(\theta)=(1-\lambda_k)\tilde Q_{k-1}(\theta)+\lambda_k\sum_j\frac{\omega_T^{(k,j)}}{\sum_l\omega_T^{(k,l)}}\log p_\theta(X_{1:T},\mathring Z^{(k,j)}_{1:T})$，M 步令 $\partial\tilde Q_k/\partial\theta=0$；复杂度 $O(m^3\times M\times T)$。因果边判据：估计系数 $\hat b_{ij,t}$ 的均值与方差双阈值（0.05）判边。

**④ 可识别性/理论结论**
定理1：瞬时结构无环、系数服从 AR(1)（$\alpha\in(-1,1)$）、观测噪声为平稳白噪声时，因果序与模型参数可识别——**不要求非高斯、不要求 faithfulness**（线性+高斯也可识别，靠非平稳性）。证明构造：根因的 $S(t,t+p)_i:=E[x_{i,t}^2x_{i,t+p}^2]$ 满足 $S(t,t+p)-S(t,t+p-1)=0$，非根因 < 0，据此识别根因并逐层剥离因果序，再按时变系数回归的可识别性（Wall 1987）恢复参数。推论1推广到含时滞项（时滞方向已知，退化为参数识别）。$\sigma^2_i$ 也随时间变化时仅实验证据支持可识别，无严格证明。

**⑤ 关键实验与结论**
合成数据（5 变量 Erdős-Rényi 图，T=500~2000）：因果发现 F1 全面超过 LiNGAM/CD-NOD/IB/MC；预测 RMSE 经 Wilcoxon 符号秩检验显著优于 Lasso、窗口 Lasso、Kalman、SSM(CPF)、GP（最大 p 值 0.018）。美国宏观数据（1965-2017，GDP/通胀/增长/失业）：恢复的同期因果图符合经济学常识（增长→通胀→GDP、GDP+通胀→失业），通胀 2007-2017 一步预测 RMSE 0.32 为所有方法最低。

**⑥ 作者自述局限**
未来工作：扩展到非线性因果关系、部分可观测过程（Geiger et al. 2015）、含瞬时环的因果模型；$\sigma^2_i$ 时变情形的严格证明缺失。

**⑦ 与我们项目的关联点**
给出"非平稳有助于识别"的构造性工具：把阀位→汽温的时变增益 $b_{a\to T,t}$ 显式建模为 AR 过程（或由负荷 $r$ 驱动），而非学静态黑箱 $g_{response}$——我们的"工况参考 $r$"可直接作为系数 AR 的外生驱动。根因统计量 $S(t,t+p)$ 差分符号检验可用于**验证分解正确性**：若 $f_{free}$ 已彻底剥离阀位影响，残余对阀位应呈"根因"特性。线性+高斯+非平稳即可识别，契合 PID 闭环下近似线性增益、近高斯扰动的主汽温对象。预测端"马尔可夫毯+Metropolis-Hastings"贝叶斯外推可作为 MPC 中时变增益不确定性的轻量传播。注意闭环数据中 $a_t$ 与汽温偏差同期相关（动作内生），该文同时建模瞬时+时滞边，可用于显式分离反馈耦合。

---

## 三、Temporally Disentangled Representation Learning under Unknown Nonstationarity (NCTRL, NeurIPS 2023, Song, Yao et al.)

**① 问题设定（数据/假设）**
与 LEAP 同族，但**工况指标不可观测**：$x_t=g(z_t)$ 可逆混合，$z_{it}=f_i(\{z_{j,t-\tau}|z_{j,t-\tau}\in Pa(z_{it})\},c_t,\epsilon_{it})$，其中 $c_t$ 为离散隐变量（$|c_t|=C$）服从一阶 Markov 链（转移矩阵 $A$）；$z_t$ 各分量在给定历史 $z_{Hx}$ 与 $c_t$ 下条件独立。不观测任何辅助变量/域标签。

**② 模型架构（生成过程/图模型）**
NCTRL 三模块：(1) ARHMM（自回归隐马尔可夫模块）——估计各域条件发射分布 $p(x_t|x_{t-1},c_t)$ 与转移矩阵 $A$，用 Viterbi 解码最优域序列 $\hat c_{1:T}$；(2) 先验网络——学习整体逆动力学 $\hat f_z^{-1}(\hat z_t,\hat z_{Hx},\hat\theta_{c_t})$，下三角 Jacobian 变量替换得 $\log p(\hat z_t|\hat z_{Hx},c_t)=\sum_i\log p(\hat\epsilon_i|c_t)+\sum_i\log|\partial\hat f_i^{-1}/\partial\hat z_{it}|$；(3) 编码器-解码器 VAE 保证 $\hat g$ 可逆。

**③ 损失函数与优化**
两段目标：HMM 自由能下界 $L(q(c),\theta_{HMM})=\mathbb E_{q(c)}[\log p_\theta(x_{1:T},c)]-H(q(c))$，前向-后向算法可微地最大化得 $q(c^\star)$；VAE 侧 ELBO = 重建（MSE）+ 采样估计的 KL（后验 vs 以解码 $c_t$ 为条件的转移先验），$L_{KLD}=\mathbb E_{\hat z_t\sim q}[\log q(\hat z_t|x_t)-\log p(\hat z_t|\hat z_{Hx},c_t)]$。

**④ 可识别性/理论结论**
定理1（推广 Gassiat et al. 2016 的 HMM 可识别性到自回归发射）：$C$ 已知、$A$ 满秩时，仅由 ≥4 个连续观测的联合分布即可识别转移矩阵 $A$ 与条件发射分布 $p(x_t|x_{t-1},c)$，至标签置换。定理2：若先验分解 $p(z_t|z_{t-1},c_t)=\prod_k p(z_{kt}|z_{t-1},c_t)$ 且由跨导数 $v_{kt}=\partial^2\eta_{kt}/\partial z_{kt}\partial z_{l,t-1}$、三阶导 $\mathring v_{kt}$ 及域间差分构造的 $2n$ 个函数向量 $\{s_{kt},\mathring s_{kt}\}$ 线性无关（充分变异性），则 $z_t$ 可识别到置换+分量可逆变换——全程无需观测域指标。

**⑤ 关键实验与结论**
合成数据 A/B（2 层/3 层混合 MLP）：MCC≈0.99，远超 BetaVAE(0.46)/iVAE(0.67)/HMNLICA(0.59)/TDRL(0.78) 等基线；$c_t$ 推断准确率≈0.90，$A$ 的 MSE≈$10^{-3}$。Modified CartPole（5 种物理参数域、隐域按 Markov 链切换）：MCC 0.96 vs TDRL 0.85、SKD 0.73，恢复的两个隐变量对应小车位置与摆角。MoSeq 小鼠行为视频：无监督分出 active/inactive 两个行为相，恢复的分量与推断域指标一致。

**⑥ 作者自述局限**
(1) 域指标假设为离散 Markov 链；(2) 只允许时延因果、无瞬时依赖——时间分辨率低时该假设被破坏，需另行处理瞬时因果。

**⑦ 与我们项目的关联点**
三篇中与我们最贴的一篇：我们有 $r$ 作工况参考，但真实工况（煤质、结焦、负荷区间）部分是隐的——ARHMM 可直接嵌在灰箱模型顶层，从主汽温历史**无监督切分工况段**并估计工况转移矩阵，无需人工打标签。先验网络的下三角 Jacobian 技巧与 LEAP 相同，可用于把 $g_{response}$ 变成可计算精确转移似然的机制模块。"时延因果掩码"即 $z_{it}$ 只以其父母 $Pa(z_{it})\subset z_{Hx}$ 为条件——可对 $a_{1:H}$ 各滞后施加同样结构：阀位仅在纯时延之后的若干滞后进入（结构稀疏且理论友好）。定理2 的充分变异性启示：工况切换要足够多样（覆盖负荷区间/吹灰周期）才能识别增益变化。其局限同样适用于我们：PID 闭环使 $a_t$ 与当前偏差瞬时相关，需显式处理（如把 $a_t$ 作观测输入而非隐变量，或补瞬时耦合模块）。
