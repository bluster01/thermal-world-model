# adhoc2_lumped_enthalpy — 精度–控制保真度张力研究仓（论文 A）

从单机组闭环运行数据构建世界模型时，数据驱动精度与动作通道物理保真度存在**可量化的内在张力**；本仓给出张力的机制解释（残差=隐式闭环控制器）、测量协议（动态响应域验证 F3）与可调折中方案（物理嵌入状态模型 + 动作隔离残差 + 注入位置门控），并在真实超临界机组上画出完整折中曲线。

**目标期刊**：Applied Energy。**规划**：[PAPER_A_FRAMEWORK_V2.md](PAPER_A_FRAMEWORK_V2.md)（现行）；[ADHOC_PAPER_PLAN.md](ADHOC_PAPER_PLAN.md)（两文方案，论文 A 部分已被 v2 取代）。

## 两文定位（2026-08-18 grill 决策）

- **论文 A（本仓）**：精度–控制保真度张力 + 物理嵌入折中曲线。A 先发表，完全独立。
- **论文 B（主仓 thermal-world-model，后行）**：系统层世界模型框架（架构稿 `docs/INDUSTRIAL_WORLD_MODEL_PAPER_CORE_ARCHITECTURE_2026-08-17.md` 为骨架），引用 A 的动力学与残差边界结论。
- **引用纪律**：跨仓引用带 `[PHASE-REF]`；跨仓数字不可比（协议不同），入文前必须同协议重测。

## 折中曲线采样点

| 点 | 定位 | 状态 |
|---|---|---|
| Direct WM（phase1 移植） | 纯数据精度端点（β-NLL 概率输出） | `36_direct_wm.py` 移植中 |
| double_w | 精度上界（读 W 捷径） | Q32 已测（旧协议） |
| qnav | 动作隔离残差上蒸发底座 | rollout 2.463（旧协议） |
| h_now | 诚实配置（h-only 不读 W） | Q32 已测（旧协议） |
| conservative_now | 守恒型残差（精度≈evap_only） | Q32 已测 |
| evap_only | 纯物理端点（FIXB 蒸发干燥灰盒） | B1-B5 已测 |

统一协议（已定）：残差/神经组件内部 RevIN，物理灰盒真实尺度；对比口径 = **物理空间 rollout/MAE，同一数据切分（开发段 [0,40000)，reserved [40000,50000) 禁触）**。

## 关键结果速查

- FIXB 蒸发干燥：B1 湿态两相偏差 21-25→**3.07°C**；B4 干态正确退化
- qnav（残差上蒸发底座）：E1-E5 全 PASS——湿 τ63=**480s** 进窗、rollout **2.463**（历史最佳）
- Q32 筛查：精度层=捷径（读W+非守恒注热），物理层 h_now 成立
- Q32-R 归因：干态符号翻转 = 残差对动作后代状态的反馈反应（replay 切断 8/8 恢复）
- Q32-S：shared≈physical——扰动/动作分离 replay 方案被否
- Q32-T 三面板：live 增益压缩 30-75%、干态翻转复现；换向抖动=控制器实现效应；初始化 180 步回放误差 18°C（→observer 需求，论文 B 接口）

## 数据协议

- 源：`A侧主汽温全数据_cleaned_10s.csv`（41 列 = date + 40 数值，10s 采样）
- 加载：`iloc[WIN_START : WIN_START+40000]`，WIN_START=70686
- 折：F0 训练 [0,20000)/评测 [25000,30000)；F1 训练 [0,30000)/评测 [35000,40000)
- reserved [40000,50000) 冻结，任何实验不得访问

## 工作纪律

1. 预注册：每个实验先写门槛/判定标准，跑前冻结，跑后如实记录（含 FAIL）
2. 审计独立性：Codex 驱动的冻结命令原样执行，raw artifacts 原样回传，执行方不写科学裁决
3. 提交：commit 与 push 拆两条命令；push 走直连 `-c http.proxy= -c https.proxy= -c http.sslVerify=false`；fetch 必须定向分支（`fetch origin adhoc/lumped-enthalpy`），通配 fetch 会把 main/pinn-features 历史卷进来
4. 数字纪律：报告数字逐项回溯 summary JSON/日志；跨仓数字不混用
