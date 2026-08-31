# PREREG：JEPA-B5 动作盲慢态（v1，2026-08-31）

## Material Passport
- Origin: execution-side, user-authorized follow-up（用户 2026-08-31 指令"做 B5"）
- Upstream Dependencies: `jepa_architecture_note_20260826.md`（B2 慢态提案）、
  `PREREG_jepa_b_series_20260830.md`（B 系列 v1 判据）、B2 报告（v1 回传）
- Verification Status: UNVERIFIED（本文件 Linux 训练前冻结）

## 动机（B2 的失败模式）

B2（4 维慢态、6 步更新）在固定协议下是唯一精度提升臂（H18 −5.3%，
负荷极差 +0.7% 双过），但**方向门 valve2 断裂**：

- c0 基线（原轨迹口径）：valve2 H18 mean −0.105°C（开阀降温 ✓，全部 4 格过门）
- b2：valve2 H18 mean **+0.010°C**、H60 **+0.079°C**（开阀升温），frac 0.29/0.30

机制判读：慢态更新输入含物理状态（state 是已记录动作的函数）——慢态把
"喷水阀降温"解释吸收进自己的功率修正（steam/metal 守恒注入），稀释了
动作→温度的因果响应，甚至反号。**表示增益挤占了动作因果通道**。

## B5 机制（唯一改动）

B2 同款结构（slow_dim=4、stride=6、守恒功率注入、Gaussian-CF 0.01），
唯一差异：**慢态更新输入动作盲化**——`update` 只读 `[slow, boundary]`，
**不读物理状态**。boundary 7 通道不包含喷水阀动作，慢态无法从动作效果中
吸收降温因果；部署接口不变（7+2 → 5 温度；内部状态 11+4）。

## 冻结矩阵

同 B 系列 v1 全部训练合同（v2.2 + A5 门、seed0、120ep/patience20/batch32/
200 batches/epoch、lr 1e-3、clip 10、hybrid + conservative_norew、oracle）：
- 臂：`c0`（匹配对照，复用 B 系列 v1 的 c0 报告）+ `b5`（动作盲慢态）
- 损失：observation_nll 1.0 + gaussian_cf 0.01（jepa_prediction/b4 项 0）
- 判定：**方向门采用原轨迹口径**（base = 窗口真实 future_actions /
  future_boundary；step = base+Δ；支持域、逐日 bootstrap、mean<0 & CI 上界<0
  & frac≥0.60 不变）——该口径已在 cde385e 修复并验证（c0 全过）

## 判定与停止（与 B 系列 v1 完全一致）

1. 身份门：b5 机制关闭（slow_mechanism_scale=0）时 rollout 与 c0 逐位相同。
2. 主门：H18 末端 MAE ≤ 0.95 × c0。
3. 稳健门：四负荷箱 max/min 相对恶化 ≤ 10%。
4. 方向门：两阀 × H18/H60 原轨迹口径 v0.3 规则全过。
5. B5 额外报告 H36/H60、drift、UTC-day 偏置（不设事后阈值）。
6. seed0 三门全过 → `PROMOTE_TO_FIXED_SEEDS_1_2`；主 MAE 恶化 ≥5% 或任一
   方向门失败 → `REJECT_EXPLORATORY_SEED0`；其余 `INCONCLUSIVE_EXPLORATORY_SEED0`。
7. 不自动重试、不补跑搜索；本批不升级论文 verdict。

## 预期判读（先验，非门）

- 若 B5 方向门过且 H18 收益保持（≤0.95×c0）→ "慢态因果隔离可行"，
  B2 的破坏可修复，状态记忆机制保留——**升 PROMOTE 并推种子 1/2**。
- 若 B5 方向门过但 H18 收益消失（→INCONCLUSIVE）→ 因果隔离有代价：
  B2 的 −5.3% 部分来自"读动作效果"，记忆机制本身收益有限。
- 若 B5 方向门仍失败 → 因果破坏不是输入路径而是注入路径
  （power 守恒注入本身干扰方向），需换注入点设计。
