# final_wm 判别矩阵 — Linux 执行报告（2026-08-20 更新）

## 1. 执行状态

- 执行 HEAD：`e9af3cc`（Codex matrix v0.2：接受 3 个 executor-side fix 并加回归测试、
  修复 StateLayout.dim / ResidualInjection.latent_step、续跑逻辑、T1 重训预算 60 epochs）
- 前置：pytest tests/final_wm/ **101/101 passed**（40.48s）→ D-SYN 3/3 PASS 不重跑 →
  split-sides SHA256 一致 → 侧 A matrix v0.2 完整执行
- **侧 A 完成**（2026-08-20 ~11:30，exit 0，R1 无崩溃）：
  - O1 9/9 RESUMED 复用；T1 12 臂重训（v0.2 预算）；B1/J1 复用；R1 复跑
  - 判决落盘 `artifacts/final_wm/matrix_summary_sideA.json` + `r1_report.json`
  - T1 各臂 best_val（seed0/1/2）：physics_only 1.453/1.463/1.467；closure_cons
    1.369/1.324/1.345；closure_steam 1.339/1.341/1.412(seed2 latent4 早停 1.412)；
    latent4 1.345/1.412/1.412
- **侧 B 暂缓**（用户指示：等本地侧 A 判决审计闭合）
- artifacts/final_wm 整目录已 commit+push（83MB，含 30 ckpt、ledger、records、判决 json）
- IAPWS 代理网格 `artifacts/final_wm/iapws_surrogate.npz`（GridThermoProperties 训练用网格，
  1.56MB）已随 a980cc1 回传 —— 解除对侧模型探针 provisional 标记（审计 §6）

## 2. 判别矩阵侧 A 判决（v0.2，详见 matrix_summary_sideA.json）

| 单元 | 判决 | 要点 |
|---|---|---|
| O1 | learned MIXED / hybrid REJECTED | learned 2/3 seed −30% 置信、1 seed +13% 退化；hybrid 3/3 CI 过零 |
| T1 | closure_cons SUPPORTED (2/3)；steam/latent REJECTED | +5.7/+6.9% 置信；steam −6.4/−2.3/−3.7%；latent CI 全过零 |
| B1 | REJECTED | GRU 边界预报 3/3 显著差于 persistence（CRPS gap 1.7-2.0） |
| J1 | SUPPORTED 3/3 | joint 优于 staged +13.5/+16.4/+33.3% |
| R1 | REJECTED（direction） | runtime-blind 3/3 过、leakage 3/3 干净、方向探针 frac_negative 0.19-0.34（要求 1.0） |

## 3. 动作通道证据链（用户驱动审计，2026-08-20）

v0.2 重分析发现两处重大变化 + 一次代码级根因定位，详见
`results/final_wm/evidence_chain.md`（权威口径）与 `experiments/final_wm/audit/`（脚本溯源）：

- 60-epoch 预算修掉了 v0.1 的水平标定偏差（final 分箱 +2.1~+5.4 → ±0.2°C）；
  ①参数 MLP 主证据失效，路线图优先级重排（optimization_roadmap.md v1）
- per-channel：final H18 1.0°C 已近 persistence；sh1_in 9.4-9.8°C 集中爆发（38× persistence）
- **R1 方向失败根因 = 再湿反馈项**：aW=0 消融后正确方向占比 0.12→0.94
- **论文旧口径撤回**：25-450× 与 −0.005~−0.015 均不成立（Direct WM v2 审计 §3）；
  物理参考独立求解为 v1 0.04-0.16 / v2 0.6-0.8°C per 2%

## 4. 训练耗时（ledger wall_seconds，执行方观测）

- T1 每臂 66-93 min（55-67 s/epoch），12 臂 ≈ 13-15 h/侧；瓶颈在 Python 循环
  （GPU util 14-20%，18 步递归 rollout + IAPWS 网格查询串行）
- 侧 A 总墙钟 ~15 h，GPU 预算内（≤36h）但墙钟偏重
- 加速项（torch.compile / 向量化 / DataLoader 多进程）已列项，侧 B 前由 Codex 裁决，
  不改协议/参数/种子

## 5. 论文状态（FMTS 2026）

- 工作稿 `docs/fmts2026/paper/fmts_main.tex`（6 页，A 侧判决 + 三图）已 push
- **目录冻结**（用户指示）：解冻条件 = 双侧判决审计闭合 + 撤回数字清零 + H1 表述重写
  + 数值全溯源。证据链齐全前不补全（见 paper/FROZEN.md）
