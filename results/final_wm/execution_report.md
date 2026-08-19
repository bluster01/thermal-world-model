# final_wm 判别矩阵 — Linux 执行报告（2026-08-18 晚）

## 1. 执行状态

- 执行 HEAD：`f8ec07f`（含 3 个 executor-side fix，均标注 per user instruction; Supervisor review required）
- 已完成：96 测试全绿 → **D-SYN PASS 3/3**（改善 99.3/107.0/136.5）→ split-sides 双侧过门、
  SHA256 溯源一致（23a89ea6…）→ matrix 侧 A 首跑 O1/T1/B1/J1 完成、R1 崩
- **状态更新（用户指示）**：修复后的侧 A 完整重跑已中止——等 Codex 修复/冻结后再统一执行
  （含 §4 持久化改造与 latent 预算裁决）。**注意：中止的重跑已部分覆盖 artifacts 里的
  checkpoints/metrics 并追加了 ledger 重复块**，下次执行前需按 Codex 新指示清理或重跑覆盖。
- 侧 B 未执行，等同样条件

## 2. Supervisor 代码缺陷（环境差异类 ×2 + 纯缺陷 ×1，均已最小修复）

1. **GPU 设备搬运缺失**（Codex 本地 CPU 测试覆盖不到）：`synthetic.py` 教师 rollout 输入、
   `properties.py` 物性网格、`transition.py` 的 properties 属性均不随 `model.to(device)` 迁移
   ——`_apply` 钩子 + 物性类 `.to()/_apply` 修复（commit cc81cb3），96 测试保持全绿，
   真实 IAPWS 网格 GPU 冒烟通过
2. **CHANNEL_INDEX import 不存在**（纯缺陷，微冒烟未覆盖 R1 路径）：run_matrix.py L344 引用
   contracts 不存在的名字，R1 单元 ImportError → 改为 `BOUNDARY_ELEMENTS.index(...)`
   （commit f8ec07f），语义不变
3. **runner 无断点续跑**（未修，提案见 §4）：单元无条件重训 + verdict 内存态最后统一落盘，
   R1 崩导致 o1/t1/b1/j1 判决丢失、必须完整重跑

## 3. 数据观察（供审计阶段，非判决）

**latent4 训练充分性疑点**（用户假设 + ledger 证据）：
- latent4 seed2：epochs_run=30（cap）best_epoch=28，val 曲线到收尾仍在下降
  （27:1.818 → 28:1.728）；seed0/1 在 24 轮停于浅谷高原（~1.94-2.03），best_epoch=18
- closure 臂 uniform 24 轮 best_epoch=18，无深谷问题
- 解释：统一训练预算（24-30 轮×200 batch）对含潜变量网络偏小，seed2 靠初始化运气探到
  深谷（1.728 < closure_steam 1.88-1.90），seed0/1 未收敛到同一洼地
- 影响：T1 latent4 的"负贡献"判决可能被训练预算混淆——**建议审计阶段加入收敛诊断**
  （patience-hit 标志、曲线平坦度）后再对 latent 下结论

**ledger 说明**：R1 崩溃后完整重跑向 ledger.jsonl 追加了 o1/t1/b1/j1 重复块；
**以第二次出现（重跑块）为准**，R1 块只存在一次。

## 4. 待 Codex 决定

1. 复核 3 个 executor-side fix（cc81cb3 GPU 搬运 / f8ec07f R1 import / 见 §2）
2. runner 持久化改造三提案：skip-if-ckpt-exists / 单元级增量 verdict 落盘 /
   `--units verdict-only`（从 metrics 复算判决，裁决与训练解耦）
3. latent4 训练预算：统一加预算重测（或 latent 臂单独预算）后再裁决
4. 侧 B 执行确认（按冻结命令顺序，A 收尾后自动接上）
