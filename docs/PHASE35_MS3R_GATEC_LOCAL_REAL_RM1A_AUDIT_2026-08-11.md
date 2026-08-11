# Phase 3.5 MS3-R Gate C 真实 RM1-A Supervisor Audit

> 日期：2026-08-11。范围：真实数据 1/100、train/validation、seed 0、单次归因筛查。本文不是独立 test、因果识别或 operator 冠军判决。

## 判决

```text
RM1A_ATTRIBUTION_COMPLETE /
CAPACITY_COLLAPSE_NOT_OBSERVED /
LOCAL_SUPERVISION_REQUIRED
```

保留预注册主参考 `C3_sched_base`，但不称经验冠军。Linux、test、MS4 和任意 `do(valve)` 声明继续冻结。

## 完整性

- 执行提交：`5566a56ac3bec391bb0dabb0843c24f8b369feed`；
- 数据：冻结真实源 SHA `85a3f926...e4da6`，6957 个 train anchors、2048 个 validation anchors；
- 六候选 train/validation anchor SHA 完全一致；
- 8 个 ledger 产物逐字节复核通过，6/6 finite，未访问 test；
- 每候选仅一次 seed 0、180 optimizer updates，不补跑、不看结果调参。

## 结果表

| 候选 | 共享预测分数↓ | local MAE °C↓ | local/persistence↓ | terminal MAE °C↓ | logged response |·| °C | logged−shuffled 优势 °C |
|---|---:|---:|---:|---:|---:|---:|
| `C0_paired_free` | 0.355289 | 2.2314 | 0.9704 | 1.4760 | 0 | 0 |
| `C1_additive_base` | 0.355004 | 2.2249 | 0.9676 | 1.4759 | 0.03497 | 0.05157 |
| `C2_sched_small` | 0.356781 | 2.2368 | 0.9727 | 1.4759 | 0.04372 | 0.06443 |
| `C3_sched_base` | 0.354881 | 2.2238 | 0.9671 | 1.4757 | 0.04381 | 0.06658 |
| `C4_sched_large` | 0.353957 | 2.2022 | 0.9577 | 1.4755 | 0.04369 | 0.06848 |
| `C5_terminal_only` | 1.445559 | 21.6378 | 9.4097 | 1.3899 | 0.03197 | 0.03118 |

“共享预测分数”对所有候选使用同一套 valve/Tin/local/terminal/rollout 权重；不同监督目标的训练 loss 不跨候选比较。

## 四个冻结对比

1. **显式响应 vs paired-free。** `C3` 相对 `C0` 的共享分数只改善 0.115%，local MAE 只改善 0.00754°C，terminal MAE 只改善 0.00026°C。显式响应没有带来可称实质性的 observed-policy 预测增益；其价值仍须由局部响应时序/方向和跨日稳定性证明。
2. **scheduled vs additive。** `C3` 与 `C1` 的预测几乎相同；scheduled 的 logged response 幅值高 25.3%，logged-vs-shuffled 优势多 0.0150°C。单 seed 不足以宣称 scheduled 优越，因此只保留预注册 scheduled-base 作为参考。
3. **free capacity 扫描。** small/base/large 的 logged response 为 0.043723/0.043807/0.043692°C，相对极差仅 0.263%，且不随容量单调消失。本扫描未观察到“free 越大、response 越小”的容量吞噬现象；这不证明真实分解全局唯一。
4. **local supervision 消融。** terminal-only 的 terminal MAE 表面改善 5.82%，但 local MAE 从 2.2238°C 爆到 21.6378°C，是 persistence 的 9.41 倍。模型把失真的中间温降当成无语义 latent 通道换取末温拟合，因此该候选必须淘汰，不能拿其末温改善冒充物理闭环改善。

## 与 RM0-B 的关系

`C3_sched_base` 与 RM0-B 的 A1phys 使用相同 anchors、预算和外壳；8 个核心预测/响应指标最大绝对差为 0。RM0-B 已在同一外壳下跑完 A1phys、LPV-Koopman、PI-ODE、DeepONet，并得到“预测近同、response 幅值约 2.5 倍差异”的不可排名结论。因此再次执行相同的单-seed RM1-B 没有新增信息，不重跑。

## 下一步

下一批只设计 RM2 的真实稳健性门：以 `C3_sched_base` 为主参考、`C0_paired_free` 为负控，并保留容量稳健性对照；增加 UTC 日/连续时间块、60/180 s 局部方向与时序、opening/closing、common/differential 支持域和两个 rolling folds。先完成本地设计、代码与 smoke，再决定是否签发 Linux 批次。operator 路线在获得跨日/跨 fold 响应证据前继续禁止排名。

机器审计见 `results/phase3_5/ms3r_gatec_local_real_rm1a/supervisor_audit_validation.json`。
