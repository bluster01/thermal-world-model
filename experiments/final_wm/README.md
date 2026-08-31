# Final-WM v0.6 / v0.7 全量重发入口

当前唯一有执行权的正式批次是 `final_wm_v07_full_reissue_v1`。它只训练一次：v0.6 提供 canonical v2.2 与 120/20 训练底座，v0.7 提供可信度判决和内容寻址证据合同。旧 v0.2-v0.6 命令只作历史追溯，不得继续执行。

完整命令见 [v0.7 Linux 冻结执行单](../../results/final_wm/v07_full_reissue_runbook_20260901.md)。协议谱系与排除项见 [v0.6/v0.7 审计](../../docs/FINAL_WM_V06_V07_PROTOCOL_AUDIT_2026-09-01.md)。

## 执行合同

- 数据：双侧 corrected canonical v2.2；正式模型只读 7 通道 base view。
- 物性：必须显式传 `artifacts/final_wm/iapws_surrogate.npz`，不得使用解析 fallback 生成权威结果。
- 预算：所有正式训练臂 `epochs=120`、`patience=20`；seeds=`0,1,2`。
- 单元：D-SYN、O1、T1、B1、J1、R1；R1 正式栈固定为 `closure_cons_norew`。
- 输出：quick 与 full 目录强制隔离；完整 full run 自动生成 `manifest.json`。
- 数据范围：validation only；split id 2 保持锁定。

## 本地 smoke

本地 smoke 只能写入独立 quick 目录，产物状态固定为 `SMOKE`：

```bash
python experiments/final_wm/run_matrix.py --phase dsyn \
  --quick --device cpu --out /tmp/final_wm_v07_quick

python experiments/final_wm/run_matrix.py --phase matrix \
  --quick --device cpu \
  --record <canonical_sideA_v2.npz> --side A \
  --out /tmp/final_wm_v07_quick
```

quick、partial seeds、`--arm-filter` 均不会生成 authoritative manifest，也不得用于论文判决。

## 独立验收

Linux 回传后，本地只读验证：

```bash
python experiments/final_wm/audit_manifest.py \
  --manifest <returned-side>/manifest.json
```

任何输入/产物哈希不匹配、run/seed 缺失、D-SYN no-op、unit 未执行/仍为 SMOKE 或 test 解锁都会 fail-closed；完整执行所得科学 `INCOMPLETE` 会如实进入 manifest，不改判据补跑。
